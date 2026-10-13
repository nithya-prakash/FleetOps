"""MLflow adapter failure paths: HTTP 5xx, auth rejection, hangs/timeouts, partial failures (compensation)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from sqlalchemy import select

from fleetops.config.settings import ControlPlaneSettings
from fleetops.persistence.db import make_engine, make_session_factory, session_scope, upgrade_database
from fleetops.persistence.tables import ModelVersion
from fleetops.registry.artifact_store import ArtifactStore, ArtifactStoreError
from fleetops.registry.mlflow_registry import MlflowRegistry, RegistryUnavailableError
from fleetops.registry.service import ModelRegistrationService
from fleetops.training.pipeline import TrainParams, train_model

MAX_SECONDS = 10.0  # every failure mode must surface within this bound


class FakeMlflow:
    """Local HTTP server with a configurable misbehaviour."""

    def __init__(self, behaviour: Callable[[BaseHTTPRequestHandler], None]) -> None:
        outer = behaviour

        class Handler(BaseHTTPRequestHandler):
            def _serve(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                if length:
                    self.rfile.read(length)
                outer(self)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _serve  # noqa: N815

            def log_message(self, *args: object) -> None:
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True  # never block interpreter/test exit on a deliberately hung handler
            block_on_close = False

        self.server = Server(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> FakeMlflow:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()


def respond(status: int, body: dict | None = None) -> Callable[[BaseHTTPRequestHandler], None]:  # type: ignore[type-arg]
    def handler(h: BaseHTTPRequestHandler) -> None:
        payload = json.dumps(body or {"error_code": "INTERNAL_ERROR", "message": "boom"}).encode()
        h.send_response(status)
        h.send_header("content-type", "application/json")
        h.send_header("content-length", str(len(payload)))
        h.end_headers()
        h.wfile.write(payload)

    return handler


def hang(h: BaseHTTPRequestHandler) -> None:
    time.sleep(40)  # far longer than MAX_SECONDS: only a working client timeout lets the test pass


@pytest.fixture(autouse=True)
def fast_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "1")
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)


@pytest.fixture(scope="module")
def trained():  # type: ignore[no-untyped-def]
    return train_model(TrainParams(n_samples=300, epochs=1))


@pytest.fixture
def service(cp_settings: ControlPlaneSettings, tmp_path: Path):  # type: ignore[no-untyped-def]
    upgrade_database(cp_settings.database_url)
    sessions = make_session_factory(make_engine(cp_settings.database_url))
    store = ArtifactStore(cp_settings.artifact_root)

    def build(registry: MlflowRegistry) -> ModelRegistrationService:
        return ModelRegistrationService(sessions, store, registry)

    return build, sessions, store


def nothing_registered(sessions, store: ArtifactStore) -> None:  # type: ignore[no-untyped-def]
    with session_scope(sessions) as db:
        assert db.scalars(select(ModelVersion)).all() == []
    with pytest.raises(ArtifactStoreError):
        store.get("anomaly-detector", 1)


@pytest.mark.parametrize("status", [500, 502, 503])
def test_server_errors_fail_fast_and_register_nothing(service, trained, status: int) -> None:  # type: ignore[no-untyped-def]
    build, sessions, store = service
    with FakeMlflow(respond(status)) as srv:
        start = time.monotonic()
        with pytest.raises(RegistryUnavailableError, match="MLflow registration failed"):
            build(MlflowRegistry(srv.url, "x")).register_training_result("anomaly-detector", trained)
        assert time.monotonic() - start < MAX_SECONDS
    nothing_registered(sessions, store)


def test_authentication_failure_is_reported_and_registers_nothing(service, trained) -> None:  # type: ignore[no-untyped-def]
    build, sessions, store = service
    unauthorized = respond(401, {"error_code": "UNAUTHENTICATED", "message": "You are not authenticated."})
    with FakeMlflow(unauthorized) as srv:
        with pytest.raises(RegistryUnavailableError) as exc:
            build(MlflowRegistry(srv.url, "x")).register_training_result("anomaly-detector", trained)
        assert "401" in str(exc.value) or "authenticated" in str(exc.value).lower()
    nothing_registered(sessions, store)


def test_hanging_server_times_out_within_bound(service, trained) -> None:  # type: ignore[no-untyped-def]
    build, sessions, store = service
    with FakeMlflow(hang) as srv:
        start = time.monotonic()
        with pytest.raises(RegistryUnavailableError):
            build(MlflowRegistry(srv.url, "x")).register_training_result("anomaly-detector", trained)
        elapsed = time.monotonic() - start
    assert elapsed < MAX_SECONDS, f"timeout not enforced ({elapsed:.1f}s)"
    nothing_registered(sessions, store)


def test_ping_reflects_availability_without_raising() -> None:
    with FakeMlflow(respond(200, {"experiments": []})) as ok:
        assert MlflowRegistry(ok.url, "x").ping() is True
    with FakeMlflow(respond(500)) as bad:
        assert MlflowRegistry(bad.url, "x").ping() is False
    with FakeMlflow(hang) as slow:
        start = time.monotonic()
        assert MlflowRegistry(slow.url, "x").ping() is False
        assert time.monotonic() - start < MAX_SECONDS
    assert MlflowRegistry("http://127.0.0.1:1", "x").ping() is False


def test_list_versions_failure_is_wrapped() -> None:
    with FakeMlflow(respond(500)) as srv, pytest.raises(RegistryUnavailableError):
        MlflowRegistry(srv.url, "x").list_versions("anomaly-detector")


@pytest.fixture
def local_registry(cp_settings: ControlPlaneSettings, tmp_path: Path) -> MlflowRegistry:
    return MlflowRegistry(
        cp_settings.mlflow_tracking_uri, "failure-tests", artifact_location=(tmp_path / "mlruns").as_uri()
    )


def test_failure_creating_the_version_marks_the_run_failed(service, trained, local_registry, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    build, sessions, store = service

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("registry write failed")

    monkeypatch.setattr(local_registry._client, "create_model_version", boom)
    with pytest.raises(RegistryUnavailableError, match="registry write failed"):
        build(local_registry).register_training_result("anomaly-detector", trained)
    runs = local_registry._client.search_runs([local_registry._experiment()])
    assert [r.info.status for r in runs] == ["FAILED"]  # the run is not left looking successful
    nothing_registered(sessions, store)


def test_artifact_store_failure_after_registration_compensates(service, trained, local_registry, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    build, sessions, store = service

    def disk_full(*_a: object, **_k: object) -> None:
        raise ArtifactStoreError("no space left on device")

    monkeypatch.setattr(store, "put", disk_full)
    with pytest.raises(ArtifactStoreError):
        build(local_registry).register_training_result("anomaly-detector", trained)
    assert local_registry.list_versions("anomaly-detector") == []  # MLflow version removed (compensation)
    with session_scope(sessions) as db:
        assert db.scalars(select(ModelVersion)).all() == []


def test_database_failure_after_registration_compensates(service, trained, local_registry, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    build, _sessions, store = service
    svc = build(local_registry)

    def failing_scope(_factory: object) -> Iterator[None]:
        raise RuntimeError("database is locked")
        yield

    monkeypatch.setattr("fleetops.registry.service.session_scope", contextmanager(failing_scope))
    with pytest.raises(RuntimeError, match="database is locked"):
        svc.register_training_result("anomaly-detector", trained)
    assert local_registry.list_versions("anomaly-detector") == []
    with pytest.raises(ArtifactStoreError):
        store.get("anomaly-detector", 1)  # artifact removed too
