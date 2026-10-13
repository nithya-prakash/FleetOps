"""HTTPS end to end: control plane, agent downloads, agent API. Certificates are verified (chain + hostname)."""

from __future__ import annotations

import ssl
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from scripts.devcerts import KeyPair, issue, make_ca

from fleetops.config.settings import ControlPlaneSettings
from fleetops.edge.api import create_edge_app
from tests.harness import Stack, free_port, wait_until

HEALTHY = [2.0, 65.0, 6.0, 12.0, 1480.0, 70.0]
SANS = ["control-plane", "localhost", "127.0.0.1"]


def write(tmp: Path, name: str, data: bytes) -> Path:
    p = tmp / name
    p.write_bytes(data)
    return p


def ctx(ca: Path) -> ssl.SSLContext:
    return ssl.create_default_context(cafile=str(ca))


@contextmanager
def https_server(app: FastAPI, cert: Path, key: Path) -> Iterator[str]:
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning", ssl_certfile=str(cert), ssl_keyfile=str(key)
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_until(lambda: server.started, 10, msg="https server")
    try:
        yield f"https://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def tiny_app() -> FastAPI:
    app = FastAPI()

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"ok": "yes"}

    return app


def issued(tmp: Path, ca: KeyPair, names: list[str], **kw: object) -> tuple[Path, Path]:
    pair = issue(ca, "svc", names, **kw)  # type: ignore[arg-type]
    return write(tmp, f"{len(list(tmp.iterdir()))}.crt", pair.cert_pem), write(
        tmp, f"{len(list(tmp.iterdir()))}.key", pair.key_pem
    )


# ---- certificate verification on an HTTPS server --------------------------------------------------------------


def test_valid_certificate_is_accepted_and_every_defect_is_rejected(tmp_path: Path) -> None:
    ca, other = make_ca(), make_ca("Other CA")
    ca_file, other_file = write(tmp_path, "ca.crt", ca.cert_pem), write(tmp_path, "other.crt", other.cert_pem)
    past = datetime.now(UTC) - timedelta(days=2)
    cases = {
        "valid": (SANS, {}, ca_file, None),
        "wrong hostname": (["control-plane"], {}, ca_file, r"Hostname mismatch|IP address mismatch"),
        "untrusted CA": (SANS, {}, other_file, r"certificate verify failed"),
        "expired": (SANS, {"not_before": past - timedelta(days=30), "not_after": past}, ca_file, r"expired"),
    }
    for label, (names, kw, trust, expected_error) in cases.items():
        cert, key = issued(tmp_path, ca, names, **kw)
        with https_server(tiny_app(), cert, key) as url:
            if expected_error is None:
                assert httpx.get(url + "/ping", verify=ctx(trust)).json() == {"ok": "yes"}, label
                with pytest.raises(httpx.RequestError):  # plaintext HTTP to the TLS port is not served
                    httpx.get(url.replace("https", "http") + "/ping", timeout=3)
            else:
                with pytest.raises(httpx.ConnectError, match=expected_error):
                    httpx.get(url + "/ping", verify=ctx(trust), timeout=5)


def test_clients_do_not_fall_back_to_unverified_connections(tmp_path: Path) -> None:
    ca = make_ca()
    cert, key = issued(tmp_path, ca, SANS)
    with (
        https_server(tiny_app(), cert, key) as url,
        pytest.raises(httpx.ConnectError, match="certificate verify failed"),
    ):
        httpx.get(url + "/ping", timeout=5)  # default trust store does not contain the development CA


# ---- the whole platform over HTTPS ------------------------------------------------------------------------------


@pytest.fixture
def tls_stack(tmp_path: Path, cp_settings: ControlPlaneSettings) -> Iterator[Stack]:
    ca = make_ca()
    tls = tmp_path / "tls"
    tls.mkdir()
    cert, key = issued(tls, ca, SANS)
    settings = cp_settings.model_copy(update={"tls_certfile": cert, "tls_keyfile": key})
    stack = Stack(tmp_path, settings)
    stack.ca = write(tls, "ca.crt", ca.cert_pem)
    stack.start()
    try:
        yield stack
    finally:
        stack.stop()


def test_full_lifecycle_over_https_with_verified_artifact_download(tls_stack: Stack) -> None:
    stack = tls_stack
    assert stack.cp.url.startswith("https://")
    version = stack.train()
    agent = stack.add_agent("edge-01")  # the agent verifies the control plane certificate with the CA it was given
    stack.wait_online("edge-01")
    final = stack.wait_state(stack.deploy(version)["deployment_id"], {"COMPLETED", "FAILED", "ROLLED_BACK"})
    assert final["state"] == "COMPLETED", (
        final["reason"],
        [(t["device_id"], t["state"], t["last_error"]) for t in final["targets"]],
    )
    active = agent.store.snapshot().active
    assert active is not None and active.version == version
    assert agent.infer(HEALTHY).model.version == version
    assert (
        httpx.get(stack.cp.url + "/api/v1/devices", verify=ctx(stack.ca)).status_code == 401
    )  # TLS is not authentication  # type: ignore[arg-type]


def test_agent_refuses_a_control_plane_it_cannot_verify(tls_stack: Stack) -> None:
    stack = tls_stack
    version = stack.train()
    agent = stack.add_agent("edge-01", control_plane_ca=None)  # no trust anchor for the dev CA: system store only
    stack.wait_online("edge-01")
    final = stack.wait_state(stack.deploy(version)["deployment_id"], {"COMPLETED", "FAILED", "ROLLED_BACK"})
    target = final["targets"][0]
    assert final["state"] == "ROLLED_BACK" and target["state"] == "FAILED"
    assert "DOWNLOAD_FAILED" in target["last_error"] and "certificate" in target["last_error"].lower()
    snap = agent.store.snapshot()
    assert snap.active is None and snap.staged is None  # nothing was downloaded, staged or activated
    assert not list(agent.store.staging_dir.iterdir())


# ---- the agent's own API over HTTPS --------------------------------------------------------------------------------


def test_agent_api_over_https(tmp_path: Path) -> None:
    from fleetops.training.pipeline import TrainParams, train_model  # noqa: PLC0415
    from tests.unit.conftest import Bundle  # noqa: PLC0415
    from tests.unit.test_edge_agent import deploy_and_activate, make_agent  # noqa: PLC0415

    bundles = {1: Bundle(tmp_path, 1, train_model(TrainParams(n_samples=600, epochs=3, seed=1)))}
    agent, _, _ = make_agent(tmp_path, bundles, settings_overrides={"api_token": "plc-token-0123456789abcdef"})
    deploy_and_activate(agent, bundles[1])
    ca = make_ca()
    cert, key = issued(tmp_path, ca, ["edge-01", "localhost", "127.0.0.1"])
    ca_file = write(tmp_path, "agent-ca.crt", ca.cert_pem)
    with https_server(create_edge_app(agent), cert, key) as url:
        ok = httpx.post(
            url + "/predict",
            json={"features": HEALTHY},
            headers={"Authorization": "Bearer plc-token-0123456789abcdef"},
            verify=ctx(ca_file),
        )
        assert ok.status_code == 200
        assert httpx.get(url + "/livez", verify=ctx(ca_file)).json() == {"status": "alive"}
        with pytest.raises(httpx.ConnectError, match="certificate verify failed"):
            httpx.get(url + "/livez", timeout=5)  # unverifiable without the CA
