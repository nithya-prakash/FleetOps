"""Control-plane API hardening: body limits, rate limiting, brute-force lockout, audited denials, metrics token."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from fleetops.config.settings import ControlPlaneSettings
from fleetops.domain.messages import DeviceModelState, Heartbeat, PredictionBatch, PredictionSample
from fleetops.domain.models import DeviceCapabilities, utcnow
from tests.harness import KEYS, Stack

CAPS = DeviceCapabilities(agent_version="0.1.0", formats=["mlp-v1"])


@pytest.fixture
def make_stack(tmp_path: Path, cp_settings: ControlPlaneSettings) -> Iterator[Any]:
    started: list[Stack] = []

    def build(**overrides: Any) -> Stack:
        s = Stack(tmp_path, cp_settings.model_copy(update=overrides)).start()
        started.append(s)
        return s

    yield build
    for s in started:
        s.stop()


def audit(stack: Stack, action: str) -> list[dict[str, Any]]:
    return list(stack.api("GET", f"/api/v1/audit/events?action={action}&limit=200", role="viewer").json()["items"])


# ---- request bodies ---------------------------------------------------------------------------------------


def test_streamed_oversize_body_is_rejected_even_without_content_length(make_stack: Any) -> None:
    stack = make_stack(max_request_bytes=10_000)

    def chunks() -> Iterator[bytes]:
        for _ in range(50):
            yield b"x" * 1000  # 50 kB sent chunked: no Content-Length header for the server to check

    r = httpx.post(
        stack.cp.url + "/api/v1/devices/register",
        headers={"Authorization": f"Bearer {KEYS['admin']}"},
        content=chunks(),
    )
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
    lying = httpx.post(
        stack.cp.url + "/api/v1/devices/register",
        headers={"Authorization": f"Bearer {KEYS['admin']}"},
        content=b"x" * 20_000,
    )
    assert lying.status_code == 413
    assert (
        'fleetops_requests_rejected_total{reason="too_large"} 2.0' in stack.api("GET", "/metrics", role="metrics").text
    )
    ok = stack.api("POST", "/api/v1/devices/register", role="admin", json={"device_id": "edge-01", "group": "line-a"})
    assert ok.status_code == 201  # normal requests are unaffected


def test_slow_body_is_cut_off_with_408(make_stack: Any) -> None:
    stack = make_stack(request_body_timeout_seconds=0.6)
    with socket.create_connection(("127.0.0.1", stack.cp.port), timeout=5) as sock:
        sock.sendall(
            b"POST /api/v1/devices/register HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            + f"Authorization: Bearer {KEYS['admin']}\r\n".encode()
            + b"Content-Length: 100\r\n\r\n{"  # promises 100 bytes, sends 1, then stalls
        )
        started = time.monotonic()
        reply = b""
        while b"request_timeout" not in reply:  # headers and body may arrive in separate segments
            data = sock.recv(4096)
            if not data:
                break
            reply += data
    assert b" 408 " in reply.split(b"\r\n", 1)[0] and b"request_timeout" in reply
    assert time.monotonic() - started < 4  # released the connection instead of waiting forever


# ---- rate limiting and brute force ---------------------------------------------------------------------------


def test_api_rate_limit_returns_429_with_retry_after_but_spares_probes(make_stack: Any) -> None:
    stack = make_stack(rate_limit_per_second=0.2, rate_limit_burst=5)
    statuses = [
        httpx.get(stack.cp.url + "/api/v1/devices", headers={"Authorization": f"Bearer {KEYS['viewer']}"}).status_code
        for _ in range(12)
    ]
    assert statuses[:5] == [200] * 5 and statuses.count(429) >= 6
    limited = httpx.get(stack.cp.url + "/api/v1/devices", headers={"Authorization": f"Bearer {KEYS['viewer']}"})
    assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1
    assert limited.json()["error"]["code"] == "rate_limited"
    assert httpx.get(stack.cp.url + "/healthz").status_code == 200  # liveness probes are not throttled
    text = httpx.get(stack.cp.url + "/metrics", headers={"Authorization": f"Bearer {KEYS['metrics']}"}).text
    assert 'fleetops_rate_limited_total{scope="api"}' in text


def test_repeated_authentication_failures_lock_the_client_out(make_stack: Any) -> None:
    stack = make_stack(auth_failure_limit_per_minute=3)
    bad = {"Authorization": "Bearer definitely-wrong-key"}
    codes = [httpx.get(stack.cp.url + "/api/v1/devices", headers=bad).status_code for _ in range(5)]
    assert codes == [401, 401, 401, 429, 429]
    good = httpx.get(stack.cp.url + "/api/v1/devices", headers={"Authorization": f"Bearer {KEYS['viewer']}"})
    assert good.status_code == 429 and "Retry-After" in good.headers  # even a valid key waits out the lockout
    text = httpx.get(stack.cp.url + "/metrics", headers={"Authorization": f"Bearer {KEYS['metrics']}"}).text
    assert 'fleetops_rate_limited_total{scope="auth_failures"}' in text


def test_insufficient_role_does_not_count_towards_lockout(make_stack: Any) -> None:
    stack = make_stack(auth_failure_limit_per_minute=3)
    for _ in range(6):  # a legitimate but under-privileged user retrying must not lock themselves out
        assert stack.api("POST", "/api/v1/devices/register", role="viewer", json={}).status_code == 403
    assert stack.api("GET", "/api/v1/devices", role="viewer").status_code == 200


# ---- audited denials -----------------------------------------------------------------------------------------


def test_unauthorized_management_actions_are_rejected_and_audited(stack: Stack) -> None:
    body = {"model_name": "anomaly-detector", "model_version": 1, "group": "line-a"}
    assert stack.api("POST", "/api/v1/deployments", role="viewer", json=body).status_code == 403
    assert stack.api("POST", "/api/v1/deployments/abc/rollback", role="viewer", json={}).status_code == 403
    assert httpx.post(stack.cp.url + "/api/v1/deployments/abc/cancel").status_code == 401
    wrong = httpx.post(
        stack.cp.url + "/api/v1/deployments", headers={"Authorization": "Bearer SUPER-SECRET-GUESS"}, json=body
    )
    assert wrong.status_code == 401

    events = audit(stack, "auth.denied")
    by_entity = {(e["actor"], e["entity_id"]): e["details"] for e in events}
    assert by_entity[("viewer", "POST /api/v1/deployments")]["required"] == "operator"
    assert by_entity[("viewer", "POST /api/v1/deployments/abc/rollback")]["reason"] == "insufficient role"
    assert by_entity[("anonymous", "POST /api/v1/deployments/abc/cancel")]["reason"] == "missing or invalid API key"
    assert all(e["details"]["client"] for e in events)
    everything = json.dumps(events)
    assert "SUPER-SECRET-GUESS" not in everything  # the presented credential is never recorded
    assert 'fleetops_auth_denied_total{reason="insufficient role"}' in stack.api("GET", "/metrics", role="metrics").text


def test_override_attempt_by_operator_is_rejected_and_audited(stack: Stack) -> None:
    stack.train()
    stack.add_agent("edge-01")
    stack.wait_online("edge-01")
    body = {
        "model_name": "anomaly-detector",
        "model_version": 1,
        "group": "line-a",
        "quality_gate": {"metrics": {"bench_f1": {"min": 0.0}}},
        "quality_override": {"reason": "operator trying to skip the gate"},
    }
    r = stack.api("POST", "/api/v1/deployments", role="operator", json=body)
    assert r.status_code == 403
    (event,) = audit(stack, "deployment.override_denied")
    assert event["actor"] == "operator" and event["details"]["requested_reason"] == "operator trying to skip the gate"
    assert stack.api("GET", "/api/v1/deployments", role="viewer").json()["total"] == 0  # nothing was created or sent


# ---- metrics / readiness exposure -----------------------------------------------------------------------------


def test_metrics_require_a_credential_and_the_token_grants_nothing_else(stack: Stack) -> None:
    url = stack.cp.url
    assert httpx.get(url + "/metrics").status_code == 401
    assert httpx.get(url + "/metrics", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert httpx.get(url + "/metrics", headers={"Authorization": f"Bearer {KEYS['metrics']}"}).status_code == 200
    assert (
        httpx.get(url + "/metrics", headers={"Authorization": f"Bearer {KEYS['viewer']}"}).status_code == 200
    )  # humans can curl
    # least privilege: the scrape token is not an API key
    assert stack.api("GET", "/api/v1/devices", role="metrics").status_code == 401
    assert stack.api("POST", "/api/v1/deployments", role="metrics", json={}).status_code == 401
    denied = audit(stack, "auth.denied")
    assert any(e["entity_id"] == "GET /metrics" for e in denied)


def test_readiness_is_minimal_for_anonymous_callers(stack: Stack) -> None:
    anon = httpx.get(stack.cp.url + "/readyz")
    assert anon.status_code == 200 and anon.json() == {"status": "ready"}
    detail = stack.api("GET", "/readyz", role="viewer").json()
    assert detail["database"] is True and detail["mlflow"] is True


def test_internal_errors_do_not_leak_details(stack: Stack, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("secret-internal-detail /srv/db/password=hunter2")

    monkeypatch.setattr("fleetops.controlplane.routes.session_scope", boom)
    r = stack.api("GET", "/api/v1/devices", role="viewer")
    assert r.status_code == 500 and r.json()["error"] == {
        "code": "internal_error",
        "message": "internal server error",
        "details": None,
    }
    assert "hunter2" not in r.text and "Traceback" not in r.text


# ---- device identity on the uplink -----------------------------------------------------------------------------


def test_revoked_device_cannot_feed_telemetry_predictions_or_state(stack: Stack) -> None:
    stack.api("POST", "/api/v1/devices/register", role="admin", json={"device_id": "edge-01", "group": "line-a"})
    assert stack.api("POST", "/api/v1/devices/edge-01/revoke", role="admin").status_code == 200
    ingest = stack.cp.server.config.app.state.ctx.ingest  # type: ignore[attr-defined]
    metrics = stack.cp.server.config.app.state.ctx.metrics  # type: ignore[attr-defined]
    sample = PredictionSample(
        ts=utcnow(),
        model_name="anomaly-detector",
        model_version=1,
        features=[1.0] * 6,
        score=0.1,
        anomaly=False,
        latency_ms=0.1,
    )
    ingest.handle(
        "fleetops/devices/edge-01/predictions",
        PredictionBatch(device_id="edge-01", sent_at=utcnow(), samples=[sample]).model_dump_json().encode(),
    )
    hb = Heartbeat(
        device_id="edge-01",
        sent_at=utcnow(),
        boot_id="b",
        agent_version="0.1.0",
        uptime_seconds=1,
        state=DeviceModelState(),
        capabilities=CAPS,
        cpu_percent=1,
        memory_mb=1,
    )
    ingest.handle("fleetops/devices/edge-01/heartbeat", hb.model_dump_json().encode())
    assert metrics.mqtt_failures.labels("predictions", "revoked")._value.get() == 1
    assert metrics.mqtt_failures.labels("heartbeat", "unregistered")._value.get() == 1
    assert stack.api("GET", "/api/v1/devices/edge-01", role="viewer").json()["status"] == "REVOKED"
    assert stack.api("POST", "/api/v1/monitoring/drift/run").json() == []  # no samples were stored


def test_key_fixture_sanity() -> None:
    assert len(set(KEYS.values())) == 4


# ---- device-facing artifact endpoint --------------------------------------------------------------------------------


def test_artifact_denials_and_bad_device_credentials_are_audited(stack: Stack) -> None:
    stack.train()
    agent = stack.add_agent("edge-01")
    stack.add_agent("edge-02")
    stack.wait_online("edge-01", "edge-02")
    path = "/api/v1/device/artifacts/anomaly-detector/1"
    good = {"Authorization": f"Bearer {agent.s.device_secret.get_secret_value()}", "X-Device-ID": "edge-01"}

    assert (
        httpx.get(stack.cp.url + path, headers={**good, "Authorization": "Bearer SECRET-GUESS-123"}).status_code == 401
    )
    assert httpx.get(stack.cp.url + path).status_code == 401
    assert httpx.get(stack.cp.url + path, headers=good).status_code == 403  # authentic device, but nothing assigned

    denied = audit(stack, "artifact.denied")
    assert (
        len(denied) == 1 and denied[0]["actor"] == "device:edge-01" and denied[0]["entity_id"] == "anomaly-detector:1"
    )
    rejected = {e["details"]["reason"] for e in audit(stack, "auth.denied")}
    assert {"invalid device credentials", "device credentials missing"} <= rejected
    assert "SECRET-GUESS-123" not in json.dumps(audit(stack, "auth.denied"))  # credentials never recorded
    assert stack.metric("fleetops_artifact_downloads_total", result="denied") == 1


def test_bad_device_credentials_trigger_the_same_lockout(make_stack: Any) -> None:
    stack = make_stack(auth_failure_limit_per_minute=3)
    path = stack.cp.url + "/api/v1/device/artifacts/anomaly-detector/1"
    bad = {"Authorization": "Bearer wrong", "X-Device-ID": "edge-01"}
    assert [httpx.get(path, headers=bad).status_code for _ in range(5)] == [401, 401, 401, 429, 429]
