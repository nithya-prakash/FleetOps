"""Edge agent HTTP API (/predict, /healthz, /metrics, /admin/network) via FastAPI TestClient."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fleetops.domain.messages import CommandType
from fleetops.edge.api import create_edge_app
from tests.helpers import faults
from tests.unit.conftest import Bundle
from tests.unit.test_edge_agent import (
    HEALTHY,
    SECRET,
    deploy_and_activate,
    make_agent,
    run,
)


@pytest.fixture
def client_no_model(tmp_path: Path, bundles: dict[int, Bundle]) -> TestClient:
    agent, _, _ = make_agent(tmp_path, bundles)
    return TestClient(create_edge_app(agent))


@pytest.fixture
def served(tmp_path: Path, bundles: dict[int, Bundle]):  # type: ignore[no-untyped-def]
    agent, bus, _ = make_agent(tmp_path, bundles)
    deploy_and_activate(agent, bundles[1])
    return agent, TestClient(create_edge_app(agent)), bus


def test_predict_without_model_is_503_not_a_wrong_answer(client_no_model: TestClient) -> None:
    r = client_no_model.post("/predict", json={"features": HEALTHY})
    assert r.status_code == 503 and r.json() == {"detail": "no active model"}
    h = client_no_model.get("/healthz").json()
    assert h["status"] == "no_model" and h["active"] is None and h["stats"] is None


def test_valid_prediction_runs_locally_and_reports_model(served) -> None:  # type: ignore[no-untyped-def]
    agent, client, _ = served
    r = client.post("/predict", json={"features": HEALTHY})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"score", "anomaly", "latency_ms", "model"} and body["model"] == {
        "name": "anomaly-detector",
        "version": 1,
    }
    assert 0.0 <= body["score"] <= 1.0 and isinstance(body["anomaly"], bool)
    h = client.get("/healthz").json()
    assert h["status"] == "ok" and h["active"]["version"] == 1 and h["stats"]["inference_count"] == 1
    assert agent.engine.stats().inference_count == 1  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        ({"features": []}, "at least 1"),
        ({"features": "abc"}, "valid list"),
        ({"features": [1.0, 2.0]}, "expected 6 features, got 2"),
        ({"features": [1.0] * 7}, "expected 6 features, got 7"),
        ({"features": [1.0] * 300}, "at most 256"),
        ({"features": [2.0, 65.0, 6.0, 12.0, 1480.0, 1e9]}, "outside plausible range"),
        ({"features": [2.0, 65.0, 6.0, 12.0, 1480.0, None]}, "valid number"),
        ({}, "required"),
    ],
)
def test_malformed_or_implausible_input_is_422_and_model_unaffected(served, payload: dict, needle: str) -> None:  # type: ignore[no-untyped-def,type-arg]
    agent, client, _ = served
    r = client.post("/predict", json=payload)
    assert r.status_code == 422 and needle in r.text
    stats = agent.engine.stats()
    assert stats is not None and stats.inference_errors == 0  # bad input is not a model failure
    assert client.post("/predict", json={"features": HEALTHY}).status_code == 200  # still serving


def test_non_finite_and_non_json_bodies(served) -> None:  # type: ignore[no-untyped-def]
    _, client, _ = served
    r = client.post(
        "/predict", content='{"features": [NaN, 1, 1, 1, 1, 1]}', headers={"content-type": "application/json"}
    )
    assert r.status_code == 422 and "NaN" in r.text
    assert client.post("/predict", content=b"not json", headers={"content-type": "application/json"}).status_code == 422
    assert client.post("/predict", content=b"\xff\xfe", headers={"content-type": "application/json"}).status_code == 422
    assert client.get("/predict").status_code == 405


def test_model_failure_is_500_without_leaking_internals(served) -> None:  # type: ignore[no-untyped-def]
    agent, client, _ = served
    assert run(agent, CommandType.SET_FAULTS, faults=faults(fail_inference_versions=[1])).outcome.value == "ACCEPTED"
    r = client.post("/predict", json={"features": HEALTHY})
    assert r.status_code == 500 and r.json() == {"detail": "inference failed"}
    assert "injected" not in r.text and "Traceback" not in r.text
    stats = agent.engine.stats()
    assert stats is not None and (stats.inference_count, stats.inference_errors) == (1, 1)
    metrics = client.get("/metrics").text
    assert 'fleetops_edge_inference_failures_total{device_id="edge-01",model_version="1"} 1.0' in metrics


def test_metrics_endpoint_exposes_expected_series(served) -> None:  # type: ignore[no-untyped-def]
    _, client, _ = served
    for _ in range(3):
        client.post("/predict", json={"features": HEALTHY})
    client.post("/predict", json={"features": [1.0]})  # rejected input
    text = client.get("/metrics").text
    assert 'fleetops_edge_inference_requests_total{device_id="edge-01",model_version="1"} 3.0' in text
    assert 'fleetops_edge_inputs_rejected_total{device_id="edge-01"} 1.0' in text
    assert 'fleetops_edge_model_version{device_id="edge-01",model="anomaly-detector"} 1.0' in text
    assert "fleetops_edge_inference_latency_seconds_bucket" in text


def test_admin_network_requires_the_device_credential(served) -> None:  # type: ignore[no-untyped-def]
    agent, client, bus = served
    assert client.post("/admin/network", json={"connected": False}).status_code == 401
    assert (
        client.post("/admin/network", json={"connected": False}, headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )
    assert agent.network_enabled and bus.connected  # unauthenticated calls changed nothing
    ok = client.post("/admin/network", json={"connected": False}, headers={"Authorization": f"Bearer {SECRET}"})
    assert ok.status_code == 200 and ok.json() == {"network_enabled": False}
    assert not bus.connected
    assert client.post("/predict", json={"features": HEALTHY}).status_code == 200  # inference unaffected by the outage
    assert client.get("/healthz").json()["network_enabled"] is False
    client.post("/admin/network", json={"connected": True}, headers={"Authorization": f"Bearer {SECRET}"})
    assert bus.connected


def test_admin_network_disabled_when_simulation_not_allowed(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles, allow_faults=False)
    client = TestClient(create_edge_app(agent))
    r = client.post("/admin/network", json={"connected": False}, headers={"Authorization": f"Bearer {SECRET}"})
    assert r.status_code == 403 and agent.network_enabled
    assert (
        client.post(
            "/admin/network", json={"connected": "banana"}, headers={"Authorization": f"Bearer {SECRET}"}
        ).status_code
        == 422
    )
