"""Edge agent API with tokens configured: authentication scopes, size limits, rate limiting."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fleetops.edge.api import create_edge_app
from tests.unit.conftest import Bundle
from tests.unit.test_edge_agent import HEALTHY, SECRET, deploy_and_activate, make_agent

API, SCRAPE = "plc-gateway-token-0123456789", "scrape-token-0123456789abcd"
BODY = {"features": HEALTHY}


def client(tmp_path: Path, bundles: dict[int, Bundle], **overrides: object) -> TestClient:
    settings = {"api_token": API, "metrics_token": SCRAPE, **overrides}
    agent, _, _ = make_agent(tmp_path, bundles, settings_overrides=settings)
    deploy_and_activate(agent, bundles[1])
    return TestClient(create_edge_app(agent))


def auth(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token is not None else {}


def test_predict_and_healthz_require_the_api_token(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    c = client(tmp_path, bundles)
    for headers in ({}, auth("wrong"), auth(""), auth(SCRAPE), auth(SECRET), {"Authorization": API}):
        assert c.post("/predict", json=BODY, headers=headers).status_code == 401, headers
        assert c.get("/healthz", headers=headers).status_code == 401, headers
    ok = c.post("/predict", json=BODY, headers=auth(API))
    assert ok.status_code == 200 and ok.json()["model"]["version"] == 1
    assert c.get("/healthz", headers=auth(API)).json()["status"] == "ok"


def test_unauthenticated_requests_do_not_run_inference_or_leak_state(
    tmp_path: Path, bundles: dict[int, Bundle]
) -> None:
    c = client(tmp_path, bundles)
    r = c.post("/predict", json=BODY)
    assert r.status_code == 401 and "anomaly" not in r.text and "edge-01" not in r.text
    assert c.get("/livez").json() == {"status": "alive"}  # probes work without credentials and reveal nothing
    assert "edge-01" not in c.get("/livez").text
    metrics = c.get("/metrics", headers=auth(SCRAPE)).text
    assert 'fleetops_edge_inference_requests_total{device_id="edge-01",model_version="1"}' not in metrics  # nothing ran


def test_metrics_scrape_token_is_read_only(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    c = client(tmp_path, bundles)
    assert c.get("/metrics").status_code == 401 and c.get("/metrics", headers=auth("wrong")).status_code == 401
    assert c.get("/metrics", headers=auth(SCRAPE)).status_code == 200
    assert c.get("/metrics", headers=auth(API)).status_code == 200
    # least privilege: the scrape token cannot run inference, read state, or use admin controls
    assert c.post("/predict", json=BODY, headers=auth(SCRAPE)).status_code == 401
    assert c.get("/healthz", headers=auth(SCRAPE)).status_code == 401
    assert c.post("/admin/network", json={"connected": False}, headers=auth(SCRAPE)).status_code == 401


def test_admin_controls_need_the_device_credential_not_the_api_token(
    tmp_path: Path, bundles: dict[int, Bundle]
) -> None:
    c = client(tmp_path, bundles)
    assert c.post("/admin/network", json={"connected": False}, headers=auth(API)).status_code == 401
    assert c.post("/admin/network", json={"connected": False}, headers=auth(SECRET)).status_code == 200


def test_oversized_and_malformed_bodies_are_rejected_before_the_model(
    tmp_path: Path, bundles: dict[int, Bundle]
) -> None:
    c = client(tmp_path, bundles, max_request_bytes=2048)
    big = c.post(
        "/predict", content=b"[" + b"1," * 5000 + b"1]", headers={**auth(API), "content-type": "application/json"}
    )
    assert big.status_code == 413 and big.json()["error"]["code"] == "payload_too_large"
    assert c.post("/predict", json={"features": [1.0] * 300}, headers=auth(API)).status_code == 422  # schema cap too
    stats = c.get("/healthz", headers=auth(API)).json()["stats"]
    assert stats["inference_count"] == 0  # none of the rejected requests reached the model


def test_predict_is_rate_limited_without_affecting_probes(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    c = client(tmp_path, bundles, predict_rate_per_second=0.1, predict_burst=3)
    codes = [c.post("/predict", json=BODY, headers=auth(API)).status_code for _ in range(8)]
    assert codes[:3] == [200, 200, 200] and set(codes[3:]) == {429}
    limited = c.post("/predict", json=BODY, headers=auth(API))
    assert limited.headers["retry-after"] == "1"
    assert c.get("/livez").status_code == 200 and c.get("/healthz", headers=auth(API)).status_code == 200


def test_api_documentation_is_not_exposed_by_the_agent(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    c = client(tmp_path, bundles)
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert c.get(path, headers=auth(API)).status_code == 404


def test_without_tokens_the_agent_stays_open_for_development(tmp_path: Path, bundles: dict[int, Bundle]) -> None:
    agent, _, _ = make_agent(tmp_path, bundles)
    deploy_and_activate(agent, bundles[1])
    c = TestClient(create_edge_app(agent))
    assert c.post("/predict", json=BODY).status_code == 200 and c.get("/metrics").status_code == 200


@pytest.mark.parametrize("bad", ["Bearer", "Bearer ", "bearer x", "Basic abc", "Bearer " + "x" * 5000])
def test_odd_authorization_headers_are_just_401(tmp_path: Path, bundles: dict[int, Bundle], bad: str) -> None:
    c = client(tmp_path, bundles)
    assert c.post("/predict", json=BODY, headers={"Authorization": bad}).status_code == 401
