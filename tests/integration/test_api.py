from __future__ import annotations

import httpx

from tests.harness import KEYS, Stack, wait_until

SLOW = {"observation_seconds": 60, "min_inferences": 10, "ack_timeout_seconds": 30, "health_timeout_seconds": 120}


def test_authentication_and_authorization(stack: Stack) -> None:
    url = stack.cp.url
    assert httpx.get(f"{url}/api/v1/devices").status_code == 401
    assert httpx.get(f"{url}/api/v1/devices", headers={"Authorization": "Bearer nope"}).status_code == 401
    body = {"device_id": "edge-01", "group": "line-a"}
    assert stack.api("POST", "/api/v1/devices/register", role="viewer", json=body).status_code == 403
    assert stack.api("POST", "/api/v1/devices/register", role="operator", json=body).status_code == 403
    assert stack.api("GET", "/api/v1/devices", role="viewer").status_code == 200
    assert stack.api("POST", "/api/v1/models/train", role="viewer", json={}).status_code == 403
    assert stack.api("POST", "/api/v1/devices/edge-01/simulation", role="operator", json={}).status_code == 403
    assert httpx.get(f"{url}/healthz").status_code == 200  # liveness is unauthenticated


def test_consistent_error_format_and_validation(stack: Stack) -> None:
    r = stack.api("GET", "/api/v1/devices/missing", role="viewer")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = stack.api("POST", "/api/v1/devices/register", role="admin", json={"device_id": "../x", "group": "line-a"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    r = stack.api(
        "POST", "/api/v1/devices/register", role="admin", json={"device_id": "edge-01", "group": "a", "extra": 1}
    )
    assert r.status_code == 422  # unknown fields rejected
    r = stack.api("GET", "/api/v1/devices?limit=1000", role="viewer")
    assert r.status_code == 422
    big = stack.api(
        "POST",
        "/api/v1/devices/register",
        role="admin",
        content=b"x" * 2_000_000,
        headers={"Content-Type": "application/json"},
    )
    assert big.status_code == 413


def test_device_registration_listing_and_pagination(stack: Stack) -> None:
    for i in range(1, 4):
        r = stack.api(
            "POST", "/api/v1/devices/register", role="admin", json={"device_id": f"edge-0{i}", "group": "line-a"}
        )
        assert r.status_code == 201 and r.json()["credentials"]["secret"]
    assert (
        stack.api(
            "POST", "/api/v1/devices/register", role="admin", json={"device_id": "edge-01", "group": "line-a"}
        ).status_code
        == 409
    )
    stack.api("POST", "/api/v1/devices/register", role="admin", json={"device_id": "press-01", "group": "line-b"})
    page = stack.api("GET", "/api/v1/devices?limit=2&offset=1", role="viewer").json()
    assert page["total"] == 4 and [d["device_id"] for d in page["items"]] == ["edge-02", "edge-03"]
    assert stack.api("GET", "/api/v1/devices?group=line-b", role="viewer").json()["total"] == 1
    assert "secret" not in stack.api("GET", "/api/v1/devices/edge-01", role="viewer").text


def test_models_train_list_and_versions(stack: Stack) -> None:
    assert stack.train() == 1 and stack.train(seed=3) == 2
    models = stack.api("GET", "/api/v1/models", role="viewer").json()
    assert models["items"] == [{"name": "anomaly-detector", "latest_version": 2, "version_count": 2}]
    versions = stack.api("GET", "/api/v1/models/anomaly-detector/versions", role="viewer").json()
    assert [v["version"] for v in versions["items"]] == [2, 1]
    v = versions["items"][0]
    assert len(v["sha256"]) == 64 and "test_f1" in v["metrics"] and v["compatibility"]["format"] == "mlp-v1"
    assert "path" not in str(v).lower().replace("compat", "")  # no filesystem paths exposed
    assert stack.api("GET", "/api/v1/models/nope/versions", role="viewer").status_code == 404
    r = stack.api("POST", "/api/v1/models/train", json={"params": {"epochs": 9999}})
    assert r.status_code == 422


def test_deployment_idempotency_conflicts_and_eligibility(stack: Stack) -> None:
    v1 = stack.train(seed=1)
    stack.add_agent("edge-01")
    stack.wait_online("edge-01")
    body = {"model_name": "anomaly-detector", "model_version": v1, "group": "line-a", "health_criteria": SLOW}
    first = stack.api("POST", "/api/v1/deployments", json=body, headers={"Idempotency-Key": "k-1"})
    assert first.status_code == 201
    replay = stack.api("POST", "/api/v1/deployments", json=body, headers={"Idempotency-Key": "k-1"})
    assert replay.status_code == 200 and replay.json()["deployment_id"] == first.json()["deployment_id"]
    other = stack.api(
        "POST",
        "/api/v1/deployments",
        json={**body, "model_version": 1, "group": "line-b"},
        headers={"Idempotency-Key": "k-1"},
    )
    assert other.status_code == 409
    assert stack.api("GET", "/api/v1/deployments", role="viewer").json()["total"] == 1
    # unknown model / unknown group / unknown device
    assert stack.api("POST", "/api/v1/deployments", json={**body, "model_version": 99}).status_code == 404
    assert stack.api("POST", "/api/v1/deployments", json={**body, "group": "ghost"}).status_code == 422
    assert stack.api("POST", "/api/v1/deployments", json={**body, "device_ids": ["nope-1"]}).status_code == 422


def test_new_deployment_supersedes_old_and_cancel(stack: Stack) -> None:
    v1, v2 = stack.train(seed=1), stack.train(seed=2)
    for d in ("edge-01", "edge-02"):
        stack.add_agent(d)
    stack.wait_online("edge-01", "edge-02")
    first = stack.deploy(v1, health_criteria=SLOW)
    wait_until(lambda: stack.deployment(first["deployment_id"])["targets"][0]["state"] == "ACTIVE", msg="canary active")
    second = stack.deploy(v2, health_criteria=SLOW)
    old = stack.deployment(first["deployment_id"])
    assert old["state"] == "CANCELLED" and old["superseded_by"] == second["deployment_id"]
    # the superseded rollout can no longer promote: edge-02 never received v1
    wait_until(
        lambda: stack.deployment(second["deployment_id"])["targets"][0]["state"] == "ACTIVE", msg="v2 canary active"
    )
    assert stack.agents["edge-02"].store.snapshot().active is None
    r = stack.api("POST", f"/api/v1/deployments/{second['deployment_id']}/cancel")
    assert r.status_code == 202 and r.json()["state"] == "CANCELLED"
    assert stack.api("POST", f"/api/v1/deployments/{second['deployment_id']}/cancel").status_code == 409
    assert stack.agents["edge-02"].store.snapshot().active is None  # cancelled before promotion


def test_artifact_download_is_authenticated_and_least_privilege(stack: Stack) -> None:
    v1 = stack.train()
    agent = stack.add_agent("edge-01")
    stack.add_agent("edge-02")
    stack.wait_online("edge-01", "edge-02")
    path = f"/api/v1/device/artifacts/anomaly-detector/{v1}"
    secret = agent.s.device_secret.get_secret_value()
    hdr = {"Authorization": f"Bearer {secret}", "X-Device-ID": "edge-01"}
    assert httpx.get(stack.cp.url + path).status_code == 401  # anonymous
    assert httpx.get(stack.cp.url + path, headers={**hdr, "Authorization": "Bearer wrong"}).status_code == 401
    assert (
        httpx.get(stack.cp.url + path, headers={**hdr, "X-Device-ID": "edge-02"}).status_code == 401
    )  # secret is per-device
    assert httpx.get(stack.cp.url + path, headers=hdr).status_code == 403  # authentic but nothing assigned yet
    assert stack.api("GET", path, role="admin").status_code == 401  # API keys are not device credentials
    dep = stack.deploy(v1, device_ids=["edge-01"], health_criteria=SLOW)
    r = httpx.get(stack.cp.url + path, headers=hdr)
    sha = stack.api("GET", "/api/v1/models/anomaly-detector/versions", role="viewer").json()["items"][0]["sha256"]
    assert r.status_code == 200 and r.headers["etag"].strip('"') == sha and dep["deployment_id"]
    other = {
        "Authorization": f"Bearer {stack.agents['edge-02'].s.device_secret.get_secret_value()}",
        "X-Device-ID": "edge-02",
    }
    assert httpx.get(stack.cp.url + path, headers=other).status_code == 403  # not a target of the deployment
    assert httpx.get(
        stack.cp.url + "/api/v1/device/artifacts/anomaly-detector/..%2f..%2fetc", headers=hdr
    ).status_code in (404, 422)
    # revoked devices lose access immediately
    assert stack.api("POST", "/api/v1/devices/edge-01/revoke", role="admin").status_code == 200
    assert httpx.get(stack.cp.url + path, headers=hdr).status_code == 401


def test_simulation_endpoint_disabled_by_config(stack: Stack) -> None:
    stack.cp.stop()
    stack.settings.simulation_enabled = False
    stack.restart_control_plane()
    stack.api("POST", "/api/v1/devices/register", role="admin", json={"device_id": "edge-01", "group": "line-a"})
    assert stack.api("POST", "/api/v1/devices/edge-01/simulation", role="admin", json={}).status_code == 403


def test_drift_alert_creates_review_recommendation_without_deploying(stack: Stack) -> None:
    v1 = stack.train(seed=1, n_samples=3000)
    stack.add_agent("edge-01", sensor_scenario="drift", sensor_seed=5, label_fraction=0.5)
    stack.wait_online("edge-01")
    stack.wait_state(stack.deploy(v1)["deployment_id"], {"COMPLETED"})

    def alert() -> bool:
        reports = stack.api("POST", "/api/v1/monitoring/drift/run").json()
        return any(r["status"] == "ALERT" for r in reports)

    wait_until(alert, timeout=30, interval=0.5, msg="drift alert")
    rep = stack.api("GET", "/api/v1/monitoring/drift?status=alert", role="viewer").json()["items"][0]
    flagged = {f["feature"] for f in rep["features"] if f["status"] == "ALERT"}
    assert "temperature_c" in flagged and "pressure_bar" not in flagged
    assert rep["quality"]["measured"] is True and "does not by itself mean" in rep["quality"]["note"]
    recs = stack.api("GET", "/api/v1/monitoring/recommendations?status=open", role="viewer").json()
    assert recs["total"] == 1 and "No automatic deployment" in recs["items"][0]["message"]
    assert stack.api("GET", "/api/v1/deployments", role="viewer").json()["total"] == 1  # nothing was auto-deployed
    rid = recs["items"][0]["id"]
    assert (
        stack.api("POST", f"/api/v1/monitoring/recommendations/{rid}/resolve", json={"action": "acknowledge"}).json()[
            "status"
        ]
        == "ACKNOWLEDGED"
    )
    assert (
        stack.api("POST", f"/api/v1/monitoring/recommendations/{rid}/resolve", json={"action": "dismiss"}).status_code
        == 409
    )


def test_no_drift_alert_for_normal_traffic(stack: Stack) -> None:
    v1 = stack.train(seed=1, n_samples=3000)
    stack.add_agent("edge-01", sensor_scenario="normal", sensor_seed=9)
    stack.wait_online("edge-01")
    stack.wait_state(stack.deploy(v1)["deployment_id"], {"COMPLETED"})
    seen: list[str] = []

    def enough() -> bool:
        seen[:] = [r["status"] for r in stack.api("POST", "/api/v1/monitoring/drift/run").json()]
        return bool(seen) and seen[-1] != "INSUFFICIENT_DATA"

    wait_until(enough, timeout=30, interval=0.5, msg="enough samples")
    assert seen[-1] in ("OK", "WARN")
    assert stack.api("GET", "/api/v1/monitoring/recommendations", role="viewer").json()["total"] == 0


def test_keys_fixture_sanity() -> None:
    assert len(set(KEYS.values())) == 4
