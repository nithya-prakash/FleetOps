"""End-to-end: real control plane + MLflow + SQLite + edge agents (in-process MQTT broker)."""

from __future__ import annotations

from tests.harness import Stack, wait_until

HEALTHY = [2.0, 65.0, 6.0, 12.0, 1480.0, 70.0]
GROUP = ["edge-01", "edge-02", "edge-03"]


def active_version(stack: Stack, device_id: str) -> int | None:
    st = stack.agents[device_id].store.snapshot().active
    return st.version if st else None


def setup_fleet(stack: Stack) -> tuple[int, int]:
    v1, v2 = stack.train(seed=1), stack.train(seed=2)
    assert (v1, v2) == (1, 2)
    for d in GROUP:
        stack.add_agent(d)
    stack.wait_online(*GROUP)
    return v1, v2


def test_canary_failure_halts_and_rolls_back_to_previous_model(stack: Stack) -> None:
    v1, v2 = setup_fleet(stack)

    # 1-3: deploy A to all devices and verify local inference on each
    dep_a = stack.deploy(v1, strategy={"canary_count": 1, "batch_percent": 100})
    assert stack.wait_state(dep_a["deployment_id"], {"COMPLETED"})["state"] == "COMPLETED"
    for d in GROUP:
        r = stack.agents[d].infer(HEALTHY)
        assert r.model.version == v1

    # 6: deterministic failure on the canary (edge-01 sorts first): v2 inference always fails
    r = stack.api("POST", "/api/v1/devices/edge-01/simulation", role="admin", json={"fail_inference_versions": [v2]})
    assert r.status_code == 202
    wait_until(lambda: stack.agents["edge-01"].engine.faults.fail_inference_versions == [v2], msg="faults applied")

    # 5: roll out B with a canary
    dep_b = stack.deploy(v2, strategy={"canary_count": 1, "batch_percent": 100})
    final = stack.wait_state(dep_b["deployment_id"], {"ROLLED_BACK", "FAILED", "COMPLETED"})

    # 7: promotion stopped; 8: canary reverted; untouched devices never left A
    assert final["state"] == "ROLLED_BACK", final
    targets = {t["device_id"]: t for t in final["targets"]}
    assert targets["edge-01"]["state"] == "ROLLED_BACK"
    assert "health check failed" in (targets["edge-01"]["last_error"] or "")
    assert targets["edge-02"]["state"] == targets["edge-03"]["state"] == "CANCELLED"
    for d in ("edge-02", "edge-03"):
        assert stack.agents[d].store.snapshot().active.version == v1  # type: ignore[union-attr]
        assert stack.agents[d].store.snapshot().staged is None

    # 9: version A is genuinely loaded on the canary and producing predictions
    agent = stack.agents["edge-01"]
    assert active_version(stack, "edge-01") == v1
    assert agent.infer(HEALTHY).model.version == v1
    assert agent.engine.active is not None and agent.engine.active.version == v1

    # 10: events persisted (and queryable) with the rollback reason
    events = stack.api(
        "GET", f"/api/v1/audit/events?deployment_id={dep_b['deployment_id']}&limit=200", role="viewer"
    ).json()
    actions = {e["action"] for e in events["items"]}
    assert {
        "deployment.created",
        "deployment.rolling_back",
        "deployment.rolled_back",
        "target.health_failed",
    } <= actions
    rb = next(e for e in events["items"] if e["action"] == "deployment.rolling_back")
    assert "edge-01" in rb["details"]["reason"] and "error rate" in rb["details"]["reason"]

    # metrics reflect the rollback
    text = stack.api("GET", "/metrics", role="viewer").text
    assert 'fleetops_rollbacks_total{trigger="automatic"} 1.0' in text


def test_successful_staged_rollout_promotes_all_batches(stack: Stack) -> None:
    v1, v2 = setup_fleet(stack)
    stack.wait_state(stack.deploy(v1)["deployment_id"], {"COMPLETED"})
    dep = stack.deploy(v2, strategy={"canary_count": 1, "batch_percent": 50})
    final = stack.wait_state(dep["deployment_id"], {"COMPLETED", "ROLLED_BACK", "FAILED"})
    assert final["state"] == "COMPLETED", final
    assert [t["state"] for t in final["targets"]] == ["ACTIVE"] * 3
    assert all(t["verified_at"] for t in final["targets"])
    assert {active_version(stack, d) for d in GROUP} == {v2}
    # previous model retained on every device as last known-good
    assert all(stack.agents[d].store.snapshot().previous.version == v1 for d in GROUP)  # type: ignore[union-attr]
    # operator-initiated rollback of a completed deployment really switches models back
    r = stack.api("POST", f"/api/v1/deployments/{dep['deployment_id']}/rollback", json={"reason": "operator decision"})
    assert r.status_code == 202
    final = stack.wait_state(dep["deployment_id"], {"ROLLED_BACK", "FAILED"})
    assert final["state"] == "ROLLED_BACK"
    assert {active_version(stack, d) for d in GROUP} == {v1}


def test_corrupted_artifact_never_activates(stack: Stack) -> None:
    v1, v2 = setup_fleet(stack)
    stack.wait_state(stack.deploy(v1)["deployment_id"], {"COMPLETED"})
    stack.api("POST", "/api/v1/devices/edge-01/simulation", role="admin", json={"corrupt_download_versions": [v2]})
    wait_until(lambda: stack.agents["edge-01"].engine.faults.corrupt_download_versions == [v2])
    dep = stack.deploy(v2)
    final = stack.wait_state(dep["deployment_id"], {"ROLLED_BACK", "FAILED", "COMPLETED"})
    assert final["state"] == "ROLLED_BACK"
    t = {x["device_id"]: x for x in final["targets"]}["edge-01"]
    assert t["state"] == "FAILED" and "CHECKSUM_MISMATCH" in t["last_error"]
    assert {active_version(stack, d) for d in GROUP} == {v1}


def test_offline_device_keeps_inferring_and_reconciles_on_reconnect(stack: Stack) -> None:
    v1, _ = setup_fleet(stack)
    stack.wait_state(stack.deploy(v1)["deployment_id"], {"COMPLETED"})
    agent = stack.agents["edge-02"]

    agent.set_network(False)  # 11: disconnect
    for _ in range(20):
        assert agent.infer(HEALTHY).model.version == v1  # local inference unaffected
    wait_until(
        lambda: stack.api("GET", "/api/v1/devices/edge-02/health", role="viewer").json()["connectivity"] == "offline",
        timeout=15,
        msg="offline detected",
    )  # LWT marks it immediately
    health = stack.api("GET", "/api/v1/devices/edge-02/health", role="viewer").json()
    assert health["assessment"] == "offline"

    agent.set_network(True)  # 12: reconnect
    wait_until(
        lambda: stack.api("GET", "/api/v1/devices/edge-02", role="viewer").json()["status"] == "ONLINE",
        msg="back online",
    )
    h = stack.api("GET", "/api/v1/devices/edge-02/health", role="viewer").json()
    assert h["active_model"]["version"] == v1  # state reconciled: control plane sees the true model
    assert agent.infer(HEALTHY).model.version == v1


def test_control_plane_restart_resumes_in_flight_rollout(stack: Stack) -> None:
    v1, _ = setup_fleet(stack)
    dep_id = stack.deploy(v1, strategy={"canary_count": 1, "batch_percent": 100})["deployment_id"]
    stack.restart_control_plane()  # in-flight rollout state lives in the database, acks queue in the broker
    stack.wait_online(*GROUP)
    final = stack.wait_state(dep_id, {"COMPLETED", "ROLLED_BACK", "FAILED"})
    assert final["state"] == "COMPLETED", final
    assert {active_version(stack, d) for d in GROUP} == {v1}
    events = stack.api("GET", f"/api/v1/audit/events?deployment_id={dep_id}&limit=200", role="viewer").json()
    assert any(e["action"] == "deployment.created" for e in events["items"])  # survived the restart


def test_lost_acknowledgements_are_reconciled_from_heartbeats(stack: Stack) -> None:
    v1, _ = setup_fleet(stack)
    stack.api("POST", "/api/v1/devices/edge-01/simulation", role="admin", json={"drop_acks": True})
    wait_until(lambda: stack.agents["edge-01"].engine.faults.drop_acks, msg="drop_acks applied")
    dep = stack.deploy(v1, device_ids=["edge-01"])
    final = stack.wait_state(dep["deployment_id"], {"COMPLETED", "ROLLED_BACK", "FAILED"})
    assert final["state"] == "COMPLETED", final  # no ack ever arrived; device-reported state proved it
    assert active_version(stack, "edge-01") == v1
