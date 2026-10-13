"""When the rollback itself fails the deployment must end FAILED, say so, and be recoverable."""

from __future__ import annotations

from tests.harness import Stack, wait_until

HEALTHY = [2.0, 65.0, 6.0, 12.0, 1480.0, 70.0]


def active_version(stack: Stack, device_id: str) -> int | None:
    a = stack.agents[device_id].store.snapshot().active
    return a.version if a else None


def audit_actions(stack: Stack, dep_id: str) -> list[str]:
    items = stack.api("GET", f"/api/v1/audit/events?deployment_id={dep_id}&limit=200", role="viewer").json()["items"]
    return [e["action"] for e in reversed(items)]


def test_failed_rollback_marks_deployment_failed_and_is_recoverable(stack: Stack) -> None:
    v1, v2 = stack.train(seed=1), stack.train(seed=2)
    for d in ("edge-01", "edge-02"):
        stack.add_agent(d)
    stack.wait_online("edge-01", "edge-02")
    stack.wait_state(stack.deploy(v1)["deployment_id"], {"COMPLETED"})

    # canary will fail its health check on v2 AND refuse to roll back (deterministic injected faults)
    r = stack.api(
        "POST",
        "/api/v1/devices/edge-01/simulation",
        role="admin",
        json={"fail_inference_versions": [v2], "fail_rollback": True},
    )
    assert r.status_code == 202
    wait_until(lambda: stack.agents["edge-01"].engine.faults.fail_rollback, msg="faults applied")

    dep_id = stack.deploy(v2, strategy={"canary_count": 1, "batch_percent": 100})["deployment_id"]
    final = stack.wait_state(dep_id, {"FAILED", "ROLLED_BACK", "COMPLETED"})

    # -- accurate representation: FAILED (not ROLLED_BACK), with an actionable reason
    assert final["state"] == "FAILED", final
    assert "rollback failed on ['edge-01']" in final["reason"] and "manual intervention" in final["reason"]
    targets = {t["device_id"]: t for t in final["targets"]}
    assert targets["edge-01"]["state"] == "FAILED"
    assert (
        targets["edge-01"]["last_error"].startswith("rollback failed:")
        and "ROLLBACK_FAILED" in targets["edge-01"]["last_error"]
    )
    assert targets["edge-02"]["state"] == "CANCELLED"

    # -- reality on the devices: the canary is STILL on the bad model (the failure is real); the rest are untouched
    assert active_version(stack, "edge-01") == v2
    assert active_version(stack, "edge-02") == v1
    assert stack.agents["edge-01"].store.snapshot().previous is not None  # known-good still retained for recovery

    # -- persisted state and audit trail
    actions = audit_actions(stack, dep_id)
    for expected in ("deployment.created", "target.health_failed", "deployment.rolling_back", "deployment.failed"):
        assert expected in actions, actions
    assert "deployment.rolled_back" not in actions
    ack_url = f"/api/v1/audit/events?deployment_id={dep_id}&action=device.ack&limit=50"
    acks = stack.api("GET", ack_url, role="viewer").json()["items"]
    assert any(e["details"]["outcome"] == "FAILED" and e["details"]["error_code"] == "ROLLBACK_FAILED" for e in acks)
    metrics = stack.api("GET", "/metrics", role="viewer").text
    assert 'fleetops_deployments_total{state="FAILED"} 1.0' in metrics
    assert 'fleetops_deployments_total{state="ROLLED_BACK"}' not in metrics

    stack.restart_control_plane()  # FAILED is durable across a restart and no automatic retry loop restarts it
    stack.wait_online("edge-01", "edge-02")
    again = stack.deployment(dep_id)
    assert again["state"] == "FAILED" and again["targets"][0]["state"] == "FAILED"
    assert active_version(stack, "edge-01") == v2

    # -- the documented recovery: fix the device, then request the rollback again
    stack.api("POST", "/api/v1/devices/edge-01/simulation", role="admin", json={"fail_inference_versions": [v2]})
    wait_until(lambda: not stack.agents["edge-01"].engine.faults.fail_rollback, msg="rollback fault cleared")
    r = stack.api("POST", f"/api/v1/deployments/{dep_id}/rollback", json={"reason": "device repaired, retry rollback"})
    assert r.status_code == 202
    recovered = stack.wait_state(dep_id, {"ROLLED_BACK", "FAILED"})
    assert recovered["state"] == "ROLLED_BACK", recovered
    assert active_version(stack, "edge-01") == v1
    assert stack.agents["edge-01"].infer(HEALTHY).model.version == v1  # v1 really loaded and predicting
    assert "deployment.rolled_back" in audit_actions(stack, dep_id)
