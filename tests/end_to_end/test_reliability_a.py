"""Reliability scenarios 5, 7, 8, 11, 12 (see docs/reliability-test-results.md for the expectations written first)."""

from __future__ import annotations

import threading
from typing import Any

import httpx
import pytest

from tests.harness import KEYS, Stack, wait_until
from tests.scenarios import DEVICES, TERMINAL, fleet, manual_rollout, still_serving

# ---------------------------------------------------------------------------------------------------- scenario 5


def test_s5_artifact_corrupted_at_rest_is_never_served_or_activated(make_stack: Any) -> None:
    stack: Stack = make_stack()
    v1, v2 = fleet(stack, DEVICES[:2])
    path = stack.settings.artifact_root / "anomaly-detector" / str(v2) / "model.safetensors"
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))  # one flipped bit in the stored artifact
    ok_before = stack.metric("fleetops_artifact_downloads_total", result="ok")

    dep = stack.deploy(v2)
    final = stack.wait_state(dep["deployment_id"], TERMINAL)

    assert final["state"] == "ROLLED_BACK" and final["targets"][0]["state"] == "FAILED"
    assert "DOWNLOAD_FAILED" in final["targets"][0]["last_error"]
    still_serving(stack, v1, DEVICES[:2])
    assert all(stack.agents[d].store.snapshot().staged is None for d in DEVICES[:2])
    assert stack.metric("fleetops_artifact_integrity_failures_total") >= 1  # detected server-side
    assert stack.metric("fleetops_artifact_downloads_total", result="corrupt") >= 1
    assert (
        stack.metric("fleetops_artifact_downloads_total", result="ok") == ok_before
    )  # not one corrupt byte was served
    (event,) = stack.audit("artifact.corrupt")[:1]
    assert event["entity_id"] == f"anomaly-detector:{v2}" and "registered" in event["details"]["detail"]
    assert stack.audit("deployment.rolled_back", deployment_id=dep["deployment_id"])

    path.write_bytes(original)  # repaired: the integrity cache must notice the change and allow the deployment
    again = stack.wait_state(stack.deploy(v2)["deployment_id"], TERMINAL)
    assert again["state"] == "COMPLETED"
    still_serving(stack, v2, DEVICES[:2])


# ---------------------------------------------------------------------------------------------------- scenario 7


def test_s7_duplicate_commands_and_acks_have_no_side_effects(make_stack: Any) -> None:
    stack: Stack = make_stack()
    _, v2 = fleet(stack, DEVICES[:2])
    dep = stack.deploy(v2)
    stack.wait_state(dep["deployment_id"], {"COMPLETED"})
    dep_id = dep["deployment_id"]
    staged_before = stack.agent_metric("edge-01", "fleetops_edge_commands_total", type="DEPLOY", outcome="STAGED")
    active_before = stack.agent_metric("edge-01", "fleetops_edge_commands_total", type="ACTIVATE", outcome="ACTIVE")
    dups_before = sum(
        stack.agent_metric("edge-01", "fleetops_edge_duplicate_commands_total", type=t) for t in ("DEPLOY", "ACTIVATE")
    )
    audit_acks_before = len(stack.audit("device.ack", deployment_id=dep_id))

    mine = [c for c in stack.wire_commands("edge-01") if c.deployment_id == dep_id]
    assert {c.type.value for c in mine} == {"DEPLOY", "ACTIVATE"}
    for cmd in mine:  # the broker (or a retry storm) delivers every command five more times
        for _ in range(5):
            stack.broker.route("fleetops/devices/edge-01/commands", cmd.model_dump_json().encode(), 1, False)
    assert stack.broker.flush()
    # replay every acknowledgement the device sent for this deployment, five more times each
    acks = [
        p
        for t, p in list(stack.broker.published)
        if t == "fleetops/devices/edge-01/status" and dep_id.encode() in p and b'"kind":"ack"' in p
    ]
    assert acks
    for payload in acks:
        for _ in range(5):
            stack.broker.route("fleetops/devices/edge-01/status", payload, 1, False)
    assert stack.broker.flush()

    # no command ran twice; the duplicates were answered from the ack cache and counted
    assert (
        stack.agent_metric("edge-01", "fleetops_edge_commands_total", type="DEPLOY", outcome="STAGED") == staged_before
    )
    assert (
        stack.agent_metric("edge-01", "fleetops_edge_commands_total", type="ACTIVATE", outcome="ACTIVE")
        == active_before
    )
    dups_after = sum(
        stack.agent_metric("edge-01", "fleetops_edge_duplicate_commands_total", type=t) for t in ("DEPLOY", "ACTIVATE")
    )
    assert dups_after - dups_before == 5 * len(mine)
    # the control plane ignored the replayed acks: state and audit trail unchanged
    assert stack.deployment(dep_id)["state"] == "COMPLETED"
    assert len(stack.audit("device.ack", deployment_id=dep_id)) == audit_acks_before
    still_serving(stack, v2, DEVICES[:2])
    assert stack.agents["edge-01"].store.snapshot().previous is not None


def test_s7_parallel_duplicate_deployment_requests(make_stack: Any) -> None:
    stack: Stack = make_stack()
    versions = fleet(stack, DEVICES[:2], models=4)
    body = {
        "model_name": "anomaly-detector",
        "model_version": versions[1],
        "group": "line-a",
        "health_criteria": {"observation_seconds": 1, "min_inferences": 5, "max_cpu_percent": 100},
    }

    def post(key: str | None, out: list[httpx.Response]) -> None:
        headers = {"Authorization": f"Bearer {KEYS['operator']}", **({"Idempotency-Key": key} if key else {})}
        out.append(httpx.post(stack.cp.url + "/api/v1/deployments", json=body, headers=headers, timeout=60))

    # (a) five parallel submissions of the same request with one Idempotency-Key -> exactly one deployment
    results: list[httpx.Response] = []
    threads = [threading.Thread(target=post, args=("same-key", results)) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(r.status_code for r in results) == [200, 200, 200, 200, 201]
    assert len({r.json()["deployment_id"] for r in results}) == 1
    stack.wait_state(results[0].json()["deployment_id"], {"COMPLETED"})
    total = stack.api("GET", "/api/v1/deployments", role="viewer").json()["total"]
    assert total == 2  # the baseline rollout + exactly one for the duplicated request
    activations = {
        d: stack.agent_metric(d, "fleetops_edge_commands_total", type="ACTIVATE", outcome="ACTIVE") for d in DEVICES[:2]
    }

    # (b) the same request WITHOUT a key, in parallel: newer supersedes older; one activation per device
    body["model_version"] = versions[2]
    results.clear()
    threads = [threading.Thread(target=post, args=(None, results)) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert all(r.status_code in (201, 422, 409) for r in results), [r.text for r in results]
    created = [r.json()["deployment_id"] for r in results if r.status_code == 201]
    assert created
    wait_until(
        lambda: any(stack.deployment(i)["state"] == "COMPLETED" for i in created), 30, msg="one request completes"
    )
    states = sorted(stack.deployment(i)["state"] for i in created)
    assert states.count("COMPLETED") == 1 and set(states) <= {"COMPLETED", "CANCELLED"}, states
    assert {stack.active_version(d) for d in DEVICES[:2]} == {versions[2]}
    for d in DEVICES[:2]:  # each device switched exactly once for the new version
        assert (
            stack.agent_metric(d, "fleetops_edge_commands_total", type="ACTIVATE", outcome="ACTIVE") - activations[d]
            == 1
        )


# ---------------------------------------------------------------------------------------------------- scenario 8


def test_s8_insufficient_storage_refuses_the_update_and_keeps_serving(make_stack: Any) -> None:
    stack: Stack = make_stack()
    v1, v2 = fleet(stack, DEVICES[:2])
    r = stack.api("POST", "/api/v1/devices/edge-01/simulation", role="admin", json={"fake_free_disk_mb": 10})
    assert r.status_code == 202
    wait_until(lambda: stack.agents["edge-01"].engine.faults.fake_free_disk_mb == 10, msg="fault applied")
    stack.agents["edge-01"].send_heartbeat()  # the low value is reported, then exported by the control plane
    wait_until(
        lambda: (
            stack.metric("fleetops_device_disk_free_mb", device_id="edge-01")
            == pytest.approx(10 * 1024 * 1024 / 1e6, rel=0.01)
        ),
        msg="disk gauge",
    )

    dep = stack.deploy(v2)
    final = stack.wait_state(dep["deployment_id"], TERMINAL)

    assert final["state"] == "ROLLED_BACK"
    target = {t["device_id"]: t for t in final["targets"]}["edge-01"]
    assert target["state"] == "FAILED" and "INSUFFICIENT_STORAGE" in target["last_error"]
    still_serving(stack, v1, DEVICES[:2])
    assert not list(stack.agents["edge-01"].store.staging_dir.iterdir())  # nothing partial was ever written
    assert stack.metric("fleetops_deployment_targets_total", outcome="failed") >= 1
    assert stack.metric("fleetops_deployments_total", state="ROLLED_BACK") == 1
    assert (
        stack.audit("device.ack", deployment_id=dep["deployment_id"])[0]["details"]["error_code"]
        == "INSUFFICIENT_STORAGE"
    )

    stack.api("POST", "/api/v1/devices/edge-01/simulation", role="admin", json={})  # space is available again
    wait_until(lambda: stack.agents["edge-01"].engine.faults.fake_free_disk_mb is None, msg="fault cleared")
    assert stack.wait_state(stack.deploy(v2)["deployment_id"], TERMINAL)["state"] == "COMPLETED"
    still_serving(stack, v2, DEVICES[:2])


# ---------------------------------------------------------------------------------------------- scenario 9 (extension)


def test_s9_rollback_failure_is_counted(make_stack: Any) -> None:
    stack: Stack = make_stack()
    v1, v2 = fleet(stack, DEVICES[:2])
    stack.api(
        "POST",
        "/api/v1/devices/edge-01/simulation",
        role="admin",
        json={"fail_inference_versions": [v2], "fail_rollback": True},
    )
    wait_until(lambda: stack.agents["edge-01"].engine.faults.fail_rollback, msg="faults applied")
    final = stack.wait_state(stack.deploy(v2)["deployment_id"], TERMINAL)
    assert final["state"] == "FAILED"
    assert stack.active_version("edge-01") == v2  # honest: the device is still on the bad model
    assert stack.active_version("edge-02") == v1
    assert stack.metric("fleetops_rollback_failures_total") == 1
    assert stack.metric("fleetops_deployments_total", state="FAILED") == 1
    # bounded: the failed rollback was not hammered
    rollbacks = [c for c in stack.wire_commands("edge-01", "ROLLBACK") if c.deployment_id == final["deployment_id"]]
    assert 1 <= len({c.command_id for c in rollbacks}) <= 2 and len(rollbacks) <= 4


# ---------------------------------------------------------------------------------------------------- scenario 11


def test_s11_inventory_is_fresh_after_a_failed_deployment_even_without_heartbeats(make_stack: Any) -> None:
    stack: Stack = make_stack(offline_after_seconds=120)
    v1, v2 = stack.train(seed=1), stack.train(seed=2)
    stack.add_agent("edge-01", heartbeat_interval_seconds=30)  # heartbeats are 30 s apart
    stack.wait_online("edge-01")
    crit = {
        "observation_seconds": 0,
        "min_inferences": 3,
        "ack_timeout_seconds": 15,
        "health_timeout_seconds": 15,
        "max_cpu_percent": 100,
    }

    manual_rollout(stack, "edge-01", v1)
    dep = manual_rollout(stack, "edge-01", v2)
    dep = {"deployment_id": dep["deployment_id"]}
    assert stack.api("GET", "/api/v1/devices/edge-01", role="viewer").json()["active_model_version"] == v2

    rb = stack.api("POST", f"/api/v1/deployments/{dep['deployment_id']}/rollback", json={"reason": "operator rollback"})
    assert rb.status_code == 202
    stack.wait_state(dep["deployment_id"], {"ROLLED_BACK"})
    assert stack.active_version("edge-01") == v1
    # no heartbeat was sent since the rollback, yet the inventory already shows the truth (taken from the ack)
    assert stack.api("GET", "/api/v1/devices/edge-01", role="viewer").json()["active_model_version"] == v1
    assert stack.api("GET", "/api/v1/devices/edge-01/health", role="viewer").json()["active_model"]["version"] == v1

    again = stack.deploy(
        v2, health_criteria=crit
    )  # immediately: not skipped as 'already running', rollback target known
    assert again["skipped"] == {} and again["targets"][0]["previous_model"]["version"] == v1
    stack.api("POST", f"/api/v1/deployments/{again['deployment_id']}/cancel")


# ---------------------------------------------------------------------------------------------------- scenario 12


def test_s12_partial_fleet_failure_rolls_everything_back_leaving_no_mixed_fleet(make_stack: Any) -> None:
    stack: Stack = make_stack()
    v1, v2 = fleet(stack)
    stack.api("POST", "/api/v1/devices/edge-03/simulation", role="admin", json={"fail_activation_versions": [v2]})
    wait_until(lambda: stack.agents["edge-03"].engine.faults.fail_activation_versions == [v2], msg="fault applied")

    dep = stack.deploy(
        v2, strategy={"canary_count": 1, "batch_percent": 100}
    )  # canary edge-01, then edge-02 + edge-03 together
    final = stack.wait_state(dep["deployment_id"], TERMINAL)

    assert final["state"] == "ROLLED_BACK", final
    targets = {t["device_id"]: t for t in final["targets"]}
    assert targets["edge-03"]["state"] == "ROLLED_BACK" or targets["edge-03"]["state"] == "FAILED"
    assert "ACTIVATION_FAILED" in (targets["edge-03"]["last_error"] or "")
    assert (
        targets["edge-01"]["state"] == "ROLLED_BACK" and targets["edge-01"]["verified_at"] is not None
    )  # verified, still reverted
    assert {stack.active_version(d) for d in DEVICES} == {v1}  # nobody is left on v2: no mixed fleet
    still_serving(stack, v1)
    assert stack.metric("fleetops_rollbacks_total", trigger="automatic") == 1
    assert stack.metric("fleetops_deployments_total", state="ROLLED_BACK") == 1
    assert stack.metric("fleetops_deployment_targets_total", outcome="rolled_back") >= 2
    actions = [e["action"] for e in stack.audit(deployment_id=dep["deployment_id"])]
    assert "deployment.rolling_back" in actions and "deployment.rolled_back" in actions
