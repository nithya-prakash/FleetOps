"""Shared helpers for the failure-injection scenarios (tests/end_to_end/test_reliability_*.py)."""

from __future__ import annotations

from tests.harness import Stack

HEALTHY = [2.0, 65.0, 6.0, 12.0, 1480.0, 70.0]
DEVICES = ("edge-01", "edge-02", "edge-03")
TERMINAL = {"COMPLETED", "ROLLED_BACK", "FAILED", "CANCELLED"}
FAST = {
    "observation_seconds": 1,
    "min_inferences": 10,
    "ack_timeout_seconds": 15,
    "health_timeout_seconds": 15,
    "max_cpu_percent": 100,
}


def fleet(stack: Stack, devices: tuple[str, ...] = DEVICES, models: int = 2, **agent_kw: object) -> list[int]:
    """Train ``models`` versions, bring the devices online and roll the first version out to all of them."""
    versions = [stack.train(seed=i + 1) for i in range(models)]
    for d in devices:
        stack.add_agent(d, **agent_kw)
    stack.wait_online(*devices)
    stack.wait_state(stack.deploy(versions[0])["deployment_id"], {"COMPLETED"})
    assert {stack.active_version(d) for d in devices} == {versions[0]}
    return versions


def still_serving(stack: Stack, version: int, devices: tuple[str, ...] = DEVICES) -> None:
    """The model recorded on each device is the expected one AND actually loads and predicts."""
    for d in devices:
        agent = stack.agents[d]
        active = agent.store.snapshot().active
        assert active is not None and active.version == version, (d, active)
        assert agent.infer(HEALTHY).model.version == version


def manual_rollout(stack: Stack, device_id: str, version: int, **criteria: object) -> dict[str, object]:
    """Roll a version out to an agent whose heartbeat interval is long: feed the health check one explicit heartbeat."""
    from tests.harness import wait_until  # noqa: PLC0415

    crit = {
        "observation_seconds": 0,
        "min_inferences": 3,
        "ack_timeout_seconds": 15,
        "health_timeout_seconds": 15,
        "max_cpu_percent": 100,
        **criteria,
    }
    dep = stack.deploy(version, health_criteria=crit)
    wait_until(lambda: stack.active_version(device_id) == version, 15, msg=f"v{version} activated")
    wait_until(
        lambda: (
            stack.agent_metric(device_id, "fleetops_edge_inference_requests_total", model_version=str(version)) >= 5
        ),
        10,
        msg="traffic on the new model",
    )
    stack.agents[device_id].send_heartbeat()
    return dict(stack.wait_state(dep["deployment_id"], {"COMPLETED"}))
