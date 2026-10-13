"""Scenario 1: MQTT broker outage/restart: real broker, real MqttBus on both sides, real persistent sessions."""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from pydantic import SecretStr

from fleetops.config.settings import ControlPlaneSettings
from fleetops.security.device_identity import derive_device_secret
from tests.harness import Stack, wait_until
from tests.real_broker import Mosquitto, docker_available
from tests.scenarios import DEVICES, FAST, HEALTHY, TERMINAL, still_serving

pytestmark = pytest.mark.skipif(not docker_available(), reason="docker is not available")

CP_PASSWORD = "reliability-cp-password-0123"


def cmd_count(stack: Stack, device: str, ctype: str, outcome: str) -> float:
    return stack.agent_metric(device, "fleetops_edge_commands_total", type=ctype, outcome=outcome)


@pytest.fixture
def real(tmp_path, cp_settings: ControlPlaneSettings) -> Iterator[tuple[Stack, Mosquitto]]:  # type: ignore[no-untyped-def]
    master = cp_settings.device_key_master.get_secret_value()
    users = {"control-plane": CP_PASSWORD, **{d: derive_device_secret(master, d, 1) for d in DEVICES[:2]}}
    broker = Mosquitto(tmp_path, users)
    broker.start()
    settings = cp_settings.model_copy(
        update={
            "mqtt_enabled": True,
            "mqtt_host": "127.0.0.1",
            "mqtt_port": broker.port,
            "mqtt_username": "control-plane",
            "mqtt_password": SecretStr(CP_PASSWORD),
            "mqtt_keepalive_seconds": 3,
            "mqtt_reconnect_max_seconds": 2,
            "offline_after_seconds": 40.0,
        }
    )
    stack = Stack(tmp_path, settings, real_mqtt=("127.0.0.1", broker.port)).start()
    try:
        yield stack, broker
    finally:
        stack.stop()
        broker.close()


def bring_up(stack: Stack) -> tuple[int, int]:
    v1, v2 = stack.train(seed=1), stack.train(seed=2)
    for d in DEVICES[:2]:
        stack.add_agent(d)
    stack.wait_online(*DEVICES[:2])
    wait_until(lambda: stack.metric("fleetops_mqtt_connected") == 1, 15, msg="control plane connected to the broker")
    stack.wait_state(stack.deploy(v1)["deployment_id"], {"COMPLETED"}, timeout=60)
    return v1, v2


def test_s1_commands_issued_while_the_control_plane_is_cut_off_from_the_broker_are_delivered_afterwards(
    real: tuple[Stack, Mosquitto],
) -> None:
    """Partition between the control plane and a healthy broker: devices stay connected (no last wills), the API still
    accepts the deployment, the engine cannot publish and keeps retrying; when the link returns it completes."""
    stack, _ = real
    v1, v2 = bring_up(stack)
    cp_bus = stack.cp.server.config.app.state.ctx.bus  # type: ignore[attr-defined]
    reconnects = stack.metric("fleetops_mqtt_reconnects_total")
    staged = {d: cmd_count(stack, d, "DEPLOY", "STAGED") for d in DEVICES[:2]}

    cp_bus.drop_network()
    wait_until(lambda: stack.metric("fleetops_mqtt_connected") == 0, 10, msg="control plane sees its link is down")
    assert all(
        stack.api("GET", f"/api/v1/devices/{d}", role="viewer").json()["status"] == "ONLINE" for d in DEVICES[:2]
    )
    dep = stack.deploy(v2, health_criteria={**FAST, "ack_timeout_seconds": 60, "health_timeout_seconds": 60})
    wait_until(
        lambda: stack.metric("fleetops_commands_published_total", type="DEPLOY", result="link_down") >= 2,
        15,
        msg="engine keeps retrying while it cannot publish",
    )
    assert stack.deployment(dep["deployment_id"])["state"] == "IN_PROGRESS"  # patient: not failed by a short outage
    for d in DEVICES[:2]:  # and the edge never needed the control plane or the broker for inference
        assert stack.agents[d].infer(HEALTHY).model.version == v1
    time.sleep(2)
    assert (
        stack.metric("fleetops_commands_published_total", type="DEPLOY", result="link_down") <= 40
    )  # paced, not a busy loop
    assert stack.deployment(dep["deployment_id"])["targets"][0]["attempts"] == 0  # failed publishes burned no attempts

    cp_bus.restore_network()
    final = stack.wait_state(dep["deployment_id"], TERMINAL, timeout=60)

    assert final["state"] == "COMPLETED"
    still_serving(stack, v2, DEVICES[:2])
    for d in DEVICES[:2]:  # delivered after recovery, executed exactly once per device
        assert cmd_count(stack, d, "DEPLOY", "STAGED") - staged[d] == 1
    assert stack.metric("fleetops_mqtt_reconnects_total") > reconnects and stack.metric("fleetops_mqtt_connected") == 1
    assert final["targets"][0]["attempts"] <= final["health_criteria"]["max_command_attempts"]
    assert "deployment.completed" in [e["action"] for e in stack.audit(deployment_id=dep["deployment_id"])]


def test_s1_full_broker_outage_refuses_new_rollouts_clearly_and_heals_when_the_broker_returns(
    real: tuple[Stack, Mosquitto],
) -> None:
    stack, broker = real
    v1, v2 = bring_up(stack)
    broker.stop()
    # the broker publishes the devices' last wills as it goes down: they are unreachable, so they are not eligible
    wait_until(
        lambda: all(
            stack.api("GET", f"/api/v1/devices/{d}", role="viewer").json()["status"] == "OFFLINE" for d in DEVICES[:2]
        ),
        15,
        msg="devices considered unreachable",
    )
    for d in DEVICES[:2]:
        assert stack.agents[d].infer(HEALTHY).model.version == v1  # still inferring with no broker at all
    refused = stack.api(
        "POST", "/api/v1/deployments", json={"model_name": "anomaly-detector", "model_version": v2, "group": "line-a"}
    )
    assert (
        refused.status_code == 422 and "not online" in refused.json()["error"]["message"]
    )  # clear, nothing half-started
    assert stack.api("GET", "/api/v1/deployments", role="viewer").json()["total"] == 1  # only the baseline exists

    started = time.monotonic()
    broker.start_again()
    wait_until(
        lambda: all(
            stack.api("GET", f"/api/v1/devices/{d}", role="viewer").json()["status"] == "ONLINE" for d in DEVICES[:2]
        ),
        30,
        msg="devices re-announce after reconnecting",
    )
    recovery = time.monotonic() - started
    print(f"RESULT broker-outage recovery to all devices ONLINE: {recovery:.1f}s")  # noqa: T201 - recorded in the results doc
    assert recovery < 20  # bounded reconnect backoff (max 2 s in this test) + keepalive
    assert stack.metric("fleetops_mqtt_connected") == 1
    final = stack.wait_state(stack.deploy(v2)["deployment_id"], TERMINAL, timeout=60)
    assert final["state"] == "COMPLETED"
    still_serving(stack, v2, DEVICES[:2])


def test_s1_broker_restart_during_the_health_window(real: tuple[Stack, Mosquitto]) -> None:
    stack, broker = real
    _, v2 = bring_up(stack)
    activations = {d: cmd_count(stack, d, "ACTIVATE", "ACTIVE") for d in DEVICES[:2]}
    reconnects = stack.metric("fleetops_mqtt_reconnects_total")
    crit = {**FAST, "observation_seconds": 6, "health_timeout_seconds": 60, "ack_timeout_seconds": 60}
    dep_id = stack.deploy(v2, strategy={"canary_count": 1, "batch_percent": 100}, health_criteria=crit)["deployment_id"]
    wait_until(
        lambda: stack.deployment(dep_id)["targets"][0]["state"] == "ACTIVE", 20, msg="canary active, health window open"
    )

    broker.restart()  # all connections drop; sessions and queued messages survive on the volume

    final = stack.wait_state(dep_id, TERMINAL, timeout=60)
    assert final["state"] == "COMPLETED"
    still_serving(stack, v2, DEVICES[:2])
    for d in DEVICES[:2]:  # nothing was executed twice by the reconnection
        assert cmd_count(stack, d, "ACTIVATE", "ACTIVE") - activations[d] == 1
    assert stack.metric("fleetops_mqtt_reconnects_total") > reconnects
    assert all(stack.agent_metric(d, "fleetops_edge_mqtt_connected") == 1 for d in DEVICES[:2])
