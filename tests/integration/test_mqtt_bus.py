"""MqttBus against a real Mosquitto. Set FLEETOPS_TEST_MQTT_HOST/PORT (CI service container or
``docker run`` with mqtt/mosquitto.conf, mqtt/acl.conf and scripts/provision.py output)."""

from __future__ import annotations

import os
import socket
import threading
import time
import uuid

import pytest

from fleetops.config.settings import ControlPlaneSettings
from fleetops.messaging.bus import BusDisconnectedError, MqttBus, MqttConfig, Will
from fleetops.security.device_identity import derive_device_secret
from tests.harness import wait_until

HOST = os.environ.get("FLEETOPS_TEST_MQTT_HOST")
PORT = int(os.environ.get("FLEETOPS_TEST_MQTT_PORT", "1883"))
MASTER = os.environ.get("FLEETOPS_TEST_DEVICE_MASTER", "dev-device-master-key-change-me-0004")
CP_PASS = os.environ.get("FLEETOPS_TEST_MQTT_PASSWORD", "dev-mqtt-password-change-me-05")

pytestmark = [
    pytest.mark.mqtt,
    pytest.mark.skipif(not HOST, reason="FLEETOPS_TEST_MQTT_HOST not set"),
]


def bus(client_id: str, user: str, password: str, will: Will | None = None, clean: bool = False) -> MqttBus:
    assert HOST
    return MqttBus(
        MqttConfig(
            host=HOST,
            port=PORT,
            client_id=client_id,
            username=user,
            password=password,
            clean_session=clean,
            reconnect_delay_min=1,
            reconnect_delay_max=2,
        ),
        will=will,
    )


def started(b: MqttBus) -> MqttBus:
    b.start()
    assert b.wait_connected(10), "could not connect/authenticate"
    return b


def test_authentication_required_and_acl_enforced(cp_settings: ControlPlaneSettings) -> None:
    bad = bus(f"bad-{uuid.uuid4().hex[:6]}", "edge-01", "wrong-password")
    bad.start()
    assert not bad.wait_connected(3)
    bad.stop()

    secret = derive_device_secret(MASTER, "edge-01", 1)
    device = started(bus(f"edge-01-{uuid.uuid4().hex[:6]}", "edge-01", secret))
    cp = started(bus(f"cp-{uuid.uuid4().hex[:6]}", "control-plane", CP_PASS))
    got: list[tuple[str, bytes]] = []
    cp.subscribe("fleetops/devices/+/heartbeat", lambda t, p: got.append((t, p)), qos=1)
    cmds: list[bytes] = []
    device.subscribe("fleetops/devices/edge-01/commands", lambda _t, p: cmds.append(p), qos=1)
    other: list[bytes] = []
    device.subscribe("fleetops/devices/edge-02/commands", lambda _t, p: other.append(p), qos=1)  # not permitted

    time.sleep(0.5)

    device.publish("fleetops/devices/edge-01/heartbeat", b"hello", qos=1)
    wait_until(lambda: got, 5, msg="heartbeat delivered")
    device.publish("fleetops/devices/edge-02/heartbeat", b"spoof", qos=1)  # ACL silently drops
    cp.publish("fleetops/devices/edge-01/commands", b"cmd", qos=1)
    cp.publish("fleetops/devices/edge-02/commands", b"secret-cmd", qos=1)
    wait_until(lambda: cmds, 5, msg="command delivered")
    time.sleep(0.5)
    assert [t for t, _ in got] == ["fleetops/devices/edge-01/heartbeat"]  # spoofed publish never arrived
    assert other == []  # device cannot read another device's commands
    for b in (device, cp):
        b.stop()


def test_persistent_session_queues_commands_and_last_will(cp_settings: ControlPlaneSettings) -> None:
    secret = derive_device_secret(MASTER, "edge-02", 1)
    cid = f"edge-02-{uuid.uuid4().hex[:6]}"
    will = Will("fleetops/devices/edge-02/status", b'{"kind":"offline"}', qos=1, retain=True)
    cp = started(bus(f"cp-{uuid.uuid4().hex[:6]}", "control-plane", CP_PASS))
    statuses: list[bytes] = []
    cp.subscribe("fleetops/devices/+/status", lambda _t, p: statuses.append(p), qos=1)

    time.sleep(0.3)

    received: list[bytes] = []
    first = bus(cid, "edge-02", secret, will=will)
    first.subscribe("fleetops/devices/edge-02/commands", lambda _t, p: received.append(p), qos=1)
    started(first)
    time.sleep(0.5)
    # unclean network loss: kill the socket without DISCONNECT -> broker publishes the last will
    first._client.socket().close()  # type: ignore[attr-defined]
    wait_until(lambda: any(b"offline" in s for s in statuses), 20, msg="last will published")
    first._client.loop_stop()

    cp.publish("fleetops/devices/edge-02/commands", b"queued-while-offline", qos=1)
    second = bus(cid, "edge-02", secret, will=will)  # same client id, persistent session
    delivered = threading.Event()
    second.subscribe("fleetops/devices/edge-02/commands", lambda _t, p: (received.append(p), delivered.set()), qos=1)
    started(second)
    assert delivered.wait(10), "queued QoS1 command was not delivered after reconnect"
    assert received[-1] == b"queued-while-offline"
    second.stop()
    cp.stop()


def test_publish_raises_when_disconnected() -> None:
    assert HOST
    b = bus("never-connected", "x", "y")
    with pytest.raises(BusDisconnectedError):
        b.publish("fleetops/devices/x/heartbeat", b"1")
    _ = socket  # keep import for environments that probe the port
