"""Full-stack harness: real control plane over HTTP (uvicorn thread), real MLflow (sqlite),
real SQLite, real edge agents; only the MQTT broker is the in-process one."""

from __future__ import annotations

import socket
import ssl
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from prometheus_client.parser import text_string_to_metric_families

from fleetops.config.settings import ControlPlaneSettings, EdgeSettings
from fleetops.controlplane.app import create_app
from fleetops.domain.messages import Command
from fleetops.edge.agent import EdgeAgent
from fleetops.messaging.bus import InMemoryBroker, InMemoryBus, MqttBus, MqttConfig
from fleetops.registry.mlflow_registry import MlflowRegistry

KEYS = {
    "metrics": "metrics-token-for-tests-0007",
    "admin": "admin-key-for-tests-0001",
    "operator": "operator-key-for-tests-02",
    "viewer": "viewer-key-for-tests-0003",
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_until(cond: Callable[[], Any], timeout: float = 20.0, interval: float = 0.05, msg: str = "") -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = cond()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"timeout waiting for condition {msg} (last={last!r})")


class LiveControlPlane:
    def __init__(
        self,
        settings: ControlPlaneSettings,
        broker: InMemoryBroker,
        registry: MlflowRegistry,
        port: int | None = None,
        real_mqtt: bool = False,
    ) -> None:
        self.settings, self.broker, self.registry = settings, broker, registry
        self.port = port or free_port()
        # real_mqtt: create_app builds its production MqttBus (reconnect hooks, metrics) against a real broker
        self.bus = None if real_mqtt else InMemoryBus(broker, "control-plane", persistent=True)
        app = create_app(settings, bus=self.bus, registry=registry)
        self.tls = settings.tls_certfile is not None
        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
                lifespan="on",
                ssl_certfile=str(settings.tls_certfile) if settings.tls_certfile else None,
                ssl_keyfile=str(settings.tls_keyfile) if settings.tls_keyfile else None,
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"{'https' if self.tls else 'http'}://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.thread.start()
        wait_until(lambda: self.server.started, 20, msg="control plane start")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
        if self.bus is not None:
            self.bus.stop()


class Stack:
    def __init__(self, tmp: Path, settings: ControlPlaneSettings, real_mqtt: tuple[str, int] | None = None) -> None:
        self.tmp, self.settings = tmp, settings
        self.real_mqtt = real_mqtt  # (host, port) of a real Mosquitto: agents and control plane then use MqttBus
        self.broker = InMemoryBroker()
        self.registry = MlflowRegistry(
            settings.mlflow_tracking_uri, settings.mlflow_experiment, artifact_location=(tmp / "mlruns").as_uri()
        )
        self.cp = LiveControlPlane(settings, self.broker, self.registry, real_mqtt=real_mqtt is not None)
        self.agents: dict[str, EdgeAgent] = {}
        self.ca: Path | None = None  # trust anchor when the control plane serves HTTPS
        self.buses: dict[str, InMemoryBus] = {}

    # -- lifecycle
    def start(self) -> Stack:
        self.cp.start()
        return self

    def restart_control_plane(self) -> None:
        self.cp.stop()
        self.cp = LiveControlPlane(
            self.settings, self.broker, self.registry, port=self.cp.port, real_mqtt=self.real_mqtt is not None
        )
        self.cp.start()

    def stop(self) -> None:
        for a in self.agents.values():
            a.stop()
        self.cp.stop()
        self.broker.close()

    # -- API
    def api(self, method: str, path: str, role: str = "operator", **kw: Any) -> httpx.Response:
        headers = {"Authorization": f"Bearer {KEYS[role]}", **kw.pop("headers", {})}
        verify: Any = ssl.create_default_context(cafile=str(self.ca)) if self.ca else True
        return httpx.request(method, self.cp.url + path, headers=headers, timeout=60, verify=verify, **kw)

    def train(self, **params: Any) -> int:
        r = self.api(
            "POST",
            "/api/v1/models/train",
            json={"model_name": "anomaly-detector", "params": {"n_samples": 800, "epochs": 5, **params}},
        )
        assert r.status_code == 201, r.text
        return int(r.json()["version"])

    def add_agent(
        self,
        device_id: str,
        group: str = "line-a",
        pre_start: Callable[[EdgeAgent], None] | None = None,
        **settings: Any,
    ) -> EdgeAgent:
        r = self.api("POST", "/api/v1/devices/register", role="admin", json={"device_id": device_id, "group": group})
        assert r.status_code == 201, r.text
        secret = r.json()["credentials"]["secret"]
        settings.setdefault("control_plane_ca", self.ca)
        defaults: dict[str, Any] = {
            "device_id": device_id,
            "device_secret": secret,
            "data_dir": self.tmp / device_id,
            "control_plane_url": self.cp.url,
            "allow_faults": True,
            "heartbeat_interval_seconds": 0.2,
            "telemetry_interval_seconds": 1.0,
            "prediction_flush_seconds": 0.3,
            "sensor_hz": 40,
            "download_max_attempts": 2,
        }
        s = EdgeSettings(_env_file=None, **{**defaults, **settings})
        holder: dict[str, EdgeAgent] = {}

        def announce() -> None:  # the agent exists by the time the first connection completes
            holder["agent"].announce_online()

        bus: Any
        if self.real_mqtt is not None:
            host, port = self.real_mqtt
            bus = MqttBus(
                MqttConfig(
                    host=host,
                    port=port,
                    client_id=f"fleetops-{device_id}",
                    username=device_id,
                    password=secret,
                    keepalive=3,
                    reconnect_delay_min=1,
                    reconnect_delay_max=2,
                ),
                will=EdgeAgent.will_for(device_id),
                on_connect=announce,
            )
        else:
            bus = InMemoryBus(self.broker, device_id, will=EdgeAgent.will_for(device_id))
        agent = EdgeAgent(s, bus, sleep=lambda _s: None)
        holder["agent"] = agent
        self.agents[device_id], self.buses[device_id] = agent, bus
        if pre_start is not None:
            pre_start(agent)
        agent.start()
        if self.real_mqtt is None:
            agent.announce_online()  # (a real MqttBus announces itself from its on_connect hook)
        return agent

    def restart_agent(self, device_id: str, crash: bool = False) -> EdgeAgent:
        """New agent process on the same data directory. ``crash`` = no clean shutdown (like kill -9): the broker
        publishes the last will and nothing is flushed or cleaned up by the old instance."""
        old, old_bus = self.agents[device_id], self.buses[device_id]
        if crash:
            old._stop.set()
            old_bus.drop_network()
        else:
            old.stop()
        bus = InMemoryBus(self.broker, device_id, will=EdgeAgent.will_for(device_id))
        agent = EdgeAgent(old.s, bus, sleep=lambda _s: None)
        self.agents[device_id], self.buses[device_id] = agent, bus
        agent.start()
        agent.announce_online()
        return agent

    # -- observation helpers
    def metric(self, name: str, **labels: str) -> float:
        """Current value of a control-plane Prometheus sample (0.0 if the series does not exist)."""
        text = self.api("GET", "/metrics", role="metrics").text
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                if sample.name == name and all(sample.labels.get(k) == v for k, v in labels.items()):
                    return float(sample.value)
        return 0.0

    def audit(
        self, action: str | None = None, deployment_id: str | None = None, device_id: str | None = None
    ) -> list[dict[str, Any]]:
        query = "&".join(
            f"{k}={v}" for k, v in (("action", action), ("deployment_id", deployment_id), ("device_id", device_id)) if v
        )
        return list(self.api("GET", f"/api/v1/audit/events?limit=200&{query}", role="viewer").json()["items"])

    def wire_commands(self, device_id: str | None = None, ctype: str | None = None) -> list[Command]:
        """Every command the control plane put on the (in-memory) broker, in publication order."""
        out = []
        for topic_name, payload in list(self.broker.published):
            if topic_name.endswith("/commands") and (device_id is None or f"/{device_id}/" in topic_name):
                cmd = Command.model_validate_json(payload)
                if ctype is None or cmd.type.value == ctype:
                    out.append(cmd)
        return out

    def agent_metric(self, device_id: str, name: str, **labels: str) -> float:
        value = self.agents[device_id].metrics.registry.get_sample_value(name, {"device_id": device_id, **labels})
        return float(value or 0.0)

    def active_version(self, device_id: str) -> int | None:
        active = self.agents[device_id].store.snapshot().active
        return active.version if active else None

    def deployment(self, dep_id: str) -> dict[str, Any]:
        r = self.api("GET", f"/api/v1/deployments/{dep_id}", role="viewer")
        assert r.status_code == 200, r.text
        return dict(r.json())

    def wait_state(self, dep_id: str, states: set[str], timeout: float = 30.0) -> dict[str, Any]:
        try:
            wait_until(
                lambda: self.deployment(dep_id)["state"] in states, timeout, msg=f"deployment {dep_id} in {states}"
            )
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n{self.deployment(dep_id)}") from exc
        return self.deployment(dep_id)

    def wait_online(self, *device_ids: str) -> None:
        for d in device_ids:
            wait_until(
                lambda d=d: self.api("GET", f"/api/v1/devices/{d}", role="viewer").json()["status"] == "ONLINE",
                10,
                msg=f"{d} online",
            )

    def deploy(self, version: int, group: str = "line-a", **kw: Any) -> dict[str, Any]:
        body = {
            "model_name": "anomaly-detector",
            "model_version": version,
            "group": group,
            "health_criteria": {
                "observation_seconds": 1,
                "min_inferences": 10,
                "ack_timeout_seconds": 15,
                "health_timeout_seconds": 15,
                "max_p95_latency_ms": 200,
                "max_cpu_percent": 100,  # agents share the test process: CPU is not meaningful here
            },
            **kw,
        }
        r = self.api("POST", "/api/v1/deployments", json=body)
        assert r.status_code in (200, 201), r.text
        return dict(r.json())
