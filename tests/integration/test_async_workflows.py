"""Asynchronous control-plane workflows (lifespan, background ticker) under pytest-asyncio."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import httpx
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy import select

from fleetops.config.settings import ControlPlaneSettings
from fleetops.controlplane.app import BACKGROUND_WORKERS, create_app
from fleetops.controlplane.context import AppContext
from fleetops.domain.messages import Command, CommandType, DeviceModelState, Heartbeat, ModelStats, StatusMessage
from fleetops.domain.models import Compatibility, DeviceCapabilities, ModelIdentity, utcnow
from fleetops.messaging.bus import InMemoryBroker, InMemoryBus
from fleetops.messaging.topics import Channel, topic
from fleetops.persistence.db import session_scope
from fleetops.persistence.tables import ModelVersion
from fleetops.registry.mlflow_registry import MlflowRegistry
from fleetops.security.device_identity import verify_command
from tests.harness import KEYS

CAPS = DeviceCapabilities(agent_version="0.1.0", formats=["mlp-v1"])


@dataclass
class Rig:
    app: FastAPI
    ctx: AppContext
    client: httpx.AsyncClient
    broker: InMemoryBroker
    commands: list[Command]
    secret: str = ""

    async def api(self, method: str, path: str, role: str = "operator", **kw: object) -> httpx.Response:
        return await self.client.request(method, path, headers={"Authorization": f"Bearer {KEYS[role]}"}, **kw)  # type: ignore[arg-type]

    def heartbeat(self, device: str, active: ModelIdentity | None = None) -> None:
        hb = Heartbeat(
            device_id=device,
            sent_at=utcnow(),
            boot_id="b",
            agent_version="0.1.0",
            uptime_seconds=1,
            state=DeviceModelState(active=active),
            capabilities=CAPS,
            cpu_percent=1,
            memory_mb=1,
            model_stats=ModelStats(
                name=active.name,
                version=active.version,
                inference_count=50,
                inference_errors=0,
                latency_p50_ms=0.1,
                latency_p95_ms=0.2,
                active_seconds=10,
            )
            if active
            else None,
        )
        self.broker.route(topic(device, Channel.HEARTBEAT), hb.model_dump_json().encode(), 0, False)

    def ack(self, cmd: Command, outcome: str) -> None:
        msg = StatusMessage.model_validate(
            {
                "kind": "ack",
                "device_id": cmd.device_id,
                "sent_at": utcnow().isoformat(),
                "command_id": cmd.command_id,
                "command_type": cmd.type.value,
                "deployment_id": cmd.deployment_id,
                "generation": cmd.generation,
                "outcome": outcome,
            }
        )
        self.broker.route(topic(cmd.device_id, Channel.STATUS), msg.model_dump_json().encode(), 1, False)


async def eventually(cond: Callable[[], object], within: float = 6.0, interval: float = 0.05) -> object:
    deadline = asyncio.get_running_loop().time() + within
    while asyncio.get_running_loop().time() < deadline:
        if result := cond():
            return result
        await asyncio.sleep(interval)
    raise AssertionError(f"condition not met within {within}s")


@pytest_asyncio.fixture
async def rig(cp_settings: ControlPlaneSettings, tmp_path) -> AsyncIterator[Rig]:  # type: ignore[no-untyped-def]
    cp_settings.offline_after_seconds = 0.8
    cp_settings.tick_interval_seconds = 0.05
    cp_settings.retry_backoff_base_seconds = 0.2
    cp_settings.retry_backoff_max_seconds = 0.4
    broker = InMemoryBroker()
    bus = InMemoryBus(broker, "control-plane")
    registry = MlflowRegistry(
        cp_settings.mlflow_tracking_uri, cp_settings.mlflow_experiment, (tmp_path / "ml").as_uri()
    )
    app = create_app(cp_settings, bus=bus, registry=registry)
    observer = InMemoryBus(broker, "observer", persistent=False)
    commands: list[Command] = []
    observer.subscribe("fleetops/devices/+/commands", lambda _t, p: commands.append(Command.model_validate_json(p)))
    observer.start()
    async with app.router.lifespan_context(app):
        ctx: AppContext = app.state.ctx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://cp.test") as client:
            rig = Rig(app, ctx, client, broker, commands)
            reg = await rig.api(
                "POST", "/api/v1/devices/register", role="admin", json={"device_id": "edge-01", "group": "line-a"}
            )
            assert reg.status_code == 201
            rig.secret = reg.json()["credentials"]["secret"]
            with session_scope(
                ctx.sessions
            ) as db:  # a registered model version (no training needed for orchestration tests)
                db.add(
                    ModelVersion(
                        name="anomaly",
                        version=1,
                        sha256="1" * 64,
                        size_bytes=10,
                        framework="safetensors-numpy",
                        compatibility=Compatibility().model_dump(),
                        input_schema={"features": [{"name": "f", "min": 0, "max": 1}]},
                        metrics={},
                        params={},
                        reference_stats={},
                        evaluation={},
                        mlflow_run_id="r",
                        created_at=utcnow(),
                    )
                )
            yield rig
    broker.close()


async def test_ticker_marks_silent_device_offline_and_heartbeat_revives_it(rig: Rig) -> None:
    rig.heartbeat("edge-01")
    await eventually(lambda: rig.broker.flush(1) and True)

    async def current() -> str:
        return str((await rig.api("GET", "/api/v1/devices/edge-01", role="viewer")).json()["status"])

    assert await current() == "ONLINE"
    deadline = asyncio.get_running_loop().time() + 6
    while await current() != "OFFLINE":  # the background ticker (not a request) flips the status
        assert asyncio.get_running_loop().time() < deadline, "ticker did not mark the device offline"
        await asyncio.sleep(0.1)
    rig.heartbeat("edge-01")
    deadline = asyncio.get_running_loop().time() + 3
    while await current() != "ONLINE":
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.05)
    health = (await rig.api("GET", "/api/v1/devices/edge-01/health", role="viewer")).json()
    assert health["connectivity"] == "online"


async def test_ticker_resends_unacknowledged_command_with_same_id_then_proceeds_on_ack(rig: Rig) -> None:
    rig.heartbeat("edge-01")
    await asyncio.sleep(0.2)
    created = await rig.api(
        "POST",
        "/api/v1/deployments",
        json={
            "model_name": "anomaly",
            "model_version": 1,
            "group": "line-a",
            "health_criteria": {"ack_timeout_seconds": 30, "observation_seconds": 0, "min_inferences": 0},
        },
    )
    assert created.status_code == 201, created.text
    dep_id = created.json()["deployment_id"]

    await eventually(
        lambda: len([c for c in rig.commands if c.type is CommandType.DEPLOY]) >= 3
    )  # ticker retries, no ack yet
    deploys = [c for c in rig.commands if c.type is CommandType.DEPLOY]
    assert len({c.command_id for c in deploys}) == 1 and len({c.generation for c in deploys}) == 1  # idempotent resend
    assert verify_command(rig.secret, deploys[0])  # authentic: signed with the device's derived credential

    rig.ack(deploys[0], "STAGED")  # the ack arrives over MQTT; the engine reacts without waiting for the next tick
    await eventually(lambda: any(c.type is CommandType.ACTIVATE for c in rig.commands))
    activate = next(c for c in rig.commands if c.type is CommandType.ACTIVATE)
    assert activate.generation > deploys[0].generation  # monotonic per-device fence
    rig.ack(activate, "ACTIVE")

    ident = ModelIdentity(name="anomaly", version=1, sha256="1" * 64)
    rig.heartbeat("edge-01", active=ident)  # then health observation arrives from heartbeats
    final: dict = {}  # type: ignore[type-arg]

    async def done() -> bool:
        final.update((await rig.api("GET", f"/api/v1/deployments/{dep_id}", role="viewer")).json())
        return final["state"] in ("COMPLETED", "FAILED", "ROLLED_BACK")

    deadline = asyncio.get_running_loop().time() + 6
    while not await done():
        assert asyncio.get_running_loop().time() < deadline, final
        await asyncio.sleep(0.1)
    assert final["state"] in ("COMPLETED", "ROLLED_BACK")  # progressed via ticker/heartbeats, not stuck
    with session_scope(rig.ctx.sessions) as db:
        from fleetops.persistence.tables import AuditEvent  # noqa: PLC0415

        actions = {a for (a,) in db.execute(select(AuditEvent.action).where(AuditEvent.deployment_id == dep_id))}
    assert {"deployment.created", "command.issued", "device.ack"} <= actions


async def test_lifespan_shutdown_cancels_background_tasks_cleanly(cp_settings: ControlPlaneSettings, tmp_path) -> None:  # type: ignore[no-untyped-def]
    before = {t for t in asyncio.all_tasks() if not t.done()}
    registry = MlflowRegistry(
        cp_settings.mlflow_tracking_uri, cp_settings.mlflow_experiment, (tmp_path / "ml").as_uri()
    )
    app = create_app(cp_settings, bus=InMemoryBus(InMemoryBroker(), "cp"), registry=registry)
    async with app.router.lifespan_context(app):
        assert any("_ticker" in repr(t) for t in asyncio.all_tasks())  # the ticker is running
        resp = await httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t").get("/healthz")
        assert resp.status_code == 200
    await asyncio.sleep(0.05)
    leftover = {t for t in asyncio.all_tasks() if not t.done()} - before
    assert not [t for t in leftover if "_ticker" in repr(t)]  # cancelled and awaited, not leaked


async def test_requests_before_startup_are_rejected_not_crashing(cp_settings: ControlPlaneSettings) -> None:
    app = create_app(cp_settings, bus=None)  # lifespan not entered yet: context missing
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        r = await client.get("/api/v1/devices", headers={"Authorization": f"Bearer {KEYS['viewer']}"})
    assert r.status_code == 503 and json.loads(r.text)["error"]["message"] == "control plane starting"


async def test_background_work_runs_on_a_small_fixed_pool(cp_settings: ControlPlaneSettings, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Regression: the default executor grew one thread at a time up to cpu_count + 4 (flagged by the soak test)."""
    app = create_app(cp_settings, bus=InMemoryBus(InMemoryBroker(), "cp"), run_background=False)
    async with app.router.lifespan_context(app):
        names = await asyncio.gather(*(asyncio.to_thread(lambda: threading.current_thread().name) for _ in range(40)))
    assert all(n.startswith("fleetops-bg") for n in names)
    assert len(set(names)) <= BACKGROUND_WORKERS
