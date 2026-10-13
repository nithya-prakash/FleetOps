"""Rollout engine against a real SQLite database with a fake command publisher and a fake clock."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from fleetops.deployment.engine import ConflictError, DeploymentSpec, InvalidRequestError, RolloutEngine
from fleetops.domain.messages import (
    AckOutcome,
    Command,
    CommandType,
    DeviceModelState,
    Heartbeat,
    ModelStats,
    StatusMessage,
)
from fleetops.domain.models import (
    Compatibility,
    DeviceCapabilities,
    HealthCriteria,
    ModelIdentity,
    RolloutStrategy,
    utcnow,
)
from fleetops.monitoring.metrics import ControlPlaneMetrics
from fleetops.persistence.db import make_engine, make_session_factory, session_scope, upgrade_database
from fleetops.persistence.tables import Deployment, DeploymentTarget, Device, ModelVersion

MASTER = "engine-test-master-key-000000"
SHA = {1: "1" * 64, 2: "2" * 64}
CAPS = DeviceCapabilities(agent_version="0.1.0", formats=["mlp-v1"])
CRIT = HealthCriteria(
    ack_timeout_seconds=30,
    observation_seconds=5,
    min_inferences=10,
    health_timeout_seconds=60,
    max_command_attempts=3,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = utcnow()

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class FakePublisher:
    def __init__(self) -> None:
        self.sent: list[Command] = []
        self.up = True

    def publish(self, command: Command) -> bool:
        self.sent.append(command)
        return self.up

    def of(self, ctype: CommandType, device: str | None = None) -> list[Command]:
        return [c for c in self.sent if c.type == ctype and (device is None or c.device_id == device)]


class Env:
    def __init__(self, tmp: Path) -> None:
        url = f"sqlite:///{tmp / 'e.db'}"
        upgrade_database(url)
        self.sessions = make_session_factory(make_engine(url))
        self.clock, self.pub = FakeClock(), FakePublisher()
        self.engine = self.make_engine(self.pub)
        with session_scope(self.sessions) as db:
            for name, ver in [("anomaly", 1), ("anomaly", 2), ("other", 1)]:
                db.add(
                    ModelVersion(
                        name=name,
                        version=ver,
                        sha256=SHA.get(ver, "3" * 64),
                        size_bytes=100,
                        framework="safetensors-numpy",
                        compatibility=Compatibility().model_dump(),
                        input_schema={"features": [{"name": "f", "dtype": "float32", "min": 0, "max": 1}]},
                        metrics={},
                        params={},
                        reference_stats={},
                        mlflow_run_id="r",
                        created_at=utcnow(),
                    )
                )
            for i in (1, 2):
                prev = ModelIdentity(name="anomaly", version=1, sha256=SHA[1]).model_dump()
                db.add(
                    Device(
                        device_id=f"edge-0{i}",
                        group_name="line-a",
                        status="ONLINE",
                        registered_at=utcnow(),
                        capabilities=CAPS.model_dump(),
                        last_seen_at=utcnow(),
                        last_heartbeat={"state": {"active": prev}},
                        active_model_name="anomaly",
                        active_model_version=1,
                    )
                )

    def make_engine(self, publisher: FakePublisher) -> RolloutEngine:
        return RolloutEngine(
            self.sessions,
            publisher,
            ControlPlaneMetrics(),
            MASTER,
            command_ttl_seconds=100,
            backoff_base=5,
            backoff_max=20,
            clock=self.clock,
        )

    def spec(self, version: int = 2, **kw: object) -> DeploymentSpec:
        return DeploymentSpec(
            model_name="anomaly",
            model_version=version,
            group="line-a",
            strategy=RolloutStrategy(canary_count=1, batch_percent=100),
            health_criteria=CRIT,
            **kw,  # type: ignore[arg-type]
        )

    def dep(self, dep_id: str) -> Deployment:
        with session_scope(self.sessions) as db:
            d = db.get(Deployment, dep_id)
            assert d is not None
            return d

    def targets(self, dep_id: str) -> dict[str, DeploymentTarget]:
        with session_scope(self.sessions) as db:
            rows = db.scalars(select(DeploymentTarget).where(DeploymentTarget.deployment_id == dep_id)).all()
            return {t.device_id: t for t in rows}

    def ack(self, cmd: Command, outcome: AckOutcome, engine: RolloutEngine | None = None, **kw: object) -> None:
        (engine or self.engine).record_ack(
            StatusMessage(
                kind="ack",
                device_id=cmd.device_id,
                sent_at=utcnow(),
                command_id=cmd.command_id,
                command_type=cmd.type,
                deployment_id=cmd.deployment_id,
                generation=cmd.generation,
                outcome=outcome,
                **kw,  # type: ignore[arg-type]
            )
        )

    def heartbeat(
        self, device: str, version: int, count: int, errors: int = 0, secs: float = 10, tick: bool = True
    ) -> None:
        """Deliver a heartbeat; ``tick`` also runs the periodic ticker pass (production: every second), which is what
        evaluates health observations now that a heartbeat no longer advances the whole deployment by itself."""
        ident = ModelIdentity(name="anomaly", version=version, sha256=SHA[version])
        self.engine.observe_heartbeat(
            Heartbeat(
                device_id=device,
                sent_at=utcnow(),
                boot_id="b",
                agent_version="0.1.0",
                uptime_seconds=1,
                state=DeviceModelState(active=ident),
                capabilities=CAPS,
                model_stats=ModelStats(
                    name="anomaly",
                    version=version,
                    inference_count=count,
                    inference_errors=errors,
                    latency_p50_ms=1,
                    latency_p95_ms=2,
                    active_seconds=secs,
                ),
                cpu_percent=5,
                memory_mb=50,
            )
        )
        if tick:
            self.engine.advance_all()


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def stage_and_activate(env: Env, dep_id: str, device: str = "edge-01") -> None:
    deploy = env.pub.of(CommandType.DEPLOY, device)[-1]
    env.ack(deploy, AckOutcome.STAGED)
    activate = env.pub.of(CommandType.ACTIVATE, device)[-1]
    env.ack(activate, AckOutcome.ACTIVE)
    assert env.targets(dep_id)[device].state == "ACTIVE"


def test_create_sends_signed_deploy_to_canary_only(env: Env) -> None:
    created = env.engine.create_deployment(env.spec(), "op")
    deploys = env.pub.of(CommandType.DEPLOY)
    assert [c.device_id for c in deploys] == ["edge-01"]
    cmd = deploys[0]
    assert cmd.manifest and cmd.manifest.artifact.sha256 == SHA[2] and cmd.signature
    assert cmd.generation == 1
    t = env.targets(created.deployment_id)
    assert (t["edge-01"].state, t["edge-02"].state) == ("DISTRIBUTING", "PENDING")
    assert t["edge-01"].previous_model["version"] == 1  # rollback destination captured


def test_unacked_command_is_resent_with_backoff_then_fails_and_halts(env: Env) -> None:
    dep_id = env.engine.create_deployment(env.spec(), "op").deployment_id
    first = env.pub.of(CommandType.DEPLOY)[0]
    env.engine.advance(dep_id)
    assert len(env.pub.sent) == 1  # before backoff elapses: nothing new
    env.clock.advance(6)
    env.engine.advance(dep_id)
    assert [c.command_id for c in env.pub.sent] == [first.command_id] * 2  # same id => device dedups
    env.clock.advance(11)
    env.engine.advance(dep_id)
    assert len(env.pub.sent) == 3
    env.clock.advance(21)  # attempts exhausted (max 3) -> fail, halt, nothing activated so nothing to roll back
    env.engine.advance(dep_id)
    d = env.dep(dep_id)
    assert d.state == "ROLLED_BACK" and "no STAGED acknowledgement" in (d.reason or "") + (
        env.targets(dep_id)["edge-01"].last_error or ""
    )
    assert env.targets(dep_id)["edge-02"].state == "CANCELLED"
    assert not env.pub.of(CommandType.ACTIVATE)


def test_expiry_reissue_when_timeout_is_longer_than_ttl(env: Env) -> None:
    spec = env.spec()
    spec.health_criteria = CRIT.model_copy(update={"ack_timeout_seconds": 3000})
    dep_id = env.engine.create_deployment(spec, "op").deployment_id
    first = env.pub.of(CommandType.DEPLOY)[0]
    env.clock.advance(101)
    env.engine.advance(dep_id)
    second = env.pub.of(CommandType.DEPLOY)[-1]
    assert second.command_id != first.command_id and second.generation > first.generation


def test_unknown_duplicate_and_stale_acks_are_ignored(env: Env) -> None:
    dep_id = env.engine.create_deployment(env.spec(), "op").deployment_id
    deploy = env.pub.of(CommandType.DEPLOY)[0]
    stale = deploy.model_copy(update={"command_id": "not-the-pending-command"})
    env.ack(stale, AckOutcome.STAGED)
    assert env.targets(dep_id)["edge-01"].state == "DISTRIBUTING"
    env.ack(deploy, AckOutcome.STAGED)
    env.ack(deploy, AckOutcome.STAGED)  # duplicate delivery
    assert env.targets(dep_id)["edge-01"].state == "STAGED"
    assert len(env.pub.of(CommandType.ACTIVATE)) == 1  # exactly one activation was requested


def test_no_health_observation_times_out_and_rollback_is_retried_until_acked(env: Env) -> None:
    dep_id = env.engine.create_deployment(env.spec(), "op").deployment_id
    stage_and_activate(env, dep_id)
    env.clock.advance(61)  # health timeout 60, no heartbeat observations arrived
    env.engine.advance(dep_id)
    assert env.dep(dep_id).state == "ROLLING_BACK"
    rb = env.pub.of(CommandType.ROLLBACK, "edge-01")
    assert len(rb) == 1 and rb[0].rollback_to is not None
    assert rb[0].rollback_to.model is not None and rb[0].rollback_to.model.version == 1
    assert rb[0].rollback_to.manifest is not None  # allows re-fetch if the local copy vanished
    for _ in range(5):  # unacknowledged rollback keeps being resent (no attempt cap on rollback)
        env.clock.advance(25)
        env.engine.advance(dep_id)
    assert len(env.pub.of(CommandType.ROLLBACK, "edge-01")) >= 5
    sent = env.pub.of(CommandType.ROLLBACK, "edge-01")
    assert len({c.command_id for c in sent}) == 2  # resent unchanged until the 100s ttl, then reissued fresh
    assert sent[-1].generation > rb[0].generation
    assert env.dep(dep_id).state == "ROLLING_BACK"
    env.ack(sent[-1], AckOutcome.ROLLED_BACK)
    assert env.dep(dep_id).state == "ROLLED_BACK"
    assert env.targets(dep_id)["edge-01"].state == "ROLLED_BACK"
    assert env.targets(dep_id)["edge-02"].state == "CANCELLED"


def test_healthy_observation_promotes_next_batch_and_completes(env: Env) -> None:
    dep_id = env.engine.create_deployment(env.spec(), "op").deployment_id
    stage_and_activate(env, dep_id)
    env.heartbeat("edge-01", 2, count=3, secs=1)  # too early to pass
    assert env.targets(dep_id)["edge-02"].state == "PENDING"
    env.heartbeat("edge-01", 2, count=50, secs=10)
    assert env.targets(dep_id)["edge-01"].verified_at is not None
    assert env.targets(dep_id)["edge-02"].state == "DISTRIBUTING"  # promoted only after the canary passed
    stage_and_activate(env, dep_id, "edge-02")
    env.heartbeat("edge-02", 2, count=50, secs=10)
    assert env.dep(dep_id).state == "COMPLETED"


def test_failing_observation_triggers_automatic_rollback_with_reason(env: Env) -> None:
    dep_id = env.engine.create_deployment(env.spec(), "op").deployment_id
    stage_and_activate(env, dep_id)
    env.heartbeat("edge-01", 2, count=20, errors=20, secs=1)
    d = env.dep(dep_id)
    assert d.state == "ROLLING_BACK" and "error rate" in (d.reason or "")
    assert env.targets(dep_id)["edge-02"].state == "CANCELLED"  # never touched
    assert not env.pub.of(CommandType.DEPLOY, "edge-02")


def test_rollback_without_known_good_model_deactivates(env: Env) -> None:
    with session_scope(env.sessions) as db:
        for dev in db.scalars(select(Device)):
            dev.last_heartbeat = {"state": {"active": None}}
            dev.active_model_name = dev.active_model_version = None
    dep_id = env.engine.create_deployment(env.spec(1), "op").deployment_id
    stage_and_activate(env, dep_id)
    env.engine.rollback(dep_id, "op", "operator decision")
    rb = env.pub.of(CommandType.ROLLBACK, "edge-01")[0]
    assert rb.rollback_to is not None and rb.rollback_to.model is None


def test_crash_recovery_resumes_from_database_and_resends(env: Env) -> None:
    env.pub.up = False  # broker down while creating: command is persisted but cannot be sent
    dep_id = env.engine.create_deployment(env.spec(), "op").deployment_id
    assert env.targets(dep_id)["edge-01"].state == "DISTRIBUTING"
    pending = env.targets(dep_id)["edge-01"].pending_command
    assert pending is not None
    # "restart": a brand-new engine instance with no memory of the earlier one
    fresh_pub = FakePublisher()
    fresh = env.make_engine(fresh_pub)
    env.clock.advance(6)
    fresh.advance_all()
    resent = fresh_pub.of(CommandType.DEPLOY)
    assert [c.command_id for c in resent] == [pending["command_id"]]
    env.ack(resent[0], AckOutcome.STAGED, engine=fresh)
    assert env.targets(dep_id)["edge-01"].state == "STAGED"
    assert fresh_pub.of(CommandType.ACTIVATE)


def test_cancel_aborts_unactivated_targets_and_blocks_second_cancel(env: Env) -> None:
    dep_id = env.engine.create_deployment(env.spec(), "op").deployment_id
    env.engine.cancel(dep_id, "op")  # DEPLOY sent, no ack yet
    assert env.dep(dep_id).state == "CANCELLED"
    assert env.pub.sent[-1].type is CommandType.ABORT
    t = env.targets(dep_id)
    assert t["edge-01"].state == t["edge-02"].state == "CANCELLED"
    with pytest.raises(ConflictError):
        env.engine.cancel(dep_id, "op")


def test_cancel_refused_while_activation_in_flight_and_keeps_active_targets(env: Env) -> None:
    dep_id = env.engine.create_deployment(env.spec(), "op").deployment_id
    env.ack(env.pub.of(CommandType.DEPLOY)[0], AckOutcome.STAGED)  # engine immediately requests ACTIVATE
    with pytest.raises(ConflictError, match="activation in flight"):
        env.engine.cancel(dep_id, "op")
    assert env.dep(dep_id).state == "IN_PROGRESS"
    env.ack(env.pub.of(CommandType.ACTIVATE)[0], AckOutcome.ACTIVE)
    env.engine.cancel(dep_id, "op")  # promotion stops; the already-active canary stays (rollback is separate)
    t = env.targets(dep_id)
    assert (t["edge-01"].state, t["edge-02"].state) == ("ACTIVE", "CANCELLED")
    assert env.dep(dep_id).state == "CANCELLED"


def test_overlapping_deployment_of_other_model_conflicts_and_same_model_supersedes(env: Env) -> None:
    first = env.engine.create_deployment(env.spec(2), "op").deployment_id
    other = DeploymentSpec(model_name="other", model_version=1, group="line-a")
    with pytest.raises(ConflictError, match="in-flight"):
        env.engine.create_deployment(other, "op")
    second = env.engine.create_deployment(env.spec(2, device_ids=["edge-01", "edge-02"]), "op").deployment_id
    d = env.dep(first)
    assert d.state == "CANCELLED" and d.superseded_by == second


def test_idempotency_key_semantics_and_eligibility(env: Env) -> None:
    a = env.engine.create_deployment(env.spec(), "op", "key-1")
    b = env.engine.create_deployment(env.spec(), "op", "key-1")
    assert (a.deployment_id, b.created) == (b.deployment_id, False)
    with pytest.raises(ConflictError, match="Idempotency-Key"):
        env.engine.create_deployment(env.spec(1), "op", "key-1")
    with session_scope(env.sessions) as db:
        for dev in db.scalars(select(Device)):
            dev.status = "OFFLINE"
    with pytest.raises(InvalidRequestError, match="no eligible devices"):
        env.engine.create_deployment(env.spec(1), "op")


def test_heartbeat_without_a_state_change_does_not_advance_the_deployment(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: every heartbeat used to run a full advance over all targets (O(targets) per heartbeat on the single
    MQTT ingest thread), which saturated it at ~100 devices and made rollouts time out under load."""
    dep_id = env.engine.create_deployment(env.spec(), "op").deployment_id
    stage_and_activate(env, dep_id)
    calls: list[str] = []
    original = env.engine._advance
    monkeypatch.setattr(env.engine, "_advance", lambda *a, **k: (calls.append("advance"), original(*a, **k))[1])
    for count in (1, 2, 3, 4, 5):
        env.heartbeat("edge-01", 2, count=count, secs=1, tick=False)
    env.heartbeat("edge-02", 1, count=5, tick=False)  # a device that is not the canary yet
    assert calls == []  # observations alone are left to the ticker
    assert env.targets(dep_id)["edge-01"].observation is not None  # ...but they are still recorded
    env.engine.advance_all()
    assert calls  # the ticker does the work


def test_heartbeat_that_heals_a_lost_ack_still_advances_immediately(env: Env) -> None:
    dep_id = env.engine.create_deployment(env.spec(), "op").deployment_id
    ident = ModelIdentity(name="anomaly", version=2, sha256=SHA[2])
    env.engine.observe_heartbeat(
        Heartbeat(
            device_id="edge-01",
            sent_at=utcnow(),
            boot_id="b",
            agent_version="0.1.0",
            uptime_seconds=1,
            state=DeviceModelState(active=ModelIdentity(name="anomaly", version=1, sha256=SHA[1]), staged=ident),
            capabilities=CAPS,
            cpu_percent=5,
            memory_mb=50,
        )
    )
    assert env.targets(dep_id)["edge-01"].state == "STAGED"  # healed from the heartbeat...
    assert env.pub.of(
        CommandType.ACTIVATE, "edge-01"
    )  # ...and the next command went out without waiting for the ticker
