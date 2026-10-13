"""Rollout engine: persisted, restart-safe, idempotent canary/batch deployments with rollback.

The engine is a function over database rows. ``advance`` can be called at any time, from any
thread (a per-deployment lock serialises it): by the ticker, by device events, or after a
control-plane restart. Commands are collected in an outbox and published only after the
transaction commits; a lost publish is healed by the retry timer stored on the target.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from pydantic import Field
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from fleetops.config.logging import log_context
from fleetops.domain.health import HealthObservation, Verdict, evaluate_health
from fleetops.domain.messages import (
    AckOutcome,
    Command,
    CommandType,
    DeviceModelState,
    Heartbeat,
    RollbackTarget,
    StatusMessage,
)
from fleetops.domain.models import (
    ArtifactRef,
    Compatibility,
    DeviceCapabilities,
    HealthCriteria,
    InputSchema,
    Manifest,
    ModelIdentity,
    ModelRef,
    RolloutStrategy,
    StrictModel,
    compatibility_problems,
    utcnow,
)
from fleetops.domain.planning import plan_batches
from fleetops.domain.quality import (
    Failure,
    ModelEvidence,
    QualityGate,
    QualityOverride,
    QualityReport,
    evaluate_quality_gate,
)
from fleetops.domain.states import (
    IN_FLIGHT_DEPLOYMENT_STATES,
    TERMINAL_DEPLOYMENT_STATES,
    DeploymentState,
    TargetState,
    ensure_deployment_transition,
    ensure_target_transition,
)
from fleetops.monitoring.metrics import ControlPlaneMetrics
from fleetops.persistence.audit import record_audit
from fleetops.persistence.db import session_scope
from fleetops.persistence.tables import Deployment, DeploymentTarget, Device, ModelVersion
from fleetops.resilience import backoff_delay
from fleetops.security.device_identity import derive_device_secret, sign_command

logger = logging.getLogger(__name__)

D, T = DeploymentState, TargetState


class EngineError(Exception):
    """Base for errors the API maps to HTTP statuses."""


class NotFoundError(EngineError):
    pass


class ConflictError(EngineError):
    pass


class InvalidRequestError(EngineError):
    pass


class ForbiddenError(EngineError):
    pass


class QualityGateError(EngineError):
    """The candidate model failed the requested quality policy (nothing was created)."""

    def __init__(self, report: QualityReport) -> None:
        super().__init__("model quality gate failed: " + report.summary())
        self.report = report


class CommandPublisher(Protocol):
    def publish(self, command: Command) -> bool: ...


class DeploymentSpec(StrictModel):
    model_name: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,62}$")
    model_version: int = Field(ge=1)
    group: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,62}$")
    device_ids: list[str] | None = Field(default=None, max_length=1000)
    strategy: RolloutStrategy = Field(default_factory=RolloutStrategy)
    health_criteria: HealthCriteria = Field(default_factory=HealthCriteria)
    quality_gate: QualityGate | None = None
    quality_override: QualityOverride | None = None  # admin-only; always audited


@dataclass
class CreatedDeployment:
    deployment_id: str
    created: bool
    skipped: dict[str, str] = field(default_factory=dict)
    superseded: list[str] = field(default_factory=list)  # older in-flight deployments of the same model + group


@dataclass
class _Outbox:
    commands: list[Command] = field(default_factory=list)
    touched: set[str] = field(default_factory=set)


def request_hash(spec: DeploymentSpec) -> str:
    return hashlib.sha256(json.dumps(spec.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


def _needs_rollback(t: DeploymentTarget) -> bool:
    return t.state in (T.DISTRIBUTING, T.STAGED, T.ACTIVE) or (t.state == T.FAILED and t.rollback_requested)


class RolloutEngine:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        publisher: CommandPublisher,
        metrics: ControlPlaneMetrics,
        device_key_master: str,
        *,
        command_ttl_seconds: float = 300.0,
        require_quality_gate: bool = False,
        backoff_base: float = 5.0,
        backoff_max: float = 60.0,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._sessions = sessions
        self._publisher = publisher
        self._metrics = metrics
        self._master = device_key_master
        self._ttl = timedelta(seconds=command_ttl_seconds)
        self._backoff_base, self._backoff_max = backoff_base, backoff_max
        self._clock = clock
        self._require_gate = require_quality_gate
        self._locks: defaultdict[str, threading.RLock] = defaultdict(threading.RLock)
        self._create_lock = threading.Lock()

    # ================================================================== public API

    def create_deployment(
        self, spec: DeploymentSpec, actor: str, idempotency_key: str | None = None, *, may_override: bool = False
    ) -> CreatedDeployment:
        outbox = _Outbox()
        with self._create_lock:
            with session_scope(self._sessions) as db:
                result = self._create(db, spec, actor, idempotency_key, outbox, may_override)
            self._publish(outbox)
            # Superseding changes *other* deployments, so it must happen under each one's own lock (not inside the
            # creating transaction): otherwise a concurrent advance() could overwrite the cancellation with stale
            # state (a lost update that once left two deployments of the same model IN_PROGRESS/COMPLETED).
            for old_id in result.superseded:
                self._supersede_locked(old_id, result.deployment_id)
        if result.created:
            self.advance(result.deployment_id)
        return result

    def cancel(self, deployment_id: str, actor: str) -> None:
        self._mutate(deployment_id, lambda db, dep, ob: self._cancel(db, dep, actor, ob))

    def rollback(self, deployment_id: str, actor: str, reason: str = "manual rollback") -> None:
        self._mutate(deployment_id, lambda db, dep, ob: self._manual_rollback(db, dep, actor, reason, ob))

    def advance(self, deployment_id: str) -> None:
        self._mutate(deployment_id, self._advance)

    def advance_all(self) -> None:
        with session_scope(self._sessions) as db:
            ids = list(
                db.scalars(
                    select(Deployment.deployment_id).where(
                        Deployment.state.in_([s.value for s in IN_FLIGHT_DEPLOYMENT_STATES])
                    )
                )
            )
        for dep_id in ids:
            try:
                self.advance(dep_id)
            except Exception:
                logger.exception("advance failed", extra={"deployment_id": dep_id})

    def send_device_command(self, device_id: str, ctype: CommandType, actor: str, **fields: Any) -> Command:
        """Issue a one-off signed command not tied to a deployment (e.g. simulation faults)."""
        outbox = _Outbox()
        with session_scope(self._sessions) as db:
            device = db.get(Device, device_id)
            if device is None:
                raise NotFoundError(f"device {device_id} not found")
            device.command_generation += 1
            now = self._clock()
            cmd = Command(
                command_id=uuid.uuid4().hex,
                type=ctype,
                device_id=device_id,
                generation=device.command_generation,
                issued_at=now,
                expires_at=now + self._ttl,
                correlation_id=uuid.uuid4().hex[:16],
                **fields,
            )
            cmd = sign_command(derive_device_secret(self._master, device_id, device.key_epoch), cmd)
            record_audit(db, actor, "command.issued", "command", cmd.command_id, device_id=device_id, type=ctype.value)
            outbox.commands.append(cmd)
        self._publish(outbox)
        return cmd

    # ================================================================== device events

    def record_ack(self, msg: StatusMessage) -> None:
        if msg.deployment_id is None or msg.outcome is None:
            return
        self._mutate(msg.deployment_id, lambda db, dep, ob: self._apply_ack(db, dep, msg, ob), missing_ok=True)

    def observe_heartbeat(self, hb: Heartbeat) -> None:
        with session_scope(self._sessions) as db:
            rows = db.execute(
                select(DeploymentTarget.deployment_id)
                .join(Deployment)
                .where(
                    DeploymentTarget.device_id == hb.device_id,
                    Deployment.state.in_([s.value for s in IN_FLIGHT_DEPLOYMENT_STATES]),
                )
            ).all()
        for (dep_id,) in set(rows):
            self._mutate(dep_id, lambda db, dep, ob: self._apply_heartbeat(db, dep, hb, ob), missing_ok=True)

    def device_came_online(self, device_id: str) -> None:
        """Reconcile: re-issue outstanding commands now instead of waiting for the retry timer."""
        now = self._clock()
        with session_scope(self._sessions) as db:
            targets = db.scalars(
                select(DeploymentTarget)
                .join(Deployment)
                .where(
                    DeploymentTarget.device_id == device_id,
                    DeploymentTarget.pending_command.is_not(None),
                    Deployment.state.in_([s.value for s in IN_FLIGHT_DEPLOYMENT_STATES]),
                )
            ).all()
            deps = {t.deployment_id for t in targets}
            for t in targets:
                t.next_retry_at = now
        for dep_id in deps:
            self.advance(dep_id)

    # ================================================================== internals: plumbing

    def _mutate(
        self, deployment_id: str, fn: Callable[[Session, Deployment, _Outbox], None], missing_ok: bool = False
    ) -> None:
        outbox = _Outbox()
        with self._locks[deployment_id]:
            with session_scope(self._sessions) as db:
                dep = db.get(Deployment, deployment_id)
                if dep is None:
                    if missing_ok:
                        return
                    raise NotFoundError(f"deployment {deployment_id} not found")
                with log_context(deployment_id=deployment_id):
                    fn(db, dep, outbox)
            self._publish(outbox)

    def _publish(self, outbox: _Outbox) -> None:
        sent: list[Command] = []
        for cmd in outbox.commands:
            try:
                ok = self._publisher.publish(cmd)
            except Exception:
                logger.exception("publish failed", extra={"command_id": cmd.command_id})
                ok = False
            self._metrics.commands_published.labels(cmd.type.value, "sent" if ok else "link_down").inc()
            if ok:
                sent.append(cmd)
            if not ok:
                logger.warning(
                    "command not sent (link down); retry timer will resend",
                    extra={"command_id": cmd.command_id, "device_id": cmd.device_id},
                )
        self._count_sends(sent)

    def _count_sends(self, sent: list[Command]) -> None:
        """A send consumes the attempt budget only if it reached the broker. While the broker is unreachable the timer
        keeps retrying without burning attempts; the acknowledgement timeout still bounds how long we keep trying."""
        tracked = [c for c in sent if c.deployment_id is not None]
        if not tracked:
            return
        with session_scope(self._sessions) as db:
            for cmd in tracked:
                t = self._target(db, cmd.deployment_id or "", cmd.device_id)
                if t is not None and (t.pending_command or {}).get("command_id") == cmd.command_id:
                    t.attempts += 1

    def _targets(self, db: Session, deployment_id: str) -> list[DeploymentTarget]:
        return list(
            db.scalars(
                select(DeploymentTarget)
                .where(DeploymentTarget.deployment_id == deployment_id)
                .order_by(DeploymentTarget.batch_index, DeploymentTarget.device_id)
            )
        )

    def _set_target(
        self, db: Session, dep: Deployment, t: DeploymentTarget, new: TargetState, detail: str | None = None
    ) -> None:
        ensure_target_transition(TargetState(t.state), new)
        old = t.state
        t.state, t.state_since = new.value, self._clock()
        record_audit(
            db,
            "system",
            "target.state_changed",
            "deployment_target",
            f"{dep.deployment_id}/{t.device_id}",
            device_id=t.device_id,
            deployment_id=dep.deployment_id,
            **{"from": old, "to": new.value, "detail": detail},
        )

    def _set_dep(
        self, db: Session, dep: Deployment, new: DeploymentState, actor: str, reason: str | None = None
    ) -> None:
        ensure_deployment_transition(DeploymentState(dep.state), new)
        old = dep.state
        dep.state, dep.updated_at = new.value, self._clock()
        if reason:
            dep.reason = reason
        if new in TERMINAL_DEPLOYMENT_STATES:
            dep.completed_at = self._clock()
            self._metrics.deployments.labels(new.value).inc()
        record_audit(
            db,
            actor,
            f"deployment.{new.value.lower()}",
            "deployment",
            dep.deployment_id,
            deployment_id=dep.deployment_id,
            **{"from": old, "reason": reason},
        )
        logger.info("deployment %s -> %s", old, new.value, extra={"reason": reason})

    def _forward_ttl(self, crit: HealthCriteria) -> timedelta:
        """DEPLOY/ACTIVATE must not outlive the engine's patience: a command still queued for an unreachable device
        after the acknowledgement timeout would run long after the deployment gave up on it."""
        return min(self._ttl, timedelta(seconds=crit.ack_timeout_seconds + 5))

    def _issue(
        self,
        db: Session,
        dep: Deployment,
        t: DeploymentTarget,
        ctype: CommandType,
        ob: _Outbox,
        *,
        ttl: timedelta | None = None,
        track: bool = True,
        **fields: Any,
    ) -> None:
        device = db.get(Device, t.device_id)
        assert device is not None
        device.command_generation += 1
        now = self._clock()
        cmd = Command(
            command_id=uuid.uuid4().hex,
            type=ctype,
            device_id=t.device_id,
            generation=device.command_generation,
            issued_at=now,
            expires_at=now + (ttl or self._ttl),
            deployment_id=dep.deployment_id,
            correlation_id=dep.deployment_id,
            **fields,
        )
        cmd = sign_command(derive_device_secret(self._master, t.device_id, device.key_epoch), cmd)
        if track:  # untracked commands are best-effort (no retry timer, no acknowledgement awaited)
            t.pending_command = cmd.model_dump(mode="json")
            t.attempts = 0  # counts sends that actually reached the broker without being acknowledged
            t.next_retry_at = now + timedelta(seconds=backoff_delay(1, self._backoff_base, self._backoff_max))
        ob.commands.append(cmd)
        record_audit(
            db,
            "system",
            "command.issued",
            "command",
            cmd.command_id,
            device_id=t.device_id,
            deployment_id=dep.deployment_id,
            type=ctype.value,
            generation=cmd.generation,
        )

    def _resend_if_due(self, db: Session, dep: Deployment, t: DeploymentTarget, ob: _Outbox) -> None:
        if t.pending_command is None or t.next_retry_at is None or self._clock() < t.next_retry_at:
            return
        old = Command.model_validate(t.pending_command)
        self._metrics.command_retries.labels(old.type.value).inc()
        if self._clock() >= old.expires_at:  # fresh command (new id + generation); device ops are idempotent
            attempts = t.attempts
            forward = old.type in (CommandType.DEPLOY, CommandType.ACTIVATE)
            crit = HealthCriteria.model_validate(dep.manifest["health_criteria"])
            self._issue(
                db,
                dep,
                t,
                old.type,
                ob,
                ttl=self._forward_ttl(crit) if forward else None,
                manifest=old.manifest,
                activate=old.activate,
                rollback_to=old.rollback_to,
            )
            t.attempts = attempts
        else:
            t.next_retry_at = self._clock() + timedelta(
                seconds=backoff_delay(max(t.attempts, 1), self._backoff_base, self._backoff_max)
            )
            ob.commands.append(old)

    def _clear_pending(self, t: DeploymentTarget) -> None:
        t.pending_command, t.next_retry_at, t.attempts = None, None, 0

    # ================================================================== creation

    def _create(
        self, db: Session, spec: DeploymentSpec, actor: str, key: str | None, ob: _Outbox, may_override: bool = False
    ) -> CreatedDeployment:
        rhash = request_hash(spec)
        if key:
            existing = db.scalar(select(Deployment).where(Deployment.idempotency_key == key))
            if existing is not None:
                if existing.request_hash != rhash:
                    raise ConflictError("Idempotency-Key was already used with a different request")
                return CreatedDeployment(existing.deployment_id, created=False)
        mv = db.scalar(
            select(ModelVersion).where(ModelVersion.name == spec.model_name, ModelVersion.version == spec.model_version)
        )
        if mv is None:
            raise NotFoundError(f"model {spec.model_name}:{spec.model_version} is not registered")
        devices = list(db.scalars(select(Device).where(Device.group_name == spec.group).order_by(Device.device_id)))
        if not devices:
            raise InvalidRequestError(f"group {spec.group!r} has no registered devices")
        if spec.device_ids is not None:
            wanted = set(spec.device_ids)
            unknown = wanted - {d.device_id for d in devices}
            if unknown:
                raise InvalidRequestError(f"devices not in group {spec.group!r}: {sorted(unknown)}")
            devices = [d for d in devices if d.device_id in wanted]

        compat = Compatibility.model_validate(mv.compatibility)
        eligible: list[Device] = []
        skipped: dict[str, str] = {}
        for d in devices:
            if d.status == "REVOKED":
                skipped[d.device_id] = "revoked"
            elif d.status != "ONLINE":
                skipped[d.device_id] = f"not online (status {d.status})"
            elif d.capabilities is None:
                skipped[d.device_id] = "capabilities unknown"
            elif problems := compatibility_problems(compat, DeviceCapabilities.model_validate(d.capabilities)):
                skipped[d.device_id] = "; ".join(problems)
            elif (d.active_model_name, d.active_model_version) == (spec.model_name, spec.model_version):
                skipped[d.device_id] = "already running this version"
            else:
                eligible.append(d)
        if not eligible:
            raise InvalidRequestError(f"no eligible devices: {skipped}")
        report = self._quality_check(db, spec, mv, eligible, actor, may_override)

        in_flight = [s.value for s in IN_FLIGHT_DEPLOYMENT_STATES]
        busy = db.execute(
            select(DeploymentTarget.device_id, Deployment)
            .join(Deployment)
            .where(
                Deployment.state.in_(in_flight),
                DeploymentTarget.device_id.in_([d.device_id for d in eligible]),
                DeploymentTarget.state.not_in([T.CANCELLED.value, T.ROLLED_BACK.value]),
            )
        ).all()
        superseded: dict[str, Deployment] = {}
        for device_id, other in busy:
            if (other.model_name, other.group_name) == (spec.model_name, spec.group) and other.state in (
                D.PENDING,
                D.IN_PROGRESS,
            ):
                superseded[other.deployment_id] = other
            else:
                raise ConflictError(
                    f"device {device_id} is part of in-flight deployment {other.deployment_id} ({other.state})"
                )

        dep_id = uuid.uuid4().hex
        now = self._clock()
        batches = plan_batches([d.device_id for d in eligible], spec.strategy)
        manifest = self.build_manifest(mv, dep_id, spec.group, spec.strategy, spec.health_criteria, now)
        dep = Deployment(
            deployment_id=dep_id,
            idempotency_key=key,
            request_hash=rhash,
            model_name=spec.model_name,
            model_version=spec.model_version,
            group_name=spec.group,
            state=D.PENDING.value,
            manifest=manifest.model_dump(mode="json"),
            batches=batches,
            current_batch=0,
            created_by=actor,
            created_at=now,
            updated_at=now,
            quality_report=report.model_dump(mode="json") if report else None,
        )
        db.add(dep)
        db.flush()
        if report is not None and report.overridden:
            record_audit(
                db,
                actor,
                "deployment.quality_gate_overridden",
                "deployment",
                dep_id,
                deployment_id=dep_id,
                reason=report.override_reason,
                failures=[f.model_dump(mode="json") for f in report.failures],
            )
        for bi, batch in enumerate(batches):
            for device_id in batch:
                db.add(
                    DeploymentTarget(
                        deployment_id=dep_id,
                        device_id=device_id,
                        batch_index=bi,
                        state=T.PENDING.value,
                        state_since=now,
                    )
                )
        record_audit(
            db,
            actor,
            "deployment.created",
            "deployment",
            dep_id,
            deployment_id=dep_id,
            model=f"{spec.model_name}:{spec.model_version}",
            group=spec.group,
            batches=batches,
            skipped=skipped,
        )
        db.flush()
        return CreatedDeployment(dep_id, created=True, skipped=skipped, superseded=list(superseded))

    def _quality_check(
        self,
        db: Session,
        spec: DeploymentSpec,
        mv: ModelVersion,
        eligible: list[Device],
        actor: str,
        may_override: bool,
    ) -> QualityReport | None:
        """Apply the optional quality policy. Raises ``QualityGateError`` unless it passes or is overridden."""
        if spec.quality_override is not None and not may_override:
            raise ForbiddenError("quality_override requires the admin role")
        gate = spec.quality_gate
        if gate is None:
            if not self._require_gate:
                return None
            report = QualityReport(
                passed=False,
                failures=[
                    Failure(
                        code="GATE_REQUIRED", message="this control plane requires a quality_gate on every deployment"
                    )
                ],
            )
        else:
            report = self._evaluate_gate(db, spec, gate, mv, eligible)
        if report.passed:
            return report
        if spec.quality_override is None:
            raise QualityGateError(report)
        report.overridden, report.override_reason, report.override_by = True, spec.quality_override.reason, actor
        logger.warning("quality gate overridden", extra={"actor": actor, "reason": report.override_reason})
        return report

    def _evaluate_gate(
        self, db: Session, spec: DeploymentSpec, gate: QualityGate, mv: ModelVersion, eligible: list[Device]
    ) -> QualityReport:
        candidate = ModelEvidence(mv.version, dict(mv.metrics), dict(mv.evaluation or {}))
        if gate.baseline_version is not None:
            versions = {gate.baseline_version}
        else:  # the incumbent(s): what the targeted devices run today
            versions = {
                d.active_model_version
                for d in eligible
                if d.active_model_name == spec.model_name and d.active_model_version not in (None, spec.model_version)
            }
        baselines: list[ModelEvidence] = []
        extra: list[Failure] = []
        for version in sorted(v for v in versions if v is not None):
            row = db.scalar(
                select(ModelVersion).where(ModelVersion.name == spec.model_name, ModelVersion.version == version)
            )
            if row is None:
                extra.append(
                    Failure(
                        code="MISSING_BASELINE",
                        baseline_version=version,
                        message=f"baseline v{version} is not a registered version, cannot compare",
                    )
                )
            else:
                baselines.append(ModelEvidence(row.version, dict(row.metrics), dict(row.evaluation or {})))
        report = evaluate_quality_gate(gate, candidate, baselines)
        if extra:
            report.failures += extra
            report.passed = False
        return report

    def build_manifest(
        self,
        mv: ModelVersion,
        deployment_id: str,
        group: str,
        strategy: RolloutStrategy,
        criteria: HealthCriteria,
        now: datetime,
    ) -> Manifest:
        return Manifest(
            deployment_id=deployment_id,
            created_at=now,
            model=ModelRef(name=mv.name, version=mv.version),
            artifact=ArtifactRef(
                path=f"/api/v1/device/artifacts/{mv.name}/{mv.version}", sha256=mv.sha256, size_bytes=mv.size_bytes
            ),
            input_schema=InputSchema.model_validate(mv.input_schema),
            framework=mv.framework,
            compatibility=Compatibility.model_validate(mv.compatibility),
            target_group=group,
            strategy=strategy,
            health_criteria=criteria,
        )

    def _supersede_locked(self, old_id: str, new_id: str) -> None:
        def apply(db: Session, dep: Deployment, _ob: _Outbox) -> None:
            self._supersede(db, dep, new_id)

        self._mutate(old_id, apply, missing_ok=True)

    def _supersede(self, db: Session, old: Deployment, new_id: str) -> None:
        if DeploymentState(old.state) not in (D.PENDING, D.IN_PROGRESS):
            return  # finished (or already cancelled) in the meantime: nothing to supersede
        for t in self._targets(db, old.deployment_id):
            if t.state in (T.PENDING, T.DISTRIBUTING, T.STAGED):
                self._clear_pending(t)
                self._set_target(db, old, t, T.CANCELLED, f"superseded by {new_id}")
        old.superseded_by = new_id
        self._set_dep(db, old, D.CANCELLED, "system", f"superseded by {new_id}")

    # ================================================================== forward progress

    def _advance(self, db: Session, dep: Deployment, ob: _Outbox) -> None:
        if dep.state == D.PENDING:
            self._set_dep(db, dep, D.IN_PROGRESS, "system")
        if dep.state == D.IN_PROGRESS:
            self._advance_forward(db, dep, ob)
        elif dep.state == D.ROLLING_BACK:
            self._advance_rollback(db, dep, ob)

    def _advance_forward(self, db: Session, dep: Deployment, ob: _Outbox) -> None:
        manifest = Manifest.model_validate(dep.manifest)
        crit = manifest.health_criteria
        while dep.state == D.IN_PROGRESS:
            batch = [t for t in self._targets(db, dep.deployment_id) if t.batch_index == dep.current_batch]
            now = self._clock()
            for t in batch:
                self._advance_target(db, dep, t, manifest, crit, now, ob)
            failed = [t for t in batch if t.state == T.FAILED]
            if failed:
                self._begin_rollback(
                    db, dep, "system", f"{failed[0].device_id}: {failed[0].last_error}", "automatic", ob
                )
                return
            if all(t.state == T.ACTIVE and t.verified_at is not None for t in batch):
                if dep.current_batch + 1 >= len(dep.batches):
                    self._set_dep(db, dep, D.COMPLETED, "system", "all batches healthy")
                    return
                dep.current_batch += 1
                record_audit(
                    db,
                    "system",
                    "deployment.batch_promoted",
                    "deployment",
                    dep.deployment_id,
                    deployment_id=dep.deployment_id,
                    batch=dep.current_batch,
                )
                continue
            return

    def _advance_target(
        self,
        db: Session,
        dep: Deployment,
        t: DeploymentTarget,
        manifest: Manifest,
        crit: HealthCriteria,
        now: datetime,
        ob: _Outbox,
    ) -> None:
        device = db.get(Device, t.device_id)
        assert device is not None
        if t.state == T.PENDING:
            hb_active = ((device.last_heartbeat or {}).get("state") or {}).get("active")
            t.previous_model = hb_active
            self._issue(db, dep, t, CommandType.DEPLOY, ob, ttl=self._forward_ttl(crit), manifest=manifest)
            self._set_target(db, dep, t, T.DISTRIBUTING)
        elif t.state == T.DISTRIBUTING:
            if self._expired(t, crit, now, t.state_since):
                self._fail(db, dep, t, "no STAGED acknowledgement before timeout / attempts exhausted", ob)
            else:
                self._resend_if_due(db, dep, t, ob)
        elif t.state == T.STAGED:
            pending_type = (t.pending_command or {}).get("type")
            if pending_type != CommandType.ACTIVATE.value:
                t.activation_sent_at = now
                self._issue(db, dep, t, CommandType.ACTIVATE, ob, ttl=self._forward_ttl(crit), activate=manifest.model)
            elif self._expired(t, crit, now, t.activation_sent_at or t.state_since):
                self._fail(db, dep, t, "no ACTIVE acknowledgement before timeout / attempts exhausted", ob)
            else:
                self._resend_if_due(db, dep, t, ob)
        elif t.state == T.ACTIVE and t.verified_at is None:
            self._evaluate(db, dep, t, crit, now)

    def _expired(self, t: DeploymentTarget, crit: HealthCriteria, now: datetime, since: datetime) -> bool:
        return (now - since).total_seconds() > crit.ack_timeout_seconds or t.attempts > crit.max_command_attempts

    def _fail(self, db: Session, dep: Deployment, t: DeploymentTarget, reason: str, ob: _Outbox | None = None) -> None:
        # Failed before any activation was requested: the device may still hold (or later receive) a queued DEPLOY.
        # Tell it, best effort, to discard the candidate. Activation can never follow, so nothing waits on this.
        discard = ob is not None and t.state in (T.DISTRIBUTING, T.STAGED) and t.activation_sent_at is None
        self._clear_pending(t)
        t.last_error = reason
        self._metrics.target_outcomes.labels("failed").inc()
        self._set_target(db, dep, t, T.FAILED, reason)
        if discard and ob is not None:
            self._issue(
                db, dep, t, CommandType.ABORT, ob, track=False, activate=ModelRef.model_validate(dep.manifest["model"])
            )

    def _evaluate(self, db: Session, dep: Deployment, t: DeploymentTarget, crit: HealthCriteria, now: datetime) -> None:
        obs = HealthObservation.model_validate(t.observation["metrics"]) if t.observation else None
        result = evaluate_health(crit, obs) if obs else None
        if result and result.verdict == Verdict.PASS:
            t.verified_at = now
            self._metrics.target_outcomes.labels("verified").inc()
            record_audit(
                db,
                "system",
                "target.health_passed",
                "deployment_target",
                f"{dep.deployment_id}/{t.device_id}",
                device_id=t.device_id,
                deployment_id=dep.deployment_id,
                observation=t.observation,
            )
        elif result and result.verdict == Verdict.FAIL:
            record_audit(
                db,
                "system",
                "target.health_failed",
                "deployment_target",
                f"{dep.deployment_id}/{t.device_id}",
                device_id=t.device_id,
                deployment_id=dep.deployment_id,
                reasons=result.reasons,
                observation=t.observation,
            )
            self._fail(db, dep, t, "health check failed: " + "; ".join(result.reasons))
            t.rollback_requested = True
        elif t.activated_at and (now - t.activated_at).total_seconds() > crit.health_timeout_seconds:
            self._fail(db, dep, t, "no healthy observation within health timeout")
            t.rollback_requested = True

    # ================================================================== rollback

    def _begin_rollback(self, db: Session, dep: Deployment, actor: str, reason: str, trigger: str, ob: _Outbox) -> None:
        self._set_dep(db, dep, D.ROLLING_BACK, actor, reason)
        dep.rollback_started_at = self._clock()
        self._metrics.rollbacks.labels(trigger).inc()
        for t in self._targets(db, dep.deployment_id):
            if t.state == T.PENDING:
                self._set_target(db, dep, t, T.CANCELLED, "rollout halted")
            elif t.state == T.FAILED and t.activation_sent_at is not None:
                t.rollback_requested = True
                self._clear_pending(t)
        self._advance_rollback(db, dep, ob)

    def _advance_rollback(self, db: Session, dep: Deployment, ob: _Outbox) -> None:
        manifest = Manifest.model_validate(dep.manifest)
        unresolved = False
        for t in self._targets(db, dep.deployment_id):
            if t.state == T.PENDING:
                self._set_target(db, dep, t, T.CANCELLED, "rollout halted")
                continue
            if not _needs_rollback(t):
                continue
            unresolved = True
            pending_type = (t.pending_command or {}).get("type")
            if pending_type in (CommandType.ROLLBACK.value, CommandType.ABORT.value):
                self._resend_if_due(db, dep, t, ob)  # rollback is retried until acknowledged
            elif t.activation_sent_at is None and t.state in (T.DISTRIBUTING, T.STAGED):
                self._issue(db, dep, t, CommandType.ABORT, ob, activate=manifest.model)
            else:
                self._issue(db, dep, t, CommandType.ROLLBACK, ob, rollback_to=self._rollback_target(db, dep, t))
                t.rollback_requested = True
        if not unresolved:
            failed = [
                t for t in self._targets(db, dep.deployment_id) if (t.last_error or "").startswith("rollback failed")
            ]
            if failed:
                self._set_dep(
                    db,
                    dep,
                    D.FAILED,
                    "system",
                    f"rollback failed on {[t.device_id for t in failed]}; manual intervention required",
                )
            else:
                self._set_dep(db, dep, D.ROLLED_BACK, "system")

    def _rollback_target(self, db: Session, dep: Deployment, t: DeploymentTarget) -> RollbackTarget:
        if not t.previous_model:
            return RollbackTarget(model=None)
        ident = ModelIdentity.model_validate(t.previous_model)
        mv = db.scalar(
            select(ModelVersion).where(ModelVersion.name == ident.name, ModelVersion.version == ident.version)
        )
        manifest = None
        if mv is not None:
            old = Manifest.model_validate(dep.manifest)
            manifest = self.build_manifest(
                mv, dep.deployment_id, dep.group_name, old.strategy, old.health_criteria, self._clock()
            )
        return RollbackTarget(model=ident, manifest=manifest)

    def _manual_rollback(self, db: Session, dep: Deployment, actor: str, reason: str, ob: _Outbox) -> None:
        if DeploymentState(dep.state) not in (D.IN_PROGRESS, D.COMPLETED, D.FAILED, D.PENDING):
            raise ConflictError(f"deployment is {dep.state}; cannot roll back")
        if dep.state == D.PENDING:
            self._set_dep(db, dep, D.IN_PROGRESS, actor)
        if dep.state == D.FAILED:
            for t in self._targets(db, dep.deployment_id):
                if t.state == T.FAILED:
                    t.rollback_requested, t.last_error = True, None
        for t in self._targets(db, dep.deployment_id):
            if t.state == T.ACTIVE:
                t.activation_sent_at = t.activation_sent_at or self._clock()
        self._begin_rollback(db, dep, actor, reason, "manual", ob)

    def _cancel(self, db: Session, dep: Deployment, actor: str, ob: _Outbox) -> None:
        if DeploymentState(dep.state) not in (D.PENDING, D.IN_PROGRESS):
            raise ConflictError(f"deployment is {dep.state}; only PENDING/IN_PROGRESS can be cancelled")
        manifest = Manifest.model_validate(dep.manifest)
        targets = self._targets(db, dep.deployment_id)
        in_flight = [t.device_id for t in targets if t.state == T.STAGED and t.activation_sent_at is not None]
        if in_flight:  # outcome unknown: the device may be switching right now
            raise ConflictError(
                f"activation in flight on {in_flight}; wait for the acknowledgement or use rollback instead"
            )
        for t in targets:
            if t.state in (T.DISTRIBUTING, T.STAGED) and t.activation_sent_at is None:
                self._issue(db, dep, t, CommandType.ABORT, ob, activate=manifest.model)
                self._clear_pending(t)
                self._set_target(db, dep, t, T.CANCELLED, "deployment cancelled")
            elif t.state == T.PENDING:
                self._set_target(db, dep, t, T.CANCELLED, "deployment cancelled")
        self._set_dep(db, dep, D.CANCELLED, actor, "cancelled by operator")

    # ================================================================== events

    def _target(self, db: Session, dep_id: str, device_id: str) -> DeploymentTarget | None:
        return db.scalar(
            select(DeploymentTarget).where(
                DeploymentTarget.deployment_id == dep_id, DeploymentTarget.device_id == device_id
            )
        )

    def _apply_ack(self, db: Session, dep: Deployment, msg: StatusMessage, ob: _Outbox) -> None:
        t = self._target(db, dep.deployment_id, msg.device_id)
        pending = t.pending_command if t else None
        if t is None or pending is None or pending.get("command_id") != msg.command_id:
            return  # unknown, duplicate or superseded acknowledgement
        now = self._clock()
        outcome = msg.outcome
        record_audit(
            db,
            f"device:{msg.device_id}",
            "device.ack",
            "command",
            msg.command_id or "",
            device_id=t.device_id,
            deployment_id=dep.deployment_id,
            outcome=outcome.value if outcome else None,
            error_code=msg.error_code,
            detail=msg.detail,
        )
        if msg.duration_ms is not None and msg.command_type is not None:
            self._metrics.update_duration.labels(msg.command_type.value).observe(msg.duration_ms / 1000)
        cur = TargetState(t.state)
        if outcome == AckOutcome.STAGED and cur == T.DISTRIBUTING:
            self._clear_pending(t)
            self._set_target(db, dep, t, T.STAGED)
        elif outcome == AckOutcome.ACTIVE and cur == T.STAGED:
            self._clear_pending(t)
            t.activated_at = now
            t.update_seconds = (now - (t.activation_sent_at or now)).total_seconds()
            self._set_target(db, dep, t, T.ACTIVE)
        elif outcome == AckOutcome.ROLLED_BACK and cur in (T.ACTIVE, T.STAGED, T.DISTRIBUTING, T.FAILED):
            self._clear_pending(t)
            t.rollback_requested = False
            self._metrics.target_outcomes.labels("rolled_back").inc()
            self._set_target(db, dep, t, T.ROLLED_BACK)
        elif outcome == AckOutcome.ABORTED and cur in (T.DISTRIBUTING, T.STAGED):
            self._clear_pending(t)
            self._set_target(db, dep, t, T.CANCELLED, "aborted by device")
        elif outcome == AckOutcome.FAILED:
            reason = f"{msg.error_code}: {msg.detail}"
            if msg.command_type in (CommandType.ROLLBACK, CommandType.ABORT):
                self._clear_pending(t)
                t.rollback_requested = False
                self._metrics.rollback_failures.inc()
                t.last_error = f"rollback failed: {reason}"
                if cur != T.FAILED:
                    self._set_target(db, dep, t, T.FAILED, t.last_error)
            elif cur in (T.DISTRIBUTING, T.STAGED):
                self._fail(db, dep, t, reason, ob)
        elif outcome == AckOutcome.REJECTED and msg.error_code == "EXPIRED":
            t.next_retry_at = now  # reissue with a fresh command
        elif outcome == AckOutcome.REJECTED and cur in (T.DISTRIBUTING, T.STAGED):
            self._fail(db, dep, t, f"command rejected by device: {msg.error_code}", ob)
        self._advance(db, dep, ob)

    def _apply_heartbeat(self, db: Session, dep: Deployment, hb: Heartbeat, ob: _Outbox) -> None:
        t = self._target(db, dep.deployment_id, hb.device_id)
        if t is None:
            return
        manifest = Manifest.model_validate(dep.manifest)
        ident = ModelIdentity(name=manifest.model.name, version=manifest.model.version, sha256=manifest.artifact.sha256)
        state: DeviceModelState = hb.state
        now = self._clock()
        cur = TargetState(t.state)
        before = t.state
        # State-based reconciliation: device-reported state heals lost acknowledgements.
        if cur == T.DISTRIBUTING and state.staged == ident:
            self._clear_pending(t)
            self._set_target(db, dep, t, T.STAGED, "reconciled from heartbeat")
        elif cur in (T.DISTRIBUTING, T.STAGED) and state.active == ident and t.activation_sent_at is not None:
            self._clear_pending(t)
            t.activated_at = now
            self._set_target(db, dep, t, T.ACTIVE, "reconciled from heartbeat")
        elif dep.state == D.ROLLING_BACK and _needs_rollback(t):
            prev = ModelIdentity.model_validate(t.previous_model) if t.previous_model else None
            if t.activation_sent_at is not None and state.active == prev and state.active != ident:
                self._clear_pending(t)
                t.rollback_requested = False
                self._set_target(db, dep, t, T.ROLLED_BACK, "reconciled from heartbeat")
            elif (
                t.activation_sent_at is None
                and ident not in (state.active, state.staged)
                and (t.pending_command or {}).get("type") == CommandType.ABORT.value
            ):
                self._clear_pending(t)
                self._set_target(db, dep, t, T.CANCELLED, "reconciled from heartbeat")
        if (
            TargetState(t.state) == T.ACTIVE
            and hb.model_stats
            and (hb.model_stats.name, hb.model_stats.version) == (ident.name, ident.version)
        ):
            s = hb.model_stats
            t.observation = {
                "at": now.isoformat(),
                "metrics": HealthObservation(
                    inference_count=s.inference_count,
                    inference_errors=s.inference_errors,
                    latency_p95_ms=s.latency_p95_ms,
                    observed_seconds=s.active_seconds,
                    cpu_percent=hb.cpu_percent,
                    memory_mb=hb.memory_mb,
                ).model_dump(),
            }
        # A full advance walks every target of the deployment, so doing it for each heartbeat made the cost per
        # second grow with fleet size squared and saturated the single MQTT ingest thread at ~100 devices. Only a
        # heartbeat that changed this target's state needs an immediate advance; health observations and timers
        # are picked up by the periodic ticker (docs/load-test-results.md, "Rollout slowdown under load").
        if t.state != before:
            self._advance(db, dep, ob)
