"""Versioned REST API (``/api/v1``) plus health, metrics and the device artifact endpoint."""

from __future__ import annotations

import hmac
import logging
import threading
from collections.abc import Callable
from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from fleetops.controlplane import devices as device_ops
from fleetops.controlplane.context import AppContext
from fleetops.controlplane.schemas import (
    AuditEventOut,
    DeploymentCreateIn,
    DeploymentOut,
    DeviceCredentials,
    DeviceHealthOut,
    DeviceOut,
    DeviceRegisterIn,
    DeviceRegisterOut,
    DriftReportOut,
    ModelSummary,
    ModelVersionOut,
    Page,
    RecommendationOut,
    ResolveIn,
    RollbackIn,
    TargetOut,
    TrainIn,
)
from fleetops.deployment.engine import ConflictError, ForbiddenError, NotFoundError
from fleetops.domain.messages import CommandType
from fleetops.domain.models import FaultConfig, utcnow
from fleetops.domain.states import IN_FLIGHT_DEPLOYMENT_STATES
from fleetops.persistence.audit import record_audit
from fleetops.persistence.db import session_scope
from fleetops.persistence.tables import (
    AuditEvent,
    Deployment,
    DeploymentTarget,
    Device,
    DriftReport,
    ModelVersion,
    ReviewRecommendation,
)
from fleetops.registry.artifact_store import ArtifactIntegrityError, ArtifactStoreError
from fleetops.registry.mlflow_registry import RegistryUnavailableError
from fleetops.security.auth import Principal, Role
from fleetops.training.pipeline import train_model

router = APIRouter()
logger = logging.getLogger(__name__)
_train_lock = threading.Lock()


def ctx_of(request: Request) -> AppContext:
    ctx: AppContext | None = getattr(request.app.state, "ctx", None)
    if ctx is None:
        raise HTTPException(status_code=503, detail="control plane starting")
    return ctx


Ctx = Annotated[AppContext, Depends(ctx_of)]


def client_of(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def bearer(authorization: str | None) -> str | None:
    return authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None


def audit_denied(ctx: AppContext, request: Request, reason: str, required: str, principal: str | None = None) -> None:
    """Persist a denied management request. Never records the presented credential."""
    ctx.metrics.auth_denied.labels(reason).inc()
    try:
        with session_scope(ctx.sessions) as db:
            record_audit(
                db,
                principal or "anonymous",
                "auth.denied",
                "http",
                f"{request.method} {request.url.path}",
                reason=reason,
                required=required,
                client=client_of(request),
            )
    except Exception:
        logger.exception("could not audit denied request")


def require(role: Role) -> Callable[..., Principal]:
    def dep(request: Request, ctx: Ctx, authorization: Annotated[str | None, Header()] = None) -> Principal:
        client = client_of(request)
        if not ctx.auth_failures.has_token(client):  # brute-force protection: too many recent 401s from this client
            ctx.metrics.rate_limited.labels("auth_failures").inc()
            raise HTTPException(
                status_code=429,
                detail="too many failed authentication attempts",
                headers={"Retry-After": str(int(ctx.auth_failures.retry_after(client)) + 1)},
            )
        principal = ctx.auth.authenticate(bearer(authorization))
        if principal is None:
            ctx.auth_failures.consume(client)
            audit_denied(ctx, request, "missing or invalid API key", role.name.lower())
            raise HTTPException(
                status_code=401, detail="missing or invalid API key", headers={"WWW-Authenticate": "Bearer"}
            )
        if not principal.allows(role):
            audit_denied(ctx, request, "insufficient role", role.name.lower(), principal.name)
            raise HTTPException(status_code=403, detail=f"requires role {role.name.lower()}")
        return principal

    return dep


Viewer = Annotated[Principal, Depends(require(Role.VIEWER))]
Operator = Annotated[Principal, Depends(require(Role.OPERATOR))]
Admin = Annotated[Principal, Depends(require(Role.ADMIN))]
Limit = Annotated[int, Query(ge=1, le=200)]
Offset = Annotated[int, Query(ge=0)]


def _page(
    db: Session, stmt: Any, count_stmt: Any, limit: int, offset: int, convert: Callable[[Any], Any]
) -> dict[str, Any]:
    total = db.scalar(count_stmt) or 0
    rows = db.scalars(stmt.limit(limit).offset(offset)).all()
    return {"items": [convert(r) for r in rows], "total": total, "limit": limit, "offset": offset}


# ------------------------------------------------------------------ health / metrics


@router.get("/healthz", tags=["ops"])
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", tags=["ops"])
def readyz(ctx: Ctx, authorization: Annotated[str | None, Header()] = None) -> Any:
    """Readiness for probes. Anonymous callers get only ready / not ready; a viewer key adds component detail."""
    try:
        with session_scope(ctx.sessions) as db:
            db.execute(select(1))
    except Exception:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    if ctx.auth.authenticate(bearer(authorization)) is None:
        return {"status": "ready"}
    return {
        "status": "ready",
        "database": True,
        "mqtt": ctx.bus.connected if ctx.bus else None,
        "mlflow": ctx.registry.ping(),
    }


@router.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics(request: Request, ctx: Ctx, authorization: Annotated[str | None, Header()] = None) -> Response:
    """Prometheus scrape endpoint. Accepts the dedicated metrics token (read-only, metrics only) or any API key."""
    presented = bearer(authorization)
    configured = ctx.settings.metrics_token
    token_ok = bool(
        presented and configured and hmac.compare_digest(presented.encode(), configured.get_secret_value().encode())
    )
    if not token_ok and ctx.auth.authenticate(presented) is None:
        audit_denied(ctx, request, "missing or invalid metrics credential", "metrics")
        raise HTTPException(
            status_code=401, detail="metrics credential required", headers={"WWW-Authenticate": "Bearer"}
        )
    return Response(generate_latest(ctx.metrics.registry), media_type=CONTENT_TYPE_LATEST)


# ------------------------------------------------------------------ devices


@router.post("/api/v1/devices/register", status_code=201, response_model=DeviceRegisterOut, tags=["devices"])
def register_device(body: DeviceRegisterIn, ctx: Ctx, principal: Admin) -> DeviceRegisterOut:
    with session_scope(ctx.sessions) as db:
        dev, secret = device_ops.register_device(
            db,
            body.device_id,
            body.group,
            body.description,
            dict(body.labels),
            principal.name,
            ctx.settings.device_key_master.get_secret_value(),
        )
        db.flush()
        return DeviceRegisterOut(
            device=DeviceOut.model_validate(dev),
            credentials=DeviceCredentials(device_id=dev.device_id, secret=secret, mqtt_username=dev.device_id),
        )


@router.get("/api/v1/devices", response_model=Page[DeviceOut], tags=["devices"])
def list_devices(
    ctx: Ctx, _p: Viewer, group: str | None = None, status: str | None = None, limit: Limit = 50, offset: Offset = 0
) -> dict[str, Any]:
    filters = [
        c
        for c in (Device.group_name == group if group else None, Device.status == status if status else None)
        if c is not None
    ]
    with session_scope(ctx.sessions) as db:
        return _page(
            db,
            select(Device).where(*filters).order_by(Device.device_id),
            select(func.count()).select_from(Device).where(*filters),
            limit,
            offset,
            DeviceOut.model_validate,
        )


def _device_or_404(db: Session, device_id: str) -> Device:
    dev = db.get(Device, device_id)
    if dev is None:
        raise NotFoundError(f"device {device_id} not found")
    return dev


@router.get("/api/v1/devices/{device_id}", response_model=DeviceOut, tags=["devices"])
def get_device(device_id: str, ctx: Ctx, _p: Viewer) -> DeviceOut:
    with session_scope(ctx.sessions) as db:
        return DeviceOut.model_validate(_device_or_404(db, device_id))


@router.get("/api/v1/devices/{device_id}/health", response_model=DeviceHealthOut, tags=["devices"])
def device_health(device_id: str, ctx: Ctx, _p: Viewer) -> DeviceHealthOut:
    with session_scope(ctx.sessions) as db:
        dev = _device_or_404(db, device_id)
        now = utcnow()
        age = (now - dev.last_seen_at).total_seconds() if dev.last_seen_at else None
        hb = dev.last_heartbeat or {}
        reasons: list[str] = []
        connectivity = "unknown" if dev.last_seen_at is None else ("online" if dev.status == "ONLINE" else "offline")
        if connectivity == "offline":
            assessment = "offline"
            reasons.append("no recent heartbeat; device may still be running inference locally")
        elif connectivity == "unknown":
            assessment = "unknown"
        elif dev.active_model_name is None:
            assessment = "no_model"
            reasons.append("no active model")
        else:
            assessment = "healthy"
            stats = hb.get("model_stats") or {}
            if stats.get("inference_count", 0) and stats.get("inference_errors", 0) / stats["inference_count"] > 0.05:
                assessment = "degraded"
                reasons.append("inference error rate above 5%")
        inflight = [
            {"deployment_id": t.deployment_id, "state": t.state, "deployment_state": d.state}
            for t, d in db.execute(
                select(DeploymentTarget, Deployment)
                .join(Deployment)
                .where(
                    DeploymentTarget.device_id == device_id,
                    Deployment.state.in_([s.value for s in IN_FLIGHT_DEPLOYMENT_STATES]),
                )
            )
        ]
        return DeviceHealthOut(
            device_id=device_id,
            status=dev.status,
            connectivity=connectivity,
            heartbeat_age_seconds=age,
            active_model=(hb.get("state") or {}).get("active"),
            model_stats=hb.get("model_stats"),
            cpu_percent=hb.get("cpu_percent"),
            memory_mb=hb.get("memory_mb"),
            inflight_deployments=inflight,
            assessment=assessment,
            reasons=reasons,
        )


@router.post("/api/v1/devices/{device_id}/revoke", response_model=DeviceOut, tags=["devices"])
def revoke_device(device_id: str, ctx: Ctx, principal: Admin) -> DeviceOut:
    with session_scope(ctx.sessions) as db:
        dev = _device_or_404(db, device_id)
        dev.status, dev.key_epoch = "REVOKED", dev.key_epoch + 1  # old secret no longer verifies
        record_audit(db, principal.name, "device.revoked", "device", device_id, device_id=device_id)
        return DeviceOut.model_validate(dev)


@router.post("/api/v1/devices/{device_id}/simulation", status_code=202, tags=["simulation"])
def set_simulation(device_id: str, body: FaultConfig, ctx: Ctx, principal: Admin) -> dict[str, Any]:
    """Deterministic failure injection (development only; disabled unless FLEETOPS_SIMULATION_ENABLED)."""
    if not ctx.settings.simulation_enabled:
        raise HTTPException(status_code=403, detail="simulation endpoints are disabled")
    cmd = ctx.engine.send_device_command(device_id, CommandType.SET_FAULTS, principal.name, faults=body)
    return {"command_id": cmd.command_id, "generation": cmd.generation}


# ------------------------------------------------------------------ models


@router.post("/api/v1/models/train", status_code=201, response_model=ModelVersionOut, tags=["models"])
def train(body: TrainIn, ctx: Ctx, principal: Operator) -> ModelVersionOut:
    if not _train_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="another training job is running")
    try:
        try:
            result = train_model(body.params)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            row = ctx.registration.register_training_result(body.model_name, result)
        except RegistryUnavailableError as exc:
            raise HTTPException(status_code=502, detail=f"model registry unavailable: {exc}") from exc
        except ArtifactStoreError as exc:
            raise HTTPException(status_code=500, detail="artifact store failure") from exc
        with session_scope(ctx.sessions) as db:
            record_audit(
                db,
                principal.name,
                "model.registered",
                "model",
                f"{row.name}:{row.version}",
                sha256=row.sha256,
                metrics=row.metrics,
            )
        return ModelVersionOut.model_validate(row)
    finally:
        _train_lock.release()


@router.get("/api/v1/models", response_model=Page[ModelSummary], tags=["models"])
def list_models(ctx: Ctx, _p: Viewer, limit: Limit = 50, offset: Offset = 0) -> dict[str, Any]:
    with session_scope(ctx.sessions) as db:
        rows = db.execute(
            select(ModelVersion.name, func.max(ModelVersion.version), func.count())
            .group_by(ModelVersion.name)
            .order_by(ModelVersion.name)
        ).all()
        items = [ModelSummary(name=n, latest_version=v, version_count=c) for n, v, c in rows]
        return {"items": items[offset : offset + limit], "total": len(items), "limit": limit, "offset": offset}


@router.get("/api/v1/models/{model_name}/versions", response_model=Page[ModelVersionOut], tags=["models"])
def list_versions(model_name: str, ctx: Ctx, _p: Viewer, limit: Limit = 50, offset: Offset = 0) -> dict[str, Any]:
    with session_scope(ctx.sessions) as db:
        page = _page(
            db,
            select(ModelVersion).where(ModelVersion.name == model_name).order_by(ModelVersion.version.desc()),
            select(func.count()).select_from(ModelVersion).where(ModelVersion.name == model_name),
            limit,
            offset,
            ModelVersionOut.model_validate,
        )
        if page["total"] == 0:
            raise NotFoundError(f"model {model_name} not found")
        return page


# ------------------------------------------------------------------ deployments


def deployment_view(db: Session, dep: Deployment, skipped: dict[str, str] | None = None) -> DeploymentOut:
    targets = db.scalars(
        select(DeploymentTarget)
        .where(DeploymentTarget.deployment_id == dep.deployment_id)
        .order_by(DeploymentTarget.batch_index, DeploymentTarget.device_id)
    ).all()
    manifest = dep.manifest
    return DeploymentOut(
        deployment_id=dep.deployment_id,
        model_name=dep.model_name,
        model_version=dep.model_version,
        group=dep.group_name,
        state=dep.state,
        reason=dep.reason,
        current_batch=dep.current_batch,
        batches=dep.batches,
        created_by=dep.created_by,
        created_at=dep.created_at,
        updated_at=dep.updated_at,
        completed_at=dep.completed_at,
        superseded_by=dep.superseded_by,
        strategy=manifest["strategy"],
        health_criteria=manifest["health_criteria"],
        targets=[
            TargetOut(
                device_id=t.device_id,
                batch_index=t.batch_index,
                state=t.state,
                previous_model=t.previous_model,
                attempts=t.attempts,
                activated_at=t.activated_at,
                verified_at=t.verified_at,
                observation=t.observation,
                last_error=t.last_error,
                update_seconds=t.update_seconds,
            )
            for t in targets
        ],
        skipped=skipped or {},
        quality=dep.quality_report,
    )


@router.post("/api/v1/deployments", response_model=DeploymentOut, status_code=201, tags=["deployments"])
def create_deployment(
    body: DeploymentCreateIn,
    response: Response,
    ctx: Ctx,
    principal: Operator,
    idempotency_key: Annotated[str | None, Header(max_length=128)] = None,
) -> DeploymentOut:
    try:
        created = ctx.engine.create_deployment(
            body.to_spec(), principal.name, idempotency_key, may_override=principal.allows(Role.ADMIN)
        )
    except ForbiddenError:
        ctx.metrics.auth_denied.labels("override not permitted").inc()
        with session_scope(ctx.sessions) as db:
            record_audit(
                db,
                principal.name,
                "deployment.override_denied",
                "deployment",
                f"{body.model_name}:{body.model_version}",
                reason="quality_override requires the admin role",
                requested_reason=body.quality_override.reason if body.quality_override else None,
            )
        raise
    if not created.created:
        response.status_code = 200  # idempotent replay
    with session_scope(ctx.sessions) as db:
        dep = db.get(Deployment, created.deployment_id)
        assert dep is not None
        return deployment_view(db, dep, created.skipped)


@router.get("/api/v1/deployments", response_model=Page[DeploymentOut], tags=["deployments"])
def list_deployments(
    ctx: Ctx,
    _p: Viewer,
    state: str | None = None,
    model_name: str | None = None,
    model_version: int | None = None,
    device_id: str | None = None,
    limit: Limit = 50,
    offset: Offset = 0,
) -> dict[str, Any]:
    filters: list[Any] = []
    if state:
        filters.append(Deployment.state == state.upper())
    if model_name:
        filters.append(Deployment.model_name == model_name)
    if model_version is not None:
        filters.append(Deployment.model_version == model_version)
    if device_id:
        filters.append(
            Deployment.deployment_id.in_(
                select(DeploymentTarget.deployment_id).where(DeploymentTarget.device_id == device_id)
            )
        )
    with session_scope(ctx.sessions) as db:
        return _page(
            db,
            select(Deployment).where(*filters).order_by(Deployment.created_at.desc()),
            select(func.count()).select_from(Deployment).where(*filters),
            limit,
            offset,
            lambda d: deployment_view(db, d),
        )


def _dep_or_404(db: Session, deployment_id: str) -> Deployment:
    dep = db.get(Deployment, deployment_id)
    if dep is None:
        raise NotFoundError(f"deployment {deployment_id} not found")
    return dep


@router.get("/api/v1/deployments/{deployment_id}", response_model=DeploymentOut, tags=["deployments"])
def get_deployment(deployment_id: str, ctx: Ctx, _p: Viewer) -> DeploymentOut:
    with session_scope(ctx.sessions) as db:
        return deployment_view(db, _dep_or_404(db, deployment_id))


@router.post(
    "/api/v1/deployments/{deployment_id}/cancel", response_model=DeploymentOut, status_code=202, tags=["deployments"]
)
def cancel_deployment(deployment_id: str, ctx: Ctx, principal: Operator) -> DeploymentOut:
    ctx.engine.cancel(deployment_id, principal.name)
    with session_scope(ctx.sessions) as db:
        return deployment_view(db, _dep_or_404(db, deployment_id))


@router.post(
    "/api/v1/deployments/{deployment_id}/rollback", response_model=DeploymentOut, status_code=202, tags=["deployments"]
)
def rollback_deployment(
    deployment_id: str, ctx: Ctx, principal: Operator, body: RollbackIn | None = None
) -> DeploymentOut:
    ctx.engine.rollback(deployment_id, principal.name, (body or RollbackIn()).reason)
    with session_scope(ctx.sessions) as db:
        return deployment_view(db, _dep_or_404(db, deployment_id))


# ------------------------------------------------------------------ monitoring & audit


@router.get("/api/v1/monitoring/drift", response_model=Page[DriftReportOut], tags=["monitoring"])
def list_drift(
    ctx: Ctx,
    _p: Viewer,
    model_name: str | None = None,
    model_version: int | None = None,
    status: str | None = None,
    limit: Limit = 20,
    offset: Offset = 0,
) -> dict[str, Any]:
    filters: list[Any] = []
    if model_name:
        filters.append(DriftReport.model_name == model_name)
    if model_version is not None:
        filters.append(DriftReport.model_version == model_version)
    if status:
        filters.append(DriftReport.status == status.upper())
    with session_scope(ctx.sessions) as db:
        return _page(
            db,
            select(DriftReport).where(*filters).order_by(DriftReport.id.desc()),
            select(func.count()).select_from(DriftReport).where(*filters),
            limit,
            offset,
            DriftReportOut.model_validate,
        )


@router.post("/api/v1/monitoring/drift/run", response_model=list[DriftReportOut], tags=["monitoring"])
def run_drift(ctx: Ctx, _p: Operator) -> list[DriftReportOut]:
    ids = ctx.drift.run_once()
    with session_scope(ctx.sessions) as db:
        return [DriftReportOut.model_validate(db.get(DriftReport, i)) for i in ids]


@router.get("/api/v1/monitoring/recommendations", response_model=Page[RecommendationOut], tags=["monitoring"])
def list_recommendations(
    ctx: Ctx, _p: Viewer, status: str | None = None, limit: Limit = 20, offset: Offset = 0
) -> dict[str, Any]:
    filters = [ReviewRecommendation.status == status.upper()] if status else []
    with session_scope(ctx.sessions) as db:
        return _page(
            db,
            select(ReviewRecommendation).where(*filters).order_by(ReviewRecommendation.id.desc()),
            select(func.count()).select_from(ReviewRecommendation).where(*filters),
            limit,
            offset,
            RecommendationOut.model_validate,
        )


@router.post(
    "/api/v1/monitoring/recommendations/{rec_id}/resolve", response_model=RecommendationOut, tags=["monitoring"]
)
def resolve_recommendation(rec_id: int, body: ResolveIn, ctx: Ctx, principal: Operator) -> RecommendationOut:
    with session_scope(ctx.sessions) as db:
        rec = db.get(ReviewRecommendation, rec_id)
        if rec is None:
            raise NotFoundError(f"recommendation {rec_id} not found")
        if rec.status != "OPEN":
            raise ConflictError(f"recommendation already {rec.status}")
        rec.status = "ACKNOWLEDGED" if body.action == "acknowledge" else "DISMISSED"
        rec.resolved_at, rec.resolved_by = utcnow(), principal.name
        record_audit(db, principal.name, f"recommendation.{body.action}", "recommendation", str(rec_id))
        return RecommendationOut.model_validate(rec)


@router.get("/api/v1/audit/events", response_model=Page[AuditEventOut], tags=["audit"])
def list_audit(
    ctx: Ctx,
    _p: Viewer,
    device_id: str | None = None,
    deployment_id: str | None = None,
    action: str | None = None,
    limit: Limit = 50,
    offset: Offset = 0,
) -> dict[str, Any]:
    filters: list[Any] = []
    if device_id:
        filters.append(AuditEvent.device_id == device_id)
    if deployment_id:
        filters.append(AuditEvent.deployment_id == deployment_id)
    if action:
        filters.append(AuditEvent.action.like(action.replace("*", "%")))
    with session_scope(ctx.sessions) as db:
        return _page(
            db,
            select(AuditEvent).where(*filters).order_by(AuditEvent.id.desc()),
            select(func.count()).select_from(AuditEvent).where(*filters),
            limit,
            offset,
            AuditEventOut.model_validate,
        )


# ------------------------------------------------------------------ model distribution (device-authenticated)


@router.get("/api/v1/device/artifacts/{model_name}/{version}", tags=["distribution"], include_in_schema=True)
def download_artifact(
    request: Request,
    model_name: str,
    version: int,
    ctx: Ctx,
    authorization: Annotated[str | None, Header()] = None,
    x_device_id: Annotated[str | None, Header()] = None,
) -> FileResponse:
    """Authenticated per device; a device may fetch only versions referenced by its own active deployments."""
    client = client_of(request)
    if not ctx.auth_failures.has_token(client):  # same brute-force protection as the management API
        ctx.metrics.rate_limited.labels("auth_failures").inc()
        raise HTTPException(
            status_code=429, detail="too many failed authentication attempts", headers={"Retry-After": "60"}
        )
    token = bearer(authorization) or ""
    if not x_device_id or not token:
        ctx.auth_failures.consume(client)
        audit_denied(ctx, request, "device credentials missing", "device")
        raise HTTPException(status_code=401, detail="device credentials required")
    denied: str | None = None
    sha = ""
    with session_scope(ctx.sessions) as db:
        dev = device_ops.authenticate_device(db, x_device_id, token, ctx.settings.device_key_master.get_secret_value())
        if dev is None:
            denied = "auth"
        elif not _may_download(db, dev.device_id, model_name, version):
            denied = "assignment"
        else:
            mv = db.scalar(select(ModelVersion).where(ModelVersion.name == model_name, ModelVersion.version == version))
            if mv is None:
                raise NotFoundError("artifact not found")
            sha = mv.sha256
            record_audit(
                db,
                f"device:{dev.device_id}",
                "artifact.downloaded",
                "artifact",
                f"{model_name}:{version}",
                device_id=dev.device_id,
            )
    # Denials are recorded AFTER the lookup transaction: raising inside it would roll the audit event back.
    if denied == "auth":
        ctx.auth_failures.consume(client)
        audit_denied(ctx, request, "invalid device credentials", "device")
        raise HTTPException(status_code=401, detail="invalid device credentials")
    if denied == "assignment":
        ctx.metrics.artifact_downloads.labels("denied").inc()
        with session_scope(ctx.sessions) as db:
            record_audit(
                db,
                f"device:{x_device_id}",
                "artifact.denied",
                "artifact",
                f"{model_name}:{version}",
                device_id=x_device_id,
                reason="device is not a target of an in-flight deployment for this version",
            )
        raise HTTPException(status_code=403, detail="device is not assigned this model version")
    try:
        path = ctx.store.verified(model_name, version, sha)
    except ArtifactIntegrityError as exc:
        # Never serve bytes that differ from the registered checksum; make it loud and auditable.
        logger.error("artifact integrity failure", extra={"model": model_name, "version": version, "detail": str(exc)})
        ctx.metrics.artifact_integrity_failures.inc()
        ctx.metrics.artifact_downloads.labels("corrupt").inc()
        with session_scope(ctx.sessions) as db:
            record_audit(
                db,
                "system",
                "artifact.corrupt",
                "artifact",
                f"{model_name}:{version}",
                device_id=x_device_id,
                detail=str(exc),
            )
        raise HTTPException(status_code=500, detail="artifact failed integrity verification") from exc
    except ArtifactStoreError as exc:
        raise NotFoundError("artifact not found") from exc
    ctx.metrics.artifact_downloads.labels("ok").inc()
    return FileResponse(
        path, media_type="application/octet-stream", headers={"ETag": f'"{sha}"', "Cache-Control": "no-store"}
    )


def _may_download(db: Session, device_id: str, name: str, version: int) -> bool:
    rows = db.execute(
        select(Deployment.model_name, Deployment.model_version, DeploymentTarget.previous_model)
        .join(DeploymentTarget, DeploymentTarget.deployment_id == Deployment.deployment_id)
        .where(
            DeploymentTarget.device_id == device_id,
            Deployment.state.in_([s.value for s in IN_FLIGHT_DEPLOYMENT_STATES]),
            or_(
                DeploymentTarget.state.in_(["PENDING", "DISTRIBUTING", "STAGED", "ACTIVE", "FAILED"]),
            ),
        ),
    ).all()
    for m_name, m_version, prev in rows:
        if (m_name, m_version) == (name, version):
            return True
        if prev and (prev.get("name"), prev.get("version")) == (name, version):
            return True
    return False


__all__ = ["router", "timedelta"]
