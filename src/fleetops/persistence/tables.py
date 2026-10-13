"""ORM tables. Only portable column types are used so the schema moves to PostgreSQL unchanged."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


class UtcDateTime(TypeDecorator[datetime]):
    """Stores timezone-aware datetimes as UTC and always returns aware values (SQLite drops tz)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime not allowed")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, datetime: UtcDateTime}


class Device(Base):
    __tablename__ = "devices"

    device_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    group_name: Mapped[str] = mapped_column(String(64), index=True)
    description: Mapped[str] = mapped_column(String(255), default="")
    labels: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="REGISTERED")  # REGISTERED/ONLINE/OFFLINE/REVOKED
    key_epoch: Mapped[int] = mapped_column(Integer, default=1)
    registered_at: Mapped[datetime]
    last_seen_at: Mapped[datetime | None] = mapped_column(nullable=True)
    last_heartbeat: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    capabilities: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    active_model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    active_model_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Monotonic fence for commands sent to this device (see docs/architecture.md §4).
    command_generation: Mapped[int] = mapped_column(Integer, default=0)


class ModelVersion(Base):
    """Immutable facts about a registered version, written once after MLflow registration."""

    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    framework: Mapped[str] = mapped_column(String(64))
    compatibility: Mapped[dict[str, Any]] = mapped_column(JSON)
    input_schema: Mapped[dict[str, Any]] = mapped_column(JSON)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)
    params: Mapped[dict[str, Any]] = mapped_column(JSON)
    reference_stats: Mapped[dict[str, Any]] = mapped_column(JSON)
    evaluation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    mlflow_run_id: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime]


class Deployment(Base):
    __tablename__ = "deployments"

    deployment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True, nullable=True)
    request_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    model_version: Mapped[int] = mapped_column(Integer)
    group_name: Mapped[str] = mapped_column(String(64), index=True)
    state: Mapped[str] = mapped_column(String(16), index=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON)
    batches: Mapped[list[Any]] = mapped_column(JSON)  # [[device_id, ...], ...]
    current_batch: Mapped[int] = mapped_column(Integer, default=0)
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    superseded_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rollback_started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    quality_report: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class DeploymentTarget(Base):
    __tablename__ = "deployment_targets"
    __table_args__ = (
        UniqueConstraint("deployment_id", "device_id"),
        Index("ix_targets_device_state", "device_id", "state"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    deployment_id: Mapped[str] = mapped_column(ForeignKey("deployments.deployment_id"), index=True)
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.device_id"))
    batch_index: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(16))
    # Known-good model on the device when the deployment began (rollback destination).
    previous_model: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # Outstanding command bookkeeping (idempotent re-send with the same command_id).
    pending_command: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_retry_at: Mapped[datetime | None] = mapped_column(nullable=True)
    state_since: Mapped[datetime]
    activation_sent_at: Mapped[datetime | None] = mapped_column(nullable=True)
    activated_at: Mapped[datetime | None] = mapped_column(nullable=True)
    verified_at: Mapped[datetime | None] = mapped_column(nullable=True)
    rollback_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    observation: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    update_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_entity", "entity_type", "entity_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(index=True)
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(32))
    entity_id: Mapped[str] = mapped_column(String(128))
    device_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    deployment_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class FeatureSample(Base):
    """Bounded window of recent edge inputs used for drift monitoring (not a telemetry archive)."""

    __tablename__ = "feature_samples"
    __table_args__ = (Index("ix_samples_model", "model_name", "model_version", "ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[int] = mapped_column(Integer)
    ts: Mapped[datetime]
    features: Mapped[list[Any]] = mapped_column(JSON)
    score: Mapped[float] = mapped_column(Float)
    anomaly: Mapped[bool] = mapped_column(Boolean)
    label: Mapped[bool | None] = mapped_column(Boolean, nullable=True)


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    model_version: Mapped[int] = mapped_column(Integer)
    generated_at: Mapped[datetime] = mapped_column(index=True)
    window_start: Mapped[datetime]
    window_end: Mapped[datetime]
    sample_count: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24))  # OK / WARN / ALERT / INSUFFICIENT_DATA
    features: Mapped[list[Any]] = mapped_column(JSON)
    quality: Mapped[dict[str, Any]] = mapped_column(JSON)
    thresholds: Mapped[dict[str, Any]] = mapped_column(JSON)


class ReviewRecommendation(Base):
    """Human review item created by a drift ALERT. Never triggers a deployment by itself."""

    __tablename__ = "review_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drift_report_id: Mapped[int] = mapped_column(ForeignKey("drift_reports.id"))
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    model_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="OPEN")  # OPEN/ACKNOWLEDGED/DISMISSED
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime]
    resolved_at: Mapped[datetime | None] = mapped_column(nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
