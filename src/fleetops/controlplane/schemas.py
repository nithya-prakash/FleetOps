"""Strict API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from fleetops.deployment.engine import DeploymentSpec
from fleetops.domain.models import DeviceId, GroupName, HealthCriteria, ModelName, RolloutStrategy, StrictModel
from fleetops.domain.quality import QualityGate, QualityOverride
from fleetops.training.pipeline import TrainParams

T = TypeVar("T")


class Page[T](BaseModel):
    items: list[T]
    total: int
    limit: int
    offset: int


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class DeviceRegisterIn(StrictModel):
    device_id: DeviceId
    group: GroupName
    description: str = Field(default="", max_length=255)
    labels: dict[str, str] = Field(default_factory=dict, max_length=20)


class DeviceOut(ORMModel):
    device_id: str
    group_name: str
    description: str
    labels: dict[str, Any]
    status: str
    registered_at: datetime
    last_seen_at: datetime | None
    active_model_name: str | None
    active_model_version: int | None


class DeviceCredentials(BaseModel):
    device_id: str
    secret: str
    mqtt_username: str
    note: str = "Shown once by this endpoint; derive/store it securely on the device."


class DeviceRegisterOut(BaseModel):
    device: DeviceOut
    credentials: DeviceCredentials


class DeviceHealthOut(BaseModel):
    device_id: str
    status: str
    connectivity: Literal["online", "offline", "unknown"]
    heartbeat_age_seconds: float | None
    active_model: dict[str, Any] | None
    model_stats: dict[str, Any] | None
    cpu_percent: float | None
    memory_mb: float | None
    inflight_deployments: list[dict[str, Any]]
    assessment: Literal["healthy", "degraded", "offline", "no_model", "unknown"]
    reasons: list[str]


class TrainIn(StrictModel):
    model_name: ModelName = "anomaly-detector"
    params: TrainParams = Field(default_factory=TrainParams)


class ModelVersionOut(ORMModel):
    name: str
    version: int
    sha256: str
    size_bytes: int
    framework: str
    compatibility: dict[str, Any]
    input_schema: dict[str, Any]
    metrics: dict[str, Any]
    params: dict[str, Any]
    evaluation: dict[str, Any]
    mlflow_run_id: str
    created_at: datetime


class ModelSummary(BaseModel):
    name: str
    latest_version: int
    version_count: int


class DeploymentCreateIn(StrictModel):
    model_name: ModelName
    model_version: int = Field(ge=1)
    group: GroupName
    device_ids: list[DeviceId] | None = Field(default=None, max_length=1000)
    strategy: RolloutStrategy = Field(default_factory=RolloutStrategy)
    health_criteria: HealthCriteria = Field(default_factory=HealthCriteria)
    quality_gate: QualityGate | None = None
    quality_override: QualityOverride | None = None  # admin only; requires a reason; audited

    def to_spec(self) -> DeploymentSpec:
        return DeploymentSpec(**self.model_dump())


class TargetOut(BaseModel):
    device_id: str
    batch_index: int
    state: str
    previous_model: dict[str, Any] | None
    attempts: int
    activated_at: datetime | None
    verified_at: datetime | None
    observation: dict[str, Any] | None
    last_error: str | None
    update_seconds: float | None


class DeploymentOut(BaseModel):
    deployment_id: str
    model_name: str
    model_version: int
    group: str
    state: str
    reason: str | None
    current_batch: int
    batches: list[list[str]]
    created_by: str
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    superseded_by: str | None
    strategy: dict[str, Any]
    health_criteria: dict[str, Any]
    targets: list[TargetOut]
    skipped: dict[str, str] = Field(default_factory=dict)
    quality: dict[str, Any] | None = None


class RollbackIn(StrictModel):
    reason: str = Field(default="manual rollback", min_length=3, max_length=500)


class DriftReportOut(ORMModel):
    id: int
    model_name: str
    model_version: int
    generated_at: datetime
    window_start: datetime
    window_end: datetime
    sample_count: int
    status: str
    features: list[Any]
    quality: dict[str, Any]
    thresholds: dict[str, Any]


class RecommendationOut(ORMModel):
    id: int
    drift_report_id: int
    model_name: str
    model_version: int
    status: str
    message: str
    created_at: datetime
    resolved_at: datetime | None
    resolved_by: str | None


class ResolveIn(StrictModel):
    action: Literal["acknowledge", "dismiss"]


class AuditEventOut(ORMModel):
    id: int
    ts: datetime
    actor: str
    action: str
    entity_type: str
    entity_id: str
    device_id: str | None
    deployment_id: str | None
    correlation_id: str | None
    details: dict[str, Any]
