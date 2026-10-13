"""Wire schemas for MQTT messages. Strict, size-bounded, versioned."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from fleetops.domain.models import (
    DeviceCapabilities,
    DeviceId,
    FaultConfig,
    Manifest,
    ModelIdentity,
    ModelName,
    ModelRef,
    StrictModel,
)

MAX_MESSAGE_BYTES = 256 * 1024
MAX_SAMPLES_PER_BATCH = 200


class CommandType(StrEnum):
    DEPLOY = "DEPLOY"  # download, verify, stage (no traffic switch)
    ACTIVATE = "ACTIVATE"  # atomically switch staged model to active
    ROLLBACK = "ROLLBACK"  # switch back to the known-good model
    ABORT = "ABORT"  # discard staged candidate of a deployment
    SET_FAULTS = "SET_FAULTS"  # simulation only


class RollbackTarget(StrictModel):
    """Known-good model to return to; ``manifest`` allows re-fetch if the local copy is gone."""

    model: ModelIdentity | None  # None = no known-good model exists: deactivate
    manifest: Manifest | None = None


class Command(StrictModel):
    schema_version: Literal[1] = 1
    command_id: str = Field(min_length=8, max_length=64)
    type: CommandType
    device_id: DeviceId
    generation: int = Field(ge=1)
    issued_at: datetime
    expires_at: datetime
    deployment_id: str | None = Field(default=None, max_length=64)
    correlation_id: str = Field(max_length=64)
    manifest: Manifest | None = None
    activate: ModelRef | None = None
    rollback_to: RollbackTarget | None = None
    faults: FaultConfig | None = None
    signature: str = Field(default="", max_length=128)


class AckOutcome(StrEnum):
    STAGED = "STAGED"
    ACTIVE = "ACTIVE"
    ROLLED_BACK = "ROLLED_BACK"
    ABORTED = "ABORTED"
    ACCEPTED = "ACCEPTED"  # SET_FAULTS applied
    FAILED = "FAILED"
    REJECTED = "REJECTED"  # stale / expired / bad signature / duplicate-with-conflict


class DeviceModelState(StrictModel):
    active: ModelIdentity | None = None
    staged: ModelIdentity | None = None
    previous: ModelIdentity | None = None
    last_generation: int = Field(default=0, ge=0)


class StatusMessage(StrictModel):
    """``online``/``offline`` presence or an ``ack`` for a command."""

    schema_version: Literal[1] = 1
    kind: Literal["online", "offline", "ack"]
    device_id: DeviceId
    sent_at: datetime
    boot_id: str | None = Field(default=None, max_length=64)
    state: DeviceModelState | None = None
    capabilities: DeviceCapabilities | None = None
    # ack fields
    command_id: str | None = Field(default=None, max_length=64)
    command_type: CommandType | None = None
    deployment_id: str | None = Field(default=None, max_length=64)
    generation: int | None = Field(default=None, ge=0)
    outcome: AckOutcome | None = None
    error_code: str | None = Field(default=None, max_length=64)
    detail: str | None = Field(default=None, max_length=1000)
    duration_ms: float | None = Field(default=None, ge=0)


class ModelStats(StrictModel):
    """Inference statistics for the *currently active* model since it was activated."""

    name: ModelName
    version: int
    inference_count: int = Field(ge=0)
    inference_errors: int = Field(ge=0)
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)
    active_seconds: float = Field(ge=0)


class Heartbeat(StrictModel):
    schema_version: Literal[1] = 1
    device_id: DeviceId
    sent_at: datetime
    boot_id: str = Field(max_length=64)
    agent_version: str = Field(max_length=32)
    uptime_seconds: float = Field(ge=0)
    state: DeviceModelState
    capabilities: DeviceCapabilities
    model_stats: ModelStats | None = None
    cpu_percent: float = Field(ge=0)
    memory_mb: float = Field(ge=0)
    mqtt_connected: bool = True
    disk_free_mb: float | None = Field(default=None, ge=0)


class Telemetry(StrictModel):
    schema_version: Literal[1] = 1
    device_id: DeviceId
    sent_at: datetime
    inference_rate_hz: float = Field(ge=0)
    inputs_rejected: int = Field(ge=0)
    sensor_means: dict[str, float] = Field(default_factory=dict, max_length=256)


class PredictionSample(StrictModel):
    ts: datetime
    model_name: ModelName
    model_version: int
    features: list[float] = Field(max_length=256)
    score: float
    anomaly: bool
    latency_ms: float = Field(ge=0)
    label: bool | None = None  # ground truth, only when available


class PredictionBatch(StrictModel):
    schema_version: Literal[1] = 1
    device_id: DeviceId
    sent_at: datetime
    samples: list[PredictionSample] = Field(max_length=MAX_SAMPLES_PER_BATCH)
