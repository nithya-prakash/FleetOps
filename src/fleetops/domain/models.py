"""Core value objects shared by the control plane, the edge agent and the tests."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

NAME_PATTERN = r"^[a-z][a-z0-9_-]{1,62}$"
DEVICE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{1,62}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
ARTIFACT_PATH_RE = re.compile(r"^/api/v1/device/artifacts/[a-z][a-z0-9_-]{1,62}/[0-9]{1,9}$")

ModelName = Annotated[str, StringConstraints(pattern=NAME_PATTERN)]
DeviceId = Annotated[str, StringConstraints(pattern=DEVICE_ID_PATTERN)]
GroupName = Annotated[str, StringConstraints(pattern=NAME_PATTERN)]
Sha256 = Annotated[str, StringConstraints(pattern=SHA256_PATTERN)]

MODEL_FORMAT: Final = "mlp-v1"
MODEL_FORMAT_VERSION = 1
FRAMEWORK = "safetensors-numpy"


def utcnow() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class ModelRef(StrictModel):
    """A registered model version. ``version`` is the MLflow registry version number."""

    name: ModelName
    version: int = Field(ge=1, le=999_999_999)

    def __str__(self) -> str:
        return f"{self.name}:{self.version}"

    def is_newer_than(self, other: ModelRef) -> bool:
        if self.name != other.name:
            raise ValueError("cannot compare versions of different models")
        return self.version > other.version


class ModelIdentity(ModelRef):
    """A model version together with the checksum of the bytes the device actually holds."""

    sha256: Sha256


class FeatureSpec(StrictModel):
    name: str = Field(min_length=1, max_length=64)
    dtype: Literal["float32"] = "float32"
    min: float
    max: float


class InputSchema(StrictModel):
    features: list[FeatureSpec] = Field(min_length=1, max_length=256)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]


class Compatibility(StrictModel):
    format: Literal["mlp-v1"] = MODEL_FORMAT
    format_version: int = MODEL_FORMAT_VERSION
    min_agent_version: str = "0.1.0"


class ArtifactRef(StrictModel):
    """Where a device can fetch bytes. Always a path on the configured control plane, never a URL."""

    path: str
    sha256: Sha256
    size_bytes: int = Field(ge=1, le=256 * 1024 * 1024)

    @field_validator("path")
    @classmethod
    def _path_is_distribution_endpoint(cls, v: str) -> str:
        if not ARTIFACT_PATH_RE.match(v):
            raise ValueError("artifact path must be a /api/v1/device/artifacts/<name>/<version> path")
        return v


class HealthCriteria(StrictModel):
    """Conditions a freshly activated model must satisfy before the rollout may continue."""

    ack_timeout_seconds: float = Field(default=60.0, gt=0, le=3600)
    observation_seconds: float = Field(default=10.0, ge=0, le=3600)
    min_inferences: int = Field(default=20, ge=0, le=1_000_000)
    max_error_rate: float = Field(default=0.05, ge=0, le=1)
    max_p95_latency_ms: float = Field(default=50.0, gt=0)
    max_cpu_percent: float = Field(default=95.0, gt=0, le=100)
    max_memory_mb: float = Field(default=1024.0, gt=0)
    health_timeout_seconds: float = Field(default=120.0, gt=0, le=3600)
    max_command_attempts: int = Field(default=4, ge=1, le=20)


class RolloutStrategy(StrictModel):
    """First batch is the canary; remaining devices follow in batches of ``batch_percent``."""

    canary_count: int = Field(default=1, ge=1, le=1000)
    batch_percent: int = Field(default=100, ge=1, le=100)


class Manifest(StrictModel):
    """Everything a device needs (and nothing it must trust implicitly) to stage a model."""

    schema_version: Literal[1] = 1
    deployment_id: str = Field(min_length=8, max_length=64)
    created_at: datetime
    model: ModelRef
    artifact: ArtifactRef
    input_schema: InputSchema
    framework: str = FRAMEWORK
    compatibility: Compatibility
    target_group: GroupName
    strategy: RolloutStrategy
    health_criteria: HealthCriteria


class DeviceCapabilities(StrictModel):
    agent_version: str = Field(max_length=32)
    formats: list[str] = Field(default_factory=list, max_length=16)


def parse_semver(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in value.split("."))
    except ValueError as exc:
        raise ValueError(f"invalid version string: {value!r}") from exc


def compatibility_problems(compat: Compatibility, caps: DeviceCapabilities) -> list[str]:
    """Reasons a device cannot run a model with ``compat``; empty list means compatible."""
    problems: list[str] = []
    if compat.format not in caps.formats:
        problems.append(f"device does not support model format {compat.format!r}")
    try:
        if parse_semver(caps.agent_version) < parse_semver(compat.min_agent_version):
            problems.append(f"agent {caps.agent_version} older than required {compat.min_agent_version}")
    except ValueError as exc:
        problems.append(str(exc))
    return problems


class FaultConfig(StrictModel):
    """Deterministic failure injection, keyed by model version (simulation only)."""

    corrupt_download_versions: list[int] = Field(default_factory=list, max_length=64)
    fail_activation_versions: list[int] = Field(default_factory=list, max_length=64)
    fail_inference_versions: list[int] = Field(default_factory=list, max_length=64)
    extra_latency_ms: dict[str, float] = Field(default_factory=dict)  # version -> ms
    drop_acks: bool = False
    fail_rollback: bool = False  # ROLLBACK commands fail (exercises the deployment FAILED path)
    drop_ack_outcomes: list[str] = Field(default_factory=list, max_length=16)  # lose only these acks, e.g. ["ACTIVE"]
    fake_free_disk_mb: int | None = Field(default=None, ge=0)  # pretend the data volume has only this much free space
