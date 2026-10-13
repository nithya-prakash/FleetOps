"""Shared test builders (real bundles, signed commands, manifests)."""

from __future__ import annotations

import uuid
from datetime import timedelta
from pathlib import Path

from fleetops.domain.messages import Command, CommandType, RollbackTarget
from fleetops.domain.models import (
    ArtifactRef,
    Compatibility,
    FaultConfig,
    HealthCriteria,
    Manifest,
    ModelIdentity,
    ModelRef,
    RolloutStrategy,
    utcnow,
)
from fleetops.registry.modelformat import BundleMeta, save_bundle
from fleetops.security.artifact import sha256_file
from fleetops.security.device_identity import sign_command
from fleetops.training.pipeline import TrainingResult


def write_bundle(path: Path, result: TrainingResult) -> tuple[str, int]:
    save_bundle(path, result.tensors, result.meta)
    return sha256_file(path), path.stat().st_size


def make_manifest(
    name: str,
    version: int,
    sha: str,
    size: int,
    meta: BundleMeta,
    deployment_id: str = "dep-00000001",
    group: str = "line-a",
) -> Manifest:
    return Manifest(
        deployment_id=deployment_id,
        created_at=utcnow(),
        model=ModelRef(name=name, version=version),
        artifact=ArtifactRef(path=f"/api/v1/device/artifacts/{name}/{version}", sha256=sha, size_bytes=size),
        input_schema=meta.input_schema,
        compatibility=Compatibility(),
        target_group=group,
        strategy=RolloutStrategy(),
        health_criteria=HealthCriteria(),
    )


_gen = {"n": 0}


def make_command(
    secret: str, ctype: CommandType, device_id: str = "edge-01", generation: int | None = None, **kw: object
) -> Command:
    if generation is None:
        _gen["n"] += 1
        generation = _gen["n"]
    now = utcnow()
    cmd = Command(
        command_id=kw.pop("command_id", uuid.uuid4().hex),  # type: ignore[arg-type]
        type=ctype,
        device_id=device_id,
        generation=generation,
        issued_at=now,
        expires_at=kw.pop("expires_at", now + timedelta(minutes=5)),  # type: ignore[arg-type]
        correlation_id="test",
        **kw,  # type: ignore[arg-type]
    )
    return sign_command(secret, cmd)


def rollback_to(ident: ModelIdentity | None, manifest: Manifest | None = None) -> RollbackTarget:
    return RollbackTarget(model=ident, manifest=manifest)


def faults(**kw: object) -> FaultConfig:
    return FaultConfig(**kw)  # type: ignore[arg-type]
