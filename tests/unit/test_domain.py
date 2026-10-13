from __future__ import annotations

import pytest
from pydantic import ValidationError

from fleetops.domain.health import HealthObservation, Verdict, evaluate_health
from fleetops.domain.models import (
    ArtifactRef,
    Compatibility,
    DeviceCapabilities,
    HealthCriteria,
    ModelRef,
    RolloutStrategy,
    compatibility_problems,
)
from fleetops.domain.planning import plan_batches
from fleetops.domain.states import (
    DeploymentState,
    InvalidTransitionError,
    TargetState,
    ensure_deployment_transition,
    ensure_target_transition,
)


def test_model_version_comparison() -> None:
    a, b = ModelRef(name="anomaly", version=1), ModelRef(name="anomaly", version=2)
    assert b.is_newer_than(a)
    assert not a.is_newer_than(b)
    assert not a.is_newer_than(a)
    with pytest.raises(ValueError, match="different models"):
        a.is_newer_than(ModelRef(name="other", version=3))


@pytest.mark.parametrize("name", ["Upper", "1abc", "a", "has space", "../etc", "x" * 80])
def test_model_name_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        ModelRef(name=name, version=1)


@pytest.mark.parametrize(
    "path",
    [
        "https://evil.example/model",
        "/etc/passwd",
        "/api/v1/device/artifacts/../x/1",
        "/api/v1/device/artifacts/model/abc",
    ],
)
def test_manifest_artifact_path_must_be_distribution_endpoint(path: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactRef(path=path, sha256="a" * 64, size_bytes=10)


def test_artifact_ref_accepts_distribution_path() -> None:
    ref = ArtifactRef(path="/api/v1/device/artifacts/anomaly/3", sha256="a" * 64, size_bytes=10)
    assert ref.path.endswith("/3")


def test_artifact_checksum_format_enforced() -> None:
    with pytest.raises(ValidationError):
        ArtifactRef(path="/api/v1/device/artifacts/anomaly/3", sha256="XYZ", size_bytes=10)


def test_target_state_transitions() -> None:
    ensure_target_transition(TargetState.PENDING, TargetState.DISTRIBUTING)
    ensure_target_transition(TargetState.DISTRIBUTING, TargetState.STAGED)
    ensure_target_transition(TargetState.STAGED, TargetState.ACTIVE)
    ensure_target_transition(TargetState.ACTIVE, TargetState.ROLLED_BACK)
    for bad in [
        (TargetState.PENDING, TargetState.ACTIVE),
        (TargetState.ACTIVE, TargetState.STAGED),
        (TargetState.ROLLED_BACK, TargetState.ACTIVE),
        (TargetState.CANCELLED, TargetState.PENDING),
    ]:
        with pytest.raises(InvalidTransitionError):
            ensure_target_transition(*bad)


def test_deployment_state_transitions() -> None:
    ensure_deployment_transition(DeploymentState.PENDING, DeploymentState.IN_PROGRESS)
    ensure_deployment_transition(DeploymentState.IN_PROGRESS, DeploymentState.ROLLING_BACK)
    ensure_deployment_transition(DeploymentState.ROLLING_BACK, DeploymentState.ROLLED_BACK)
    with pytest.raises(InvalidTransitionError):
        ensure_deployment_transition(DeploymentState.ROLLED_BACK, DeploymentState.IN_PROGRESS)
    with pytest.raises(InvalidTransitionError):
        ensure_deployment_transition(DeploymentState.CANCELLED, DeploymentState.COMPLETED)


@pytest.mark.parametrize(
    ("n", "strategy", "expected_sizes"),
    [
        (0, RolloutStrategy(), []),
        (1, RolloutStrategy(), [1]),
        (4, RolloutStrategy(canary_count=1), [1, 3]),
        (10, RolloutStrategy(canary_count=2, batch_percent=30), [2, 3, 3, 2]),
        (3, RolloutStrategy(canary_count=5), [3]),
    ],
)
def test_plan_batches(n: int, strategy: RolloutStrategy, expected_sizes: list[int]) -> None:
    ids = [f"dev-{i:02d}" for i in range(n)]
    batches = plan_batches(list(reversed(ids)), strategy)
    assert [len(b) for b in batches] == expected_sizes
    assert [d for b in batches for d in b] == ids  # deterministic order, nothing lost/duplicated


CRIT = HealthCriteria(observation_seconds=10, min_inferences=20, max_error_rate=0.05, max_p95_latency_ms=50)


def obs(**kw: float) -> HealthObservation:
    base = {"inference_count": 100, "inference_errors": 0, "latency_p95_ms": 5.0, "observed_seconds": 30.0}
    base.update(kw)
    return HealthObservation(**base)  # type: ignore[arg-type]


def test_health_pass() -> None:
    assert evaluate_health(CRIT, obs()).verdict is Verdict.PASS


def test_health_pending_until_observed_long_enough() -> None:
    assert evaluate_health(CRIT, obs(observed_seconds=2)).verdict is Verdict.PENDING
    assert evaluate_health(CRIT, obs(inference_count=10)).verdict is Verdict.PENDING


def test_health_fails_early_on_error_rate() -> None:
    res = evaluate_health(CRIT, obs(inference_count=6, inference_errors=6, observed_seconds=1))
    assert res.verdict is Verdict.FAIL
    assert "error rate" in res.reasons[0]


def test_health_ignores_noise_with_too_few_samples() -> None:
    res = evaluate_health(CRIT, obs(inference_count=2, inference_errors=2, observed_seconds=1))
    assert res.verdict is Verdict.PENDING


def test_health_fails_on_latency_cpu_memory() -> None:
    assert evaluate_health(CRIT, obs(latency_p95_ms=80)).verdict is Verdict.FAIL
    assert evaluate_health(CRIT, obs(cpu_percent=99)).verdict is Verdict.FAIL
    assert evaluate_health(CRIT, obs(memory_mb=5000)).verdict is Verdict.FAIL


def test_compatibility() -> None:
    caps = DeviceCapabilities(agent_version="0.1.0", formats=["mlp-v1"])
    assert compatibility_problems(Compatibility(), caps) == []
    assert compatibility_problems(Compatibility(min_agent_version="0.2.0"), caps)
    assert compatibility_problems(Compatibility(), DeviceCapabilities(agent_version="0.1.0", formats=[]))
    assert compatibility_problems(Compatibility(), DeviceCapabilities(agent_version="x", formats=["mlp-v1"]))
