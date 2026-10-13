"""Pure health evaluation of a freshly activated model against configured criteria."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from fleetops.domain.models import HealthCriteria

# Below this many inferences, rates are too noisy to fail a rollout on.
EARLY_FAIL_MIN_SAMPLES = 5


class Verdict(StrEnum):
    PASS = "PASS"  # noqa: S105
    PENDING = "PENDING"
    FAIL = "FAIL"


class HealthObservation(BaseModel):
    """What a device reports about the model it has been running since activation."""

    inference_count: int = Field(ge=0)
    inference_errors: int = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)
    observed_seconds: float = Field(ge=0)
    cpu_percent: float = Field(default=0.0, ge=0)
    memory_mb: float = Field(default=0.0, ge=0)


class HealthResult(BaseModel):
    verdict: Verdict
    reasons: list[str] = Field(default_factory=list)


def evaluate_health(criteria: HealthCriteria, obs: HealthObservation) -> HealthResult:
    reasons: list[str] = []
    enough_for_rates = obs.inference_count >= EARLY_FAIL_MIN_SAMPLES
    if enough_for_rates:
        error_rate = obs.inference_errors / obs.inference_count
        if error_rate > criteria.max_error_rate:
            reasons.append(f"inference error rate {error_rate:.1%} > {criteria.max_error_rate:.1%}")
        if obs.latency_p95_ms > criteria.max_p95_latency_ms:
            reasons.append(f"p95 latency {obs.latency_p95_ms:.1f}ms > {criteria.max_p95_latency_ms:.1f}ms")
    if obs.cpu_percent > criteria.max_cpu_percent:
        reasons.append(f"cpu {obs.cpu_percent:.0f}% > {criteria.max_cpu_percent:.0f}%")
    if obs.memory_mb > criteria.max_memory_mb:
        reasons.append(f"memory {obs.memory_mb:.0f}MB > {criteria.max_memory_mb:.0f}MB")
    if reasons:
        return HealthResult(verdict=Verdict.FAIL, reasons=reasons)

    if obs.observed_seconds >= criteria.observation_seconds and (obs.inference_count >= criteria.min_inferences):
        return HealthResult(verdict=Verdict.PASS)
    return HealthResult(
        verdict=Verdict.PENDING,
        reasons=[
            f"observed {obs.observed_seconds:.0f}s/{criteria.observation_seconds:.0f}s, "
            f"{obs.inference_count}/{criteria.min_inferences} inferences"
        ],
    )
