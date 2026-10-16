"""Model-quality gate: pure evaluation of a candidate against thresholds and the incumbent model(s).

Principles:
* Direction matters: higher is better for precision/recall/F1/AUC, lower is better for latency and
  false-positive rate. A rule's bound must match the metric's direction.
* Models are compared only when they were evaluated on the *same* benchmark: identical suite, dataset
  fingerprint, metric-definition version and threshold policy (plus the same host class for latency).
  Anything else is "incomparable" and fails safely instead of comparing unlike numbers.
* Missing metrics fail. A healthy container is not evidence of a good model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from fleetops.domain.models import StrictModel

METRIC_DEFINITIONS_VERSION = 1
EVAL_SUITE = "hydraulic-benchmark-v1"


class Direction(StrEnum):
    HIGHER_IS_BETTER = "higher"
    LOWER_IS_BETTER = "lower"


METRIC_DIRECTION: dict[str, Direction] = {
    "bench_precision": Direction.HIGHER_IS_BETTER,
    "bench_recall": Direction.HIGHER_IS_BETTER,
    "bench_f1": Direction.HIGHER_IS_BETTER,
    "bench_roc_auc": Direction.HIGHER_IS_BETTER,
    "bench_false_positive_rate": Direction.LOWER_IS_BETTER,
    "bench_p95_latency_ms": Direction.LOWER_IS_BETTER,
}
HOST_DEPENDENT = frozenset({"bench_p95_latency_ms"})
# evaluation-metadata keys that must be equal for any metric comparison
COMPARABILITY_KEYS = ("suite", "dataset_fingerprint", "metric_definitions_version", "threshold_policy")
EPS = 1e-12

FailureCode = Literal[
    "GATE_REQUIRED",
    "MISSING_METRIC",
    "INCOMPARABLE",
    "MISSING_BASELINE",
    "BELOW_MINIMUM",
    "ABOVE_MAXIMUM",
    "REGRESSION",
]


class MetricRule(StrictModel):
    """``min`` for higher-is-better metrics, ``max`` for lower-is-better; ``max_regression`` vs the baseline."""

    min: float | None = None
    max: float | None = None
    max_regression: float | None = Field(default=None, ge=0)
    relative: bool = False  # interpret max_regression as a fraction of the baseline value

    @model_validator(mode="after")
    def _has_a_bound(self) -> MetricRule:
        if self.min is None and self.max is None and self.max_regression is None:
            raise ValueError("a metric rule needs at least one of min, max, max_regression")
        return self


class QualityGate(StrictModel):
    metrics: dict[str, MetricRule] = Field(min_length=1, max_length=len(METRIC_DIRECTION))
    baseline_version: int | None = Field(default=None, ge=1)  # default: the version(s) currently running
    require_baseline: bool = False  # fail instead of skipping regression checks when there is no incumbent

    @model_validator(mode="after")
    def _validate_metrics(self) -> QualityGate:
        for name, rule in self.metrics.items():
            direction = METRIC_DIRECTION.get(name)
            if direction is None:
                raise ValueError(f"unknown metric {name!r}; known: {sorted(METRIC_DIRECTION)}")
            if direction is Direction.HIGHER_IS_BETTER and rule.max is not None:
                raise ValueError(f"{name} is higher-is-better: use 'min', not 'max'")
            if direction is Direction.LOWER_IS_BETTER and rule.min is not None:
                raise ValueError(f"{name} is lower-is-better: use 'max', not 'min'")
        return self


class QualityOverride(StrictModel):
    reason: str = Field(min_length=10, max_length=500)


@dataclass(frozen=True)
class ModelEvidence:
    version: int
    metrics: dict[str, Any]
    evaluation: dict[str, Any] = field(default_factory=dict)


class Failure(BaseModel):
    code: FailureCode
    metric: str | None = None
    message: str
    candidate: float | None = None
    baseline: float | None = None
    baseline_version: int | None = None
    limit: float | None = None


class QualityReport(BaseModel):
    passed: bool
    failures: list[Failure] = Field(default_factory=list)
    compared_against: list[int] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    overridden: bool = False
    override_reason: str | None = None
    override_by: str | None = None

    def summary(self) -> str:
        return "; ".join(f.message for f in self.failures) or "quality gate passed"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def incomparable_reason(metric: str, candidate: dict[str, Any], baseline: dict[str, Any]) -> str | None:
    """Why two models' ``metric`` values cannot be compared, or None if they can."""
    if not candidate or not baseline:
        return "evaluation metadata is missing"
    keys = list(COMPARABILITY_KEYS) + (["latency_environment"] if metric in HOST_DEPENDENT else [])
    for key in keys:
        if candidate.get(key) is None or baseline.get(key) is None:
            return f"evaluation metadata lacks {key!r}"
        if candidate[key] != baseline[key]:
            return f"{key} differs (candidate {candidate[key]!r} vs baseline {baseline[key]!r})"
    return None


def evaluate_quality_gate(gate: QualityGate, candidate: ModelEvidence, baselines: list[ModelEvidence]) -> QualityReport:
    failures: list[Failure] = []
    skipped: list[str] = []
    needs_baseline = any(r.max_regression is not None for r in gate.metrics.values())
    if needs_baseline and not baselines:
        if gate.require_baseline:
            failures.append(
                Failure(
                    code="MISSING_BASELINE", message="no incumbent model to compare against (require_baseline is set)"
                )
            )
        else:
            skipped.append("no incumbent model: regression checks skipped, absolute thresholds still enforced")

    for name, rule in gate.metrics.items():
        higher = METRIC_DIRECTION[name] is Direction.HIGHER_IS_BETTER
        value = _number(candidate.metrics.get(name))
        if value is None:
            failures.append(
                Failure(
                    code="MISSING_METRIC", metric=name, message=f"candidate v{candidate.version} has no usable {name}"
                )
            )
            continue
        if higher and rule.min is not None and value < rule.min - EPS:
            failures.append(
                Failure(
                    code="BELOW_MINIMUM",
                    metric=name,
                    candidate=value,
                    limit=rule.min,
                    message=f"{name} {value:.4f} is below the required minimum {rule.min:.4f}",
                )
            )
        if not higher and rule.max is not None and value > rule.max + EPS:
            failures.append(
                Failure(
                    code="ABOVE_MAXIMUM",
                    metric=name,
                    candidate=value,
                    limit=rule.max,
                    message=f"{name} {value:.4f} exceeds the allowed maximum {rule.max:.4f}",
                )
            )
        if rule.max_regression is None:
            continue
        for base in baselines:
            reason = incomparable_reason(name, candidate.evaluation, base.evaluation)
            if reason:
                failures.append(
                    Failure(
                        code="INCOMPARABLE",
                        metric=name,
                        baseline_version=base.version,
                        message=f"cannot compare {name} of v{candidate.version} with v{base.version}: {reason}",
                    )
                )
                continue
            base_value = _number(base.metrics.get(name))
            if base_value is None:
                failures.append(
                    Failure(
                        code="MISSING_METRIC",
                        metric=name,
                        baseline_version=base.version,
                        message=f"baseline v{base.version} has no usable {name}",
                    )
                )
                continue
            allowed = rule.max_regression * abs(base_value) if rule.relative else rule.max_regression
            limit = base_value - allowed if higher else base_value + allowed
            worse = value < limit - EPS if higher else value > limit + EPS
            if worse:
                failures.append(
                    Failure(
                        code="REGRESSION",
                        metric=name,
                        candidate=value,
                        baseline=base_value,
                        baseline_version=base.version,
                        limit=limit,
                        message=(
                            f"{name} regressed: {value:.4f} vs v{base.version} {base_value:.4f} "
                            f"(allowed {'drop' if higher else 'increase'} {allowed:.4f}, limit {limit:.4f})"
                        ),
                    )
                )
    return QualityReport(
        passed=not failures, failures=failures, compared_against=[b.version for b in baselines], skipped=skipped
    )
