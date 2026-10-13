from __future__ import annotations

import pytest
from pydantic import ValidationError

from fleetops.domain.quality import (
    EVAL_SUITE,
    METRIC_DEFINITIONS_VERSION,
    METRIC_DIRECTION,
    Direction,
    MetricRule,
    ModelEvidence,
    QualityGate,
    evaluate_quality_gate,
)

EVAL = {
    "suite": EVAL_SUITE,
    "dataset_fingerprint": "abc",
    "metric_definitions_version": METRIC_DEFINITIONS_VERSION,
    "threshold_policy": "model-own-threshold",
    "latency_environment": "linux-aarch64-py3.12",
}


def ev(version: int, evaluation: dict | None = None, **metrics: float) -> ModelEvidence:  # type: ignore[type-arg]
    return ModelEvidence(version, dict(metrics), EVAL if evaluation is None else evaluation)


def gate(**rules: MetricRule) -> QualityGate:
    return QualityGate(metrics=rules)


def codes(report) -> list[str]:  # type: ignore[no-untyped-def]
    return [f.code for f in report.failures]


def test_metric_directions_are_declared_for_every_gateable_metric() -> None:
    assert METRIC_DIRECTION["bench_f1"] is Direction.HIGHER_IS_BETTER
    assert METRIC_DIRECTION["bench_p95_latency_ms"] is Direction.LOWER_IS_BETTER
    assert METRIC_DIRECTION["bench_false_positive_rate"] is Direction.LOWER_IS_BETTER


def test_rule_validation_rejects_wrong_direction_unknown_metric_and_empty_rule() -> None:
    with pytest.raises(ValidationError, match="higher-is-better"):
        gate(bench_f1=MetricRule(max=0.9))
    with pytest.raises(ValidationError, match="lower-is-better"):
        gate(bench_p95_latency_ms=MetricRule(min=1.0))
    with pytest.raises(ValidationError, match="unknown metric"):
        gate(accuracy=MetricRule(min=0.9))
    with pytest.raises(ValidationError, match="at least one"):
        MetricRule()
    with pytest.raises(ValidationError):
        MetricRule(max_regression=-0.1)


def test_passing_candidate() -> None:
    g = gate(
        bench_f1=MetricRule(min=0.7, max_regression=0.02),
        bench_p95_latency_ms=MetricRule(max=5, max_regression=0.5, relative=True),
    )
    r = evaluate_quality_gate(
        g, ev(2, bench_f1=0.80, bench_p95_latency_ms=0.2), [ev(1, bench_f1=0.79, bench_p95_latency_ms=0.19)]
    )
    assert r.passed and r.compared_against == [1] and r.failures == []


@pytest.mark.parametrize(("value", "passes"), [(0.7001, True), (0.70, True), (0.6999, False)])
def test_minimum_threshold_boundary_is_inclusive(value: float, passes: bool) -> None:
    r = evaluate_quality_gate(gate(bench_f1=MetricRule(min=0.70)), ev(2, bench_f1=value), [])
    assert r.passed is passes
    assert passes or codes(r) == ["BELOW_MINIMUM"]


@pytest.mark.parametrize(("value", "passes"), [(0.9, True), (1.0, True), (1.0001, False)])
def test_lower_is_better_maximum_boundary(value: float, passes: bool) -> None:
    r = evaluate_quality_gate(gate(bench_p95_latency_ms=MetricRule(max=1.0)), ev(2, bench_p95_latency_ms=value), [])
    assert r.passed is passes
    assert passes or codes(r) == ["ABOVE_MAXIMUM"]


@pytest.mark.parametrize(("candidate", "passes"), [(0.78, True), (0.7800001, True), (0.77, False)])
def test_higher_is_better_regression_boundary_absolute(candidate: float, passes: bool) -> None:
    r = evaluate_quality_gate(
        gate(bench_f1=MetricRule(max_regression=0.02)), ev(2, bench_f1=candidate), [ev(1, bench_f1=0.80)]
    )
    assert r.passed is passes
    assert passes or codes(r) == ["REGRESSION"]


@pytest.mark.parametrize(("candidate", "passes"), [(1.5, True), (1.51, False)])
def test_lower_is_better_regression_is_relative_increase(candidate: float, passes: bool) -> None:
    g = gate(bench_p95_latency_ms=MetricRule(max_regression=0.5, relative=True))
    r = evaluate_quality_gate(g, ev(2, bench_p95_latency_ms=candidate), [ev(1, bench_p95_latency_ms=1.0)])
    assert r.passed is passes


def test_improvement_in_lower_is_better_metric_is_not_a_regression() -> None:
    g = gate(bench_false_positive_rate=MetricRule(max_regression=0.0))
    assert evaluate_quality_gate(
        g, ev(2, bench_false_positive_rate=0.01), [ev(1, bench_false_positive_rate=0.05)]
    ).passed
    worse = evaluate_quality_gate(g, ev(2, bench_false_positive_rate=0.06), [ev(1, bench_false_positive_rate=0.05)])
    assert codes(worse) == ["REGRESSION"] and "increase" in worse.failures[0].message


def test_regression_message_is_actionable() -> None:
    r = evaluate_quality_gate(
        gate(bench_recall=MetricRule(max_regression=0.01)), ev(2, bench_recall=0.5), [ev(1, bench_recall=0.9)]
    )
    f = r.failures[0]
    assert (f.metric, f.candidate, f.baseline, f.baseline_version) == ("bench_recall", 0.5, 0.9, 1)
    assert "regressed" in f.message and "v1" in f.message and f.limit == pytest.approx(0.89)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "0.9", True])
def test_missing_or_unusable_candidate_metric_fails(bad: object) -> None:
    c = ModelEvidence(2, {"bench_f1": bad}, EVAL)
    assert codes(evaluate_quality_gate(gate(bench_f1=MetricRule(min=0.5)), c, [])) == ["MISSING_METRIC"]
    assert codes(evaluate_quality_gate(gate(bench_f1=MetricRule(min=0.5)), ModelEvidence(2, {}, EVAL), [])) == [
        "MISSING_METRIC"
    ]


def test_missing_baseline_metric_fails() -> None:
    r = evaluate_quality_gate(gate(bench_f1=MetricRule(max_regression=0.1)), ev(2, bench_f1=0.8), [ev(1)])
    assert codes(r) == ["MISSING_METRIC"] and r.failures[0].baseline_version == 1


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"dataset_fingerprint": "different"}, "dataset_fingerprint differs"),
        ({"suite": "other-suite"}, "suite differs"),
        ({"metric_definitions_version": 0}, "metric_definitions_version differs"),
        ({"threshold_policy": "fixed-0.5"}, "threshold_policy differs"),
    ],
)
def test_incomparable_evaluation_metadata_fails_safely(override: dict, fragment: str) -> None:  # type: ignore[type-arg]
    base = ev(
        1, evaluation={**EVAL, **override}, bench_f1=0.5
    )  # looks WORSE than the candidate, must still not pass silently
    r = evaluate_quality_gate(gate(bench_f1=MetricRule(max_regression=0.0)), ev(2, bench_f1=0.9), [base])
    assert codes(r) == ["INCOMPARABLE"] and fragment in r.failures[0].message


def test_missing_evaluation_metadata_is_incomparable() -> None:
    r = evaluate_quality_gate(
        gate(bench_f1=MetricRule(max_regression=0.0)), ev(2, bench_f1=0.9), [ev(1, evaluation={}, bench_f1=0.5)]
    )
    assert codes(r) == ["INCOMPARABLE"] and "missing" in r.failures[0].message
    lacking = {k: v for k, v in EVAL.items() if k != "dataset_fingerprint"}
    r = evaluate_quality_gate(
        gate(bench_f1=MetricRule(max_regression=0.0)), ev(2, bench_f1=0.9), [ev(1, evaluation=lacking, bench_f1=0.5)]
    )
    assert codes(r) == ["INCOMPARABLE"]


def test_latency_requires_the_same_host_class() -> None:
    other_host = {**EVAL, "latency_environment": "darwin-arm64"}
    g = gate(bench_p95_latency_ms=MetricRule(max_regression=1.0, relative=True))
    r = evaluate_quality_gate(
        g, ev(2, bench_p95_latency_ms=1.0), [ev(1, evaluation=other_host, bench_p95_latency_ms=1.0)]
    )
    assert codes(r) == ["INCOMPARABLE"] and "latency_environment" in r.failures[0].message
    # quality metrics do not depend on the host
    ok = evaluate_quality_gate(
        gate(bench_f1=MetricRule(max_regression=0.0)), ev(2, bench_f1=0.9), [ev(1, evaluation=other_host, bench_f1=0.9)]
    )
    assert ok.passed


def test_first_deployment_without_incumbent() -> None:
    g = gate(bench_f1=MetricRule(min=0.6, max_regression=0.02))
    r = evaluate_quality_gate(g, ev(1, bench_f1=0.7), [])
    assert r.passed and r.skipped  # absolute threshold enforced, regression skipped, and said so
    assert codes(evaluate_quality_gate(g, ev(1, bench_f1=0.5), [])) == ["BELOW_MINIMUM"]
    strict = QualityGate(metrics=g.metrics, require_baseline=True)
    assert codes(evaluate_quality_gate(strict, ev(1, bench_f1=0.7), [])) == ["MISSING_BASELINE"]


def test_regression_against_every_distinct_incumbent() -> None:
    g = gate(bench_f1=MetricRule(max_regression=0.01))
    r = evaluate_quality_gate(g, ev(3, bench_f1=0.75), [ev(1, bench_f1=0.70), ev(2, bench_f1=0.80)])
    assert [f.baseline_version for f in r.failures] == [2] and r.compared_against == [1, 2]


def test_multiple_failures_are_all_reported() -> None:
    g = gate(bench_f1=MetricRule(min=0.9), bench_recall=MetricRule(min=0.9), bench_p95_latency_ms=MetricRule(max=0.1))
    r = evaluate_quality_gate(g, ev(2, bench_f1=0.5, bench_p95_latency_ms=1.0), [])
    assert sorted(codes(r)) == ["ABOVE_MAXIMUM", "BELOW_MINIMUM", "MISSING_METRIC"]
    assert "below the required minimum" in r.summary()
