"""Quality gate integrated into deployment creation (real SQLite, fake publisher)."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from fleetops.deployment.engine import (
    DeploymentSpec,
    ForbiddenError,
    QualityGateError,
    RolloutEngine,
)
from fleetops.domain.messages import CommandType
from fleetops.domain.quality import (
    EVAL_SUITE,
    METRIC_DEFINITIONS_VERSION,
    MetricRule,
    QualityGate,
    QualityOverride,
)
from fleetops.persistence.db import session_scope
from fleetops.persistence.tables import AuditEvent, Deployment, Device, ModelVersion
from tests.unit.test_engine import CRIT, Env

EVAL = {
    "suite": EVAL_SUITE,
    "dataset_fingerprint": "fp-1",
    "metric_definitions_version": METRIC_DEFINITIONS_VERSION,
    "threshold_policy": "model-own-threshold",
    "latency_environment": "linux-arm64-py3.12",
}
BASELINE = {"bench_f1": 0.80, "bench_recall": 0.75, "bench_p95_latency_ms": 1.0}


@pytest.fixture
def env(tmp_path) -> Env:  # type: ignore[no-untyped-def]
    e = Env(tmp_path)
    set_model(e, 1, **BASELINE)  # incumbent running on both devices
    set_model(e, 2, bench_f1=0.81, bench_recall=0.76, bench_p95_latency_ms=1.1)
    return e


def set_model(e: Env, version: int, evaluation: dict | None = None, **metrics: float) -> None:  # type: ignore[type-arg]
    with session_scope(e.sessions) as db:
        row = db.scalar(select(ModelVersion).where(ModelVersion.name == "anomaly", ModelVersion.version == version))
        assert row is not None
        row.metrics = metrics
        row.evaluation = EVAL if evaluation is None else evaluation


def spec(e: Env, gate: QualityGate | None, override: str | None = None, version: int = 2) -> DeploymentSpec:
    s = e.spec(version)
    s.quality_gate = gate
    s.quality_override = QualityOverride(reason=override) if override else None
    return s


def no_regression(**tolerances: float) -> QualityGate:
    return QualityGate(metrics={m: MetricRule(max_regression=t) for m, t in tolerances.items()})


def assert_nothing_happened(e: Env) -> None:
    with session_scope(e.sessions) as db:
        assert db.scalars(select(Deployment)).all() == []
    assert e.pub.sent == []  # no command ever reached a device


def test_passing_candidate_is_deployed_and_report_persisted(env: Env) -> None:
    gate = QualityGate(
        metrics={
            "bench_f1": MetricRule(min=0.7, max_regression=0.0),
            "bench_p95_latency_ms": MetricRule(max_regression=0.2, relative=True),
        }
    )
    created = env.engine.create_deployment(spec(env, gate), "op")
    assert env.pub.of(CommandType.DEPLOY)  # normal rollout proceeds (canary first)
    report = env.dep(created.deployment_id).quality_report
    assert (
        report is not None
        and report["passed"] is True
        and report["compared_against"] == [1]
        and not report["overridden"]
    )


def test_regression_is_blocked_with_actionable_details(env: Env) -> None:
    set_model(env, 2, bench_f1=0.70, bench_recall=0.76, bench_p95_latency_ms=1.1)
    with pytest.raises(QualityGateError) as exc:
        env.engine.create_deployment(spec(env, no_regression(bench_f1=0.05)), "op")
    f = exc.value.report.failures[0]
    assert (f.code, f.metric, f.baseline_version) == ("REGRESSION", "bench_f1", 1)
    assert f.candidate == pytest.approx(0.70) and f.baseline == pytest.approx(0.80) and f.limit == pytest.approx(0.75)
    assert "regressed" in str(exc.value)
    assert_nothing_happened(env)


def test_latency_regression_uses_lower_is_better(env: Env) -> None:
    set_model(env, 2, bench_f1=0.9, bench_p95_latency_ms=2.0)  # F1 improved, latency doubled
    gate = no_regression(bench_p95_latency_ms=0.5)
    with pytest.raises(QualityGateError) as exc:
        env.engine.create_deployment(spec(env, gate), "op")
    assert exc.value.report.failures[0].metric == "bench_p95_latency_ms"
    assert_nothing_happened(env)
    set_model(env, 2, bench_f1=0.9, bench_p95_latency_ms=1.5)  # exactly at the limit: allowed
    assert env.engine.create_deployment(spec(env, gate), "op").created


def test_exact_boundary_value_passes(env: Env) -> None:
    set_model(env, 2, bench_f1=0.75)  # baseline 0.80 - tolerance 0.05 == exactly the limit
    assert env.engine.create_deployment(spec(env, no_regression(bench_f1=0.05)), "op").created


def test_missing_candidate_metric_fails_safely(env: Env) -> None:
    set_model(env, 2, bench_f1=0.9)  # no recall reported
    with pytest.raises(QualityGateError) as exc:
        env.engine.create_deployment(spec(env, no_regression(bench_recall=0.1)), "op")
    assert exc.value.report.failures[0].code == "MISSING_METRIC"
    assert_nothing_happened(env)


def test_missing_baseline_metric_fails_safely(env: Env) -> None:
    set_model(env, 1, bench_recall=0.75)  # incumbent lacks F1
    with pytest.raises(QualityGateError) as exc:
        env.engine.create_deployment(spec(env, no_regression(bench_f1=0.1)), "op")
    assert (exc.value.report.failures[0].code, exc.value.report.failures[0].baseline_version) == ("MISSING_METRIC", 1)


def test_incomparable_evaluation_metadata_fails_safely(env: Env) -> None:
    set_model(env, 2, evaluation={**EVAL, "dataset_fingerprint": "fp-OTHER"}, bench_f1=0.99)
    with pytest.raises(QualityGateError) as exc:
        env.engine.create_deployment(spec(env, no_regression(bench_f1=0.5)), "op")
    f = exc.value.report.failures[0]
    assert f.code == "INCOMPARABLE" and "dataset_fingerprint" in f.message  # a "better" number must not slip through
    assert_nothing_happened(env)


def test_legacy_model_without_evaluation_metadata_is_incomparable(env: Env) -> None:
    set_model(env, 1, evaluation={}, **BASELINE)
    with pytest.raises(QualityGateError) as exc:
        env.engine.create_deployment(spec(env, no_regression(bench_f1=0.5)), "op")
    assert exc.value.report.failures[0].code == "INCOMPARABLE"


def test_baseline_is_every_distinct_incumbent_version(env: Env) -> None:
    # edge-02 runs a different (better) version; the candidate must not regress against either
    with session_scope(env.sessions) as db:
        db.add(
            ModelVersion(
                name="anomaly",
                version=3,
                sha256="3" * 64,
                size_bytes=1,
                framework="f",
                compatibility=env.spec().model_dump(),
                input_schema={},
                metrics={"bench_f1": 0.95},
                params={},
                reference_stats={},
                evaluation=EVAL,
                mlflow_run_id="r",
                created_at=env.clock.now,
            )
        )
        dev = db.get(Device, "edge-02")
        assert dev is not None
        dev.active_model_version = 3
    set_model(env, 2, bench_f1=0.85)  # beats v1 (0.80) but regresses against v3 (0.95)
    with pytest.raises(QualityGateError) as exc:
        env.engine.create_deployment(spec(env, no_regression(bench_f1=0.05)), "op")
    assert [f.baseline_version for f in exc.value.report.failures] == [3]
    assert exc.value.report.compared_against == [1, 3]


def test_explicit_baseline_version_and_unregistered_baseline(env: Env) -> None:
    explicit = QualityGate(metrics={"bench_f1": MetricRule(max_regression=0.0)}, baseline_version=1)
    assert env.engine.create_deployment(spec(env, explicit), "op").created
    ghost = QualityGate(metrics={"bench_f1": MetricRule(max_regression=0.0)}, baseline_version=77)
    env2_spec = spec(env, ghost)
    with pytest.raises(QualityGateError) as exc:
        env.engine.create_deployment(env2_spec, "op")
    assert exc.value.report.failures[0].code == "MISSING_BASELINE"


def test_absolute_minimum_enforced_even_for_first_deployment(env: Env) -> None:
    with session_scope(env.sessions) as db:
        for dev in db.scalars(select(Device)):
            dev.active_model_name = dev.active_model_version = None  # nothing running: no incumbent
    gate = QualityGate(metrics={"bench_f1": MetricRule(min=0.9, max_regression=0.0)})
    with pytest.raises(QualityGateError) as exc:
        env.engine.create_deployment(spec(env, gate), "op")
    assert [f.code for f in exc.value.report.failures] == ["BELOW_MINIMUM"]


def test_authorized_override_is_audited_with_reason_and_failures(env: Env) -> None:
    set_model(env, 2, bench_f1=0.60)
    reason = "regulator-approved temporary model for line 3"
    created = env.engine.create_deployment(
        spec(env, no_regression(bench_f1=0.05), reason), "alice-admin", may_override=True
    )
    report = env.dep(created.deployment_id).quality_report
    assert report is not None and report["overridden"] is True and report["override_by"] == "alice-admin"
    assert report["override_reason"] == reason and report["failures"][0]["code"] == "REGRESSION"
    with session_scope(env.sessions) as db:
        ev = db.scalar(select(AuditEvent).where(AuditEvent.action == "deployment.quality_gate_overridden"))
        assert ev is not None and ev.actor == "alice-admin" and ev.details["reason"] == reason
        assert ev.deployment_id == created.deployment_id and ev.details["failures"]
    assert env.pub.of(CommandType.DEPLOY)  # health checks / canary still apply to the overridden rollout


def test_override_without_permission_is_forbidden_even_if_gate_would_pass(env: Env) -> None:
    with pytest.raises(ForbiddenError, match="admin"):
        env.engine.create_deployment(
            spec(env, no_regression(bench_f1=0.5), "please let me through"), "op", may_override=False
        )
    assert_nothing_happened(env)


def test_override_reason_is_required_to_be_meaningful() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        QualityOverride(reason="ok")


def test_policy_can_require_a_gate_on_every_deployment(env: Env) -> None:
    strict = RolloutEngine(
        env.sessions, env.pub, env.engine._metrics, "k" * 24, require_quality_gate=True, clock=env.clock
    )
    with pytest.raises(QualityGateError) as exc:
        strict.create_deployment(spec(env, None), "op")
    assert exc.value.report.failures[0].code == "GATE_REQUIRED"
    assert_nothing_happened(env)
    assert strict.create_deployment(
        spec(env, None, "emergency rollout approved by plant manager"), "root", may_override=True
    ).created


def test_no_gate_keeps_previous_behaviour(env: Env) -> None:
    created = env.engine.create_deployment(spec(env, None), "op")
    assert env.dep(created.deployment_id).quality_report is None


def test_idempotent_replay_returns_original_without_re_evaluating(env: Env) -> None:
    s = spec(env, no_regression(bench_f1=0.05))
    first = env.engine.create_deployment(s, "op", "key-q")
    set_model(env, 2, bench_f1=0.1)  # would fail now, but the original request already succeeded
    again = env.engine.create_deployment(s, "op", "key-q")
    assert again.deployment_id == first.deployment_id and not again.created


def test_health_criteria_remain_independent_of_quality_gate(env: Env) -> None:
    created = env.engine.create_deployment(spec(env, no_regression(bench_f1=0.05)), "op")
    assert env.dep(created.deployment_id).manifest["health_criteria"]["ack_timeout_seconds"] == CRIT.ack_timeout_seconds
