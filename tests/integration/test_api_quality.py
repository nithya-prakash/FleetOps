"""Quality gate over HTTP with real trained models (real MLflow, real benchmark metrics)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from fleetops.persistence.db import make_engine
from tests.harness import Stack

GOOD = {"n_samples": 3000, "epochs": 30, "seed": 11}
BAD = {"n_samples": 800, "epochs": 1, "seed": 12, "hidden_sizes": [2], "learning_rate": 0.0001}
UPDATE_FINGERPRINT = (
    "UPDATE model_versions SET evaluation = json_set(evaluation, '$.dataset_fingerprint', 'other') WHERE version = :v"
)
F1_NO_REGRESSION = {"metrics": {"bench_f1": {"max_regression": 0.0}}}


def bench(stack: Stack, version: int) -> dict[str, Any]:
    items = stack.api("GET", "/api/v1/models/anomaly-detector/versions", role="viewer").json()["items"]
    return dict(next(v for v in items if v["version"] == version)["metrics"])


def deploy_body(version: int, **extra: Any) -> dict[str, Any]:
    return {
        "model_name": "anomaly-detector",
        "model_version": version,
        "group": "line-a",
        "health_criteria": {
            "observation_seconds": 1,
            "min_inferences": 10,
            "max_cpu_percent": 100,
            "ack_timeout_seconds": 15,
            "health_timeout_seconds": 15,
        },
        **extra,
    }


def fleet_on_baseline(stack: Stack) -> tuple[int, int]:
    good = stack.train(**GOOD)
    bad = stack.train(**BAD)
    for d in ("edge-01", "edge-02"):
        stack.add_agent(d)
    stack.wait_online("edge-01", "edge-02")
    stack.wait_state(stack.deploy(good)["deployment_id"], {"COMPLETED"})  # baseline (no gate requested)
    return good, bad


def active(stack: Stack, device: str) -> int | None:
    a = stack.agents[device].store.snapshot().active
    return a.version if a else None


def test_models_expose_comparable_benchmark_metrics_and_metadata(stack: Stack) -> None:
    a, b = stack.train(**GOOD), stack.train(**BAD)
    versions = {
        v["version"]: v
        for v in stack.api("GET", "/api/v1/models/anomaly-detector/versions", role="viewer").json()["items"]
    }
    ea, eb = versions[a]["evaluation"], versions[b]["evaluation"]
    assert ea["dataset_fingerprint"] == eb["dataset_fingerprint"]  # different training seeds, same benchmark
    assert ea["suite"] == "sensor-benchmark-v1" and "synthetic" in ea["note"]
    expected = {"bench_f1", "bench_recall", "bench_precision", "bench_p95_latency_ms", "bench_false_positive_rate"}
    assert expected <= set(versions[a]["metrics"])
    assert stack.api("POST", "/api/v1/models/train", json={"params": {"seed": 20240601}}).status_code == 422  # reserved


def test_worse_candidate_is_blocked_and_devices_are_untouched(stack: Stack) -> None:
    good, bad = fleet_on_baseline(stack)
    assert bench(stack, bad)["bench_f1"] < bench(stack, good)["bench_f1"] - 0.05, (
        "precondition: candidate must be worse"
    )
    before = stack.api("GET", "/api/v1/deployments", role="viewer").json()["total"]

    r = stack.api("POST", "/api/v1/deployments", json=deploy_body(bad, quality_gate=F1_NO_REGRESSION))
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert err["code"] == "quality_gate_failed" and "regressed" in err["message"]
    failure = err["details"]["failures"][0]
    assert failure["code"] == "REGRESSION" and failure["metric"] == "bench_f1" and failure["baseline_version"] == good
    assert failure["candidate"] < failure["baseline"] and err["details"]["compared_against"] == [good]

    assert stack.api("GET", "/api/v1/deployments", role="viewer").json()["total"] == before  # nothing created
    assert active(stack, "edge-01") == active(stack, "edge-02") == good  # nothing was sent to devices
    assert all(a.store.snapshot().staged is None for a in stack.agents.values())


def test_better_candidate_passes_and_then_rolls_out_normally(stack: Stack) -> None:
    good, _ = fleet_on_baseline(stack)
    better = stack.train(**{**GOOD, "seed": 13})
    gate = {"metrics": {"bench_f1": {"max_regression": 0.15}}}
    r = stack.api("POST", "/api/v1/deployments", json=deploy_body(better, quality_gate=gate))
    assert r.status_code == 201, r.text
    assert r.json()["quality"]["passed"] is True and r.json()["quality"]["compared_against"] == [good]
    final = stack.wait_state(r.json()["deployment_id"], {"COMPLETED", "ROLLED_BACK", "FAILED"})
    assert final["state"] == "COMPLETED"  # runtime health checks / canary still ran
    assert active(stack, "edge-01") == active(stack, "edge-02") == better


def test_threshold_validation_and_direction_rules(stack: Stack) -> None:
    fleet_on_baseline(stack)
    candidate = stack.train(**{**GOOD, "seed": 14})
    wrong_direction = {"metrics": {"bench_f1": {"max": 0.9}}}
    r = stack.api("POST", "/api/v1/deployments", json=deploy_body(candidate, quality_gate=wrong_direction))
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    unknown = {"metrics": {"accuracy": {"min": 0.9}}}
    assert (
        stack.api("POST", "/api/v1/deployments", json=deploy_body(candidate, quality_gate=unknown)).status_code == 422
    )
    impossible = {"metrics": {"bench_recall": {"min": 1.01}}}
    r = stack.api("POST", "/api/v1/deployments", json=deploy_body(candidate, quality_gate=impossible))
    assert r.status_code == 422 and r.json()["error"]["details"]["failures"][0]["code"] == "BELOW_MINIMUM"


def test_incomparable_evaluation_is_rejected_over_http(stack: Stack) -> None:
    good, bad = fleet_on_baseline(stack)
    engine = make_engine(stack.settings.database_url)
    with engine.begin() as conn:  # simulate a model that was evaluated on another dataset
        conn.execute(
            text(UPDATE_FINGERPRINT),
            {"v": bad},
        )
    r = stack.api("POST", "/api/v1/deployments", json=deploy_body(bad, quality_gate=F1_NO_REGRESSION))
    assert r.status_code == 422
    failure = r.json()["error"]["details"]["failures"][0]
    assert failure["code"] == "INCOMPARABLE" and "dataset_fingerprint" in failure["message"]
    assert active(stack, "edge-01") == good


def test_missing_metric_is_rejected_over_http(stack: Stack) -> None:
    _, bad = fleet_on_baseline(stack)
    engine = make_engine(stack.settings.database_url)
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE model_versions SET metrics = json_remove(metrics, '$.bench_f1') WHERE version = :v"),
            {"v": bad},
        )
    r = stack.api("POST", "/api/v1/deployments", json=deploy_body(bad, quality_gate=F1_NO_REGRESSION))
    assert r.status_code == 422 and r.json()["error"]["details"]["failures"][0]["code"] == "MISSING_METRIC"


def test_override_requires_admin_and_is_audited(stack: Stack) -> None:
    good, bad = fleet_on_baseline(stack)
    reason = "approved by plant manager, ticket OPS-1234"
    body = deploy_body(bad, quality_gate=F1_NO_REGRESSION, quality_override={"reason": reason})

    denied = stack.api("POST", "/api/v1/deployments", role="operator", json=body)
    assert denied.status_code == 403 and denied.json()["error"]["code"] == "forbidden"
    assert active(stack, "edge-01") == good
    short = stack.api("POST", "/api/v1/deployments", role="admin", json={**body, "quality_override": {"reason": "ok"}})
    assert short.status_code == 422  # a reason is mandatory and must be meaningful

    allowed = stack.api("POST", "/api/v1/deployments", role="admin", json=body)
    assert allowed.status_code == 201, allowed.text
    quality = allowed.json()["quality"]
    assert quality["overridden"] is True and quality["override_reason"] == reason and quality["override_by"] == "admin"
    assert quality["failures"][0]["code"] == "REGRESSION"
    events = stack.api("GET", "/api/v1/audit/events?action=deployment.quality_gate_overridden", role="viewer").json()
    assert events["total"] == 1 and events["items"][0]["details"]["reason"] == reason
    assert events["items"][0]["actor"] == "admin"
    final = stack.wait_state(allowed.json()["deployment_id"], {"COMPLETED", "ROLLED_BACK", "FAILED"})
    assert final["state"] in ("COMPLETED", "ROLLED_BACK")  # the overridden rollout still runs under health checks
