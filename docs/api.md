# API reference

Base path `/api/v1`. Interactive docs: `GET /docs` (OpenAPI at `/openapi.json`). Auth: `Authorization: Bearer <key>`.
Roles: **viewer** < **operator** < **admin** (a higher role may do everything a lower one can).

Errors always look like `{"error": {"code": "...", "message": "...", "details": ...}}`:
`401 unauthorized`, `403 forbidden`, `404 not_found`, `409 conflict`, `413 payload_too_large`,
`422 validation_error | invalid_request | quality_gate_failed`, `429`, `502` (model registry unavailable), `500 internal_error`.
Unknown JSON fields are rejected. Lists return `{items, total, limit, offset}` (`limit` 1-200).

| Method & path | Role | Notes |
|---|---|---|
| `POST /devices/register` | admin | `{device_id, group, description?, labels?}` -> 201 with the one-time device credential. 409 if it exists |
| `GET /devices` | viewer | filters: `group`, `status`; pagination |
| `GET /devices/{id}` | viewer | inventory record (never includes secrets) |
| `GET /devices/{id}/health` | viewer | connectivity, heartbeat age, active model, last stats, in-flight deployments, assessment (`healthy/degraded/offline/no_model/unknown`) |
| `POST /devices/{id}/revoke` | admin | status `REVOKED`, key epoch bumped: old credentials stop working immediately |
| `POST /devices/{id}/simulation` | admin | deterministic fault injection (`FaultConfig`); 403 unless `FLEETOPS_SIMULATION_ENABLED=true` and the agent allows faults |
| `POST /models/train` | operator | trains, evaluates, logs to MLflow, registers; 201 `ModelVersion`. Bounded parameters; one job at a time (429) |
| `GET /models` | viewer | name, latest version, version count |
| `GET /models/{name}/versions` | viewer | sha256, size, compatibility, input schema, metrics, params, MLflow run id (no filesystem paths) |
| `POST /deployments` | operator (override: admin) | body below (optional `quality_gate`, `quality_override`); header `Idempotency-Key`: same key + same body -> 200 with the original; same key + different body -> 409 |
| `GET /deployments` | viewer | filters: `state`, `model_name`, `model_version`, `device_id` |
| `GET /deployments/{id}` | viewer | state, reason, batches, per-device targets with state/observation/last_error |
| `POST /deployments/{id}/cancel` | operator | 202. Stops promotion and aborts un-activated devices. 409 if not cancellable or an activation is in flight |
| `POST /deployments/{id}/rollback` | operator | 202. `{reason?}`. Reverts every device that may run the new model to its previous model; retried until acknowledged |
| `GET /monitoring/drift` | viewer | filters `model_name`, `model_version`, `status` |
| `POST /monitoring/drift/run` | operator | evaluates now, returns new reports |
| `GET /monitoring/recommendations` | viewer | review items created by drift ALERTs |
| `POST /monitoring/recommendations/{id}/resolve` | operator | `{action: acknowledge|dismiss}` |
| `GET /audit/events` | viewer | filters `device_id`, `deployment_id`, `action` (`*` wildcard) |
| `GET /device/artifacts/{name}/{version}` | **device** | headers `Authorization: Bearer <device secret>` + `X-Device-ID`; 403 unless the device is a target of an in-flight deployment for that version (or its rollback destination) |
| `GET /healthz`, `/readyz`, `/metrics` | none | liveness, readiness (DB, MQTT, MLflow), Prometheus (keep on an internal network) |

## Create a deployment

```json
{
  "model_name": "anomaly-detector", "model_version": 2, "group": "line-a",
  "device_ids": ["edge-01", "edge-02"],
  "strategy": {"canary_count": 1, "batch_percent": 50},
  "health_criteria": {"ack_timeout_seconds": 60, "observation_seconds": 10, "min_inferences": 20,
                      "max_error_rate": 0.05, "max_p95_latency_ms": 50, "max_cpu_percent": 95,
                      "max_memory_mb": 1024, "health_timeout_seconds": 120, "max_command_attempts": 4}
}
```
Response includes `skipped` (device -> reason: offline, revoked, incompatible, already on that version).
A new deployment of the same model to the same group **supersedes** an in-flight one (the old one becomes
`CANCELLED`, `superseded_by` set). Overlap with a different in-flight deployment is `409`.

States - deployment: `PENDING, IN_PROGRESS, COMPLETED, ROLLING_BACK, ROLLED_BACK, FAILED, CANCELLED`;
target: `PENDING, DISTRIBUTING, STAGED, ACTIVE, FAILED, ROLLED_BACK, CANCELLED`.
`FAILED` on a deployment means a rollback itself failed on some device and needs an operator.

## Quality gate (optional, per deployment; can be made mandatory)

Healthy containers say nothing about model quality, so `POST /deployments` can enforce a policy **before any device is touched**.
Every trained model is evaluated on a fixed benchmark (`hydraulic-benchmark-v1`, independent of the training seed) so models are comparable;
the metrics (`bench_precision`, `bench_recall`, `bench_f1`, `bench_roc_auc`, `bench_false_positive_rate`, `bench_p95_latency_ms`) and the
evaluation metadata (suite, dataset fingerprint, metric-definition version, threshold policy, latency environment) are in
`GET /models/{name}/versions` (`metrics`, `evaluation`). The benchmark is a fixed hold-out of real recordings from one lab hydraulic rig (662 cycles, 33% faulty); it says nothing about a factory line.

```json
"quality_gate": {
  "metrics": {
    "bench_f1":                  {"min": 0.60, "max_regression": 0.02},
    "bench_recall":              {"max_regression": 0.05},
    "bench_p95_latency_ms":      {"max": 5.0, "max_regression": 0.5, "relative": true},
    "bench_false_positive_rate": {"max_regression": 0.0}
  },
  "baseline_version": null,
  "require_baseline": false
}
```
* **Direction**: higher-is-better metrics (precision/recall/F1/AUC) take `min`; lower-is-better (false-positive rate, latency) take `max`.
  A bound in the wrong direction or an unknown metric is a `422 validation_error`.
* **Regression**: `max_regression` is the allowed worsening versus the baseline (absolute, or a fraction of the baseline with
  `relative: true`). Boundaries are inclusive. The baseline is `baseline_version` if given, otherwise **every distinct version the targeted
  devices currently run** (the candidate must not regress against any). With no incumbent, regression checks are skipped and reported
  (`skipped`) unless `require_baseline` is set; absolute `min`/`max` always apply.
* **Fail safely**: a missing/non-finite metric (candidate or baseline), a baseline that is not registered, or models evaluated on different
  benchmarks (any of suite, dataset fingerprint, metric-definition version or threshold policy differs; for latency also the host class) is
  a failure, never a silent pass. Models registered before this feature have no evaluation metadata and are therefore incomparable.
* **Result**: `422 quality_gate_failed`, nothing is created or sent, and `error.details` lists every failure:
  `{"code": "REGRESSION", "metric": "bench_recall", "message": "bench_recall regressed: 0.5566 vs v1 0.6787 (allowed drop 0.0200, limit 0.6587)",
  "candidate": 0.5566, "baseline": 0.6787, "baseline_version": 1, "limit": 0.6587}`. Codes: `MISSING_METRIC`, `INCOMPARABLE`,
  `MISSING_BASELINE`, `BELOW_MINIMUM`, `ABOVE_MAXIMUM`, `REGRESSION`, `GATE_REQUIRED`.
* **Pass**: the deployment proceeds exactly as before (canary, health criteria, automatic rollback) and the report is stored on the
  deployment (`quality`).
* **Override**: `"quality_override": {"reason": "<at least 10 characters>"}`, **admin only** (operators get `403`). The deployment proceeds,
  `quality.overridden` is true with the reason and actor, and a `deployment.quality_gate_overridden` audit event records the reason and the
  failed checks. Runtime health checks still apply.
* **Make it mandatory**: `FLEETOPS_REQUIRE_QUALITY_GATE=true` rejects deployments without a gate (`GATE_REQUIRED`); only the audited admin
  override can bypass it.
* Latency numbers are measured on the training host; compare them only between models benchmarked in the same environment (enforced) and
  use generous tolerances: they are noisy.
