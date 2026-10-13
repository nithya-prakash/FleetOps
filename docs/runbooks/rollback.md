# Runbook: rollback

**Automatic** (alert `DeploymentRolledBack`): the engine halted a rollout because a device failed to stage/activate, timed out, or
missed the health criteria. Devices were switched back to their last-known-good model.
1. `GET /api/v1/deployments/{id}`: `reason` names the device and criterion; targets show `ROLLED_BACK` (reverted), `CANCELLED` (never touched), `FAILED` (failed before activation, nothing to revert).
2. Verify on the device: `curl localhost:<agent port>/healthz` -> `active` is the old version; `POST /predict` works.
3. Fix the cause (model, criteria, device) and create a new deployment; a rolled-back deployment is final.

**Manual**: `POST /api/v1/deployments/{id}/rollback {"reason": "..."}` (operator). Works while `IN_PROGRESS` or after `COMPLETED`.
Each device that may run the new model is told to return to its previous model; this is retried until acknowledged.

**Deployment `FAILED` (rollback failed)**: a device rejected `ROLLBACK` (`ROLLBACK_UNAVAILABLE`, `ROLLBACK_FAILED`). Check the target's
`last_error` and agent logs. Options: re-register/restore the device's model store, deploy the known-good version as a new deployment,
or revoke the device. A device with no known-good model is deactivated (inference returns 503) rather than left on a failing model.

**Cancel vs rollback**: `cancel` only stops further promotion and aborts devices that have not activated; it is refused (409) while an
activation is in flight, and devices already ACTIVE keep the new model - use rollback to revert them.

## Policy: rolling back when there is no known-good model

A rollback needs somewhere to go. The engine records each device's active model when its deployment starts (`previous_model`) and the
agent keeps that model on disk. If a device had **no model** when the failing deployment began (a first-ever rollout), the rollback
command's destination is "none" and the agent **deactivates inference**: `POST /predict` returns `503 no active model`, `/healthz` reports
`no_model`, the control plane assessment is `no_model`, and no predictions are produced until a good model is deployed. Covered by
`tests/unit/test_engine.py::test_rollback_without_known_good_model_deactivates` and the matching agent test.

**Why this is the default (the trade-off).** The alternative is to leave the failing model running. That keeps an endpoint "up" but serves
a model that has just failed its health criteria (errors, latency, or resource problems), so downstream logic may act on wrong or missing
answers without knowing it. An explicit, observable outage is preferred over silent degradation, because the consumer can detect a 503 and
fall back to its own rules.

**Operational consequences.**
* A line can lose anomaly detection until a good model is deployed. Plan for it: consumers of `/predict` must handle 503.
* Seed every device with a vetted baseline model **before** rolling out candidates, so there is always a known-good destination.
* Use `quality_gate.require_baseline` and a `min` threshold for first deployments so an obviously bad model is blocked before reaching a
  device (see quality-gate.md).
* Recovery: create a new deployment of a known-good version (devices in `no_model` state are eligible). The old deployment stays `ROLLED_BACK`.
* Alert on `fleetops_device_model_version == 0` (model version 0 = none) and on the `no_model` assessment.

The policy was **not** changed; making it configurable per deployment (`deactivate` | `keep`) is listed as future work because "keep" needs
an explicit, audited acceptance of the risk above.
