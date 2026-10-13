# Runbook: failed deployment

Signals: `DeploymentRolledBack` (critical), deployment state `ROLLED_BACK` / `FAILED`, `fleetops_deployment_targets_total{state="FAILED"}`.
Devices keep serving their last known-good model throughout; a failed update never removes it.
1. `GET /api/v1/deployments/{id}`: `reason` plus per-target `state` and `last_error`. Typical causes: `no STAGED acknowledgement` (device unreachable or slow), artifact hash/size mismatch, insufficient disk, health criteria missed (`inference error rate`, latency), quality-gate rejection (see `quality-gate.md`).
2. Confirm devices: `GET /api/v1/devices` -> `active_model_version` is the previous version everywhere; if it is mixed, deployment state would be `FAILED`, not `ROLLED_BACK` - go to `failed-rollback.md`.
3. Check the audit log for `deployment.*` and `artifact.denied` entries around the time.
4. Fix the cause, then create a **new** deployment (final states are not retried). Use a smaller canary first if the cause was model behaviour.
