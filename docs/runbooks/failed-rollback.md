# Runbook: failed rollback

Signals: deployment state `FAILED` after a rollback attempt; `increase(fleetops_rollback_failures_total[10m]) > 0` (add an alert if you page on it).
Risk: some devices may still run the bad model. This is the only state needing prompt human action.
1. `GET /api/v1/deployments/{id}`: list targets whose `last_error` mentions `ROLLBACK_UNAVAILABLE` / `ROLLBACK_FAILED`; note the device ids.
2. On each device: `curl <agent>:9100/healthz` -> `active`, `previous`. If `previous` is missing there is no known-good model locally.
3. Options, least to most invasive: (a) deploy the known-good version as a **new** deployment to just those devices (`device_ids`); (b) restore the device's model store and restart the agent; (c) if the device must not serve, leave it deactivated (inference returns 503 rather than running a failing model) or `POST /devices/{id}/revoke`.
4. Record the incident: the failed rollback is audited; keep the deployment id and agent logs. Root-cause before the next rollout (disk, corrupted store, version pruned).
