# Observability review

## What was checked (2026-10-13)
* **Dashboard queries run against real data**: `python scripts/verify_dashboard.py` executes every panel query of `monitoring/grafana/dashboards/fleetops.json`
  against the live Prometheus of the running stack; all returned series (no panel queries a metric that does not exist).
* **Alert rules**: `promtool check rules` reports 16 rules valid, and Prometheus loaded all 16 with health `ok` (`/api/v1/rules`).
* **Label cardinality**: the control plane's counters and histograms use small closed label sets only (`state`, `outcome`, `trigger`, `channel`, `reason`,
  `command`, `type`, `result`, `status`, `scope`, `method` + HTTP status). **No request path, user id, model version or free text is a label**, so a
  scanner cannot create series. The only unbounded-by-design labels are `device_id` / `group` (and `model`) on the per-device gauges
  (`fleetops_device_*`), which are exported by a collector that reads the device table at scrape time. Series count = devices x ~10 gauges; fine for
  hundreds of devices (100 simulated devices were scraped throughout the load tests), a problem for tens of thousands, and every scrape does a DB read proportional to the fleet size.
* **Added in this round** (they were missing): control-plane scrape down, control-plane/broker disconnect, rollback failure, artifact integrity failure,
  denied-request spike and rate-limiting alerts, plus the matching runbooks.

## Alerts, severity and response
| Alert | Severity | Meaning | Response |
|---|---|---|---|
| ControlPlaneDown | critical | Prometheus cannot scrape the control plane for 1 min | `docs/troubleshooting.md`; devices keep serving |
| ControlPlaneMqttDisconnected | critical | control plane lost the broker for 1 min | `runbooks/mqtt-outage.md` |
| DeploymentRolledBack | critical | an automatic or manual rollback occurred | `runbooks/rollback.md`, `failed-deployment.md` |
| RollbackFailed | critical | a device rejected/failed a rollback or abort | `runbooks/failed-rollback.md` (act promptly) |
| ArtifactIntegrityFailure | critical | hash/size verification failed; artifact not served | `runbooks/artifact-corruption.md` |
| EdgeInferenceFailures | critical | a device's inference failure rate > 0.1/s for 30 s | `runbooks/rollback.md` |
| EdgeDeviceOffline / HeartbeatStale | warning | device not connected / stale heartbeat | `runbooks/device-offline.md` |
| EdgeDeviceWithoutModel | warning | device online without an active model (inference 503) | deploy a model |
| DeviceClockSkew | warning | device clock off by > 30 s (commands will be rejected) | fix NTP |
| EdgeDiskLow | warning | < 256 MB free on a device | free space; updates are refused below the headroom |
| EdgeInferenceLatencyHigh | warning | p95 > 50 ms for 1 min | check CPU, load |
| MqttMessageFailures | warning | > 5 rejected MQTT messages in 5 min | check device versions/schema; audit log |
| AuthDenialsHigh | warning | > 20 denied management requests in 5 min | check audit log, `docs/security.md` |
| RateLimitingActive | warning | > 50 HTTP 429 in 5 min | misbehaving client or limit too low |
| InputDriftDetected | info | PSI/KS drift report with status ALERT | `runbooks/drift-review.md`; no automatic action |

Thresholds are starting points chosen from the measured behaviour (e.g. inference p95 ~0.16 ms server-side, heartbeat ingest lag p99 26 ms at 100 devices), not tuned on production data.
Alerts go nowhere by default: no Alertmanager or notification channel is configured in this repository (a deliberate omission; wiring it needs an
on-call target the project does not have).

## Known gaps
* No Alertmanager, no paging, no SLOs; no log aggregation (JSON logs go to stdout only).
* `/metrics` of the control plane is open on the internal network unless `FLEETOPS_METRICS_TOKEN` is set (set it outside the dev profile).
* Per-device series scale linearly with the fleet (see above); a larger fleet would move device state into a pull API or aggregate before export.
* Load-test devices registered in the dev database persist until `make reset`; they show as offline and fire `EdgeDeviceOffline` meanwhile.
