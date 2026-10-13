# Troubleshooting

| Symptom | Likely cause / check |
|---|---|
| `make up` fails: `run make init` | `.env` or `deploy/secrets/` missing - run `make init` |
| Docker daemon unreachable / VM crash | Docker Desktop not running, or host disk < ~1 GB free (`df -h /`) |
| Port already in use | Another project uses it; override `FLEETOPS_CP_PORT`, `FLEETOPS_MLFLOW_PORT`, `FLEETOPS_MQTT_PORT`, `EDGE*_PORT`, ... |
| `POST /models/train` -> 502 `registry unavailable` | MLflow down/slow. `docker compose logs mlflow`. A "Invalid Host header" 403 means the hostname is missing from `--allowed-hosts` in the Dockerfile |
| Device stays `REGISTERED`, never `ONLINE` | Agent cannot authenticate to Mosquitto: re-run `make init` and recreate containers; check `docker compose logs mosquitto edge-01`. Heartbeats from unregistered ids are dropped (`reason="unregistered"`) |
| Deployment: "no eligible devices" | Response lists the reason per device: offline, revoked, incompatible agent/format, already on version |
| Target `FAILED` with `DOWNLOAD_FAILED` | Control plane unreachable from the device for longer than the bounded retries; `DOWNLOAD_DENIED` = device not assigned that version or credential revoked |
| Target `FAILED` with `CHECKSUM_MISMATCH` | Artifact corrupted in transit/at rest (or `corrupt_download_versions` fault is set) - the model was never activated |
| Rollout rolled back unexpectedly | `GET /audit/events?deployment_id=...`: `target.health_failed` lists the criterion that failed; tune `health_criteria` if it was too strict (e.g. CPU on shared hosts) |
| Deployment stuck `ROLLING_BACK` | A device cannot be reached; the rollback command is resent until acknowledged. See runbooks/device-offline.md |
| Deployment `FAILED` | A rollback failed on a device (`rollback failed:` in the target error). Manual intervention: see runbooks/rollback.md |
| Commands ignored by an agent | `STALE`/`EXPIRED`/`BAD_SIGNATURE` acks in audit (`device.ack`): wrong master key after re-provisioning, or clocks badly skewed |
| `POST /deployments` -> 422 `quality_gate_failed` | The candidate did not meet the requested policy; `error.details.failures` names the metric, values and limit. `INCOMPARABLE`/`MISSING_*` usually mean the model predates the benchmark: retrain it. See runbooks/quality-gate.md |
| `POST /deployments` -> 403 on `quality_override` | Only the admin role may override, with a reason of >= 10 characters |
| `models/train` -> 502 and MLflow log "401" | Control plane cannot authenticate to MLflow: check `fleetops_mlflow_password` secret and that `mlflow-init` completed (`docker compose logs mlflow-init`) |
| Control plane log `mqtt connect refused: Not authorized` | Missing/incorrect `FLEETOPS_MQTT_USERNAME` or MQTT password secret (also happened in Kubernetes when the username env var was missing) |
| Kubernetes pod `StartError ... mounting kube-api-access ... read-only file system` | A secret mounted at `/run/secrets` clashes with the service-account token; keep `automountServiceAccountToken: false` |
| Agent rejects commands with `CLOCK_SKEW`/`EXPIRED` | Device clock wrong. Logs carry `skew_seconds`/`overdue_seconds`; the control plane exposes `fleetops_device_clock_skew_seconds` and warns above `FLEETOPS_CLOCK_SKEW_WARN_SECONDS`. Fix NTP |
| Trivy fails the build | `scripts/trivy_scan.sh` output names the package and fixed version. Update the base image digest / lock (docs/supply-chain.md); only add `.trivyignore` entries with a justification and expiry |
| MQTT tests skipped | Set `FLEETOPS_TEST_MQTT_HOST` (use `make test-mqtt`) |
| `SQLite database is locked` | Only one control-plane replica may use the SQLite file |

Logs are JSON with `deployment_id`, `device_id`, `command_id`, `correlation_id`:
`docker compose logs control-plane | jq -c 'select(.deployment_id=="...")'` (lines are JSON; filter non-JSON noise with `-R 'fromjson?'`).
