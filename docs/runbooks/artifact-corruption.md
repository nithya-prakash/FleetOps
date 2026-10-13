# Runbook: artifact corruption

Signals: `increase(fleetops_artifact_integrity_failures_total[10m]) > 0`, target `last_error` with hash/size mismatch, `artifact.denied` audit entries.
The control plane verifies the stored artifact against the registered SHA-256 before serving it, and the agent verifies the download again before staging; a corrupted file is never activated.
1. Identify the version and where it failed: control-plane side (`fleetops_artifact_integrity_failures_total` increments, download 5xx/409) or device side (target `FAILED`, staging directory cleaned).
2. Compare the registered hash (`GET /api/v1/models/{name}/versions`) with `sha256sum` of the stored file in the artifact volume.
3. Corrupt at rest: restore the volume from backup or train and register a new version (`POST /api/v1/models/train`); the corrupted version must not be reused.
4. Corrupt in transit only (device side): check the network path and disk of that device; retry with a new deployment.
5. Never bypass verification or edit the stored hash to make a mismatch pass.
