# Runbook: device offline

Alerts: `EdgeDeviceOffline`, `HeartbeatStale`. **An offline device is still inferring locally with its last known-good model.**
1. `GET /api/v1/devices/{id}/health`: `connectivity`, `heartbeat_age_seconds`, `assessment` (`offline`).
2. Determine scope: one device (power/network at the cell) or all (broker/control plane: check `/readyz`, Mosquitto logs).
3. Do not assume the device is broken. On site: `curl http://<agent>:9100/healthz` shows `mqtt_connected`, active/previous model, buffered state.
4. If a deployment is in flight, the device is not promoted; the engine fails the target after `ack_timeout`/attempt limits and rolls back. A pending **rollback** is resent until the device returns.
5. On reconnect the agent announces `online`; the control plane re-sends outstanding commands (same ids, deduplicated), heals lost acks from the heartbeat, and the device's reported model becomes the source of truth. Buffered predictions are flushed.
6. Persistent loss > expected: replace/repair; if compromised, `POST /devices/{id}/revoke`.

## Offline for a long time (days)
* The device serves its last known-good model; nothing is lost on the device. Its entry stays `OFFLINE`; telemetry buffering on the device is bounded, so long outages lose old readings.
* New deployments **skip** offline devices (listed under `skipped` with the reason), so the rest of the fleet is not blocked; the device will not be on the new version until it returns and a deployment targets it again.
* On return it receives current commands only (expired/superseded commands are rejected, never replayed), then reconciles from its reported state. Check its clock (`DeviceClockSkew`): expiry checks reject commands on a skewed clock.
* Decide after a policy period (e.g. 14 days): repair/replace, or `POST /devices/{id}/revoke` so a lost or stolen device cannot rejoin with its old credential.
