# Runbook: MQTT broker outage

Signals: `fleetops_mqtt_connected == 0` (control plane) / `fleetops_edge_mqtt_connected == 0` (agents), `EdgeDeviceOffline` on many devices at once, `increase(fleetops_mqtt_reconnects_total[5m])`, `MqttMessageFailures`.
Impact: no new rollouts (the API refuses them with a clear error), no telemetry, devices keep inferring locally. Measured recovery after a broker restart in the load test: all devices back ONLINE with fresh heartbeats in about 1.6 s (50 devices; `docs/load-test-results.md`).
1. `docker compose ps mosquitto` / `docker compose logs mosquitto`: crashed, out of disk, wrong ACL/password file, expired certificate (TLS profile; see `tls-certificate-failure.md`).
2. Restart: `docker compose restart mosquitto`. A clean stop publishes last-will messages, so devices show OFFLINE until they reconnect; that is expected.
3. Both sides reconnect automatically with bounded backoff and jitter; persistent sessions keep queued commands. Do not restart the agents.
4. Verify: `/readyz` on the control plane, devices ONLINE, `fleetops_mqtt_connected == 1`. In-flight deployments are re-driven by the engine timer with the same `command_id`s (no double execution).
5. If a deployment failed during the outage it rolled back or will be retried per `failed-deployment.md`; create a new one.
