# Runbook: TLS certificate failure

Signals: clients log `certificate verify failed` / `certificate has expired`; MQTT disconnected with no reconnect; `/readyz` unreachable over HTTPS; Prometheus targets down with TLS errors.
Never fix this by disabling verification; fix the certificate or the trust.
1. Identify the failing hop and error: `openssl s_client -connect <host>:<port> -CAfile deploy/secrets/tls/ca.crt -servername <name>` (expired, wrong name, unknown CA).
2. `expired`: issue a new leaf (dev: `python scripts/devcerts.py`; production: your CA), replace the files, restart the service (`docker compose restart <service>`). Mosquitto also reloads on SIGHUP.
3. `wrong name`: the certificate SAN must contain the name the client uses (service name, `localhost`, `127.0.0.1` in the dev profile).
4. `unknown CA`: distribute the correct CA certificate to the client (agents: `FLEETOPS_EDGE_MQTT_TLS_CA`; control plane: `FLEETOPS_MQTT_TLS_CA`). When rotating the CA, ship a bundle with both CAs first.
5. mTLS: a client certificate that is expired or from another CA is refused by the broker; re-issue it.
6. Verify: `make verify-tls`; devices reconnect on their own. Devices keep inferring locally during the outage.
