# Demo: repeatable walkthrough on the real stack

Everything below runs real components (MLflow, Mosquitto, the control plane, three edge agents, Prometheus, Grafana). Nothing is mocked or pre-recorded.
Total time on the development machine: about 2.5 minutes for the scripted part plus build time.

## Run it
```bash
make init                       # once: venv, .env (dev-only secrets), broker/device credentials
make reset && make up           # clean stack (deletes volumes)
python scripts/demo.py all      # or step by step, see below
```

| Step | Command | What it proves (visible output) |
|---|---|---|
| 1 | `python scripts/demo.py bootstrap` | trains and registers v1/v2 (hash, schema, metrics in MLflow), registers 3 devices, waits until ONLINE |
| 2 | `python scripts/demo.py deploy-v1` | first rollout: canary, then the rest; devices run v1 and answer `/predict` locally; telemetry flows |
| 3 | `python scripts/demo.py failure` | v2 with an injected runtime fault: canary health fails, **automatic rollback**, untouched devices stay on v1; the audit trail is printed |
| 4 | `python scripts/demo.py gate` | quality gate blocks regressing candidates (HTTP 422 with reasons), an operator cannot override (403), admin override is audited |
| 5 | `python scripts/demo.py promote-v2` | healthy v2 under an explicit quality policy: canary, then batches, all ACTIVE |
| 6 | `python scripts/demo.py offline` | edge-02 cut off: it keeps inferring locally, the control plane shows OFFLINE, then it **reconciles** on reconnect |
| 7 | `python scripts/demo.py observe` | queries the real Prometheus: deployments by state, rollbacks, per-device outcomes, connected devices, model version per device, MQTT message counts |

Then open Grafana at http://127.0.0.1:13000 (admin password in `deploy/secrets/grafana_admin_password`) to see the same facts on the dashboard, and
http://127.0.0.1:19090/alerts for the alert rules.

Optional extras (separate, each takes a minute):
* Drift: `EDGE3_SCENARIO=drift docker compose up -d edge-03`, wait a few minutes, `GET /api/v1/drift/...` (see README "Monitoring and drift"); the alert is informational and triggers no deployment.
* Security checks against the running stack: `python scripts/verify_security.py` (52 checks); TLS: `make tls-up && make verify-tls`.
* Load: see `docs/load-test-results.md`.

## Recorded output (2026-10-13, clean stack, default rate limits)
The scripted run ended with: `deployments by final state: COMPLETED=2, ROLLED_BACK=1`, `rollbacks by trigger: automatic=1`, `per-device outcomes: verified=6, failed=1, rolled_back=1`,
all three devices on `anomaly-detector:2`, and 7 artifact downloads, each verified by hash.

## Recording checklist
- [ ] Fresh `make reset && make up`, wait for healthy; close other CPU-heavy apps.
- [ ] Terminal font large, 100+ columns; hide secrets: never show `.env`, `deploy/secrets/*` or the Grafana password on screen.
- [ ] Open Grafana first (dashboard "FleetOps"), time range "last 15 minutes", auto refresh 5 s.
- [ ] Say up front: development profile, synthetic data, single replica, not production.
- [ ] Run the steps above in order; pause on the audit trail (step 3) and the 422 reasons (step 4).
- [ ] During step 6 show `curl` against edge-02 `/predict` answering while the control plane shows it OFFLINE.
- [ ] End with `observe` and the Grafana panels; mention what is not claimed (see README "Limitations").
- [ ] Review the recording for leaked tokens before sharing it.
