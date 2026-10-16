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
- [ ] Say up front: development profile, real data from one lab rig but simulated devices, single replica, not production.
- [ ] Run the steps above in order; pause on the audit trail (step 3) and the 422 reasons (step 4).
- [ ] During step 6 show `curl` against edge-02 `/predict` answering while the control plane shows it OFFLINE.
- [ ] End with `observe` and the Grafana panels; mention what is not claimed (see README "Limitations").
- [ ] Review the recording for leaked tokens before sharing it.

## How `docs/demo.gif` was made
Recorded from a clean stack (`make reset && make up`), then rendered; nothing is edited by hand:
```bash
asciinema rec --headless --window-size 110x34 --idle-time-limit 1.5 --command "make demo" --output-format asciicast-v2 demo.cast
agg --theme monokai --font-size 14 --idle-time-limit 1.5 --fps-cap 12 --last-frame-duration 5 demo.cast docs/demo.gif
```
The recording had 65 output events over 139 s and no errors; the GIF is about 1.2 MB and 28 s because idle waits are capped at 1.5 s.

## How `docs/demo-dashboard.gif` was made
Playwright drives the installed Google Chrome against the Grafana dashboard (kiosk mode, 5 s refresh) on a fresh stack while `scripts/demo.py all` runs; a screenshot every 2 s (100 frames over 170 s), a Pillow overlay with the elapsed real time, then ffmpeg assembles the GIF at 5 frames/s (about 9x). Playwright and Pillow are installed in a throw-away virtual environment; they are not project dependencies.
```bash
python -m venv /tmp/pw && /tmp/pw/bin/pip install playwright pillow
make reset && make up
/tmp/pw/bin/python scripts/demo_media/record_dashboard.py --out /tmp/frames --interval 2 --tail 25
# overlay a "real time Ns" label with Pillow, then:
ffmpeg -framerate 5 -i /tmp/frames_labelled/frame_%04d.png -vf "scale=1000:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" docs/demo-dashboard.gif
```
While recording it, the dashboard showed a real flaw that was then fixed: counters that first appear at 1 made `increase()` report 0 for the first rollback, and counts showed as fractions (1.03). All event counters are now created at 0 and the stat panels round.
