# FleetOps

Deploy, monitor and roll back ML models on a fleet of factory edge devices: canary rollouts with automatic rollback, a model quality gate and offline inference. Python, FastAPI, MQTT, MLflow, Prometheus/Grafana.

[![CI](https://github.com/nithya-prakash/FleetOps/actions/workflows/ci.yml/badge.svg)](https://github.com/nithya-prakash/FleetOps/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

## Demo
![make demo](docs/demo.gif)

The GIF is a replay of a real `make demo` run on the full Docker stack (139 s of wall time; waits longer than 1.5 s are shortened). It shows model registration, a canary rollout, an injected fault causing **automatic rollback**, the quality gate rejecting a worse model, a device losing its network and **still inferring locally**, and Prometheus' record of it.
It does not show the Grafana dashboard or real hardware (simulated sensors only). Full text of a run: [docs/demo-output.txt](docs/demo-output.txt); how to record your own: [docs/demo.md](docs/demo.md).

## Results
All data is synthetic. Numbers come from one Apple M4 laptop (Docker Desktop), not from a production environment.

| Metric | Result | Data / workload | Reproduce |
|---|---|---|---|
| Automated tests | 363 pass, 94% line coverage | 271 unit, 66 integration, 26 end-to-end, real Mosquitto | `make test-mqtt` (coverage: `pytest --cov`) |
| Live security checks | 52 pass | running Compose stack | `make verify-security` |
| Anomaly model v1 | precision 0.66, recall 0.68, F1 0.67, ROC-AUC 0.95 | 4,000 synthetic samples, 6% anomalies (`sensor-benchmark-v1`) | `python scripts/demo.py gate` |
| Edge `/predict` | 1,202 req/s per agent, p99 30 ms (16 clients) | 15 s per level, 256 fixed payloads | `python -m scripts.load_test.run predict` |
| Rollout to 100 devices | 12-13 s (4/4 completed; 3/3 timed out before a fix) | 100 simulated agents over real MQTT | `python -m scripts.load_test.run warmup --devices 100` |
| Soak | PASS: 50 min (42 min steady state), no memory growth | 50 simulated devices, rollout every 5 min | `python -m scripts.load_test.run soak --devices 50 --duration 3000` |
| Recovery | 5.2 s after a control-plane restart, 1.6 s after a broker restart | 50 simulated devices | `python -m scripts.load_test.run restart --devices 50` |
| Failure injection | 12 scenarios pass | broker restart, crash mid-download, corrupt artifact, ... | `pytest tests/end_to_end` |

The load-test commands need simulated-device credentials and raised rate limits first: see [docs/load-test-results.md](docs/load-test-results.md).

## Honest reading
The model is a small stand-in (precision/recall around 0.66 on a synthetic benchmark of 4,000 samples); the point of the project is the rollout, rollback and monitoring machinery, not the detector.
Performance was measured on one laptop with simulated devices; nothing ran on physical edge hardware, a native Linux host or x86_64. CI ran on GitHub for the first time on 2026-10-13: the first run had two failing jobs
(real defects, since fixed); the badge above shows the current state and [docs/ci.md](docs/ci.md) the history. The 50-minute soak is not evidence of multi-day stability.

## Quickstart
```bash
make init   # venv from uv.lock, dev-only .env, credentials (needs Docker + uv)
make up     # Mosquitto, MLflow, control plane, 3 edge agents, Prometheus, Grafana
make demo   # the full lifecycle; API docs at http://localhost:18000/docs, Grafana at :13000
```
| Variable | Purpose |
|---|---|
| `FLEETOPS_ADMIN_API_KEY` / `_OPERATOR_` / `_VIEWER_` | Bearer keys for the three API roles (dev values in `.env.example`) |
| `FLEETOPS_DEVICE_KEY_MASTER` | derives each device's credential |
| `FLEETOPS_BIND` | interface for published ports (default `127.0.0.1`) |
| `FLEETOPS_ENV` | `prod` refuses to start without TLS and strong keys |

## Architecture
```mermaid
flowchart LR
  OP[Operator] -->|REST, API key| CP[Control plane<br/>FastAPI + rollout engine]
  CP --- DB[(SQLite)]
  CP --- ML[MLflow registry]
  CP -->|signed commands| MQ[Mosquitto<br/>auth + ACL + TLS]
  MQ <--> E[Edge agents<br/>local inference]
  E -->|hash-verified download| CP
  P[Prometheus + Grafana] -->|scrape| CP & E
```
Python 3.12, FastAPI, SQLAlchemy/Alembic (SQLite), MLflow, Mosquitto/paho-mqtt, PyTorch (training only), NumPy + safetensors (edge runtime), Docker Compose, k3s manifests, GitHub Actions.

## Usage
```bash
# after `make demo` (it registers edge-01..03 and the models)
KEY=$(grep FLEETOPS_OPERATOR_API_KEY .env | cut -d= -f2)
curl -s -X POST localhost:18000/api/v1/models/train -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model_name":"anomaly-detector","params":{"epochs":10}}'            # train + register a version
curl -s -X POST localhost:18000/api/v1/deployments -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model_name":"anomaly-detector","model_version":1,"group":"line-a","strategy":{"canary_count":1,"batch_percent":50}}'
```
Endpoints: `GET /api/v1/devices`, `GET /api/v1/deployments/{id}`, `POST /api/v1/deployments/{id}/rollback`, edge `POST :19101/predict`. Full reference: [docs/api.md](docs/api.md).

## Limitations
- Single control-plane replica on SQLite: **not highly available**; no PostgreSQL or leader election (design in [docs/capacity-and-scaling.md](docs/capacity-and-scaling.md)).
- Development CA only; no production PKI, no Alertmanager or paging, no external security review ([docs/security-assessment.md](docs/security-assessment.md)).
- Not run on physical devices, native Linux or x86_64 at runtime (amd64 image built and smoke-tested under emulation); the Raspberry Pi guide is unverified.
- Synthetic data only; one MLP model format; the quality gate uses a synthetic benchmark.
- Backup script exists but no restore drill has been done. Not claimed production-ready.

## Repository
`src/fleetops/` code - `tests/` - `scripts/` demos, load tests, benchmarks - `deploy/` + `compose*.yaml` - `monitoring/` - `docs/` ([details.md](docs/details.md) has the full reference, runbooks and engineering notes).
Checks: `make lint && make test-mqtt`. License: [MIT](LICENSE).
