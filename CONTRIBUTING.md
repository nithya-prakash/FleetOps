# Contributing

Setup: `make init` (uv, Python 3.12, installs from `uv.lock`). Quality gate before every change:
`make lint && make test` (plus `make test-mqtt` when touching messaging). Live checks: `make up && make demo`
and `python scripts/verify_security.py`.

## Engineering rules
- Python 3.12, venv in `.venv` (`make init`). Quality gate: `ruff format --check . && ruff check . && mypy` (strict) and `pytest`.
- `domain/`, `security/`, `messaging/topics.py` are pure (no I/O). Rollout logic lives in `deployment/engine.py`; FastAPI/paho are adapters.
- The edge package (`fleetops.edge`, `registry/modelformat.py`, `training/sensors.py`) must NOT import torch, pandas, sqlalchemy or mlflow (edge image is slim).
- Model artifacts are safetensors only; never add pickle/torch.load. Always verify sha256 before staging and before every load.
- Device commands: signed, deduplicated by command_id, fenced by per-device `generation`. Delivery is never proof of success; only device acks or device-reported state.
- No secrets in git. Dev-only values live in `.env.example`; `deploy/secrets/` and `.env` are gitignored.
- Verify in Docker (`make up`, `make demo`), not just unit tests. Do not claim a command passed unless it was run.
- Host ports are remapped (CP 18000, MLflow 15000, Prom 19090, Grafana 13000, MQTT 11883, agents 19101-3) to avoid clashes with other projects.

- Dependencies are locked: change `pyproject.toml`, run `uv lock`, commit `uv.lock`; container builds install from the lock with
  hash verification. Base images are pinned by digest (see docs/supply-chain.md).
- Quality gate for model rollouts is enforced by the control plane (see docs/api.md); never bypass it in code, use the audited override.
