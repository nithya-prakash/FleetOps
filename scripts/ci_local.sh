#!/usr/bin/env bash
# Emulates the `static` and `tests` jobs of .github/workflows/ci.yml inside a CLEAN COPY of the working tree
# (only files git would track: no .venv, no .env, no secrets), so missing files and environment assumptions show up.
#
#   scripts/ci_local.sh            # copies to a temporary directory and runs the checks there
#
# This is an emulation on the developer's machine; it does not replace a real GitHub Actions run (docs/ci.md).
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
CLEAN="$(mktemp -d)"
trap 'rm -rf "$CLEAN"; docker rm -f mosq-ci >/dev/null 2>&1 || true' EXIT
(cd "$SRC" && git ls-files -co --exclude-standard -z | rsync -a --from0 --files-from=- . "$CLEAN/")
git -C "$CLEAN" init -q   # some tests ask git whether secrets are ignored
set -- "$CLEAN"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
cd "$1"
UV_VERSION=0.12.5
export UV_PYTHON=3.12   # what actions/setup-python provides in CI
uv lock --check
uv sync --frozen --all-extras 2>&1 | tail -2
.venv/bin/ruff format --check . | tail -1
.venv/bin/ruff check . | tail -1
.venv/bin/mypy | tail -1
docker run --rm --entrypoint promtool -v "$PWD/monitoring/prometheus:/p:ro" prom/prometheus:v2.55.1@sha256:2659f4c2ebb718e7695cb9b25ffa7d6be64db013daba13e05c875451cf51b0d3 check rules /p/alerts.yml | tail -1
cp .env.example .env
.venv/bin/python scripts/provision.py | tail -1
.venv/bin/python scripts/devcerts.py | tail -1
docker compose config -q && echo compose-plain-ok
docker compose -f compose.yaml -f compose.tls.yaml config -q && echo compose-tls-ok
docker build -q --target mosquitto -t fleetops-mosquitto:dev . >/dev/null
docker rm -f mosq-ci >/dev/null 2>&1 || true
docker run -d --name mosq-ci -p 127.0.0.1:18830:1883 --read-only --cap-drop ALL --tmpfs /mosquitto/run:uid=1883,gid=1883,mode=0700 --tmpfs /tmp -v "$PWD/deploy/secrets/mosquitto_passwd:/run/secrets/mosquitto_passwd:ro" fleetops-mosquitto:dev >/dev/null
for _ in $(seq 1 30); do (exec 3<>/dev/tcp/127.0.0.1/18830) 2>/dev/null && break; sleep 1; done
FLEETOPS_TEST_MQTT_HOST=127.0.0.1 FLEETOPS_TEST_MQTT_PORT=18830 MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/pytest -q --cov --cov-report=term --cov-report=xml --cov-fail-under=90 2>&1 | grep -E "passed|failed|^TOTAL|Required|FAIL" 
docker rm -f mosq-ci >/dev/null
ls coverage.xml
