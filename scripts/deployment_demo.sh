#!/usr/bin/env bash
# Cold start to full lifecycle, for the README recording. Needs Docker, built images and the dev .env (make init).
set -u
banner() { echo ""; echo "── $1 ──"; sleep 1; }
banner "1. Clean start: remove old volumes, start Mosquitto, MLflow, control plane, 3 edge agents, Prometheus, Grafana"
make reset >/dev/null 2>&1
docker compose up -d --wait 2>&1 | grep -E "Container .*(Healthy|Started)" | sed 's/ Container //' | sort -u | head -12
docker compose ps --format "table {{.Service}}\t{{.Status}}"
sleep 2
banner "2. Full lifecycle on real hydraulic-rig data: train, canary, quality gate, rollback, offline, metrics"
.venv/bin/python scripts/demo.py all
