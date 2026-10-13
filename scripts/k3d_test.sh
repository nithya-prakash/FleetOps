#!/usr/bin/env bash
# Disposable Kubernetes check: create a k3d cluster, load the locally built images, create secrets from
# deploy/secrets, apply deploy/k3s/fleetops.yaml, wait for readiness, run the smoke test, delete the cluster.
# Prereqs: docker, k3d, kubectl, built images (`docker compose build`), `python scripts/provision.py`.
set -euo pipefail
CLUSTER="${CLUSTER:-fleetops-test}"
PY="${PY:-.venv/bin/python}"
trap 'k3d cluster delete "$CLUSTER" >/dev/null 2>&1 || true' EXIT

k3d cluster create "$CLUSTER" --servers 1 --agents 0 --no-lb --wait --timeout 240s --k3s-arg "--disable=traefik@server:0"
k3d image import -c "$CLUSTER" fleetops-edge:dev fleetops-control-plane:dev fleetops-mlflow:dev fleetops-mosquitto:dev
./scripts/k8s_secrets.sh fleetops
kubectl apply -f deploy/k3s/fleetops.yaml
for d in mosquitto mlflow; do kubectl -n fleetops rollout status "deploy/$d" --timeout=240s; done
kubectl -n fleetops wait --for=condition=complete job/mlflow-init --timeout=240s
kubectl -n fleetops rollout status deploy/control-plane --timeout=300s
kubectl -n fleetops rollout status sts/edge-01 --timeout=300s
"$PY" scripts/k8s_smoke.py
