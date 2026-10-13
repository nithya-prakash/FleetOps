# Deployment guide

## Local (Docker Compose): the supported demo
```bash
make init && make up && make bootstrap && make demo && make verify-security
```
`make init` installs from `uv.lock`, creates `.env` from `.env.example` (development-only values) and runs `scripts/provision.py`,
which writes every credential as a file under `deploy/secrets/` (gitignored, directory `0700`). It is idempotent and removes stale files;
re-run it after changing any secret in `.env`, then `docker compose up -d --force-recreate`.

Services: `mosquitto` (project image: baked config, password file from a secret copied to a private tmpfs), `mlflow` (basic-auth,
sqlite backend, `--serve-artifacts`, Host-header allow-list), `mlflow-init` (one-shot: creates/rotates the non-admin MLflow account the
control plane uses), `control-plane`, `edge-01..03`, `prometheus`, `grafana`. All are non-root with read-only root filesystems and dropped
capabilities. Persistent volumes: `cp-data` (SQLite + artifact store), `mlflow-data`, `mosquitto-data`, `edge-*-data` (model store: active +
previous), `prometheus-data`, `grafana-data`. Health checks gate the startup order (control plane waits for `mlflow-init`).

**Ports** bind to `${FLEETOPS_BIND:-127.0.0.1}`: control plane 18000, MLflow 15000, MQTT 11883, Prometheus 19090, Grafana 13000, agents
19101-19103. MLflow UI: user `admin`, password `FLEETOPS_MLFLOW_ADMIN_PASSWORD` from `.env`; Grafana: user `admin`, `GRAFANA_ADMIN_PASSWORD`.

Verification helpers (all talk to the running stack): `make demo` (full lifecycle incl. quality gate), `make verify-security`
(loopback binding, MLflow auth/authorization, no credentials in container env, broker file permissions, non-root/read-only),
`python scripts/verify_dashboard.py` (every Grafana panel query executes against live Prometheus), `make scan` (Trivy).

### Adding devices
1. `python scripts/provision.py --devices edge-01 edge-02 edge-03 edge-04` (regenerates the broker password file and per-device secrets).
2. Add an `edge-04` service and secret to `compose.yaml` (copy `edge-03`) and a Prometheus target; recreate `mosquitto`.
3. `POST /api/v1/devices/register` (admin). Heartbeats from unregistered ids are rejected (`reason="unregistered"`).

### Persistence checks
`make down` keeps all state; `make reset` deletes volumes. After `docker compose restart control-plane`, in-flight rollouts resume from the
database and queued acknowledgements are delivered from the broker's persistent session.

## Kubernetes / K3s (optional)
`deploy/k3s/fleetops.yaml`: Mosquitto, MLflow (+ `mlflow-init` Job), control plane (single replica, `Recreate`, PVC) and one edge-agent
StatefulSet. Secrets are created from the provisioned files, never from the manifest:
```bash
python scripts/provision.py && docker compose build        # images fleetops-*:dev
scripts/k3d_test.sh      # disposable k3d cluster: create, import images, secrets, apply, wait, smoke test, delete
```
or manually: `k3d cluster create ...; k3d image import ...; scripts/k8s_secrets.sh; kubectl apply -f deploy/k3s/fleetops.yaml`.
`scripts/k8s_smoke.py` registers edge-01, trains through the authenticated MLflow, deploys with a quality gate and checks inference on the pod.

Verified on k3d v5.9.0 / k3s v1.35.5 (Docker Desktop, Apple Silicon): all pods Ready, `mlflow-init` Completed, smoke test passed. That
run found and fixed two manifest bugs invisible to schema validation (service-account token mount clashing with `/run/secrets`; control
plane missing `FLEETOPS_MQTT_USERNAME`); both now have regression tests. Not verified: multi-node clusters, Linux hosts, a real factory
network, image pulls from a registry (images are imported locally), PodSecurity admission levels.

## Moving to PostgreSQL (future work)
Set `FLEETOPS_DATABASE_URL=postgresql+psycopg://...` (add the driver to the lock/image). Only portable column types are used and
migrations are Alembic (a test asserts migrations equal the ORM schema). Running more than one control-plane replica additionally needs
database advisory locks or leader election in place of the in-process per-deployment locks. This is deliberately **not** implemented: a
multi-replica architecture would need to be built and tested as a whole.

## TLS profile
`make tls-up` generates a development CA and short-lived certificates (`scripts/devcerts.py`, git-ignored under `deploy/secrets/tls/`) and starts the stack from
`compose.yaml` + `compose.tls.yaml`: MQTT on 8883, HTTPS for the control plane, MLflow and the agents, Prometheus scraping with CA verification.
`make verify-tls` checks it; helper scripts use `FLEETOPS_TLS=1`. Switch back with `docker compose up -d` (plain profile). For production use certificates from your own CA
(see `docs/security.md`) and `FLEETOPS_ENV=prod`, which refuses to start without TLS.

## Load and benchmark tooling
`make load-test` / `make bench` (see `docs/load-test-results.md` and `docs/edge-inference-benchmarks.md`); both write git-ignored output (`load_results/`, `benchmark_results/`).
