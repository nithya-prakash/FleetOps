.DEFAULT_GOAL := help
PY := .venv/bin/python
COMPOSE := docker compose

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n",$$1,$$2}'

init:  ## create venv, install deps, create .env (dev-only secrets), provision MQTT/device credentials
	test -d .venv || uv venv --python 3.12 .venv
	uv sync --frozen --all-extras
	test -f .env || cp .env.example .env
	$(PY) scripts/provision.py

up:  ## build and start the whole stack (broker, MLflow, control plane, 3 agents, Prometheus, Grafana)
	$(COMPOSE) up -d --build
	$(COMPOSE) ps

tls-up:  ## start the stack with TLS everywhere (development CA); use FLEETOPS_TLS=1 for the helper scripts
	$(PY) scripts/devcerts.py
	docker compose -f compose.yaml -f compose.tls.yaml up -d --build --wait

verify-tls:  ## assert TLS is enforced on the running TLS profile
	FLEETOPS_TLS=1 $(PY) scripts/verify_tls.py

down:  ## stop the stack (keeps volumes)
	$(COMPOSE) down

reset:  ## stop the stack AND delete all volumes
	$(COMPOSE) down -v

logs:  ## follow structured logs
	$(COMPOSE) logs -f --tail=50 control-plane edge-01 edge-02 edge-03

bootstrap:  ## train+register v1/v2, register devices, wait until online
	$(PY) scripts/demo.py bootstrap

deploy-v1:  ## roll out baseline v1
	$(PY) scripts/demo.py deploy-v1

failure-demo:  ## v2 with deterministic canary failure -> automatic rollback
	$(PY) scripts/demo.py failure

gate-demo:  ## quality gate blocks regressing candidates
	$(PY) scripts/demo.py gate

promote-v2:  ## healthy v2 rollout
	$(PY) scripts/demo.py promote-v2

offline-demo:  ## cut edge-02's network, show local inference, restore
	$(PY) scripts/demo.py offline

demo:  ## entire lifecycle against the running stack
	$(PY) scripts/demo.py all

bench:  ## NumPy vs ONNX Runtime inference benchmark (needs: uv sync --extra bench)
	$(PY) -m scripts.benchmarks.build_models
	$(PY) -m scripts.benchmarks.run --repeat 5

load-test:  ## capacity runs against the running stack (raise rate limits first; see scripts/load_test/run.py)
	$(PY) -m scripts.load_test.run predict --concurrency 1,4,16 --duration 15
	$(PY) -m scripts.load_test.run api --concurrency 4,16 --duration 15

status:  ## fleet + deployment summary
	$(PY) scripts/demo.py status

lint:  ## lockfile freshness, ruff, mypy
	uv lock --check
	.venv/bin/ruff format --check .
	.venv/bin/ruff check .
	.venv/bin/mypy

test:  ## unit + integration + end-to-end (MQTT tests skip unless FLEETOPS_TEST_MQTT_HOST is set)
	$(PY) -m pytest -q

test-mqtt:  ## run the suite against a throw-away Mosquitto built from the project's hardened image
	-docker rm -f fleetops-test-mosq
	docker build -q --target mosquitto -t fleetops-mosquitto:dev .
	docker run -d --name fleetops-test-mosq -p 127.0.0.1:18830:1883 --read-only --cap-drop ALL \
	  --tmpfs /mosquitto/run:uid=1883,gid=1883,mode=0700 --tmpfs /tmp \
	  -v "$$PWD/deploy/secrets/mosquitto_passwd:/run/secrets/mosquitto_passwd:ro" fleetops-mosquitto:dev
	sleep 2
	FLEETOPS_TEST_MQTT_HOST=127.0.0.1 FLEETOPS_TEST_MQTT_PORT=18830 $(PY) -m pytest -q; r=$$?; docker rm -f fleetops-test-mosq; exit $$r

lock:  ## re-resolve dependencies into uv.lock (commit the result)
	uv lock

scan:  ## Trivy: lockfile + all built images (fails on fixable HIGH/CRITICAL)
	./scripts/trivy_scan.sh

verify-security:  ## assert loopback binding, MLflow auth, secrets handling, hardening on the running stack
	$(PY) scripts/verify_security.py

digests:  ## print the current registry digests for the pinned base images
	./scripts/update_digests.sh

dashboard:  ## regenerate the Grafana dashboard JSON
	$(PY) scripts/gen_dashboard.py

.PHONY: bench load-test tls-up verify-tls gate-demo lock scan verify-security digests help init up down reset logs bootstrap deploy-v1 failure-demo promote-v2 offline-demo demo status lint test test-mqtt dashboard
