# Security

**The local Compose profile is for development only**: plaintext MQTT and HTTP, throw-away credentials generated from `.env`,
simulation endpoints enabled. Do not describe or run it as production.

## Network exposure
* Every published port of the Compose stack binds to `127.0.0.1` by default (`FLEETOPS_BIND`, set to `0.0.0.0` only on a trusted
  network). Service-to-service traffic uses the internal compose network and does not depend on published ports.
* Binding to loopback is **not** treated as authentication: the control plane, MLflow, Grafana and the broker all authenticate.
* Not authenticated by design (keep them on an internal network): `/healthz`, `/readyz`, `/metrics` on the control plane, and an
  agent's `/predict`, `/healthz`, `/metrics` (a PLC-side inference API on the plant network). The agent's `/admin/*` requires the device
  credential and is disabled unless simulation is allowed.

## Identities and authorisation
* **Operators**: three Bearer API keys (viewer/operator/admin), compared in constant time; never logged or returned.
  In `FLEETOPS_ENV=prod`: >= 24 chars, distinct, no `change-me`, simulation off, MLflow credentials and MQTT password required.
* **Devices**: per-device secret = `HMAC-SHA256(master, "fleetops-device:<id>:<epoch>")`, used as MQTT password and artifact bearer.
  Revocation bumps the epoch.
* **MLflow**: runs with its built-in basic-auth (`--app-name basic-auth`). Humans use `admin`; the control plane uses a separate
  non-admin account created/rotated by the one-shot `mlflow-init` service. `scripts/verify_security.py` checks that anonymous and wrong
  credentials get 401, the service account works, and it cannot create users or is not an admin. Default permission for authenticated
  users is READ. Limitation: MLflow basic-auth is coarse; for multi-team use put MLflow behind an identity-aware proxy.
* **Mosquitto**: `allow_anonymous false`, password file, topic ACLs (a device reads only its own `commands` and writes only its own
  uplink topics). Verified against a real broker in the integration tests.
* **Commands**: HMAC-signed, with id, per-device generation and expiry; the agent logs and reports clock-skew and expiry rejections.
* **Artifact download**: device authentication plus a database check that the device is a target of an in-flight deployment for exactly
  that version (or its rollback destination); denials are audited. No file paths are exposed.

## Secrets handling
* Credentials are mounted as **files** under `/run/secrets` (Compose secrets / Kubernetes Secrets) and read by the applications via
  pydantic settings (`secrets_dir`), by Grafana via `GF_SECURITY_ADMIN_PASSWORD__FILE`, and by the MLflow entrypoint into its own process
  environment. They are not in `docker inspect`, compose output or container command lines
  (`scripts/verify_security.py` searches every container's env/cmd for each credential).
* `python scripts/provision.py` writes `deploy/secrets/` (directory `0700`, gitignored; files `0644` because containers use different
  uids and the directory permission is what protects them from other host users). Stale files from older layouts are removed.
* The Mosquitto password file is copied by the container entrypoint into a private tmpfs with mode `0600` (what Mosquitto requires), so
  the world-readable-file warning is gone. The broker healthcheck is a TCP check, deliberately: an authenticated check would put the
  password on a command line.
* The Kubernetes manifest contains **no Secret objects** (a manifest with placeholder passwords is a trap). `scripts/k8s_secrets.sh`
  creates them from the provisioned files; use sealed-secrets/external-secrets on a real cluster.
* `.env` and `deploy/secrets/` are gitignored; a test enforces it. `.env.example` has only labelled development placeholders.

## Model supply chain
safetensors only (data, no code); checksum verified after download, before activation and at every load; bundle parsing is validated and
size-capped; corrupted active model falls back to the previous one. A **quality gate** (see api.md) can block regressing or unevaluated
candidates before any device is touched. Not covered: provenance signing from a build pipeline.

## Container and build hardening
Non-root (uid 10001; Mosquitto 1883), read-only root filesystems, `cap_drop: ALL`, `no-new-privileges`, tmpfs for `/tmp`, health checks,
digest-pinned bases, hash-verified locked dependencies, Trivy gate. See [supply-chain.md](supply-chain.md). Kubernetes: non-root, read-only,
no privilege escalation, capabilities dropped, service-account token **not** mounted.

## API protections
* **Rate limiting**: token bucket per client address for `/api/` requests (behind a reverse proxy the address is the proxy's), `FLEETOPS_RATE_LIMIT_PER_SECOND` (default 50) and
  `..._BURST` (100); the agent's `/predict` has its own limit (`FLEETOPS_EDGE_PREDICT_RATE_PER_SECOND`, default 500). Excess requests get 429
  with `Retry-After`. These are protections, not capacity: raise them for load tests (see `docs/load-test-results.md`).
* **Payload limits**: request bodies are counted while streamed (control plane 1 MB, agent 64 KB); oversized bodies get 413 before parsing.
* **Brute force**: after `FLEETOPS_AUTH_FAILURE_LIMIT_PER_MINUTE` (default 10) failed credentials from one client address, further requests are answered
  with 429 until the bucket refills; a correct credential is not accepted while the client is limited.
* **Audited denials**: authorisation failures, quality-gate overrides and artifact denials are written to the audit log (in their own
  transaction, so the rejected request does not roll the record back).
* **Metrics**: `/metrics` may require a read-only token (`FLEETOPS_METRICS_TOKEN`), which grants nothing else. Edge APIs accept a separate
  edge API token.

## TLS (exercised)
`make tls-up` starts the stack with TLS on every hop (`compose.tls.yaml`): MQTT on 8883 (broker, control plane, agents), HTTPS for the
control plane, MLflow and the agents' APIs, Prometheus scraping over HTTPS with CA verification. `make verify-tls` asserts that plaintext
listeners are gone, that a client without the CA is refused and that the verified connection works. `FLEETOPS_ENV=prod` refuses to start without
TLS configuration. **No component disables certificate verification**, in code or in tests.

Automated tests (`tests/integration/test_tls_mqtt.py`, `test_tls_https.py`, real Mosquitto container) cover: valid chain and hostname;
expired certificate; certificate for the wrong name; untrusted CA; plaintext client against the TLS port; mutual TLS (client certificate
required, none/untrusted refused); the whole deploy lifecycle over HTTPS with verified artifact download.

### Development vs production certificates
`scripts/devcerts.py` creates a private CA and short-lived leaf certificates under `deploy/secrets/tls/` (git-ignored). This is **not a PKI**: the
CA key lives on the developer's disk, there is no revocation and no automated renewal. For production use certificates from your organisation's
CA (or an internal CA with an HSM/KMS-held key), deliver them as files under `/run/secrets`, and distribute only the CA certificate to clients.

### Certificate rotation
1. Issue the new leaf certificate/key (same names in SAN) and replace the files; 2. restart the service (`docker compose restart <service>`;
Mosquitto also reloads on `SIGHUP`); 3. clients reconnect on their own with bounded backoff; verify with `make verify-tls`. To rotate the **CA**,
first distribute a bundle containing old and new CA, then re-issue leaves, then drop the old CA. See `docs/runbooks/tls-certificate-failure.md`.

## Remaining limitations
Single control-plane replica (SQLite, in-process locks); the default local profile is plaintext (use `make tls-up` for TLS); MLflow
basic-auth is coarse-grained and MLflow's Flask secret is passed through the environment; agent `/healthz` is unauthenticated by design; no
automated secret or certificate rotation; no revocation list for device certificates (revoke by bumping the device epoch); one lab rig's data only;
the image-vulnerability gate ignores unfixed issues; Docker Desktop on macOS was the only container runtime exercised locally (Linux is covered by
the CI workflow, which has not yet run). This project has not had an external security review or penetration test.

## Backup and restore
`python scripts/backup.py backup` takes a consistent online backup of the control-plane and MLflow SQLite databases (SQLite backup API, safe while the service runs) and artifacts, with a SHA-256 manifest; `python scripts/backup.py verify <dir>` checks hashes, SQLite integrity and row counts. Tested against the running stack, including detection of a tampered file. Restoring is a manual procedure (stop the stack, copy files into the volumes, start) and a restore drill has **not** been performed. See also `docs/security-assessment.md`.
