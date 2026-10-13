# Security and deployment-readiness assessment

Scope: the code and configuration in this repository, assessed on 2026-10-13. Method: code and configuration review, the project's own checks
(`scripts/verify_security.py` 52 checks, `scripts/verify_tls.py`, Trivy, 345 tests), and the experiments referenced below.
No external review or penetration test has been done, and nothing here is a certification. An item is called a **vulnerability** only when
it was demonstrated.

## 1. Verified defects (found by running things; all fixed)
| # | Finding | Evidence | Status |
|---|---|---|---|
| V1 | Development CA issued leaf certificates without Subject/Authority Key Identifier, so TLS verification failed on Python >= 3.13 (strict X.509 verification) | A clean checkout on Python 3.14 failed 8 TLS tests with `Missing Authority Key Identifier` | Fixed in `scripts/devcerts.py`; the 48 affected tests pass on 3.14 in the clean copy |
| V2 | Heartbeat handling ran a full deployment advance per heartbeat (O(targets) each) on the single MQTT ingest thread: with 100 devices the thread saturated, acknowledgements arrived late, command retries multiplied (150-300 per rollout) and rollouts rolled back after ~50 s. This is an availability weakness under load, not a confidentiality issue | `docs/load-test-results.md` "Rollout slowdown under load": 3/3 rollouts ROLLED_BACK before, 4/4 COMPLETED in 12-13 s after | Fixed in `engine.py` with regression tests |
| V3 | No backup/restore procedure existed | repository search | `scripts/backup.py backup|verify` added; restore remains a documented manual procedure |

## 2. Controls that were reviewed and found sound (with the evidence)
* **Authentication and authorisation**: three Bearer roles compared in constant time; every write route requires operator or admin (device registration and revocation: admin; deployments, rollback, cancel, training: operator); viewers are read-only. Rollback authorization = operator, recorded as the actor in the audit log. Tests: `tests/integration/test_api_security.py`.
* **Device identity and boundaries**: per-device secret derived by HMAC from a master key with an epoch; revocation bumps the epoch; artifact download requires the device credential and an in-flight deployment for exactly that version (denials audited). MQTT ACL: a device reads only its own `commands` topic and writes only its own uplink topics (`mqtt/acl.conf`, tested against a real broker).
* **Command integrity**: HMAC-signed, expiring, deduplicated by `command_id`, per-device generation fence.
* **Model artifact integrity and deserialisation**: models are safetensors (data only); the repository contains no `pickle`, `torch.load`, `eval`/`exec` or `yaml.load` use outside tests (grep of `src/`); the hash is verified on the control plane before serving and on the device before staging; corrupted artifacts are never activated (scenario 5).
* **TLS**: verification is never disabled; tests cover expired, wrong-name, untrusted and not-yet-valid certificates, plaintext against the TLS port, and mTLS.
* **Containers**: read-only root filesystem, `cap_drop: ALL`, `no-new-privileges`, non-root users, loopback-only published ports (test-enforced), secrets as files, no credential in logs (`verify_security.py`).
* **Input and resource limits**: strict Pydantic models, streamed body-size limits, per-client rate limits, brute-force limiting, bounded training parameters.
* **Audit**: state changes, denials, overrides and rollbacks are recorded with actor and correlation id.
* **CI/supply chain**: `permissions: contents: read`; actions pinned to commit SHAs (verified to match the `v4.4.0`, `v5.6.0`, `v4.6.2` tags); images pinned by digest; downloaded tools checksum-verified; hash-locked dependency installs; Trivy pinned with a self-test.

## 3. Hardening recommendations (not vulnerabilities)
1. Run a real CI execution and keep branch protection on `main` (required checks, no force-push) once the repository is on GitHub.
2. `pip install "uv==0.12.5"` in CI is version-pinned but not hash-verified; install uv from the pinned container image or verify its checksum.
3. Dev TLS private keys are written mode 0644 inside a 0700 directory so that container users can read them. Acceptable locally; in production deliver keys through a secret manager with per-service ownership.
4. Rate limiting is keyed by client address; behind a reverse proxy every client shares the proxy's address. Forward the real client address from a trusted proxy or limit at the ingress.
5. `/healthz`, `/readyz` and (without `FLEETOPS_METRICS_TOKEN`) `/metrics` are unauthenticated by design; set the token and keep these on an internal network.
6. Add Alertmanager (or another channel) so critical alerts reach a person; alerts currently fire only inside Prometheus.
7. Scheduled backups of `scripts/backup.py` output to storage outside the host, and a periodic restore drill.
8. Rotate the dev credentials in `.env` on any shared machine; the example values are public.

## 4. Unverified assumptions
* Behaviour on Linux hosts and on physical devices beyond what Docker Desktop shows.
* That MLflow's basic-auth permission model is sufficient for multiple teams (it is coarse).
* That device clocks are within the skew limits in the field (alerts exist, no field data).
* That the in-process rate limiter and lockout behave identically with several replicas (there is only one).
* Absence of vulnerabilities in dependencies beyond what Trivy's database knew on the scan date.

## 5. Controls that require infrastructure outside this repository
Production PKI and HSM/KMS key protection; an identity provider for operators; network segmentation and firewalling between plant and cloud; a managed
PostgreSQL and object store; log aggregation and SIEM; on-call paging; backup storage; an external penetration test.

## 6. Production PKI migration plan (design only; no CA was deployed and no credential invented)
1. **Trust roots**: create an offline root CA (key in an HSM or air-gapped, ceremony documented) and an online intermediate "FleetOps Issuing CA" (key in a KMS/HSM). Clients trust the root; the development CA is never reused.
2. **Server certificates** (broker, control plane, MLflow, agent APIs): issued by the intermediate with DNS SANs for the real names, 90-day lifetime, issued and renewed automatically (ACME-style or the organisation's enrolment protocol) and delivered as files to `/run/secrets`. Services reload certificates on SIGHUP or restart (already supported; see the TLS runbook).
3. **Device identity**: replace the shared-secret MQTT password with per-device client certificates (mTLS is already implemented and tested) whose subject is the device id; Mosquitto maps the certificate common name to the user (`use_identity_as_username`) so the existing ACL patterns keep working. Keys are generated on the device (TPM/secure element where available) and only the CSR leaves it.
4. **Enrolment**: an operator registers the device (existing admin endpoint) which creates a one-time enrolment token; the device submits a CSR with the token; the CA returns a certificate with a 30-90 day lifetime.
5. **Renewal**: the agent renews at two thirds of the lifetime over the authenticated channel; failure to renew triggers the existing offline behaviour (the device keeps inferring) plus an alert on expiry.
6. **Revocation**: short lifetimes are the primary control; additionally maintain a CRL (or OCSP) that the broker checks (`crlfile`), and keep the existing "revoke" action which bumps the device epoch so the control plane rejects the device even if its certificate is still valid.
7. **Migration**: run both mechanisms in parallel (password and certificate) per device, switch devices group by group through the existing deployment mechanism for configuration, then disable password authentication on the broker.
8. **Auditing**: log every issuance and revocation with the actor; alert on issuance outside expected windows.

## 7. Is SQLite with one replica suitable?
For the intended scale of a pilot (tens to a few hundred devices at 5-30 s heartbeats) yes, with the heartbeat fix (V2): 100 simulated devices were ingested and rolled out in about 12 s on a laptop (`docs/load-test-results.md`). Move to PostgreSQL and more than one replica when any of these is true: a requirement for availability during a control-plane restart or host failure; more devices than the ingest thread can keep up with (watch `fleetops_mqtt_ingest_lag_seconds` and heartbeat processing time); or a need for managed backups and point-in-time recovery. Before horizontal scaling these coordination problems must be solved first, because today they are solved by being single-process: rollout-engine locks (needs leader election or database row locks), MQTT consumption (shared subscriptions or a single consumer), rate-limit and lockout state, local artifact and MLflow storage (object storage and database-backed MLflow). No database migration was performed in this work.

## 8. Local-demo readiness vs production readiness
| | Local demo | Production |
|---|---|---|
| Readiness | Ready: reproducible, tested, documented | **Not ready** |
| Missing for production | | real PKI, HA control plane and database, Alertmanager and on-call, backup schedule and restore drills, physical-device validation, external security review, GitHub CI history |
