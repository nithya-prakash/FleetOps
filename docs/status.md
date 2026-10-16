# Project status (verified 2026-10-13, updated after the hardening round)

FleetOps is a **development / portfolio system**: not claimed production-ready, not highly available (single control-plane replica), real data from one lab rig only (no fleet data).

## Verified on this checkout (commands in the README and docs)
* `pytest` with a real Mosquitto container: **367 passed** (275 unit, 66 integration, 26 end-to-end; 94% line coverage was measured before the real-data switch and not re-run); `uv lock --check`, `ruff format --check`, `ruff check`, strict `mypy` (78 files) clean.
* Live stack (plain and TLS profiles): `python scripts/demo.py all` (canary failure -> automatic rollback, quality gate, override rules, healthy rollout, offline inference + reconciliation, Prometheus observations); `scripts/verify_security.py` (52 checks), `make verify-tls`, `scripts/verify_dashboard.py`; a rollout over TLS.
* Trivy 0.75.0 on `uv.lock` and the images: exit 0, no fixable HIGH/CRITICAL on the scan date.
* 12 failure-injection scenarios with assertions on device, persisted state, audit and metrics: `docs/reliability-test-results.md` (includes defects found and fixed).
* Measurements on one Apple M4 laptop with Docker: `docs/load-test-results.md`, `docs/edge-inference-benchmarks.md`.
* Edge image built for arm64 (native) and amd64 (QEMU) and run offline with matching outputs.
* Kubernetes manifest on a disposable k3d cluster (verified in the previous round; not re-run in this one).

## Hardening round 2 (2026-10-13)
* **Found and fixed**: (1) heartbeat handling did O(targets) work per heartbeat and saturated the MQTT ingest thread at 100 devices - rollouts timed out and rolled back (3/3) before, 4/4 completed in ~12 s after; (2) the dev CA's certificates lacked key identifiers, so TLS failed on Python >= 3.13; (3) the control plane's default thread pool grew slowly (looked like a leak in the soak test; bounded now); (4) no backup tooling (`scripts/backup.py`, verified against the live stack).
* **Soak**: harness with explicit PASS/FAIL/INCONCLUSIVE rules; a 50-minute run (42 min steady state, 50 simulated devices) PASSED; an earlier 1 h run FAILED on thread growth, which was investigated with `py-spy` (details in `docs/load-test-results.md`).
* **CI**: workflow reworked (split jobs, timeouts, concurrency, TLS job, Prometheus rule and Compose validation, coverage artifact), actionlint-clean, action SHAs verified; a clean-copy local emulation found the certificate issue. **Not run on GitHub** (decision: stay local); steps in `docs/ci.md`.
* Alertmanager (not implemented by decision), native Linux and physical-device validation (no such environment available): remain unverified. Security assessment and PKI plan: `docs/security-assessment.md`.

## Added in this round
Security hardening (rate limits, body limits, brute-force limiting, audited denials, metrics token, edge API tokens), TLS/mTLS for MQTT and HTTPS with verification never disabled,
failure-injection suite and the engine fixes it drove, load/soak tooling, ONNX benchmark, 6 new alerts, 5 new runbooks, documentation set listed in the README.

## Not verified / needs an external environment
* **GitHub Actions** has never run (nothing was pushed; the workflow was validated locally only). Docker Desktop on macOS/arm64 is the only container runtime exercised.
* Physical edge hardware (Raspberry Pi, mini-PC, PLC integration); x86_64 performance; the amd64 control-plane image.
* Multi-node or managed Kubernetes; PostgreSQL; multi-replica control plane (designed in `docs/capacity-and-scaling.md`, not built).
* Production PKI, Alertmanager/paging, external security review.
* Soak beyond 10 minutes (inconclusive for memory growth: 314 -> 336 MB); one 22 s warm-up deployment in the scale test was not investigated.
* The vision extension was not implemented (licence and design in `docs/vision-extension-design.md`).

## Known limitations
See `docs/security.md`, `docs/capacity-and-scaling.md` and the README "Limitations" section.
