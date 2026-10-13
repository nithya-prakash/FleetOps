# Advanced validation plan

Status of each item is tracked at the end of this document; results live in the linked documents. Nothing here claims production readiness.

## Baseline (re-established independently, 2026-10-12)
| Check | Result on this checkout |
|---|---|
| `uv lock --check`, `ruff format --check`, `ruff check`, `mypy` (strict) | clean (110 files formatted, 66 source files typed) |
| `pytest` with a real Mosquitto container | 255 passed, 93% coverage (209 unit / 39 integration / 7 end-to-end) |
| `engine.py` (restored from a built image after an accidental truncation) | 957 lines, 37 `RolloutEngine` methods, parses and ends cleanly; the tests cover 93% of it. The 35 uncovered lines (heartbeat reconciliation of rollbacks, ABORT acks, expired-command reissue, FAILED-rollback acks, send_device_command) were read and are consistent with the tested paths. Phase 3 scenarios are designed to execute them. |
| Environment | macOS arm64 (Apple Silicon), Docker Desktop 29.5.3 (10 CPUs, 8 GiB), Python 3.12 venv (system Python 3.14), 23 GiB disk free |

Git: no commits exist (by instruction nothing is committed or pushed in this task).

## Review of existing security / reliability limitations (inputs to this plan)
Unauthenticated: control-plane `/metrics`, `/readyz` details, agent `/predict`, `/metrics`; no rate limiting; body-size limit trusts the
`Content-Length` header (chunked bodies bypass it); authentication failures and denied management requests are not audited; a revoked
device's predictions were still ingested; plaintext MQTT/HTTP only; no failure-injection coverage of broker restart, storage exhaustion,
partial fleet outcomes; no load/soak numbers; one inference runtime; no ARM64/x86 image evidence; operational runbooks missing for several
scenarios; single replica, SQLite.

## Sequence and acceptance criteria
| # | Phase | Deliverable | Measurable acceptance |
|---|---|---|---|
| 2 | Security | Rate limiting + streamed body limits + auth-denial auditing; metrics/predict tokens; revoked-device ingestion fix; **TLS end to end** (MQTT + HTTPS) with dev CA tooling | New tests pass; TLS tests reject wrong hostname, untrusted CA, expired cert, and plain-text fallback; `verify_security.py` extended and passing on a live TLS stack; no TLS verification is ever disabled |
| 3 | Reliability | 12 deterministic failure scenarios | Each scenario: expectation written first, actual active model + persisted state + audit + metrics asserted; results in `docs/reliability-test-results.md` |
| 4 | Load / soak | `scripts/load_test/` runner | Reproducible command lines; recorded p50/p95/p99, throughput, errors, CPU/RSS, MQTT ingestion, deploy duration vs fleet size (3, 10, N), restart recovery, soak trend (RSS, queue depth, stale state) |
| 5 | Edge inference | NumPy vs ONNX Runtime (and INT8 if meaningful) | Export verified numerically; same inputs for all runtimes; decision recorded with evidence in `docs/edge-inference-benchmarks.md` |
| 6 | Linux edge devices | Multi-arch audit and builds, provisioning guide | arm64 and amd64 images built (amd64 by emulation only if it works) and smoke-tested where possible; physical-device guide labelled **unverified on hardware** |
| 7 | Vision extension | Design (implementation only if it does not distract) | License of the dataset checked and recorded; interface sketched; clearly separated from the sensor workflow |
| 8 | Observability | Low-cardinality metrics review, alert severity/response, runbooks | Dashboard/alert queries still validated against exported metrics; new runbooks for the seven listed scenarios |
| 9 | Architecture review | Capacity and scaling document, target diagram | Labelled "not implemented"; no new infrastructure introduced |
| 10 | Portfolio | README, engineering decisions, reproducible demo script | Demo uses real functionality end to end; recording checklist provided |

## Known limitations carried into this work
Single control-plane replica; local-only evidence (Docker Desktop on macOS/arm64, no Linux host, no physical edge device); synthetic data;
GitHub Actions workflow never executed; Kubernetes verified on a single-node k3d only.

## Progress log
2026-10-13: phases 1-10 completed except where a phase explicitly allowed a design-only outcome (7).
1 baseline (344 -> 345 tests); 2 security + TLS/mTLS exercised end to end; 3 twelve scenarios, 5 defects fixed; 4 load/soak measured; 5 ONNX measured (NumPy kept);
6 arm64 + amd64 images, offline inference verified, physical-device guide labelled unverified; 7 licence checked, design only; 8 alerts/runbooks, dashboard queries re-verified;
9 capacity review + target diagram (not implemented); 10 docs, demo, final verification. Final state and remaining unverified items: `docs/status.md`.
