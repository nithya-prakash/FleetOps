# Load, scale and soak results

Everything below was **measured on one laptop with the Compose stack running in Docker Desktop**; it is a development-machine
characterisation, not a production capacity statement. No target numbers were assumed in advance. Raw JSON (with the full environment)
is written by the tools to `load_results/` (git-ignored).

## Environment of these runs (2026-10-13)
| | |
|---|---|
| Host | Apple M4 (arm64), 10 cores, 16 GB RAM, macOS (Darwin 24.6.0) |
| Container runtime | Docker Desktop 29.5.3, VM limits 10 CPUs / 8 GB, linux/arm64 images |
| Software | Python 3.12.14, control plane (FastAPI/uvicorn, SQLite), MLflow (SQLite), Mosquitto 2.0.22, edge agents (NumPy runtime, 1 uvicorn worker) |
| Transport | plaintext profile (TLS off), loopback ports, rate limits raised (`FLEETOPS_RATE_LIMIT_PER_SECOND=5000`, burst 10000, edge predict 5000/10000) |
| Workload | model `anomaly-detector` (MLP 16-8, 6 features); deterministic request payloads (seed 7); simulated devices are real `EdgeAgent`s over real MQTT, 2-3 farm processes on the host |
| Caveat | load generator and system under test share the same machine and compete for CPU |

## Reproduce
```bash
make init up                                   # as in README; then raise the rate limits and recreate (see header of scripts/load_test/run.py)
python scripts/demo.py bootstrap
python scripts/provision.py --load-devices 100 # broker credentials for simulated devices, then recreate mosquitto
python -m scripts.load_test.run predict --concurrency 1,4,16 --duration 15
python -m scripts.load_test.run api --concurrency 4,16 --duration 15
python -m scripts.load_test.run ingest --devices 100 --duration 30
python -m scripts.load_test.run deploy-scale --sizes 3,10,30,100
python -m scripts.load_test.run restart --devices 50
python -m scripts.load_test.run soak --devices 50 --duration 600
```

## Edge inference API (`POST /predict` on edge-01, closed loop, 15 s per level)
| clients | req/s | client p50 / p95 / p99 ms | server-side inference p50 / p95 / p99 ms | 429 | errors | agent CPU avg/max % | RSS MB |
|---|---|---|---|---|---|---|---|
| 1 | 552 | 1.66 / 2.74 / 4.28 | 0.053 / 0.100 / 0.427 | 0 | 0 | 53 / 72 | 64 |
| 4 | 1012 | 3.65 / 6.19 / 9.99 | 0.053 / 0.176 / 0.440 | 0 | 0 | 105 / 111 | 65 |
| 16 | 1202 | 12.45 / 21.23 / 30.03 | 0.053 / 0.159 / 0.435 | 0 | 0 | 123 / 124 | 66 |

Reading: the model itself takes ~0.05 ms; client-observed latency is dominated by HTTP handling in a single-worker process. Throughput
saturates at ~1.2k req/s per agent (one core), after which added clients only add queueing delay.

## Control-plane management API (read mix over 6 endpoints)
| clients | req/s | p50 / p95 / p99 ms | 429 | errors | CP CPU avg/max % | CP RSS MB |
|---|---|---|---|---|---|---|
| 4 | 308 | 9.97 / 31.6 / 54.7 | 0 | 0 | 127 / 152 | 490 |
| 16 | 228 | 51.2 / 172.8 / 264.2 | 0 | 0 | 157 / 168 | 506 |

Reading: throughput **drops** when concurrency rises from 4 to 16 and CPU is above 100% (the process is CPU-bound on this host, with
synchronous SQLAlchemy/SQLite work). It is far above what an operator console needs but it is the number to watch when adding dashboards or automation.

## MQTT ingestion (100 simulated devices, heartbeat every 2 s, 1 Hz inference, 30 s window)
* All 100 devices online after 3.2 s; 100/100 online at the end; 0 failures.
* Uplink: ~51 heartbeats/s, ~10 telemetry/s, ~0.6 predictions/s.
* Control-plane processing time per heartbeat p50 1.59 / p95 2.47 / p99 4.67 ms; predictions p50 4.55 / p99 9.87 ms.
* Heartbeat ingest lag (device send -> processed) p50 2.6 / p95 5.3 / p99 26 ms.
* API probe during ingestion p50 6.2 / p95 10.8 / p99 11.8 ms. Control plane CPU avg 12%, Mosquitto <1% CPU, 3 MB RSS.

## Deployment duration vs fleet size
Canary of 1, then the rest in one batch; health window 2 s; each device downloads and verifies the artifact.
| devices | final state | seconds |
|---|---|---|
| 100 (first rollout after the farm started; warm-up) | COMPLETED | 22.1 |
| 3 | COMPLETED | 6.1 |
| 10 | COMPLETED | 8.2 |
| 30 | COMPLETED | 8.2 |
| 100 | COMPLETED | 8.2 |

Reading: after the first rollout the duration is dominated by the fixed canary health window plus the batch step and barely grows from 10 to
100 devices on this machine. The 22 s warm-up row was not investigated further; it is a one-off measured once, so do not quote it as a typical figure.
Simulated devices run on one host, so this says nothing about WAN artifact-download time.

## Restart recovery under a running fleet (50 devices; times from the start of `docker compose restart`)
| restarted | restart command | control plane ready | all devices ONLINE with a post-restart heartbeat |
|---|---|---|---|
| control-plane | 1.5 s | 5.2 s | 5.2 s |
| mosquitto | 0.5 s | 0.5 s | 1.6 s |

(An earlier version of this scenario measured the wrong thing - its waits started after the restart and its "online" check was already true; it was
fixed to require heartbeats newer than the restart before these numbers were recorded.)

## Soak (50 devices, 600 s, 40 samples)
* No device offline in any sample; 0 new MQTT processing failures; API probe p50 12.6 / p95 22.5 / p99 25.8 ms.
* Control-plane RSS 314 -> 336 MB (fitted slope +120 MB/h), Mosquitto +0.9 MB/h, database file +0.5 MB/h.
* **Not conclusive.** A 10-minute window mostly shows warm-up (caches, connection pools). Whether the control plane leaks cannot be decided from
  it; run `soak --duration 14400` (4 h) before relying on long-running behaviour. Only this short soak was run.

## What is not measured
Multi-host or WAN behaviour; fleets above 100 simulated devices; PostgreSQL; TLS overhead (the TLS profile was functionally verified, not benchmarked);
sustained multi-hour operation; behaviour on x86_64 or on real edge hardware.

---

# Update 2026-10-13 (second round): rollout slowdown under load

The earlier sections above are kept as first recorded. This round investigated the unexplained 22 s first rollout in "Deployment duration vs
fleet size" and found a real scalability defect; the results below supersede the "rollout of 100 devices ~8 s" statement for the pre-fix code.

## Method
`python -m scripts.load_test.run warmup --devices 100 --settle <s> --rollouts <n>` rolls v1/v2 out to 100 simulated devices (canary of 1, then the
other 99) and, for every rollout, reconstructs a per-stage timeline from the control plane's own **audit events** (server timestamps, so client
polling is excluded) and records command retries, acknowledgement count, and deltas of the MQTT ingest-lag and processing-time histograms.
Same machine, same stack, same workload before and after the fix; the host was under memory pressure during these runs (macOS reported ~100 MB free,
7 GB compressed), which makes absolute numbers pessimistic but affected both sides equally.

## What was ruled out (each with a run)
| Hypothesis | Test | Result |
|---|---|---|
| Farm start-up / cold devices | `--settle 60` before the first rollout | first rollout still 52.9 s: not a warm-up effect |
| Default rate limit (all simulated devices share one source address) | raised limits to 5000/10000 and recreated the control plane; no `fleetops_rate_limited_total` series existed anyway | still rolled back: not rate limiting |
| Competing containers (MLflow, Grafana, Prometheus) | stopped them | still rolled back (65, 55, 50 s) |
| Artifact download problem | `fleetops_artifact_downloads_total{result="ok"}` matched the target count, no HTTP 429/5xx | downloads were fine |
| Idle ingest capacity | 100-device `ingest` scenario | heartbeat processing p50 2.0 ms, lag p50 2.8 ms: fine when no rollout is running |

## Finding
During a rollout the audit log showed that almost all STAGED/ACTIVE transitions were "reconciled from heartbeat" instead of coming from
acknowledgements, 150-300 DEPLOY/ACTIVATE commands were re-sent per rollout, and heartbeat **processing time rose from 2 ms to 17 ms (p50)** with
heartbeat ingest lag at 1.5-5.5 s. At ~50 heartbeats/s that is ~85% of the single MQTT ingest thread. Cause: `RolloutEngine._apply_heartbeat` ended with
a full `_advance()` of the deployment (loads and evaluates every target) for **each heartbeat**, i.e. work per second proportional to devices x targets.
Near saturation, queueing makes the outcome bimodal (fast when the thread keeps up, timeouts and rollback when it does not), which is consistent with
one slow row among otherwise fast ones in the first round. The original 22 s row was **not instrumented**, so this mechanism is the best-supported
explanation for it, not a proven cause of that exact number.

## Fix
A heartbeat now triggers an immediate advance only when it changed that target's state (healing a lost acknowledgement). Health observations and timers
are handled by the existing 1 s ticker. Regression tests: `tests/unit/test_engine.py::test_heartbeat_without_a_state_change_does_not_advance_the_deployment`
(fails if the old behaviour is restored - checked by mutation) and `..._heals_a_lost_ack_still_advances_immediately`.

## Before / after (100 devices, same workload)
| | Before | After |
|---|---|---|
| Rollout outcome | 3 of 3 `ROLLED_BACK` ("no ACTIVE/STAGED acknowledgement before timeout / attempts exhausted") in the instrumented run; earlier uninstrumented runs 9.8-52.9 s | 4 of 4 `COMPLETED` |
| Duration | 46-65 s (rolled back) | 12.0, 13.4, 12.2, 13.1 s |
| Commands re-sent per rollout | DEPLOY 137-297, ACTIVATE 100-299 | 0 / 0 |
| Heartbeat processing p50 / p99 | 17.2 ms / 65 ms | 2.2-2.6 ms / 36-50 ms |
| Heartbeat ingest lag p50 | 1.5-5.5 s | 0.013-0.28 s |

The post-fix duration (~12 s) is dominated by the canary health window (2 s) plus staging/activation of 99 devices and the batch health window; the
earlier "~8 s" figure was measured with a different, lightly loaded host state and is not directly comparable.

## Soak, second round (2026-10-13): harness, verdict rules, results
The 10-minute soak in the first round was inconclusive. `python -m scripts.load_test.run soak` now has a warm-up segment, a steady-state segment with a
scheduled rollout every 5 minutes (so the deployment engine is part of the soak, not only ingest), a controlled control-plane restart at the end, and
**explicit rules** (`scripts/load_test/soak.py`, unit-tested in `tests/unit/test_soak_verdict.py`):
* judged on the steady state only (after warm-up); RSS growth counts as a leak only if the fitted slope exceeds 10 MB/h **and** the trend is linear (R^2 >= 0.7);
* file descriptors or threads growing more than 10% (first- vs last-quartile median), any container restart, HTTP 5xx, MQTT processing failures, rollout that did not complete, offline devices, or recovery slower than 60 s = **FAIL**;
* a steady state shorter than 40 minutes can never PASS (**INCONCLUSIVE**); PASS means "nothing found in this window", not a long-term guarantee.
The application process (PID 1 in the container: RSS, open fds, threads) is measured separately from the load generator's process tree.

| Run | Duration (steady) | Verdict | Notes |
|---|---|---|---|
| Smoke (20 devices) | 5 min (4) | INCONCLUSIVE | too short by design; 3 rollouts completed, recovery 5.3 s |
| 1 h soak, 50 devices | 60 min (50) | **FAIL**: threads grew 22% (27 -> 33) | RSS flat (410 -> 374 MB), 11/11 rollouts completed, recovery 6.5 s |
| Investigation | 7 min, 16 rollouts; 30 drift runs; 25 min with `py-spy` | - | thread count did not grow per rollout or per drift run; `py-spy dump` showed the growing threads were `asyncio_0..asyncio_8`, all idle |
| 50 min soak after the fix, 50 devices | 50 min (42) | **PASS** | RSS 369 -> 307 MB (no growth), threads 25 -> 25 (quartile medians; transient rollout peaks return to baseline), fds 37 -> 39, 9/9 rollouts completed, 0 restarts, 0 5xx, 0 MQTT failures, API probe p50/p95/p99 21 / 46 / 73 ms, control-plane CPU median 6%, recovery after a controlled restart 7.2 s |

**The thread "growth" was not a leak, but it was a real wart.** `asyncio.to_thread` uses the loop's default executor, which creates workers lazily up to
`min(32, cpu_count + 4)` (14 in this container). The ticker only ever needs one at a time, yet the pool crept up by one thread every few minutes. The
control plane now installs a fixed 4-thread executor (`BACKGROUND_WORKERS`, regression test `test_background_work_runs_on_a_small_fixed_pool`, checked by mutation), so the thread count is deterministic and the soak rule stays strict.

Limits of this evidence: one machine, 42 minutes of steady state, 50 simulated devices; it does not show multi-day behaviour. A negative RSS slope here means memory was
still settling after start-up (the control plane imports pandas/scipy/MLflow), not that memory is shrinking forever.
