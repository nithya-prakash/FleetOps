# Reliability test results

Each scenario has an **expected behaviour written before the test or the code change**, a deterministic way to inject the failure, and
assertions on the *actual* state: the model really running on the device, the persisted deployment/target state, audit events, and metrics.
Results are filled in from real runs (see "Results" below). The control plane is a **single replica**; nothing here claims high availability.

## Expected behaviour (written first)

| # | Scenario | Injection | Expected behaviour |
|---|---|---|---|
| 1 | MQTT broker restart | Real Mosquitto container stopped / restarted (persistent session + volume), also while a health window is open | Both sides reconnect on their own with bounded backoff; commands published while the broker is down are retried by the engine's timer (same `command_id`); rollout completes; no command executed twice; devices keep inferring throughout; reconnects are counted |
| 2 | Control-plane restart during a deployment | Restart while the canary is in its health window and while a DEPLOY is un-acked | State resumes from the database; no duplicate execution on devices (dedup by `command_id`); deployment completes; audit trail continuous |
| 3 | Edge-agent restart during artifact download | Download blocked after a partial write, agent "killed" (new instance on the same data dir) | Old model stays active and serving; stale partial download removed at startup; engine retry makes the new instance download again; target ends STAGED/ACTIVE; no leftover files |
| 4 | Network loss before / during / after activation | Cut device link at each point | *Before*: DEPLOY queued, ack timeout fails the target, rollout rolled back, queued DEPLOY is neutralised (ABORT / expiry) so the device never activates it. *During*: activation may happen on the device while the control plane is blind; health cannot be verified -> target fails -> rollback command reissued after expiry; the device keeps serving the new model until reconnect (documented limitation) then reverts. *After*: deployment stays COMPLETED; device keeps inferring |
| 5 | Corrupted / mismatched artifacts | Corrupt the stored artifact on the control plane; corrupt download; truncate model on device | Control plane refuses to serve an artifact whose hash differs from the registry (audited + counted); device verifies the manifest hash anyway; nothing is staged or activated; last known-good keeps serving |
| 6 | Expired and superseded commands | Short TTL + device offline past expiry; supersede while a DEPLOY is queued; replay an old ACTIVATE | Expired commands are rejected by the device and reissued fresh by the engine; the device never activates the superseded version; a replayed older ACTIVATE is rejected as STALE |
| 7 | Duplicate MQTT messages / duplicate requests | Re-deliver the same command and ack bytes N times; 5 parallel POSTs with one Idempotency-Key | A command executes once (replayed ack, counted); duplicate acks cause no extra transition; one deployment is created; no duplicate side effects |
| 8 | Disk exhaustion | Free-space override fault; injected ENOSPC during write and during state persistence | Agent refuses to start a download it cannot store (`INSUFFICIENT_STORAGE`), cleans partial files on ENOSPC, state file stays valid; last known-good keeps serving; free disk is exported |
| 9 | Rollback failure | `fail_rollback` fault | Deployment ends FAILED (not ROLLED_BACK), device remains on the bad model, failure counted, recoverable with a second rollback |
| 10 | Device reconnects after several versions were deployed | Device offline while v2 then v3 roll out to the others | Offline device is excluded, receives no stale commands; on reconnect it reports its true (old) version and can be brought current by a fresh deployment |
| 11 | Stale device inventory after failed deployment | Agent heartbeat interval much longer than the test | Inventory reflects the real model immediately after a rollback (from the ack, not a stale heartbeat); a following deployment captures the right rollback destination and does not skip the device |
| 12 | Partial fleet rollout | One device of the second batch fails activation | Whole deployment halts and **every** activated device (including the already verified canary) is rolled back: no mixed fleet is left behind |

Cross-cutting expectations: retries are bounded (`max_command_attempts` for forward commands; rollback is retried until acknowledged by design),
every state change is audited, and a failed update never removes the last known-good model.

## Results

Run: `pytest tests/end_to_end tests/unit tests/integration` (in-memory broker harness: real control plane over HTTP(S), real MLflow with SQLite, real edge agents; scenarios 1 additionally run against a real Mosquitto container via `FLEETOPS_TEST_MQTT_PORT`). Final full suite on 2026-10-13 with a real Mosquitto container: 345 passed, coverage 94%. Assertions read real device state (active model served, staged files on disk), persisted control-plane state (deployment/target rows), the audit log and Prometheus metrics, not mocks of them.

| # | Scenario | Tests | Outcome |
|---|---|---|---|
| 1 | Broker restart | `test_reliability_mqtt.py` (3 tests, real Mosquitto) | Pass. Commands issued while the control plane is cut off are delivered after reconnect; a full broker outage refuses *new* rollouts with a clear error and heals when the broker returns (measured recovery 0.8 s in that test); a restart inside the health window does not fail the rollout; reconnects are counted. Note: a clean broker shutdown publishes the devices' last-will messages, so the control plane marks them OFFLINE until they reconnect (safe, by design). |
| 2 | Control-plane restart mid-deployment | `test_reliability_b.py::test_s2_*` (2) | Pass. State resumes from the database; un-acked commands are re-sent with the same `command_id`; no duplicate execution; audit continuous. |
| 3 | Agent killed during download | `test_s3_*` | Pass. Old model keeps serving, partial download removed at startup, retry completes, no leftover files. |
| 4 | Link loss before / during / after activation | `test_s4_*` (3) | Pass. *Before*: no STAGED ack, the target fails, the rollout is ROLLED_BACK and the untouched device is CANCELLED; one best-effort ABORT is sent; on reconnect the expired DEPLOY is REJECTED, nothing is downloaded and the device still serves v1. *After activation / completion*: the device keeps serving while the control plane is blind and is reconciled on reconnect; a completed rollout stays COMPLETED. |
| 5 | Corrupted artifact | `test_reliability_a.py::test_s5_*` | Pass. Hash/size mismatch is never served or activated; `artifact.denied` audited. |
| 6 | Expired and superseded commands | `test_s6_*` (2) | Pass. Expired command rejected and reissued fresh; superseded deployment never activates on a reconnecting device. |
| 7 | Duplicate messages / parallel requests | `test_s7_*` (2) | Pass. Re-delivered commands/acks have no side effects; 5 parallel unkeyed POSTs yield exactly one active deployment. |
| 8 | Disk exhaustion | `test_s8_*` + unit tests | Pass. Update refused, last-known-good keeps serving, state-write failure does not corrupt the stored state. |
| 9 | Rollback failure | `test_s9_*`, `test_rollback_failure.py` | Pass. Deployment ends FAILED (not ROLLED_BACK); failure counted in metrics; recoverable by a new deployment. |
| 10 | Reconnect after several versions | `test_s10_*` | Pass. Device converges to the current desired version, never to an intermediate one. |
| 11 | Stale inventory after failed deployment | `test_s11_*` | Pass. Inventory shows the true active version without waiting for a heartbeat. |
| 12 | Partial fleet failure | `test_s12_*` | Pass. Whole deployment halts and every device returns to the previous version; no mixed fleet. |

### Defects the scenarios found (all fixed, each with a regression test)

1. **Lost update when superseding** — parallel unkeyed deployment POSTs could both supersede the same predecessor. Fixed by performing supersede under the engine lock.
2. **Denial audit rolled back with the 403** — `artifact.denied` rows were written in the request transaction that the error then rolled back. Now written in its own transaction.
3. **Failed publishes consumed the attempt budget** — a broker outage could exhaust `max_command_attempts` without any command having left. Only successful sends count now.
4. **Queued DEPLOY outliving the engine's patience** — a command sitting in the broker queue could be executed long after the engine had given up. Forward-command TTL is now `min(ttl, ack_timeout + 5)` and a best-effort ABORT is sent when a target is failed.
5. **Staging leftovers after a crash** — partial downloads survived an agent restart. Removed at startup.
6. (Earlier) stale inventory after rollback and CPU % above 100 in agent metrics.

### Test-harness artifact found and fixed
`test_full_lifecycle_over_https_with_verified_artifact_download` ended ROLLED_BACK in two full-suite runs run with coverage and passed in every isolated run (more than 20). The assertion was changed to print the deployment record, which showed the cause: `health check failed: memory 1083MB > 1024MB`. The harness runs agents inside the pytest process, so the memory they report is the whole test process's RSS, which grows past the default 1 GB limit late in a long run. This is not a product defect (real agents are separate processes, about 65 MB), and the health check behaved correctly. The test now sets an explicit, documented `max_memory_mb` for in-process agents; the final full run passed.

### Remaining limitations
- Single control-plane replica and SQLite: a control-plane outage stops new rollouts and monitoring until it restarts (devices keep serving).
- Scenarios run against an in-memory broker harness except scenario 1; other scenarios have not been repeated on a real broker.
- No test of multi-node network partitions, clock skew beyond the command expiry check, or physical power loss during a write.
