# Continuous integration

**Status:** the workflow has run on GitHub. The first run ([run 38401322687](https://github.com/nithya-prakash/FleetOps/actions/runs/38401322687), 2026-10-13, `ubuntu-24.04`) had
**4 of 6 jobs passing**; the two failures were real defects (below) and have fixes in the following commit. The outcome of the run after the fixes is recorded in "Run history" at the bottom of this file.

## Jobs (`.github/workflows/ci.yml`)
| Job | Purpose | Timeout |
|---|---|---|
| `static` | `uv lock --check`, locked install, Ruff format + lint, mypy, workflow lint (actionlint), Prometheus rule validation (`promtool`), Compose config for the plain and TLS profiles | 15 min |
| `tests` | unit + integration + end-to-end against the project's hardened Mosquitto image, coverage XML artifact, `--cov-fail-under=90` | 30 min (25 for pytest) |
| `containers` | build all images, Trivy scan (fixable HIGH/CRITICAL), boot the stack, `verify_security.py`, `demo.py all`, dashboard queries | 40 min |
| `tls-profile` | start the TLS profile, `verify_tls.py`, a bootstrap and rollout over TLS | 30 min |
| `kubernetes-manifests` | kubeconform (checksum-verified binary) | 10 min |
| `kubernetes-e2e` | disposable k3d cluster + smoke test (after `static`) | 40 min |

Design: `permissions: contents: read`; actions pinned to commit SHAs (verified against the `v4.4.0` checkout, `v5.6.0` setup-python and `v4.6.2` upload-artifact tags);
tool images pinned by digest; downloaded binaries checksum-verified; `concurrency` cancels superseded pull-request runs; no secrets are used (the
`.env` is copied from the public `.env.example`, whose values are throw-away development credentials); triggers: push to `main`, pull requests, weekly schedule,
and manual `workflow_dispatch`. The MQTT broker is started with `docker run` rather than a service container because the hardened image needs a mounted password file and
custom options, which service containers cannot express; readiness is polled instead of sleeping.

## What was validated locally, and what that does and does not prove
| Check | Result |
|---|---|
| `actionlint` (pinned image) on the workflow | clean |
| Pinned action SHAs resolve to the expected tags (GitHub API) | verified |
| `promtool check rules` | 16 rules valid |
| `docker compose config` plain and TLS profiles | valid |
| **Clean-copy emulation** `scripts/ci_local.sh` (copies only files git would track into a temporary directory, no `.venv`/`.env`/secrets, then runs the `static` and `tests` steps) | see "Findings"; final run recorded at the bottom |

The emulation runs on macOS/arm64 with Docker Desktop, not on an `ubuntu-24.04` GitHub runner. It cannot prove runner-specific behaviour (permissions, Docker
version, network egress, timing). It *can* and did catch repository problems that a developer's working tree hides.

### Findings from the first clean-copy run (2026-10-13)
The first run, which unintentionally used Python 3.14 (the host default), failed 10 tests:
* 8 TLS tests: **a real defect** - the development CA issued certificates without Subject/Authority Key Identifier, which Python >= 3.13 rejects. Fixed in `scripts/devcerts.py`; the same tests pass on 3.14 afterwards.
* 1 test (`test_secret_directory_is_gitignored`) needs a git repository: an artefact of the copy having no `.git`; the script now runs `git init` in the copy.
* 1 test (`test_missing_metric_is_rejected_over_http`) failed once in the full run and passed in isolation on 3.14. Not reproduced; watched in later runs (see below).
CI pins Python 3.12 (`actions/setup-python`), so these would not all have appeared on GitHub, but the project declares `requires-python >= 3.12`.

## Run it yourself
Locally (about 15 minutes, needs Docker and `uv`):
```bash
scripts/ci_local.sh
```
On GitHub:
```bash
git push origin main                  # CI starts automatically on every push to main and on pull requests
gh run list --limit 3                 # recent runs
gh run watch                          # follow the latest run
gh workflow run ci.yml                # manual re-run (workflow_dispatch)
``` Expected first-run risks: the `containers`,
`tls-profile` and `kubernetes-e2e` jobs use Docker builds and timing-dependent waits that have only run on Docker Desktop; Trivy's database may report new findings;
GitHub's runners are x86_64 while all local runs were arm64.

### Re-validation after the fixes
After the `devcerts.py` fix, the affected test files (`test_tls_mqtt.py`, `test_tls_https.py`, `test_api_quality.py`, `test_deploy_config.py`) passed on Python 3.14 in the clean copy (48 passed).
The complete clean-copy run on Python 3.12 (`scripts/ci_local.sh`, the closest local match to CI) was **not repeated** after the later engine and control-plane
changes; the full suite was run in the working tree instead (see `docs/status.md`). Run `scripts/ci_local.sh` before the first push.
`scripts/ci_local.sh` was generalised (temporary directory, `git init`, `UV_PYTHON=3.12`) after the run that produced the findings above and has only been syntax-checked in that final form.

## Run history
### Run 1 - first push (2026-10-13)
| Job | Result | Time |
|---|---|---|
| `static` | passed | 42 s |
| `kubernetes-manifests` | passed | 5 s |
| `tls-profile` | passed | 3 min |
| `containers` (build, Trivy, live stack, security checks, `demo.py all`, dashboard queries) | passed | 5 min |
| `tests` | **failed**: 8 failed, 355 passed, 10 min 45 s | |
| `kubernetes-e2e` | **failed** at the last step of the smoke test | 4 min |

Causes and fixes:
* `kubernetes-e2e`: the whole lifecycle worked (devices online, model trained, deployment COMPLETED on the pod); the final `/predict` call failed with `api token required`. The edge API requires its token and `scripts/k8s_smoke.py` did not send it (the script predates that protection and had not been re-run since). Fixed.
* `tests`: all 8 failures were `deployment ... in {'COMPLETED'}` timeouts in `test_api.py` and `test_api_quality.py`. In-process test agents report the whole pytest process's memory, which on the runner (after ~80 tests with coverage) exceeded the default 1 GB health limit, so rollouts rolled back. The same mechanism had been found locally in one test; the fix is now in the shared harness (`Stack.deploy` relaxes the memory limit unless a test sets its own) and failure messages now include the deployment's state, reason and per-device errors. This explanation fits the pattern and the earlier local reproduction, but the CI log did not record the rollback reason, so the next run is the confirmation.
* The suite is also slow on the runner (10 min 45 s vs ~6.5 min locally); the timeout budget (25 min) is sufficient.

### Run 2 - after the first fixes (2026-10-13)
| Job | Result |
|---|---|
| `static`, `kubernetes-manifests`, `containers` | passed |
| `kubernetes-e2e` | **passed** (confirms the edge-token fix; the whole lifecycle runs on a k3d cluster in CI) |
| `tls-profile` | failed: `verify_tls.py` asserted instantly that Prometheus had scraped every target; `edge-02` was still `down` right after start-up. A timing race in the check, not a TLS defect (the same job passed in run 1). The check now polls for up to 60 s and reports each target's last scrape error |
| `tests` | 1 failed, 362 passed (6 min 37 s): `test_better_candidate_passes_and_then_rolls_out_normally` posts its own deployment body, so the harness-level memory relaxation from run 1's fix did not reach it. The relaxation now lives in the harness's shared `api()` call so every deployment request gets it |
The run-1 explanation for the failing `tests` job is supported by this run: 7 of the 8 earlier failures disappeared with the harness change, and the failing test used a code path the change did not cover.
