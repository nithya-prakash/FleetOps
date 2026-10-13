# Supply chain: reproducible builds and vulnerability scanning

## What is pinned, and how it is verified
| Layer | Pin | Verified by |
|---|---|---|
| Python dependencies | `uv.lock` (every transitive package, with hashes) | `uv lock --check` (CI); `pip install --require-hashes` in every image build |
| Local dev environment | `uv sync --frozen` from the same lock | `make init` |
| Container base images | `ARG ...@sha256:` in the `Dockerfile` (Python, uv, Mosquitto); `image: ...@sha256:` in `compose.yaml` (Prometheus, Grafana) | `tests/unit/test_deploy_config.py` fails if any pin is missing |
| Scanner | Trivy image pinned by version **and** digest in `scripts/trivy_scan.sh` | same test |
| GitHub Actions | pinned to full commit SHAs (version in a comment) | same test |
| Downloaded CI tools (k3d, kubeconform) | version + SHA-256 checked in the workflow | CI fails on mismatch |

On Linux (containers, CI) PyTorch resolves from the CPU-only index so images do not pull CUDA wheels; macOS uses the regular
wheel. The project itself is copied into the image (no build backend is downloaded at build time). Python >= 3.12 is
required (NumPy 2.5 dropped 3.11).

## Updating
* **Python packages**: edit `pyproject.toml` (or `uv lock --upgrade-package <name>`), run `make lock`, run `make lint test`, commit `uv.lock`.
* **Base images**: `make digests` prints the registry digests for the pinned tags. Review upstream notes, change the tag + digest in the
  `Dockerfile`/`compose.yaml` (and `scripts/update_digests.sh`), rebuild, then `make scan test`. Dependabot (uv, docker, github-actions;
  `.github/dependabot.yml`) opens weekly PRs that follow the same checklist.
* **Why a weekly scheduled CI run**: new CVEs are published without any code change, so the scan also runs on a cron.

## Vulnerability scanning
`make scan` (= `scripts/trivy_scan.sh`, also run in CI) scans `uv.lock` and the four built images (OS packages **and** Python packages).

* **Policy**: fail on **HIGH/CRITICAL findings that have a fix available** (`--ignore-unfixed`). Unfixed findings are not hidden by
  accident: run `TRIVY_SEVERITY=LOW,MEDIUM,HIGH,CRITICAL scripts/trivy_scan.sh` (or drop `--ignore-unfixed` locally) to see them.
* **Exceptions**: only in `.trivyignore`, one ID per line with an `exp:YYYY-MM-DD` expiry (a test rejects entries without one). Write the
  justification in a comment above the entry. There are currently none.
* **Self-test**: `scripts/trivy_scan.sh --selftest` scans a deliberately vulnerable lockfile and fails unless the scanner reports
  findings, so a scanner that silently stops parsing `uv.lock` is detected. (This was needed: Trivy 0.58 did not parse `uv.lock` at all and
  reported "no findings"; 0.75 does.)

### What a clean result does and does not mean
"0 findings" means *no fixable HIGH/CRITICAL vulnerability known to Trivy's database on the scan date*. It does not mean the images are
vulnerability-free: unfixed issues are excluded, MEDIUM/LOW are not gating, the database lags disclosures, and application-logic flaws are
out of scope. Treat it as one control among several (digest pinning, non-root/read-only containers, least-privilege credentials, review).

Results when this was last run (2026-10-12, Trivy 0.75.0): `uv.lock` and the edge, control-plane and MLflow images (Debian 13.7) and the
Mosquitto image (Alpine 3.23.6): 0 fixable HIGH/CRITICAL. An earlier scan of Mosquitto 2.0.20 (Alpine 3.20) found 2 CRITICAL + 16 HIGH
(openssl, musl, zlib, cJSON) with fixes available; the fix was moving the pin to 2.0.22 - evidence that the gate works.
