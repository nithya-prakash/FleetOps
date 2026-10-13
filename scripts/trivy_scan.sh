#!/usr/bin/env bash
# Scan the built images and the dependency lockfile with Trivy (run from the repo root).
#
#   scripts/trivy_scan.sh                     # fail (exit 1) on fixable HIGH/CRITICAL findings
#   TRIVY_SEVERITY=MEDIUM,HIGH,CRITICAL scripts/trivy_scan.sh
#
# Policy (see docs/supply-chain.md): block HIGH and CRITICAL findings that have a fixed version available
# (--ignore-unfixed). Accepted exceptions live in .trivyignore, each with an expiry and a justification.
# "No findings" means "nothing known to this database at scan time" - not "vulnerability-free".
set -euo pipefail

TRIVY_IMAGE="${TRIVY_IMAGE:-aquasec/trivy:0.75.0@sha256:af6acf9a6b85dfe389a1941505c0ce9efef52a4719635e1a962f022a3d855daa}"
SEVERITY="${TRIVY_SEVERITY:-HIGH,CRITICAL}"
IMAGES=(${FLEETOPS_IMAGES:-fleetops-edge:dev fleetops-control-plane:dev fleetops-mlflow:dev fleetops-mosquitto:dev})

trivy() {
  docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v trivy-cache:/root/.cache \
    -v "$PWD":/work:ro -w /work "$TRIVY_IMAGE" "$@"
}

if [[ "${1:-}" == "--selftest" ]]; then
  # Prove the scanner can actually see lockfile vulnerabilities: scan a lock that pins a known-vulnerable package.
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT
  printf '[project]\nname = "selftest"\nversion = "0"\nrequires-python = ">=3.12"\ndependencies = ["urllib3==1.25.8"]\n' > "$work/pyproject.toml"
  (cd "$work" && uv lock >/dev/null)
  if docker run --rm -v trivy-cache:/root/.cache -v "$work":/w:ro -w /w "$TRIVY_IMAGE" fs --severity HIGH,CRITICAL \
       --exit-code 1 --scanners vuln uv.lock >"$work/out.txt" 2>&1; then
    echo "SELFTEST FAILED: a known-vulnerable lockfile produced no findings"; cat "$work/out.txt"; exit 2
  fi
  grep -E "Total:|CVE-" "$work/out.txt" | head -3
  echo "selftest ok: the scanner detects vulnerabilities in uv.lock"
  exit 0
fi

status=0
echo "== lockfile (uv.lock) =="
trivy fs --severity "$SEVERITY" --ignore-unfixed --ignorefile .trivyignore --exit-code 1 --scanners vuln \
  --skip-dirs .venv --skip-dirs data uv.lock || status=1
for img in "${IMAGES[@]}"; do
  echo "== image $img =="
  trivy image --severity "$SEVERITY" --ignore-unfixed --ignorefile .trivyignore --exit-code 1 --scanners vuln "$img" || status=1
done
exit $status
