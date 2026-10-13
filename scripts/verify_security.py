"""Assert the security posture of the *running* Compose stack. Exit code 1 if any check fails.

    python scripts/verify_security.py

Checks: loopback-only published ports, MLflow authentication/authorization, no credentials in container
environments, Mosquitto auth + password-file permissions + non-root, non-root application containers.
"""

from __future__ import annotations

import json
import os
import ssl
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent

# TLS profile (compose.tls.yaml): FLEETOPS_TLS=1 switches to https:// and verifies against the development CA.
TLS = os.environ.get("FLEETOPS_TLS") == "1"
SCHEME = "https" if TLS else "http"
VERIFY: ssl.SSLContext | bool = (
    ssl.create_default_context(cafile=str(ROOT / "deploy" / "secrets" / "tls" / "ca.crt")) if TLS else True
)
H = httpx.Client(verify=VERIFY, timeout=30)
ENV = {
    k.strip(): v.strip()
    for k, _, v in (ln.partition("=") for ln in (ROOT / ".env").read_text().splitlines() if "=" in ln and ln[0] != "#")
}
MLFLOW = f"{SCHEME}://127.0.0.1:{ENV.get('FLEETOPS_MLFLOW_PORT', '15000')}"
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{(' - ' + detail) if detail and not ok else ''}")
    if not ok:
        failures.append(name)


def sh(*args: str) -> str:
    """stdout and stderr combined (tools such as mosquitto_pub report refusals on stderr)."""
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.stdout + proc.stderr


def exec_in(service: str, *cmd: str) -> str:
    return sh("docker", "compose", "exec", "-T", service, *cmd)


# 1. published ports -----------------------------------------------------------------------------
rows = [json.loads(line) for line in sh("docker", "compose", "ps", "--format", "json").splitlines() if line.strip()]
published = [(r["Service"], p) for r in rows for p in r.get("Publishers") or [] if p.get("PublishedPort")]
check("some ports are published", bool(published))
for svc, p in published:
    check(f"{svc}:{p['PublishedPort']} bound to loopback", p["URL"] in ("127.0.0.1", "::1"), f"bound to {p['URL']}")

# 2. MLflow authentication / authorization -------------------------------------------------------
search = "/api/2.0/mlflow/experiments/search"
admin = ("admin", ENV["FLEETOPS_MLFLOW_ADMIN_PASSWORD"])
svc = (ENV.get("FLEETOPS_MLFLOW_USERNAME", "fleetops-cp"), ENV["FLEETOPS_MLFLOW_PASSWORD"])
body = {"max_results": 1}
check("mlflow rejects anonymous requests", H.post(MLFLOW + search, json=body).status_code == 401)
check("mlflow rejects wrong password", H.post(MLFLOW + search, auth=("admin", "wrong"), json=body).status_code == 401)
check("mlflow accepts the service account", H.post(MLFLOW + search, auth=svc, json=body).status_code == 200)
check("mlflow /health is reachable for probes", H.get(MLFLOW + "/health").status_code == 200)
new_user = {"username": "intruder", "password": "x" * 16}
code = H.post(MLFLOW + "/api/2.0/mlflow/users/create", auth=svc, json=new_user).status_code
check("service account cannot manage users (least privilege)", code in (401, 403), f"HTTP {code}")
check(
    "service account is not an admin",
    H.get(MLFLOW + "/api/2.0/mlflow/users/get", params={"username": svc[0]}, auth=svc)
    .json()
    .get("user", {})
    .get("is_admin")
    is False,
)

# 3. credentials must not be in container environments -------------------------------------------
secret_values = [
    ENV[k]
    for k in (
        "FLEETOPS_ADMIN_API_KEY",
        "FLEETOPS_OPERATOR_API_KEY",
        "FLEETOPS_VIEWER_API_KEY",
        "FLEETOPS_DEVICE_KEY_MASTER",
        "FLEETOPS_MQTT_PASSWORD",
        "FLEETOPS_MLFLOW_PASSWORD",
        "FLEETOPS_MLFLOW_ADMIN_PASSWORD",
        "GRAFANA_ADMIN_PASSWORD",
    )
]
inspected = sh("docker", "compose", "ps", "-q")
leaks = []
for cid in inspected.split():
    cfg = sh(
        "docker", "inspect", "--format", "{{json .Config.Env}} {{json .Config.Cmd}} {{json .Config.Entrypoint}}", cid
    )
    leaks += [v[:4] + "…" for v in secret_values if v in cfg]
check("no credential appears in container env/command (docker inspect)", not leaks, f"{leaks}")

# 4. Mosquitto -----------------------------------------------------------------------------------
perm = exec_in("mosquitto", "stat", "-c", "%a %U", "/mosquitto/run/passwd").strip()
check("mosquitto password file is 0600 and owned by mosquitto", perm == "600 mosquitto", perm)
check("mosquitto runs as non-root", exec_in("mosquitto", "id", "-u").strip() == "1883")
anon_args = ["--cafile", "/run/secrets/tls/ca.crt", "-p", "8883", "-h", "localhost"] if TLS else ["-h", "127.0.0.1"]
anon = exec_in("mosquitto", "mosquitto_pub", *anon_args, "-t", "x", "-m", "x")
check("mosquitto refuses anonymous clients", "not authorised" in anon.lower(), anon[:80])
ro = sh(
    "docker",
    "inspect",
    "--format",
    "{{.HostConfig.ReadonlyRootfs}} {{.HostConfig.CapDrop}}",
    sh("docker", "compose", "ps", "-q", "mosquitto").strip(),
).strip()
check("mosquitto root fs is read-only with capabilities dropped", ro.startswith("true") and "ALL" in ro, ro)

# 5. application containers ----------------------------------------------------------------------
for service in ("control-plane", "edge-01", "mlflow"):
    check(f"{service} runs as non-root", exec_in(service, "id", "-u").strip() == "10001")
    cid = sh("docker", "compose", "ps", "-q", service).strip()
    ro = sh("docker", "inspect", "--format", "{{.HostConfig.ReadonlyRootfs}} {{.HostConfig.CapDrop}}", cid).strip()
    check(f"{service} read-only rootfs and cap_drop ALL", ro.startswith("true") and "ALL" in ro, ro)

# 6. control-plane and agent APIs require credentials; scoped tokens stay scoped ---------------------------
CP = f"{SCHEME}://127.0.0.1:{ENV.get('FLEETOPS_CP_PORT', '18000')}"
metrics_tok, api_tok = ENV["FLEETOPS_METRICS_TOKEN"], ENV["FLEETOPS_EDGE_API_TOKEN"]


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


check("control plane /metrics rejects anonymous scrapes", H.get(CP + "/metrics").status_code == 401)
check(
    "control plane /metrics accepts the scrape token",
    H.get(CP + "/metrics", headers=bearer(metrics_tok)).status_code == 200,
)
check(
    "scrape token cannot call the management API",
    H.get(CP + "/api/v1/devices", headers=bearer(metrics_tok)).status_code == 401,
)
check("anonymous /readyz is minimal", H.get(CP + "/readyz").json() == {"status": "ready"})
check(
    "anonymous artifact download is refused",
    H.get(CP + "/api/v1/device/artifacts/anomaly-detector/1").status_code == 401,
)
for dev, port in (("edge-01", ENV.get("EDGE1_PORT", "19101")), ("edge-02", ENV.get("EDGE2_PORT", "19102"))):
    url = f"{SCHEME}://127.0.0.1:{port}"
    check(
        f"{dev} /predict rejects anonymous callers",
        H.post(url + "/predict", json={"features": [1.0]}).status_code == 401,
    )
    check(f"{dev} /healthz rejects anonymous callers", H.get(url + "/healthz").status_code == 401)
    check(f"{dev} /metrics rejects anonymous scrapes", H.get(url + "/metrics").status_code == 401)
    check(f"{dev} /livez is open and reveals nothing", H.get(url + "/livez").json() == {"status": "alive"})
    check(f"{dev} accepts the API token", H.get(url + "/healthz", headers=bearer(api_tok)).status_code == 200)
    check(
        f"{dev} scrape token reads metrics only",
        H.get(url + "/metrics", headers=bearer(metrics_tok)).status_code == 200
        and H.post(url + "/predict", json={"features": [1.0]}, headers=bearer(metrics_tok)).status_code == 401,
    )
    check(f"{dev} API docs are not exposed", H.get(url + "/openapi.json", headers=bearer(api_tok)).status_code == 404)

# 7. Prometheus really scrapes with the token (otherwise the dashboards would be empty) ----------------------
targets = H.get(f"http://127.0.0.1:{ENV.get('FLEETOPS_PROM_PORT', '19090')}/api/v1/targets", timeout=10).json()["data"][
    "activeTargets"
]
check(
    "all Prometheus targets are up with token authentication",
    bool(targets) and all(t["health"] == "up" for t in targets),
    str([(t["labels"]["instance"], t["health"]) for t in targets if t["health"] != "up"]),
)

# 8. credentials in process environments / command lines / logs --------------------------------------------
# MLflow offers no file option for the Flask session key, so that single value is the documented exception.
flask_secret = ENV["FLEETOPS_MLFLOW_FLASK_SECRET"]
all_secrets = [*secret_values, metrics_tok, api_tok]
for device in ("edge-01", "edge-02", "edge-03"):
    all_secrets.append((ROOT / "deploy" / "secrets" / f"{device}.secret").read_text().strip())
for service in ("control-plane", "mlflow", "edge-01", "mosquitto"):
    dump = exec_in(service, "sh", "-c", "cat /proc/[0-9]*/environ /proc/[0-9]*/cmdline 2>/dev/null | tr '\\0' '\\n'")
    leaked = [s[:4] + "…" for s in all_secrets if s in dump]
    check(f"{service}: no credential in any process environment or command line", not leaked, str(leaked))
mlflow_dump = exec_in("mlflow", "sh", "-c", "cat /proc/[0-9]*/environ 2>/dev/null | tr '\\0' '\\n'")
check("mlflow: only the documented Flask session key is in the environment", flask_secret in mlflow_dump)
logs = sh("docker", "compose", "logs", "--no-color")
leaked_logs = [s[:4] + "…" for s in {*all_secrets, flask_secret} if s in logs]
check("no credential appears in any container log", not leaked_logs, str(leaked_logs))

print(f"\n{len(failures)} failed" if failures else "\nall security checks passed")
sys.exit(1 if failures else 0)
