"""Drive the running Compose stack through the full lifecycle (uses only the public APIs).

python scripts/demo.py bootstrap     # train v1 + v2, register edge-01..03, wait until online
python scripts/demo.py deploy-v1     # roll out the baseline to the fleet
python scripts/demo.py failure       # v2 with a deterministic canary failure -> automatic rollback
python scripts/demo.py gate          # quality gate blocks regressing candidates before any device is touched
python scripts/demo.py promote-v2    # healthy v2 rollout under an explicit quality policy (clears faults first)
python scripts/demo.py offline       # cut edge-02's network, show local inference, restore
python scripts/demo.py status        # fleet + deployments summary
python scripts/demo.py all           # everything above, in order
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent

# TLS profile (compose.tls.yaml): FLEETOPS_TLS=1 switches to https:// and verifies against the development CA.
TLS = os.environ.get("FLEETOPS_TLS") == "1"
SCHEME = "https" if TLS else "http"
VERIFY: ssl.SSLContext | bool = (
    ssl.create_default_context(cafile=str(ROOT / "deploy" / "secrets" / "tls" / "ca.crt")) if TLS else True
)
DEVICES = ["edge-01", "edge-02", "edge-03"]
EDGE_PORTS = {"edge-01": 19101, "edge-02": 19102, "edge-03": 19103}
MODEL = "anomaly-detector"
HEALTHY = [2.0, 65.0, 6.0, 12.0, 1480.0, 70.0]
CRITERIA = {"observation_seconds": 5, "min_inferences": 15, "ack_timeout_seconds": 60, "health_timeout_seconds": 90}


def env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return values


E = env()
BASE = f"{SCHEME}://127.0.0.1:{E.get('FLEETOPS_CP_PORT', '18000')}"


def api(method: str, path: str, role: str = "operator", **kw: Any) -> httpx.Response:
    key = E[f"FLEETOPS_{role.upper()}_API_KEY"]
    return httpx.request(
        method, BASE + path, headers={"Authorization": f"Bearer {key}"}, verify=VERIFY, timeout=120, **kw
    )


def edge_secret(device: str) -> str:
    return (ROOT / "deploy" / "secrets" / f"{device}.secret").read_text().strip()


def edge(device: str, method: str, path: str, **kw: Any) -> httpx.Response:
    if path.startswith("/admin"):
        headers = {"Authorization": f"Bearer {edge_secret(device)}"}  # device credential
    elif path == "/livez":
        headers = {}
    else:  # the agent API token that a PLC / gateway client would hold
        token = (ROOT / "deploy" / "secrets" / "fleetops_edge_api_token").read_text().strip()
        headers = {"Authorization": f"Bearer {token}"}
    return httpx.request(
        method, f"{SCHEME}://127.0.0.1:{EDGE_PORTS[device]}{path}", headers=headers, verify=VERIFY, timeout=10, **kw
    )


def say(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def wait(cond: Any, what: str, timeout: float = 120.0) -> Any:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        res = cond()
        if res:
            return res
        time.sleep(1)
    raise SystemExit(f"timed out waiting for {what}")


def versions() -> list[int]:
    r = api("GET", f"/api/v1/models/{MODEL}/versions", "viewer")
    return [v["version"] for v in r.json()["items"]] if r.status_code == 200 else []


def bootstrap() -> None:
    wait(lambda: httpx.get(BASE + "/readyz", verify=VERIFY, timeout=5).status_code == 200, "control plane ready")
    existing = versions()
    for seed, label in ((1, "baseline v1"), (2, "candidate v2")):
        if len(existing) >= seed:
            continue
        say(f"training {label} (seed={seed}) and registering in MLflow")
        r = api("POST", "/api/v1/models/train", json={"model_name": MODEL, "params": {"seed": seed}})
        r.raise_for_status()
        v = r.json()
        print(
            f"  registered {v['name']}:{v['version']} sha256={v['sha256'][:16]}… test_f1={v['metrics']['test_f1']:.2f} "
            f"roc_auc={v['metrics']['test_roc_auc']:.2f}  (synthetic data)"
        )
    say("registering devices (group line-a)")
    for d in DEVICES:
        r = api("POST", "/api/v1/devices/register", "admin", json={"device_id": d, "group": "line-a"})
        outcome = {201: "registered", 409: "already registered"}.get(r.status_code, r.text)
        print(f"  {d}: {outcome}")
    wait(
        lambda: all(api("GET", f"/api/v1/devices/{d}", "viewer").json()["status"] == "ONLINE" for d in DEVICES),
        "devices online",
    )
    status()


def deploy(version: int, **extra: Any) -> dict[str, Any]:
    body = {"model_name": MODEL, "model_version": version, "group": "line-a", "health_criteria": CRITERIA, **extra}
    r = api("POST", "/api/v1/deployments", json=body)
    r.raise_for_status()
    dep = r.json()
    print(f"  deployment {dep['deployment_id'][:8]} created: batches={dep['batches']} skipped={dep['skipped']}")
    last = ""

    def done() -> bool:
        nonlocal last
        d = api("GET", f"/api/v1/deployments/{dep['deployment_id']}", "viewer").json()
        line = f"{d['state']:12} " + " ".join(f"{t['device_id']}={t['state']}" for t in d["targets"])
        if line != last:
            print("  " + line, flush=True)
            last = line
        return d["state"] in ("COMPLETED", "ROLLED_BACK", "FAILED", "CANCELLED")

    wait(done, "deployment to finish")
    return dict(api("GET", f"/api/v1/deployments/{dep['deployment_id']}", "viewer").json())


def show_models() -> None:
    for d in DEVICES:
        h = edge(d, "GET", "/healthz").json()
        p = edge(d, "POST", "/predict", json={"features": HEALTHY})
        pred = p.json() if p.status_code == 200 else {"error": p.status_code}
        active = h["active"]
        label = f"{active['name']}:{active['version']}" if active else None
        print(f"  {d}: active={label} previous={h['previous']['version'] if h['previous'] else None} predict={pred}")


def deploy_v1() -> None:
    say("rolling out baseline v1 to line-a (canary: 1 device, then the rest)")
    final = deploy(1, strategy={"canary_count": 1, "batch_percent": 100})
    assert final["state"] == "COMPLETED" and len(final["targets"]) == len(DEVICES), final
    say("local inference on every edge agent")
    show_models()


def clear_faults() -> None:
    for d in DEVICES:
        api("POST", f"/api/v1/devices/{d}/simulation", "admin", json={})


def failure() -> None:
    say("injecting deterministic fault: canary edge-01 will fail inference on v2")
    api("POST", "/api/v1/devices/edge-01/simulation", "admin", json={"fail_inference_versions": [2]}).raise_for_status()
    time.sleep(3)
    say("rolling out candidate v2 (canary first)")
    final = deploy(2, strategy={"canary_count": 1, "batch_percent": 100})
    print(f"  final state: {final['state']} - {final['reason']}")
    assert final["state"] == "ROLLED_BACK", final
    say("verify: rollback was a real model switch; untouched devices never left v1")
    show_models()
    say("audit trail for this deployment")
    ev = api("GET", f"/api/v1/audit/events?deployment_id={final['deployment_id']}&limit=100", "viewer").json()["items"]
    for e in reversed(ev):
        if e["action"].startswith("deployment.") or e["action"] in ("target.health_failed",):
            print(f"  {e['ts']} {e['action']:28} {json.dumps(e['details'])[:110]}")


def bench(version: int) -> dict[str, float]:
    items = api("GET", f"/api/v1/models/{MODEL}/versions", "viewer").json()["items"]
    return {
        k: v for k, v in next(i for i in items if i["version"] == version)["metrics"].items() if k.startswith("bench_")
    }


def show_failure(resp: httpx.Response) -> dict[str, Any]:
    err = resp.json()["error"]
    print(f"  HTTP {resp.status_code} {err['code']}")
    for f in err["details"]["failures"]:
        print(f"    - [{f['code']}] {f['message']}")
    return dict(err)


def gate() -> None:
    say("benchmark metrics (same fixed evaluation suite for every model, so they are comparable)")
    for v in (1, 2):
        print(
            f"  v{v}: "
            + " ".join(f"{k.removeprefix('bench_')}={x:.3f}" for k, x in bench(v).items() if "latency" not in k)
        )
    before = {d: edge(d, "GET", "/healthz").json()["active"]["version"] for d in DEVICES}
    strict = {"metrics": {"bench_f1": {"max_regression": 0.02}, "bench_recall": {"max_regression": 0.02}}}
    say(
        "v2 vs the running v1 with a strict policy (recall may drop at most 0.02): blocked before any device is touched"
    )
    r = api(
        "POST",
        "/api/v1/deployments",
        json={"model_name": MODEL, "model_version": 2, "group": "line-a", "quality_gate": strict},
    )
    assert r.status_code == 422, r.text
    show_failure(r)
    say("training a deliberately weak candidate (v3) and gating it")
    t = api(
        "POST",
        "/api/v1/models/train",
        json={
            "model_name": MODEL,
            "params": {"seed": 3, "epochs": 1, "hidden_sizes": [2], "learning_rate": 0.0001, "n_samples": 800},
        },
    )
    t.raise_for_status()
    v3 = t.json()["version"]
    r = api(
        "POST",
        "/api/v1/deployments",
        json={
            "model_name": MODEL,
            "model_version": v3,
            "group": "line-a",
            "quality_gate": {"metrics": {"bench_f1": {"max_regression": 0.02}}},
        },
    )
    assert r.status_code == 422, r.text
    show_failure(r)
    say("an operator cannot override the gate; only an admin can, with an audited reason")
    body = {
        "model_name": MODEL,
        "model_version": v3,
        "group": "line-a",
        "quality_gate": strict,
        "quality_override": {"reason": "demonstration only - not deployed"},
    }
    r = api("POST", "/api/v1/deployments", "operator", json=body)
    print(f"  operator override -> HTTP {r.status_code} {r.json()['error']['code']}")
    assert r.status_code == 403
    after = {d: edge(d, "GET", "/healthz").json()["active"]["version"] for d in DEVICES}
    assert before == after, (before, after)
    print(f"  devices unchanged: {after}")


def promote_v2() -> None:
    say("clearing injected faults, then rolling out v2 under an explicit quality policy (canary, then 50% batches)")
    clear_faults()
    time.sleep(3)
    policy = {
        "metrics": {
            "bench_f1": {"max_regression": 0.02},
            "bench_recall": {"max_regression": 0.15},
            "bench_false_positive_rate": {"max_regression": 0.0},
        }
    }
    final = deploy(2, strategy={"canary_count": 1, "batch_percent": 50}, quality_gate=policy)
    print(
        f"  quality gate: passed={final['quality']['passed']} compared_against=v{final['quality']['compared_against']}"
    )
    assert final["state"] == "COMPLETED" and len(final["targets"]) == len(DEVICES), final
    show_models()


def offline() -> None:
    say("cutting edge-02's network (MQTT + control-plane HTTP) - simulated outage")
    edge("edge-02", "POST", "/admin/network", json={"connected": False}).raise_for_status()
    for i in range(3):
        p = edge("edge-02", "POST", "/predict", json={"features": HEALTHY}).json()
        print(f"  local inference during outage #{i + 1}: {p}")
    wait(
        lambda: api("GET", "/api/v1/devices/edge-02/health", "viewer").json()["connectivity"] == "offline",
        "offline detection",
        60,
    )
    print(
        "  control plane view:",
        json.dumps(
            {
                k: v
                for k, v in api("GET", "/api/v1/devices/edge-02/health", "viewer").json().items()
                if k in ("connectivity", "assessment", "reasons")
            }
        ),
    )
    say("restoring network; device state reconciles")
    edge("edge-02", "POST", "/admin/network", json={"connected": True}).raise_for_status()
    wait(lambda: api("GET", "/api/v1/devices/edge-02", "viewer").json()["status"] == "ONLINE", "device back online", 60)
    print(
        "  control plane view:",
        json.dumps(api("GET", "/api/v1/devices/edge-02/health", "viewer").json()["active_model"]),
    )


def status() -> None:
    say("fleet")
    for d in api("GET", "/api/v1/devices", "viewer").json()["items"]:
        print(f"  {d['device_id']}: {d['status']:8} model={d['active_model_name']}:{d['active_model_version']}")
    say("deployments")
    for d in api("GET", "/api/v1/deployments", "viewer").json()["items"]:
        print(f"  {d['deployment_id'][:8]} v{d['model_version']} {d['state']:12} {d['reason'] or ''}")


PROM_URL = os.environ.get("FLEETOPS_PROM_URL", "http://127.0.0.1:19090")
OBSERVATIONS = {
    "deployments by final state": "sum by (state) (fleetops_deployments_total)",
    "rollbacks by trigger": "sum by (trigger) (fleetops_rollbacks_total)",
    "per-device outcomes": "sum by (outcome) (fleetops_deployment_targets_total)",
    "devices connected": "sum(fleetops_device_connected)",
    "active model version per device": "fleetops_device_model_version",
    "mqtt messages processed": "sum by (channel) (fleetops_mqtt_messages_total)",
    "artifact downloads": "sum by (result) (fleetops_artifact_downloads_total)",
}


def observe() -> None:
    """What Prometheus (and therefore the Grafana dashboard) saw during the demo - read from the real server."""
    grafana = os.environ.get("FLEETOPS_GRAFANA_PORT", "13000")
    say(f"Prometheus observations ({PROM_URL}; dashboard: http://127.0.0.1:{grafana})")
    hidden = ("__name__", "group", "job", "instance")
    for title, query in OBSERVATIONS.items():
        data = httpx.get(f"{PROM_URL}/api/v1/query", params={"query": query}, timeout=10).json()["data"]["result"]
        rows = ", ".join(
            ",".join(v for k, v in r["metric"].items() if k not in hidden) + f"={r['value'][1]}" for r in data
        )
        print(f"  {title}: {rows or 'no data'}")


COMMANDS = {
    "bootstrap": bootstrap,
    "deploy-v1": deploy_v1,
    "failure": failure,
    "gate": gate,
    "promote-v2": promote_v2,
    "offline": offline,
    "status": status,
    "observe": observe,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=[*COMMANDS, "all"])
    cmd = ap.parse_args().command
    for fn in COMMANDS.values() if cmd == "all" else [COMMANDS[cmd]]:
        fn()
    sys.exit(0)
