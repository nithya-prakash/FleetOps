"""Load and soak scenarios against the RUNNING Compose stack (docker compose up -d --wait; python scripts/demo.py bootstrap).

    python -m scripts.load_test.run predict  --concurrency 1,4,16 --duration 15
    python -m scripts.load_test.run api      --concurrency 4,16 --duration 15
    python -m scripts.load_test.run ingest   --devices 100 --duration 30
    python -m scripts.load_test.run deploy-scale --sizes 3,10,30,100
    python -m scripts.load_test.run restart  --devices 50
    python -m scripts.load_test.run soak     --devices 50 --duration 600

Raise the stack's rate limits for capacity runs (they are protections, not capacity):
    FLEETOPS_RATE_LIMIT_PER_SECOND=5000 FLEETOPS_RATE_LIMIT_BURST=10000 FLEETOPS_EDGE_PREDICT_RATE_PER_SECOND=5000 \
    FLEETOPS_EDGE_PREDICT_BURST=10000 docker compose up -d --force-recreate control-plane edge-01 edge-02 edge-03
Every run writes JSON (with the environment) to load_results/ (git-ignored) and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from fleetops.training.sensors import generate_arrays
from scripts.load_test.common import (
    CP_URL,
    EDGE_URL,
    ROOT,
    VERIFY,
    StatsSampler,
    api_client,
    bucket_delta,
    histogram,
    metrics_text,
    percentiles,
    quantile_from_buckets,
    sample,
    save,
    secret_file,
    wait_for,
)
from scripts.load_test.soak import evaluate
from scripts.load_test.timeline import fetch_events, timeline

DATASET_SEED = 7  # deterministic request payloads for every run
ROWS = [row.tolist() for row in generate_arrays(256, 0.06, DATASET_SEED)[0]]
MODEL = "anomaly-detector"


def ms(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 1000:.2f}"


# ----------------------------------------------------------------------------------------------- generic closed-loop driver


def closed_loop(workers: int, duration: float, make_client: Any, request: Any) -> dict[str, Any]:
    """``workers`` threads issue requests back to back for ``duration`` s. ``request(client, i)`` -> status code."""
    latencies: list[float] = []
    status: dict[str, int] = {}
    lock = threading.Lock()
    stop_at = time.perf_counter() + duration

    def work(worker: int) -> None:
        client = make_client()
        i = worker
        local_lat: list[float] = []
        local_status: dict[str, int] = {}
        while time.perf_counter() < stop_at:
            t0 = time.perf_counter()
            try:
                code = str(request(client, i))
            except httpx.HTTPError as exc:
                code = type(exc).__name__
            local_lat.append(time.perf_counter() - t0)
            local_status[code] = local_status.get(code, 0) + 1
            i += workers
        client.close()
        with lock:
            latencies.extend(local_lat)
            for k, v in local_status.items():
                status[k] = status.get(k, 0) + v

    started = time.perf_counter()
    threads = [threading.Thread(target=work, args=(w,)) for w in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - started
    total = len(latencies)
    errors = sum(v for k, v in status.items() if not k.startswith("2") and k != "429")
    return {
        "workers": workers,
        "requests": total,
        "seconds": round(elapsed, 2),
        "throughput_rps": round(total / elapsed, 1),
        "latency_ms": {k: round(v * 1000, 3) for k, v in percentiles(latencies).items()},
        "mean_ms": round(statistics.fmean(latencies) * 1000, 3) if latencies else None,
        "status": status,
        "rate_limited_429": status.get("429", 0),
        "error_rate": round(errors / total, 5) if total else None,
    }


# ----------------------------------------------------------------------------------------------- predict (edge inference API)


def cmd_predict(a: argparse.Namespace) -> None:
    device = a.device
    url = EDGE_URL[device]
    token = secret_file("fleetops_edge_api_token")
    headers = {"Authorization": f"Bearer {token}"}

    def edge_metrics() -> str:
        return httpx.get(
            url + "/metrics",
            headers={"Authorization": f"Bearer {secret_file('fleetops_edge_metrics_token')}"},
            verify=VERIFY,
            timeout=10,
        ).text

    def make_client() -> httpx.Client:
        return httpx.Client(base_url=url, headers=headers, verify=VERIFY, timeout=10)

    def request(client: httpx.Client, i: int) -> int:
        return client.post("/predict", json={"features": ROWS[i % len(ROWS)]}).status_code

    results = []
    for c in [int(x) for x in a.concurrency.split(",")]:
        before = histogram(edge_metrics(), "fleetops_edge_inference_latency_seconds")
        with StatsSampler([device], interval=2.0) as stats:
            r = closed_loop(c, a.duration, make_client, request)
        after = histogram(edge_metrics(), "fleetops_edge_inference_latency_seconds")
        delta = bucket_delta(after, before)
        r["server_inference_ms"] = {
            f"p{q}": (round(v * 1000, 4) if (v := quantile_from_buckets(delta, q / 100)) is not None else None)
            for q in (50, 95, 99)
        }
        r["resources"] = stats.summary()
        results.append(r)
    path = save(
        "predict", {"device": device, "payload_dataset": f"synthetic seed {DATASET_SEED}, 256 rows", "results": results}
    )
    print(f"\n### /predict on {device} (closed loop, {a.duration}s per level)\n")
    print(
        "| clients | rps | client p50 ms | p95 | p99 | server-side inference p50/p95/p99 ms | 429 | errors | agent CPU avg/max % | RSS MB |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        res = r["resources"].get(device, {})
        s = r["server_inference_ms"]
        print(
            f"| {r['workers']} | {r['throughput_rps']} | {r['latency_ms']['p50']} | {r['latency_ms']['p95']} | {r['latency_ms']['p99']} | "
            f"{s['p50']}/{s['p95']}/{s['p99']} | {r['rate_limited_429']} | {r['error_rate']} | "
            f"{res.get('cpu_avg_pct', float('nan')):.0f}/{res.get('cpu_max_pct', float('nan')):.0f} | {res.get('mem_max_mb', float('nan')):.0f} |"
        )
    print(f"\nsaved {path}")


# ----------------------------------------------------------------------------------------------- management API latency


def cmd_api(a: argparse.Namespace) -> None:
    paths = [
        "/api/v1/devices?limit=50",
        "/api/v1/deployments?limit=20",
        "/api/v1/models",
        "/api/v1/audit/events?limit=20",
        "/api/v1/devices/edge-01/health",
        "/api/v1/monitoring/drift?limit=5",
    ]

    def make_client() -> httpx.Client:
        return api_client("viewer")

    def request(client: httpx.Client, i: int) -> int:
        return client.get(paths[i % len(paths)]).status_code

    results = []
    for c in [int(x) for x in a.concurrency.split(",")]:
        before = histogram(metrics_text(), "fleetops_mqtt_processing_seconds")
        with StatsSampler(["control-plane"], interval=2.0) as stats:
            r = closed_loop(c, a.duration, make_client, request)
        r["resources"] = stats.summary()
        results.append(r)
        _ = before
    path = save("api", {"endpoints": paths, "results": results})
    print(f"\n### Management API (read mix over {len(paths)} endpoints, closed loop, {a.duration}s per level)\n")
    print("| clients | rps | p50 ms | p95 | p99 | 429 | errors | CP CPU avg/max % | CP RSS MB |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        res = r["resources"].get("control-plane", {})
        print(
            f"| {r['workers']} | {r['throughput_rps']} | {r['latency_ms']['p50']} | {r['latency_ms']['p95']} | {r['latency_ms']['p99']} | "
            f"{r['rate_limited_429']} | {r['error_rate']} | {res.get('cpu_avg_pct', float('nan')):.0f}/{res.get('cpu_max_pct', float('nan')):.0f} | {res.get('mem_max_mb', float('nan')):.0f} |"
        )
    print(f"\nsaved {path}")


# ----------------------------------------------------------------------------------------------- device farm control


class FarmHandle:
    def __init__(self, count: int, procs: int | None = None, hb: float = 2.0, sensor_hz: float = 1.0) -> None:
        self.count, self.hb, self.sensor_hz = count, hb, sensor_hz
        self.procs = procs or max(1, min(8, count // 40 + 1))
        self.proc: subprocess.Popen[bytes] | None = None
        self.time_to_online: float | None = None

    def online(self) -> int:
        with api_client("viewer") as c:
            return int(
                c.get("/api/v1/devices", params={"group": "load", "status": "ONLINE", "limit": 1}).json()["total"]
            )

    def __enter__(self) -> FarmHandle:
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.load_test.farm",
                "--devices",
                str(self.count),
                "--procs",
                str(self.procs),
                "--hb",
                str(self.hb),
                "--sensor-hz",
                str(self.sensor_hz),
            ],
            cwd=ROOT,
        )
        started = time.monotonic()
        ok = wait_for(lambda: self.online() >= self.count, timeout=60 + self.count * 0.5, interval=1.0)
        if ok is None:
            raise SystemExit(f"only {self.online()}/{self.count} devices came online")
        self.time_to_online = time.monotonic() - started
        return self

    def __exit__(self, *exc: object) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def ensure_models() -> tuple[int, int]:
    with api_client("viewer") as c:
        items = c.get(f"/api/v1/models/{MODEL}/versions").json()["items"]
    versions = sorted(v["version"] for v in items)
    if len(versions) < 2:
        raise SystemExit("train at least two model versions first: python scripts/demo.py bootstrap")
    return versions[0], versions[1]


# ----------------------------------------------------------------------------------------------- MQTT ingestion


def cmd_ingest(a: argparse.Namespace) -> None:
    with (
        FarmHandle(a.devices, hb=a.hb, sensor_hz=a.sensor_hz) as farm,
        StatsSampler(["control-plane", "mosquitto"], interval=2.0) as stats,
    ):
        time.sleep(5)  # settle
        m0, t0 = metrics_text(), time.time()
        api_lat: list[float] = []
        end = time.time() + a.duration
        with api_client("viewer") as c:
            while time.time() < end:  # a human-style probe running during ingestion
                s = time.perf_counter()
                c.get("/api/v1/devices?limit=20")
                api_lat.append(time.perf_counter() - s)
                time.sleep(0.25)
        m1, t1 = metrics_text(), time.time()
        online = farm.online()
        ttonline = farm.time_to_online
    secs = t1 - t0
    channels = ["heartbeat", "telemetry", "predictions", "status"]
    msgs = {
        c: (
            sample(m1, "fleetops_mqtt_messages_total", channel=c)
            - sample(m0, "fleetops_mqtt_messages_total", channel=c)
        )
        / secs
        for c in channels
    }
    fails = sum(
        sample(m1, "fleetops_mqtt_processing_failures_total") - sample(m0, "fleetops_mqtt_processing_failures_total")
        for _ in [0]
    )
    proc = {}
    for chan in ("heartbeat", "predictions"):
        d = bucket_delta(
            histogram(m1, "fleetops_mqtt_processing_seconds", channel=chan),
            histogram(m0, "fleetops_mqtt_processing_seconds", channel=chan),
        )
        proc[chan] = {f"p{q}": quantile_from_buckets(d, q / 100) for q in (50, 95, 99)}
    lag = bucket_delta(
        histogram(m1, "fleetops_mqtt_ingest_lag_seconds", channel="heartbeat"),
        histogram(m0, "fleetops_mqtt_ingest_lag_seconds", channel="heartbeat"),
    )
    lag_q = {f"p{q}": quantile_from_buckets(lag, q / 100) for q in (50, 95, 99)}
    result = {
        "devices": a.devices,
        "heartbeat_interval_s": a.hb,
        "sensor_hz": a.sensor_hz,
        "seconds": round(secs, 1),
        "time_to_all_online_s": round(ttonline or 0, 1),
        "online_at_end": online,
        "messages_per_s": {k: round(v, 1) for k, v in msgs.items()},
        "processing_s": proc,
        "ingest_lag_s": lag_q,
        "failures_in_window": fails,
        "api_probe_ms": {k: round(v * 1000, 2) for k, v in percentiles(api_lat).items()},
        "resources": stats.summary(),
    }
    path = save("ingest", result)
    print(
        f"\n### MQTT ingestion: {a.devices} devices, heartbeat every {a.hb}s, {a.sensor_hz} Hz inference ({secs:.0f}s window)\n"
    )
    print(
        f"- all online after {result['time_to_all_online_s']}s; online at end: {online}/{a.devices}; failures in window: {fails:.0f}"
    )
    print("- uplink messages/s: " + ", ".join(f"{k} {v}" for k, v in result["messages_per_s"].items()))
    for chan, q in proc.items():
        print(f"- processing {chan}: p50 {ms(q['p50'])} / p95 {ms(q['p95'])} / p99 {ms(q['p99'])} ms")
    print(
        f"- heartbeat ingest lag (device send -> processed): p50 {ms(lag_q['p50'])} / p95 {ms(lag_q['p95'])} / p99 {ms(lag_q['p99'])} ms"
    )
    print(f"- API probe during ingestion: {result['api_probe_ms']}")
    for svc, r in result["resources"].items():
        print(f"- {svc}: CPU avg {r['cpu_avg_pct']:.0f}% max {r['cpu_max_pct']:.0f}%, RSS max {r['mem_max_mb']:.0f} MB")
    print(f"\nsaved {path}")


# ----------------------------------------------------------------------------------------------- deployment duration vs fleet size


def cmd_deploy_scale(a: argparse.Namespace) -> None:
    sizes = [int(x) for x in a.sizes.split(",")]
    v1, v2 = ensure_models()
    rows = []
    with (
        FarmHandle(max(sizes), hb=a.hb, sensor_hz=a.sensor_hz) as farm,
        api_client("operator", timeout=120) as op,
        StatsSampler(["control-plane"], interval=2.0) as stats,
    ):
        ids = [f"load-{i:04d}" for i in range(1, max(sizes) + 1)]
        crit = {
            "observation_seconds": 2,
            "min_inferences": 2,
            "ack_timeout_seconds": 60,
            "health_timeout_seconds": 120,
            "max_cpu_percent": 100,
        }
        # baseline: put every device on v1 first so each measured rollout is a real v1 <-> v2 switch
        target = v1
        for n in [max(sizes), *sizes]:
            for version in [target] if n != max(sizes) or target != v1 else [v1]:
                body = {
                    "model_name": MODEL,
                    "model_version": version,
                    "group": "load",
                    "device_ids": ids[:n],
                    "strategy": {"canary_count": 1, "batch_percent": 100},
                    "health_criteria": crit,
                }
                started = time.monotonic()
                r = op.post("/api/v1/deployments", json=body)
                if r.status_code not in (200, 201):
                    rows.append({"devices": n, "version": version, "error": r.text[:200]})
                    continue
                dep = r.json()["deployment_id"]
                state = wait_for(
                    lambda dep=dep: (
                        op.get(f"/api/v1/deployments/{dep}").json()["state"] in ("COMPLETED", "ROLLED_BACK", "FAILED")
                    ),
                    600,
                    0.5,
                )
                final = op.get(f"/api/v1/deployments/{dep}").json()
                rows.append(
                    {
                        "devices": n,
                        "version": version,
                        "state": final["state"],
                        "seconds": round(time.monotonic() - started, 2),
                        "targets": len(final["targets"]),
                        "skipped": len(r.json().get("skipped", {})),
                        "timeout": state is None,
                    }
                )
            target = v2 if target == v1 else v1
        _ = farm
    result = {"sizes": sizes, "rows": rows, "resources": stats.summary()}
    path = save("deploy-scale", result)
    print("\n### Deployment duration vs fleet size (canary of 1, then the rest in one batch; health window 2 s)\n")
    print("| devices | to version | final state | seconds |")
    print("|---|---|---|---|")
    for entry in rows:
        print(
            f"| {entry['devices']} | v{entry['version']} | {entry.get('state', entry.get('error'))} | {entry.get('seconds', '-')} |"
        )
    print(f"\nsaved {path}")


def cmd_warmup(a: argparse.Namespace) -> None:
    """Cold vs warm rollouts to a settled farm, with a per-stage server-side timeline of each (docs/load-test-results.md)."""
    v1, v2 = ensure_models()
    crit = {
        "observation_seconds": 2,
        "min_inferences": 2,
        "ack_timeout_seconds": 60,
        "health_timeout_seconds": 120,
        "max_cpu_percent": 100,
    }
    ids = [f"load-{i:04d}" for i in range(1, a.devices + 1)]
    rows: list[dict[str, Any]] = []
    with FarmHandle(a.devices, hb=a.hb, sensor_hz=a.sensor_hz) as farm, api_client("operator", timeout=120) as op:
        _ = farm
        if a.restart_control_plane_first:
            compose("restart", "-t", "2", "control-plane")
            wait_for(lambda: httpx.get(CP_URL + "/readyz", verify=VERIFY, timeout=3).status_code == 200, 120, 0.25)
            wait_for(lambda: farm.online() >= a.devices, 120, 0.5)
        time.sleep(a.settle)  # let heartbeats/telemetry reach steady state before the first rollout
        for version in [v1, v2, v1, v2][: a.rollouts]:
            body = {
                "model_name": MODEL,
                "model_version": version,
                "group": "load",
                "device_ids": ids,
                "strategy": {"canary_count": 1, "batch_percent": 100},
                "health_criteria": crit,
            }
            before = metrics_text()
            started = time.monotonic()
            r = op.post("/api/v1/deployments", json=body)
            r.raise_for_status()
            dep = r.json()["deployment_id"]
            wait_for(
                lambda dep=dep: (
                    op.get(f"/api/v1/deployments/{dep}").json()["state"] in ("COMPLETED", "ROLLED_BACK", "FAILED")
                ),
                600,
                0.25,
            )
            total = time.monotonic() - started
            final = op.get(f"/api/v1/deployments/{dep}").json()
            after = metrics_text()
            lag = {}
            for channel in ("status", "heartbeat"):
                d = bucket_delta(
                    histogram(after, "fleetops_mqtt_ingest_lag_seconds", channel=channel),
                    histogram(before, "fleetops_mqtt_ingest_lag_seconds", channel=channel),
                )
                lag[channel] = {
                    f"p{q}": (round(v, 3) if (v := quantile_from_buckets(d, q / 100)) is not None else None)
                    for q in (50, 95, 99)
                }
            proc = {}
            for channel in ("status", "heartbeat"):
                d = bucket_delta(
                    histogram(after, "fleetops_mqtt_processing_seconds", channel=channel),
                    histogram(before, "fleetops_mqtt_processing_seconds", channel=channel),
                )
                proc[channel] = {
                    f"p{q}": (round(v, 4) if (v := quantile_from_buckets(d, q / 100)) is not None else None)
                    for q in (50, 95, 99)
                } | {"count": d[-1][1] if d else 0}
            events = fetch_events(op, dep)
            retries = {
                t: sample(after, "fleetops_command_retries_total", type=t)
                - sample(before, "fleetops_command_retries_total", type=t)
                for t in ("DEPLOY", "ACTIVATE")
            }
            rows.append(
                {
                    "to_version": version,
                    "state": final["state"],
                    "reason": final.get("reason"),
                    "client_seconds": round(total, 2),
                    "retries": retries,
                    "acks_applied": sum(1 for e in events if e["action"] == "device.ack"),
                    "ingest_lag_s": lag,
                    "processing_s": proc,
                    **timeline(events),
                }
            )
    path = save(
        "warmup",
        {"devices": a.devices, "settle_s": a.settle, "restart_first": a.restart_control_plane_first, "rows": rows},
    )
    print(
        f"\n### Rollout timeline ({a.devices} devices, settle {a.settle}s, control plane restarted first: {a.restart_control_plane_first})\n"
    )
    for row in rows:
        print(row)
    print(f"\nsaved {path}")


# ----------------------------------------------------------------------------------------------- restart recovery


def compose(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", "compose", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def cmd_restart(a: argparse.Namespace) -> None:
    out: dict[str, Any] = {"devices": a.devices}
    with FarmHandle(a.devices, hb=a.hb, sensor_hz=a.sensor_hz) as farm:
        _ = farm
        for service in ("control-plane", "mosquitto"):
            time.sleep(8)  # steady state first
            wall = datetime.now(UTC)
            started = time.monotonic()
            compose("restart", "-t", "2", service)
            restart_cmd_s = time.monotonic() - started
            ready = wait_for(
                lambda: httpx.get(CP_URL + "/readyz", verify=VERIFY, timeout=3).status_code == 200, 120, 0.25
            )
            fresh = wait_for(
                lambda since=wall: _fresh_since(since, a.devices), 180, 0.5
            )  # every device ONLINE with last_seen after the restart
            out[service] = {
                "compose_restart_command_s": round(restart_cmd_s, 1),
                "ready_after_s": round(restart_cmd_s + ready, 1) if ready is not None else None,
                "all_devices_online_with_fresh_heartbeat_after_s": round(time.monotonic() - started, 1)
                if fresh is not None
                else None,
            }
    path = save("restart", out)
    print("\n### Restart recovery under a running fleet (all times from the start of `docker compose restart`)\n")
    print(
        "| restarted | restart command (s) | control plane ready (s) | all devices ONLINE with post-restart heartbeat (s) |"
    )
    print("|---|---|---|---|")
    for service in ("control-plane", "mosquitto"):
        r = out[service]
        print(
            f"| {service} | {r['compose_restart_command_s']} | {r['ready_after_s']} | {r['all_devices_online_with_fresh_heartbeat_after_s']} |"
        )
    print(f"\nsaved {path}")


def _fresh_since(since: datetime, expected: int) -> bool:
    with api_client("viewer") as c:
        items = c.get("/api/v1/devices", params={"group": "load", "limit": 200}).json()["items"]
    fresh = [
        d
        for d in items
        if d["status"] == "ONLINE"
        and d["last_seen_at"]
        and datetime.fromisoformat(d["last_seen_at"].replace("Z", "+00:00")) > since
    ]
    return len(fresh) >= expected


# ----------------------------------------------------------------------------------------------- soak


def container_rss(service: str) -> float:
    res = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", f"fleetops-{service}-1"],
        capture_output=True,
        text=True,
        check=False,
    )
    from scripts.load_test.common import _to_mb

    return _to_mb(res.stdout.split("/")[0].strip()) if res.stdout.strip() else float("nan")


def slope_per_hour(points: list[tuple[float, float]]) -> float:
    """Least-squares slope of (seconds, value) in value-units per hour."""
    n = len(points)
    if n < 3:
        return float("nan")
    mx, my = sum(p[0] for p in points) / n, sum(p[1] for p in points) / n
    den = sum((p[0] - mx) ** 2 for p in points)
    return (sum((p[0] - mx) * (p[1] - my) for p in points) / den) * 3600 if den else float("nan")


def _proc_counts(container: str) -> tuple[int | None, int | None, float | None]:
    """(open fds, threads, VmRSS MB) of PID 1 inside the container - the application process itself."""
    res = subprocess.run(
        [
            "docker",
            "exec",
            container,
            "sh",
            "-c",
            "ls /proc/1/fd | wc -l; ls /proc/1/task | wc -l; grep VmRSS /proc/1/status",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = res.stdout.split("\n")
    try:
        return int(lines[0]), int(lines[1]), float(lines[2].split()[1]) / 1024
    except (IndexError, ValueError):
        return None, None, None


def _started_at(container: str) -> str:
    res = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.StartedAt}}", container], capture_output=True, text=True, check=False
    )
    return res.stdout.strip()


def _cpu_percent(container: str) -> float | None:
    res = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}", container],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(res.stdout.strip().rstrip("%"))
    except ValueError:
        return None


def _tree_rss_mb(pid: int) -> float:
    import psutil

    try:
        proc = psutil.Process(pid)
        return sum(p.memory_info().rss for p in [proc, *proc.children(recursive=True)]) / (1024 * 1024)
    except psutil.Error:
        return float("nan")


def cmd_soak(a: argparse.Namespace) -> None:
    """Bounded, repeatable soak: warm-up, steady state with a periodic rollout, controlled restart, explicit verdict."""
    v1, v2 = ensure_models()
    containers = {"cp": "fleetops-control-plane-1", "broker": "fleetops-mosquitto-1"}
    ids = [f"load-{i:04d}" for i in range(1, min(a.devices, 10) + 1)]
    crit = {
        "observation_seconds": 2,
        "min_inferences": 2,
        "ack_timeout_seconds": 60,
        "health_timeout_seconds": 120,
        "max_cpu_percent": 100,
    }
    series: list[dict[str, Any]] = []
    rollouts: list[str] = []
    next_rollout, version = a.rollout_every, v1
    with FarmHandle(a.devices, hb=a.hb, sensor_hz=a.sensor_hz) as farm, api_client("operator", timeout=60) as c:
        started_at = {k: _started_at(v) for k, v in containers.items()}
        start = time.monotonic()
        m_start = m_prev = metrics_text()
        while time.monotonic() - start < a.duration:
            t = time.monotonic() - start
            s0 = time.perf_counter()
            total = c.get("/api/v1/devices", params={"group": "load", "limit": 1}).json()["total"]
            online = c.get("/api/v1/devices", params={"group": "load", "status": "ONLINE", "limit": 1}).json()["total"]
            probe = (time.perf_counter() - s0) * 1000
            m = metrics_text()
            fds, threads, rss = _proc_counts(containers["cp"])
            lag = bucket_delta(
                histogram(m, "fleetops_mqtt_ingest_lag_seconds", channel="heartbeat"),
                histogram(m_prev, "fleetops_mqtt_ingest_lag_seconds", channel="heartbeat"),
            )
            m_prev = m
            series.append(
                {
                    "t": round(t),
                    "devices": total,
                    "online": online,
                    "api_probe_ms": round(probe, 1),
                    "cp_rss_mb": rss if rss is not None else container_rss("control-plane"),
                    "cp_cpu_percent": _cpu_percent(containers["cp"]),
                    "cp_fds": fds,
                    "cp_threads": threads,
                    "broker_rss_mb": container_rss("mosquitto"),
                    "load_generator_rss_mb": round(_tree_rss_mb(farm.proc.pid), 1) if farm.proc else None,
                    "http_5xx": sum(
                        sample(m, "fleetops_http_requests_total", status=str(code)) for code in (500, 502, 503, 504)
                    ),
                    "mqtt_failures": sample(m, "fleetops_mqtt_processing_failures_total"),
                    "heartbeat_lag_p95_s": quantile_from_buckets(lag, 0.95),
                    "db_mb": _db_mb(),
                }
            )
            if t >= next_rollout:  # periodic rollout keeps the deployment engine in the soak, not just ingest
                next_rollout += a.rollout_every
                version = v2 if version == v1 else v1
                body = {
                    "model_name": MODEL,
                    "model_version": version,
                    "group": "load",
                    "device_ids": ids,
                    "strategy": {"canary_count": 1, "batch_percent": 100},
                    "health_criteria": crit,
                }
                r = c.post("/api/v1/deployments", json=body)
                if r.status_code in (200, 201):
                    dep = r.json()["deployment_id"]
                    wait_for(
                        lambda d=dep: (
                            c.get(f"/api/v1/deployments/{d}").json()["state"] in ("COMPLETED", "ROLLED_BACK", "FAILED")
                        ),
                        300,
                        0.5,
                    )
                    rollouts.append(c.get(f"/api/v1/deployments/{dep}").json()["state"])
                else:
                    rollouts.append(f"HTTP {r.status_code}")
            time.sleep(max(0.0, a.interval - (time.monotonic() - start - t)))
        m_end = metrics_text()
        restarts = sum(1 for k, v in containers.items() if _started_at(v) != started_at[k])
        # Recovery: a controlled control-plane restart under the running fleet.
        compose("restart", "-t", "2", "control-plane")
        wall = datetime.now(UTC)
        t_restart = time.monotonic()
        ready = wait_for(lambda: httpx.get(CP_URL + "/readyz", verify=VERIFY, timeout=3).status_code == 200, 120, 0.25)
        fresh = wait_for(lambda: _fresh_since(wall, a.devices), 180, 0.5)
        recovery = (time.monotonic() - t_restart) if (ready is not None and fresh is not None) else None
    warmup = a.warmup_minutes * 60
    verdict = evaluate(
        series,
        warmup,
        restarts=restarts,
        http_5xx=series[-1]["http_5xx"] - series[0]["http_5xx"],
        mqtt_failures=sample(m_end, "fleetops_mqtt_processing_failures_total")
        - sample(m_start, "fleetops_mqtt_processing_failures_total"),
        rollouts=rollouts,
        recovery_seconds=recovery,
        offline_samples=len([s for s in series if s["online"] < a.devices and s["t"] >= warmup]),
    )
    result = {
        "devices": a.devices,
        "duration_s": a.duration,
        "interval_s": a.interval,
        "warmup_s": warmup,
        "rollout_every_s": a.rollout_every,
        "samples": len(series),
        "rollouts": rollouts,
        "recovery_seconds": recovery,
        "container_restarts": restarts,
        "verdict": verdict.verdict,
        "reasons": verdict.reasons,
        "details": verdict.details,
        "api_probe_ms": {k: round(v, 1) for k, v in percentiles([s["api_probe_ms"] for s in series]).items()},
        "cp_cpu_percent_median": statistics.median(
            [s["cp_cpu_percent"] for s in series if s["cp_cpu_percent"] is not None] or [float("nan")]
        ),
        "load_generator_rss_mb_last": series[-1]["load_generator_rss_mb"],
        "series": series,
    }
    path = save("soak", result)
    print(
        f"\n### Soak: {a.devices} simulated devices, {a.duration:.0f}s, warm-up {a.warmup_minutes} min, {len(series)} samples\n"
    )
    print(f"- **verdict: {verdict.verdict}**")
    for reason in verdict.reasons:
        print(f"  - {reason}")
    print(f"- steady-state details: {verdict.details}")
    print(
        f"- rollouts: {rollouts or 'none scheduled'}; recovery after a controlled control-plane restart: {recovery if recovery is None else round(recovery, 1)} s"
    )
    print(
        f"- control-plane RSS {series[0]['cp_rss_mb']:.0f} -> {series[-1]['cp_rss_mb']:.0f} MB; load generator RSS {series[-1]['load_generator_rss_mb']} MB (separate process tree)"
    )
    print(f"- API probe ms {result['api_probe_ms']}; CP CPU median {result['cp_cpu_percent_median']:.0f}%")
    print(f"\nsaved {path}")


def _db_mb() -> float:
    res = subprocess.run(
        [
            "docker",
            "exec",
            "fleetops-control-plane-1",
            "sh",
            "-c",
            "du -sk /data/fleetops.db* | awk '{s+=$1} END {print s}'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return float(res.stdout.strip() or 0) / 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--device", default="edge-01")
    p.add_argument("--concurrency", default="1,4,16")
    p.add_argument("--duration", type=float, default=15)
    p.set_defaults(fn=cmd_predict)
    p = sub.add_parser("api")
    p.add_argument("--concurrency", default="4,16")
    p.add_argument("--duration", type=float, default=15)
    p.set_defaults(fn=cmd_api)
    for name, fn in (
        ("ingest", cmd_ingest),
        ("restart", cmd_restart),
        ("soak", cmd_soak),
        ("deploy-scale", cmd_deploy_scale),
    ):
        p = sub.add_parser(name)
        p.add_argument("--devices", type=int, default=50)
        p.add_argument("--hb", type=float, default=2.0)
        p.add_argument("--sensor-hz", type=float, default=1.0)
        p.add_argument("--duration", type=float, default=30 if name == "ingest" else 600)
        p.add_argument("--interval", type=float, default=15)
        p.add_argument("--sizes", default="3,10,30,100")
        p.add_argument("--warmup-minutes", type=float, default=10)
        p.add_argument(
            "--rollout-every", type=float, default=300, help="seconds between scheduled rollouts during the soak"
        )
        p.set_defaults(fn=fn)
    p = sub.add_parser("warmup")
    p.add_argument("--devices", type=int, default=100)
    p.add_argument("--hb", type=float, default=2.0)
    p.add_argument("--sensor-hz", type=float, default=1.0)
    p.add_argument(
        "--settle", type=float, default=0, help="seconds to wait after the farm is online before the first rollout"
    )
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--restart-control-plane-first", action="store_true")
    p.set_defaults(fn=cmd_warmup)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
