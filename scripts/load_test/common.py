"""Shared helpers for the load / soak harness (no mocks: everything talks to the running Compose stack)."""

from __future__ import annotations

import json
import os
import platform
import ssl
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil
from prometheus_client.parser import text_string_to_metric_families

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "load_results"


def read_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


ENV = read_env()
TLS = os.environ.get("FLEETOPS_TLS") == "1"
SCHEME = "https" if TLS else "http"
VERIFY: ssl.SSLContext | bool = (
    ssl.create_default_context(cafile=str(ROOT / "deploy" / "secrets" / "tls" / "ca.crt")) if TLS else True
)
CP_URL = f"{SCHEME}://127.0.0.1:{ENV.get('FLEETOPS_CP_PORT', '18000')}"
MQTT_PORT = int(ENV.get("FLEETOPS_MQTT_PORT", "11883"))
EDGE_URL = {f"edge-0{i}": f"{SCHEME}://127.0.0.1:{ENV.get(f'EDGE{i}_PORT', str(19100 + i))}" for i in (1, 2, 3)}


def secret_file(name: str) -> str:
    return (ROOT / "deploy" / "secrets" / name).read_text().strip()


def api_client(role: str = "viewer", timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(
        base_url=CP_URL,
        verify=VERIFY,
        timeout=timeout,
        headers={"Authorization": f"Bearer {ENV[f'FLEETOPS_{role.upper()}_API_KEY']}"},
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=256),
    )


def metrics_text(url: str = CP_URL, token: str | None = None) -> str:
    tok = token or secret_file("fleetops_metrics_token")
    return httpx.get(url + "/metrics", headers={"Authorization": f"Bearer {tok}"}, verify=VERIFY, timeout=15).text


def sample(text: str, name: str, **labels: str) -> float:
    total = 0.0
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                total += s.value
    return total


def histogram(text: str, name: str, **labels: str) -> list[tuple[float, float]]:
    """Cumulative (le, count) buckets of a histogram, summed over series matching ``labels``."""
    buckets: dict[float, float] = {}
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            if s.name == f"{name}_bucket" and all(s.labels.get(k) == v for k, v in labels.items()):
                le = float("inf") if s.labels["le"] == "+Inf" else float(s.labels["le"])
                buckets[le] = buckets.get(le, 0.0) + s.value
    return sorted(buckets.items())


def quantile_from_buckets(buckets: list[tuple[float, float]], q: float) -> float | None:
    """Prometheus-style histogram_quantile over cumulative buckets (linear interpolation inside a bucket)."""
    if not buckets or buckets[-1][1] <= 0:
        return None
    rank = q * buckets[-1][1]
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= rank:
            if le == float("inf"):
                return prev_le
            span = count - prev_count
            return prev_le + (le - prev_le) * ((rank - prev_count) / span if span > 0 else 1.0)
        prev_le, prev_count = le, count
    return prev_le


def bucket_delta(after: list[tuple[float, float]], before: list[tuple[float, float]]) -> list[tuple[float, float]]:
    base = dict(before)
    return [(le, c - base.get(le, 0.0)) for le, c in after]


def percentiles(values: list[float], qs: tuple[float, ...] = (50, 95, 99)) -> dict[str, float]:
    if not values:
        return {f"p{int(q)}": float("nan") for q in qs}
    ordered = sorted(values)
    out = {}
    for q in qs:
        idx = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
        out[f"p{int(q)}"] = ordered[idx]
    return out


@dataclass
class StatsSampler:
    """Samples ``docker stats`` for the given compose services (CPU %, memory) on a background thread."""

    services: list[str]
    interval: float = 3.0
    samples: list[dict[str, Any]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _containers(self) -> dict[str, str]:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=False
        ).stdout.split()
        return {s: next((c for c in out if f"-{s}-" in c or c.endswith(f"-{s}")), "") for s in self.services}

    def _loop(self) -> None:
        names = self._containers()
        while not self._stop.is_set():
            res = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{json .}}", *[n for n in names.values() if n]],
                capture_output=True,
                text=True,
                check=False,
            )
            now = time.time()
            for line in res.stdout.splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                service = next((s for s, n in names.items() if n == row["Name"]), row["Name"])
                used = row["MemUsage"].split("/")[0].strip()
                self.samples.append(
                    {"t": now, "service": service, "cpu": float(row["CPUPerc"].rstrip("%")), "mem_mb": _to_mb(used)}
                )
            self._stop.wait(self.interval)

    def __enter__(self) -> StatsSampler:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for service in {s["service"] for s in self.samples}:
            rows = [s for s in self.samples if s["service"] == service]
            cpu, mem = [r["cpu"] for r in rows], [r["mem_mb"] for r in rows]
            out[service] = {
                "cpu_avg_pct": sum(cpu) / len(cpu),
                "cpu_max_pct": max(cpu),
                "mem_avg_mb": sum(mem) / len(mem),
                "mem_max_mb": max(mem),
                "samples": float(len(rows)),
            }
        return out


def _to_mb(text: str) -> float:
    units = {"B": 1 / 1e6, "KiB": 1024 / 1e6, "MiB": 1.048576, "GiB": 1073.741824, "kB": 1e-3, "MB": 1.0, "GB": 1e3}
    for suffix in sorted(units, key=len, reverse=True):
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * units[suffix]
    return float(text)


def environment() -> dict[str, Any]:
    def sh(*cmd: str) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()

    docker = sh(
        "docker",
        "info",
        "--format",
        "{{.ServerVersion}} | {{.NCPU}} CPUs | {{.MemTotal}} bytes | {{.OperatingSystem}} {{.Architecture}}",
    )
    images = sh("docker", "compose", "images", "--format", "json")
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / 1e9, 1),
        },
        "python": platform.python_version(),
        "docker": docker,
        "tls": TLS,
        "compose_images": images[:2000],
        "note": "Docker Desktop on macOS runs Linux containers in a VM; absolute numbers are for this machine only.",
    }


def save(name: str, data: dict[str, Any]) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{name}.json"
    path.write_text(json.dumps({"environment": environment(), **data}, indent=2, default=str))
    return path


def row(label: str, **values: Any) -> str:
    cells = " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in values.values())
    return f"| {label} | {cells} |"


def wait_for(cond: Any, timeout: float, interval: float = 0.5) -> float | None:
    """Seconds until ``cond()`` is truthy, or None on timeout."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            if cond():
                return time.monotonic() - start
        except httpx.HTTPError:
            pass
        time.sleep(interval)
    return None
