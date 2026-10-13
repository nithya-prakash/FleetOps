"""Run every query of the provisioned Grafana dashboard through Grafana against the live Prometheus.

    python scripts/verify_dashboard.py

Fails if the dashboard/datasource is not provisioned, a query errors, or a panel that should have data
right after the demo returns nothing. Panels for events that may not have happened yet (rollbacks, failures,
drift) are allowed to be empty and are reported as such.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
ENV = {
    k.strip(): v.strip()
    for k, _, v in (ln.partition("=") for ln in (ROOT / ".env").read_text().splitlines() if "=" in ln and ln[0] != "#")
}
GRAFANA = f"http://127.0.0.1:{ENV.get('FLEETOPS_GRAFANA_PORT', '13000')}"
AUTH = ("admin", ENV["GRAFANA_ADMIN_PASSWORD"])
# panels that must show data once the stack has been running for a few scrape intervals
MUST_HAVE_DATA = {
    "Device connectivity",
    "Active model version by device",
    "Heartbeat age",
    "Inference latency p95",
    "Inference rate",
    "Agent CPU (% of host)",
    "Agent memory",
    "MQTT messages processed / s",
}


def main() -> int:
    http = httpx.Client(base_url=GRAFANA, auth=AUTH, timeout=30)
    found = http.get("/api/search", params={"query": "FleetOps"}).json()
    if not found:
        print("FAIL  dashboard 'FleetOps overview' is not provisioned")
        return 1
    dash: dict[str, Any] = http.get(f"/api/dashboards/uid/{found[0]['uid']}").json()["dashboard"]
    failed = 0
    for panel in dash["panels"]:
        for target in panel.get("targets", []):
            body = {
                "queries": [
                    {
                        "refId": target["refId"],
                        "datasource": target["datasource"],
                        "expr": target["expr"],
                        "instant": True,
                    }
                ],
                "from": "now-5m",
                "to": "now",
            }
            resp = http.post("/api/ds/query", json=body)
            result = resp.json().get("results", {}).get(target["refId"], {})
            frames = result.get("frames", [])
            series = len(frames)
            error = result.get("error") or (None if resp.status_code == 200 else resp.text[:120])
            if error:
                state, failed = f"FAIL  query error: {error}", failed + 1
            elif series == 0 and panel["title"] in MUST_HAVE_DATA:
                state, failed = "FAIL  no data", failed + 1
            else:
                state = f"ok    {series} series" if series else "ok    (empty: no such events yet)"
            print(f"{state:38} {panel['title']}")
    print("\nall dashboard queries executed" if not failed else f"\n{failed} problem(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
