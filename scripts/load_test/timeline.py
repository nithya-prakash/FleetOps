"""Stage-by-stage timeline of one deployment, reconstructed from the control plane's own audit events.

Offsets are seconds after ``deployment.created`` using the SERVER's timestamps, so client polling/HTTP delays are excluded.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx


def fetch_events(client: httpx.Client, deployment_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = client.get(
            "/api/v1/audit/events", params={"deployment_id": deployment_id, "limit": 200, "offset": offset}
        ).json()
        events += page["items"]
        offset += 200
        if offset >= page["total"]:
            return sorted(events, key=lambda e: (e["ts"], e["id"]))


def _t(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def timeline(events: list[dict[str, Any]]) -> dict[str, Any]:
    created = next(e for e in events if e["action"] == "deployment.created")
    t0 = _t(created["ts"])

    def at(pred: Any) -> list[float]:
        return [round(_t(e["ts"]) - t0, 2) for e in events if pred(e)]

    changes = [e for e in events if e["action"] == "target.state_changed"]

    def to(state: str) -> list[float]:
        return [round(_t(e["ts"]) - t0, 2) for e in changes if e["details"].get("to") == state]

    def span(values: list[float]) -> list[float] | None:
        return [min(values), max(values)] if values else None

    promoted = at(lambda e: e["action"] == "deployment.batch_promoted")
    done = [
        e
        for e in events
        if e["action"].startswith("deployment.") and e["action"].split(".")[1] in ("completed", "rolled_back", "failed")
    ]
    return {
        "events": len(events),
        "downloads": span(at(lambda e: e["action"] == "artifact.downloaded")),
        "distributing": span(to("DISTRIBUTING")),
        "staged": span(to("STAGED")),
        "active": span(to("ACTIVE")),
        "health_passed": span(at(lambda e: e["action"] == "target.health_passed")),
        "batch_promoted_at": promoted,
        "finished_at": round(_t(done[-1]["ts"]) - t0, 2) if done else None,
    }
