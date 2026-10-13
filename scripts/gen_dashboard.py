"""Regenerate monitoring/grafana/dashboards/fleetops.json (kept in git; edit here, not in the UI)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DS = {"type": "prometheus", "uid": "prometheus"}
OUT = Path(__file__).resolve().parent.parent / "monitoring" / "grafana" / "dashboards" / "fleetops.json"


def panel(
    pid: int,
    title: str,
    kind: str,
    exprs: list[tuple[str, str]],
    x: int,
    y: int,
    w: int = 8,
    h: int = 8,
    unit: str = "short",
    desc: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    p: dict[str, Any] = {
        "id": pid,
        "title": title,
        "type": kind,
        "datasource": DS,
        "description": desc,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [
            {"datasource": DS, "expr": e, "legendFormat": legend, "refId": chr(65 + i)}
            for i, (e, legend) in enumerate(exprs)
        ],
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
    }
    if extra:
        p.update(extra)
    return p


def row(pid: int, title: str, y: int) -> dict[str, Any]:
    return {
        "id": pid,
        "type": "row",
        "title": title,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "collapsed": False,
        "panels": [],
    }


panels = [
    row(1, "Fleet status (operational)", 0),
    panel(
        2,
        "Device connectivity",
        "stat",
        [("fleetops_device_connected", "{{device_id}}")],
        0,
        1,
        6,
        5,
        desc="1 = control plane sees heartbeats. Offline devices keep inferring locally.",
        extra={
            "options": {"colorMode": "background", "reduceOptions": {"calcs": ["lastNotNull"]}},
            "fieldConfig": {
                "defaults": {
                    "mappings": [
                        {
                            "type": "value",
                            "options": {
                                "0": {"text": "OFFLINE", "color": "red"},
                                "1": {"text": "ONLINE", "color": "green"},
                            },
                        }
                    ]
                },
                "overrides": [],
            },
        },
    ),
    panel(
        3,
        "Active model version by device",
        "stat",
        [("fleetops_device_model_version", "{{device_id}} {{model}}")],
        6,
        1,
        6,
        5,
        desc="Version reported by the device itself (0 = no model).",
    ),
    panel(
        4,
        "Heartbeat age",
        "timeseries",
        [("fleetops_device_heartbeat_age_seconds", "{{device_id}}")],
        12,
        1,
        12,
        5,
        unit="s",
    ),
    row(5, "Edge inference (operational metrics, scraped from each agent)", 6),
    panel(
        6,
        "Inference latency p95",
        "timeseries",
        [
            (
                "histogram_quantile(0.95, sum by (le, device_id) (rate(fleetops_edge_inference_latency_seconds_bucket[1m])))",
                "{{device_id}}",
            )
        ],
        0,
        7,
        unit="s",
    ),
    panel(
        7,
        "Inference rate",
        "timeseries",
        [
            (
                "sum by (device_id, model_version) (rate(fleetops_edge_inference_requests_total[1m]))",
                "{{device_id}} v{{model_version}}",
            )
        ],
        8,
        7,
        unit="reqps",
    ),
    panel(
        8,
        "Inference failures / rejected inputs",
        "timeseries",
        [
            ("sum by (device_id) (rate(fleetops_edge_inference_failures_total[1m]))", "failures {{device_id}}"),
            ("sum by (device_id) (rate(fleetops_edge_inputs_rejected_total[1m]))", "rejected {{device_id}}"),
        ],
        16,
        7,
        unit="ops",
    ),
    panel(
        9,
        "Agent CPU (% of host)",
        "timeseries",
        [("fleetops_edge_cpu_percent", "{{device_id}}")],
        0,
        15,
        12,
        7,
        unit="percent",
    ),
    panel(
        10,
        "Agent memory",
        "timeseries",
        [("fleetops_edge_memory_mb", "{{device_id}}")],
        12,
        15,
        12,
        7,
        unit="decmbytes",
    ),
    row(11, "Deployments", 22),
    panel(
        12,
        "Deployments by final state (1h)",
        "bargauge",
        [("sum by (state) (increase(fleetops_deployments_total[1h]))", "{{state}}")],
        0,
        23,
        6,
        7,
    ),
    panel(
        13,
        "Rollbacks (1h)",
        "stat",
        [("sum by (trigger) (increase(fleetops_rollbacks_total[1h]))", "{{trigger}}")],
        6,
        23,
        6,
        7,
        desc="automatic = health criteria or ack failure; manual = operator request.",
    ),
    panel(
        14,
        "Per-device outcomes (1h)",
        "bargauge",
        [("sum by (outcome) (increase(fleetops_deployment_targets_total[1h]))", "{{outcome}}")],
        12,
        23,
        6,
        7,
    ),
    panel(
        15,
        "Model update duration p95",
        "timeseries",
        [
            (
                "histogram_quantile(0.95, sum by (le, command) (rate(fleetops_model_update_duration_seconds_bucket[5m])))",
                "{{command}}",
            )
        ],
        18,
        23,
        6,
        7,
        unit="s",
    ),
    row(16, "Messaging and model quality (kept separate from operational health)", 30),
    panel(
        17,
        "MQTT processing failures",
        "timeseries",
        [
            (
                "sum by (channel, reason) (increase(fleetops_mqtt_processing_failures_total[5m]))",
                "{{channel}} {{reason}}",
            )
        ],
        0,
        31,
        8,
        7,
    ),
    panel(
        18,
        "MQTT messages processed / s",
        "timeseries",
        [("sum by (channel) (rate(fleetops_mqtt_messages_total[1m]))", "{{channel}}")],
        8,
        31,
        8,
        7,
        unit="ops",
    ),
    panel(
        19,
        "Drift reports by status (1h)",
        "bargauge",
        [("sum by (status) (increase(fleetops_drift_reports_total[1h]))", "{{status}}")],
        16,
        31,
        8,
        7,
        desc="Drift = inputs changed; it does not mean the model is inaccurate. Review via /api/v1/monitoring/recommendations.",
    ),
]

dashboard = {
    "uid": "fleetops-overview",
    "title": "FleetOps overview",
    "tags": ["fleetops"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "5s",
    "time": {"from": "now-15m", "to": "now"},
    "panels": panels,
    "templating": {"list": []},
    "annotations": {"list": []},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"wrote {OUT}")
