"""Every metric and label used by the Grafana dashboard and the Prometheus alerts must really be exported."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from prometheus_client import CollectorRegistry

from fleetops.domain.models import utcnow
from fleetops.edge.metrics import EdgeMetrics
from fleetops.monitoring.metrics import ControlPlaneMetrics
from fleetops.persistence.db import make_engine, make_session_factory, session_scope, upgrade_database
from fleetops.persistence.tables import Device

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SUFFIXES = {"counter": ["_total"], "histogram": ["_bucket", "_sum", "_count"], "gauge": [""], "unknown": [""]}


def exported(registry: CollectorRegistry) -> dict[str, set[str]]:
    """sample name -> label names, from the registry's own families (works even before any observation)."""
    out: dict[str, set[str]] = {}
    for family in registry.collect():
        suffixes = SAMPLE_SUFFIXES.get(family.type, [""])
        base = family.name.removesuffix("_total") if family.type == "counter" else family.name
        labels: set[str] = set()
        for s in family.samples:
            labels |= set(s.labels)
        for suffix in suffixes:
            out[base + suffix] = labels | ({"le"} if suffix == "_bucket" else set())
    return out


def label_names(registry: CollectorRegistry) -> dict[str, set[str]]:
    """Declared label names (needed because unlabelled-so-far metrics expose no samples)."""
    declared: dict[str, set[str]] = {}
    for collector, names in registry._collector_to_names.items():
        labels = set(getattr(collector, "_labelnames", ()) or ())
        for n in names:
            declared[n] = labels
    return declared


def build_exports(tmp_path: Path) -> dict[str, set[str]]:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    upgrade_database(url)
    sessions = make_session_factory(make_engine(url))
    with session_scope(sessions) as db:
        db.add(
            Device(
                device_id="edge-01",
                group_name="line-a",
                status="ONLINE",
                registered_at=utcnow(),
                last_seen_at=utcnow(),
                active_model_name="anomaly",
                active_model_version=1,
                last_heartbeat={
                    "sent_at": utcnow().isoformat(),
                    "cpu_percent": 1.0,
                    "memory_mb": 2.0,
                    "model_stats": {"version": 1, "inference_count": 3, "inference_errors": 0},
                },
            )
        )
    cp = ControlPlaneMetrics(sessions)
    cp.deployments.labels("COMPLETED").inc()
    edge = EdgeMetrics("edge-01")
    result: dict[str, set[str]] = {}
    for reg in (cp.registry, edge.registry):
        samples = exported(reg)
        declared = label_names(reg)
        for name, labels in samples.items():
            stem = re.sub(r"_(total|bucket|sum|count|created)$", "", name)
            result[name] = (
                labels | declared.get(name, set()) | declared.get(stem, set()) | declared.get(stem + "_total", set())
            )
    return result


def promql_metrics(expr: str) -> set[str]:
    return set(re.findall(r"\bfleetops_[a-z0-9_]+", expr))


def dashboard_targets() -> list[tuple[str, str, str]]:
    dash = json.loads((ROOT / "monitoring/grafana/dashboards/fleetops.json").read_text())
    return [(p["title"], t["expr"], t.get("legendFormat", "")) for p in dash["panels"] for t in p.get("targets", [])]


def test_dashboard_queries_reference_only_exported_metrics(tmp_path: Path) -> None:
    exports = build_exports(tmp_path)
    targets = dashboard_targets()
    assert len(targets) >= 15
    missing = {(title, m) for title, expr, _ in targets for m in promql_metrics(expr) if m not in exports}
    assert not missing, f"dashboard references metrics nobody exports: {sorted(missing)}"


def test_dashboard_legend_labels_exist_on_the_queried_metrics(tmp_path: Path) -> None:
    exports = build_exports(tmp_path)
    problems = []
    for title, expr, legend in dashboard_targets():
        wanted = set(re.findall(r"\{\{\s*([a-z_]+)\s*\}\}", legend))
        by = re.search(r"\bby\s*\(([^)]*)\)", expr)
        available = (
            {x.strip() for x in by.group(1).split(",")}
            if by
            else set().union(*(exports[m] for m in promql_metrics(expr)))
        )
        if not wanted <= available:
            problems.append((title, sorted(wanted - available)))
    assert not problems, problems


def test_alert_rules_reference_only_exported_metrics(tmp_path: Path) -> None:
    exports = build_exports(tmp_path)
    rules = yaml.safe_load((ROOT / "monitoring/prometheus/alerts.yml").read_text())
    exprs = [r["expr"] for g in rules["groups"] for r in g["rules"]]
    assert len(exprs) >= 7
    missing = {m for e in exprs for m in promql_metrics(e) if m not in exports}
    assert not missing, sorted(missing)


def test_alert_label_selectors_use_real_label_values(tmp_path: Path) -> None:
    exports = build_exports(tmp_path)
    rules = yaml.safe_load((ROOT / "monitoring/prometheus/alerts.yml").read_text())
    for group in rules["groups"]:
        for rule in group["rules"]:
            for metric, selector in re.findall(r"(fleetops_[a-z0-9_]+)\{([^}]*)\}", rule["expr"]):
                for label in re.findall(r"([a-z_]+)\s*=", selector):
                    assert label in exports[metric], (rule["alert"], metric, label)


def test_prometheus_scrapes_the_services_that_export_these_metrics() -> None:
    cfg = yaml.safe_load((ROOT / "monitoring/prometheus/prometheus.yml").read_text())
    jobs = {j["job_name"]: j["static_configs"][0]["targets"] for j in cfg["scrape_configs"]}
    assert jobs["fleetops-control-plane"] == ["control-plane:8000"]
    assert jobs["fleetops-edge"] == ["edge-01:9100", "edge-02:9100", "edge-03:9100"]


def test_event_counters_exist_at_zero_so_the_first_event_is_visible_to_increase() -> None:
    """Prometheus increase() ignores a counter series that first appears already at 1: a dashboard would show the first
    rollback as 0. Every known label value must therefore be exported from the start."""
    from prometheus_client import generate_latest  # noqa: PLC0415

    from fleetops.monitoring.metrics import ControlPlaneMetrics  # noqa: PLC0415

    text = generate_latest(ControlPlaneMetrics().registry).decode()
    for series in (
        'fleetops_deployments_total{state="ROLLED_BACK"} 0.0',
        'fleetops_deployments_total{state="COMPLETED"} 0.0',
        'fleetops_rollbacks_total{trigger="automatic"} 0.0',
        'fleetops_deployment_targets_total{outcome="rolled_back"} 0.0',
    ):
        assert series in text, series
