"""Control-plane Prometheus metrics (instance-scoped registry; device gauges read the DB at scrape)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from fleetops.domain.models import utcnow
from fleetops.persistence.db import session_scope
from fleetops.persistence.tables import Device


class DeviceCollector(Collector):
    """Computes per-device gauges from the inventory at scrape time (so heartbeat age is always fresh)."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions = sessions

    def collect(self) -> Iterable[Metric]:
        age = GaugeMetricFamily(
            "fleetops_device_heartbeat_age_seconds", "Seconds since last heartbeat", labels=["device_id", "group"]
        )
        online = GaugeMetricFamily("fleetops_device_connected", "1 if device is ONLINE", labels=["device_id", "group"])
        version = GaugeMetricFamily(
            "fleetops_device_model_version",
            "Model version reported by device (0 = none)",
            labels=["device_id", "group", "model"],
        )
        skew = GaugeMetricFamily(
            "fleetops_device_clock_skew_seconds",
            "Control-plane receive time minus device send time of the last heartbeat (includes latency)",
            labels=["device_id"],
        )
        disk = GaugeMetricFamily(
            "fleetops_device_disk_free_mb",
            "Free space on the device's model-store volume (last heartbeat)",
            labels=["device_id"],
        )
        cpu = GaugeMetricFamily("fleetops_device_cpu_percent", "Last reported agent CPU %", labels=["device_id"])
        mem = GaugeMetricFamily("fleetops_device_memory_mb", "Last reported agent RSS MB", labels=["device_id"])
        inf = GaugeMetricFamily(
            "fleetops_device_inferences_since_activation",
            "Inferences by the active model since activation",
            labels=["device_id", "model_version"],
        )
        err = GaugeMetricFamily(
            "fleetops_device_inference_errors_since_activation",
            "Inference errors since activation",
            labels=["device_id", "model_version"],
        )
        now = utcnow()
        with session_scope(self._sessions) as db:
            for d in db.scalars(select(Device)):
                age.add_metric(
                    [d.device_id, d.group_name], (now - d.last_seen_at).total_seconds() if d.last_seen_at else -1
                )
                online.add_metric([d.device_id, d.group_name], 1.0 if d.status == "ONLINE" else 0.0)
                version.add_metric(
                    [d.device_id, d.group_name, d.active_model_name or "none"], float(d.active_model_version or 0)
                )
                hb = d.last_heartbeat or {}
                if hb.get("disk_free_mb") is not None:
                    disk.add_metric([d.device_id], float(hb["disk_free_mb"]))
                sent = hb.get("sent_at")
                if sent and d.last_seen_at:
                    sent_dt = datetime.fromisoformat(sent)
                    skew.add_metric([d.device_id], (d.last_seen_at - sent_dt).total_seconds())
                cpu.add_metric([d.device_id], float(hb.get("cpu_percent", 0)))
                mem.add_metric([d.device_id], float(hb.get("memory_mb", 0)))
                stats = hb.get("model_stats")
                if stats:
                    v = str(stats["version"])
                    inf.add_metric([d.device_id, v], float(stats["inference_count"]))
                    err.add_metric([d.device_id, v], float(stats["inference_errors"]))
        yield from (age, online, version, skew, disk, cpu, mem, inf, err)


class ControlPlaneMetrics:
    def __init__(self, sessions: sessionmaker[Session] | None = None) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.deployments = Counter(
            "fleetops_deployments_total", "Deployments reaching a final state", ["state"], registry=r
        )
        self.target_outcomes = Counter(
            "fleetops_deployment_targets_total", "Per-device deployment outcomes", ["outcome"], registry=r
        )
        self.rollbacks = Counter("fleetops_rollbacks_total", "Rollbacks initiated", ["trigger"], registry=r)
        self.mqtt_failures = Counter(
            "fleetops_mqtt_processing_failures_total",
            "MQTT messages that failed validation/processing",
            ["channel", "reason"],
            registry=r,
        )
        self.mqtt_messages = Counter("fleetops_mqtt_messages_total", "MQTT messages processed", ["channel"], registry=r)
        self.update_duration = Histogram(
            "fleetops_model_update_duration_seconds",
            "Device-reported time to execute a model command",
            ["command"],
            registry=r,
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
        )
        self.http_requests = Counter(
            "fleetops_http_requests_total", "Management API requests", ["method", "status"], registry=r
        )
        self.mqtt_processing = Histogram(
            "fleetops_mqtt_processing_seconds",
            "Time to validate, persist and act on one uplink message",
            ["channel"],
            registry=r,
            buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
        )
        self.mqtt_ingest_lag = Histogram(
            "fleetops_mqtt_ingest_lag_seconds",
            "Device send time to control-plane receive time (heartbeats)",
            ["channel"],
            registry=r,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
        )
        self.commands_published = Counter(
            "fleetops_commands_published_total", "Commands handed to the broker", ["type", "result"], registry=r
        )
        self.command_retries = Counter(
            "fleetops_command_retries_total",
            "Commands re-sent because no acknowledgement arrived",
            ["type"],
            registry=r,
        )
        self.rollback_failures = Counter(
            "fleetops_rollback_failures_total", "Devices that rejected or failed a rollback/abort", registry=r
        )
        self.mqtt_connected = Gauge(
            "fleetops_mqtt_connected", "1 while the control plane is connected to the broker", registry=r
        )
        self.mqtt_reconnects = Counter(
            "fleetops_mqtt_reconnects_total", "Broker reconnections after the first connect", registry=r
        )
        self.artifact_downloads = Counter(
            "fleetops_artifact_downloads_total", "Device artifact downloads", ["result"], registry=r
        )
        self.artifact_integrity_failures = Counter(
            "fleetops_artifact_integrity_failures_total",
            "Stored artifacts whose hash differs from the registry",
            registry=r,
        )
        self.auth_denied = Counter("fleetops_auth_denied_total", "Denied management requests", ["reason"], registry=r)
        self.rate_limited = Counter("fleetops_rate_limited_total", "Requests answered with 429", ["scope"], registry=r)
        self.requests_rejected = Counter(
            "fleetops_requests_rejected_total", "Requests rejected before routing", ["reason"], registry=r
        )
        self.drift_reports = Counter("fleetops_drift_reports_total", "Drift reports generated", ["status"], registry=r)
        if sessions is not None:
            r.register(DeviceCollector(sessions))


MetricsHook = Callable[[str], None]
