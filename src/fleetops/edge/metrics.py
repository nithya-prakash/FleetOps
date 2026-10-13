"""Per-agent Prometheus metrics (own registry: no global state)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class EdgeMetrics:
    def __init__(self, device_id: str) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.device_id = device_id
        self.inference_latency = Histogram(
            "fleetops_edge_inference_latency_seconds",
            "Local inference latency",
            ["device_id", "model_version"],
            registry=r,
            buckets=(0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
        )
        self.inference_total = Counter(
            "fleetops_edge_inference_requests_total", "Inference requests", ["device_id", "model_version"], registry=r
        )
        self.inference_failures = Counter(
            "fleetops_edge_inference_failures_total",
            "Inference failures (model/runtime, not bad input)",
            ["device_id", "model_version"],
            registry=r,
        )
        self.inputs_rejected = Counter(
            "fleetops_edge_inputs_rejected_total", "Inputs rejected by schema validation", ["device_id"], registry=r
        )
        self.model_version = Gauge(
            "fleetops_edge_model_version", "Active model version (0 = none)", ["device_id", "model"], registry=r
        )
        self.mqtt_connected = Gauge("fleetops_edge_mqtt_connected", "MQTT link state", ["device_id"], registry=r)
        self.commands = Counter(
            "fleetops_edge_commands_total", "Commands handled", ["device_id", "type", "outcome"], registry=r
        )
        self.mqtt_failures = Counter(
            "fleetops_edge_mqtt_processing_failures_total", "Unprocessable MQTT messages", ["device_id"], registry=r
        )
        self.update_duration = Histogram(
            "fleetops_edge_model_update_duration_seconds",
            "Time to stage/activate a model",
            ["device_id", "command"],
            registry=r,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
        )
        self.disk_free = Gauge(
            "fleetops_edge_disk_free_mb", "Free space on the model-store volume", ["device_id"], registry=r
        )
        self.duplicate_commands = Counter(
            "fleetops_edge_duplicate_commands_total",
            "Re-delivered commands answered from the ack cache",
            ["device_id", "type"],
            registry=r,
        )
        self.cpu = Gauge("fleetops_edge_cpu_percent", "Agent process CPU %", ["device_id"], registry=r)
        self.memory = Gauge("fleetops_edge_memory_mb", "Agent process RSS MB", ["device_id"], registry=r)
