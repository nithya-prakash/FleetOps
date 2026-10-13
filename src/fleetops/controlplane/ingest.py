"""MQTT uplink handlers: validate, persist, and feed the rollout engine."""

from __future__ import annotations

import logging
import time

from pydantic import ValidationError
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from fleetops.config.logging import log_context
from fleetops.controlplane import devices
from fleetops.deployment.engine import CommandPublisher, RolloutEngine
from fleetops.domain.messages import MAX_MESSAGE_BYTES, Command, Heartbeat, PredictionBatch, StatusMessage, Telemetry
from fleetops.domain.models import utcnow
from fleetops.messaging.bus import BusDisconnectedError, MessageBus
from fleetops.messaging.topics import QOS, Channel, parse_topic, topic, uplink_wildcard
from fleetops.monitoring.metrics import ControlPlaneMetrics
from fleetops.persistence.db import session_scope
from fleetops.persistence.tables import Device, FeatureSample

logger = logging.getLogger(__name__)


class BusCommandPublisher:
    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus

    def publish(self, command: Command) -> bool:
        try:
            self._bus.publish(
                topic(command.device_id, Channel.COMMANDS),
                command.model_dump_json().encode(),
                qos=QOS[Channel.COMMANDS],
            )
        except BusDisconnectedError:
            return False
        return True


class NullPublisher:
    """Used when MQTT is disabled (API-only tests); commands remain pending and are retried."""

    def publish(self, command: Command) -> bool:
        return False


_: type[CommandPublisher] = BusCommandPublisher  # structural check


class UplinkIngest:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        engine: RolloutEngine,
        metrics: ControlPlaneMetrics,
        sample_retention: int,
        clock_skew_warn_seconds: float = 30.0,
    ) -> None:
        self._sessions = sessions
        self._engine = engine
        self._metrics = metrics
        self._retention = sample_retention
        self._skew_warn = clock_skew_warn_seconds

    def attach(self, bus: MessageBus) -> None:
        for channel in (Channel.HEARTBEAT, Channel.TELEMETRY, Channel.PREDICTIONS, Channel.STATUS):
            bus.subscribe(uplink_wildcard(channel), self.handle, qos=QOS[channel])

    def _fail(self, channel: str, reason: str, detail: str = "") -> None:
        self._metrics.mqtt_failures.labels(channel, reason).inc()
        logger.warning("uplink message rejected", extra={"channel": channel, "reason": reason, "detail": detail[:200]})

    def handle(self, raw_topic: str, payload: bytes) -> None:
        parsed = parse_topic(raw_topic)
        if parsed is None or parsed[1] == Channel.COMMANDS:
            self._fail("unknown", "bad_topic", raw_topic)
            return
        device_id, channel = parsed
        if len(payload) > MAX_MESSAGE_BYTES:
            self._fail(channel.value, "too_large")
            return
        started = time.perf_counter()
        try:
            with log_context(device_id=device_id):
                self._dispatch(device_id, channel, payload)
        except ValidationError as exc:
            self._fail(channel.value, "invalid_payload", str(exc))
        except Exception:
            logger.exception("uplink processing failed")
            self._metrics.mqtt_failures.labels(channel.value, "exception").inc()
        finally:
            self._metrics.mqtt_processing.labels(channel.value).observe(time.perf_counter() - started)

    def _dispatch(self, device_id: str, channel: Channel, payload: bytes) -> None:
        if channel == Channel.HEARTBEAT:
            hb = Heartbeat.model_validate_json(payload)
            if not self._device_matches(device_id, hb.device_id, channel):
                return
            self._check_clock_skew(device_id, hb)
            with session_scope(self._sessions) as db:
                known = devices.apply_heartbeat(db, hb) is not None
            if not known:
                self._fail(channel.value, "unregistered", device_id)
                return
            self._engine.observe_heartbeat(hb)
        elif channel == Channel.STATUS:
            msg = StatusMessage.model_validate_json(payload)
            if not self._device_matches(device_id, msg.device_id, channel):
                return
            with session_scope(self._sessions) as db:
                known = devices.apply_status(db, msg) is not None
            if not known:
                self._fail(channel.value, "unregistered", device_id)
                return
            if msg.kind == "online":
                self._engine.device_came_online(device_id)
            elif msg.kind == "ack":
                self._engine.record_ack(msg)
        elif channel == Channel.TELEMETRY:
            tel = Telemetry.model_validate_json(payload)
            if not self._device_matches(device_id, tel.device_id, channel) or not self._accepted(device_id, channel):
                return
        elif channel == Channel.PREDICTIONS:
            batch = PredictionBatch.model_validate_json(payload)
            if self._device_matches(device_id, batch.device_id, channel) and self._accepted(device_id, channel):
                self._store_samples(batch)
        self._metrics.mqtt_messages.labels(channel.value).inc()

    def _accepted(self, device_id: str, channel: Channel) -> bool:
        """Only registered, non-revoked devices may feed telemetry or predictions."""
        with session_scope(self._sessions) as db:
            dev = db.get(Device, device_id)
            status = dev.status if dev else None
        if status is None:
            self._fail(channel.value, "unregistered", device_id)
            return False
        if status == "REVOKED":
            self._fail(channel.value, "revoked", device_id)
            return False
        return True

    def _check_clock_skew(self, device_id: str, hb: Heartbeat) -> None:
        """Skew = receive time - device send time (includes network latency). Large values break command expiry."""
        skew = (utcnow() - hb.sent_at).total_seconds()
        self._metrics.mqtt_ingest_lag.labels("heartbeat").observe(max(skew, 0.0))
        if abs(skew) > self._skew_warn:
            logger.warning(
                "device clock skew exceeds tolerance",
                extra={
                    "reason": "clock_skew",
                    "device_id": device_id,
                    "skew_seconds": round(skew, 1),
                    "tolerance_seconds": self._skew_warn,
                    "hint": "device clock ahead: fresh commands look expired; behind: commands look future-dated",
                },
            )

    def _device_matches(self, topic_device: str, payload_device: str, channel: Channel) -> bool:
        if topic_device != payload_device:
            self._fail(channel.value, "device_mismatch", f"{topic_device} != {payload_device}")
            return False
        return True

    def _store_samples(self, batch: PredictionBatch) -> None:
        with session_scope(self._sessions) as db:
            if devices_known := db.get(Device, batch.device_id):
                for s in batch.samples:
                    db.add(
                        FeatureSample(
                            device_id=devices_known.device_id,
                            model_name=s.model_name,
                            model_version=s.model_version,
                            ts=s.ts,
                            features=s.features,
                            score=s.score,
                            anomaly=s.anomaly,
                            label=s.label,
                        )
                    )
                total = db.scalar(select(func.count()).select_from(FeatureSample)) or 0
                if total > self._retention:  # bounded retention: drop the oldest
                    cutoff = db.scalar(
                        select(FeatureSample.id).order_by(FeatureSample.id.desc()).offset(self._retention).limit(1)
                    )
                    if cutoff is not None:
                        db.execute(delete(FeatureSample).where(FeatureSample.id <= cutoff))
