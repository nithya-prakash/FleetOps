"""Control-plane MQTT ingestion: malformed, invalid, mismatched and recoverable-failure messages."""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from fleetops.controlplane.ingest import UplinkIngest
from fleetops.domain.messages import (
    DeviceModelState,
    Heartbeat,
    PredictionBatch,
    PredictionSample,
    StatusMessage,
)
from fleetops.domain.models import DeviceCapabilities, ModelIdentity, utcnow
from fleetops.persistence.db import session_scope
from fleetops.persistence.tables import AuditEvent, Device, FeatureSample
from tests.unit.test_engine import CAPS, Env

HB_TOPIC = "fleetops/devices/edge-01/heartbeat"
PRED_TOPIC = "fleetops/devices/edge-01/predictions"
STATUS_TOPIC = "fleetops/devices/edge-01/status"


def heartbeat(device: str = "edge-01", **kw: object) -> Heartbeat:
    return Heartbeat(
        device_id=device,
        sent_at=kw.pop("sent_at", utcnow()),
        boot_id="b",
        agent_version="0.1.0",
        uptime_seconds=1,  # type: ignore[arg-type]
        state=DeviceModelState(active=ModelIdentity(name="anomaly", version=1, sha256="1" * 64)),
        capabilities=DeviceCapabilities(agent_version="0.1.0", formats=["mlp-v1"]),
        cpu_percent=1,
        memory_mb=1,
    )


def predictions(n: int = 3, device: str = "edge-01") -> PredictionBatch:
    sample = PredictionSample(
        ts=utcnow(), model_name="anomaly", model_version=1, features=[1.0] * 6, score=0.1, anomaly=False, latency_ms=0.1
    )
    return PredictionBatch(device_id=device, sent_at=utcnow(), samples=[sample] * n)


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


@pytest.fixture
def ingest(env: Env) -> UplinkIngest:
    return UplinkIngest(env.sessions, env.engine, env.engine._metrics, sample_retention=10)


def failures(env: Env, channel: str, reason: str) -> float:
    return env.engine._metrics.mqtt_failures.labels(channel, reason)._value.get()  # type: ignore[attr-defined]


def device(env: Env, device_id: str = "edge-01") -> Device:
    with session_scope(env.sessions) as db:
        d = db.get(Device, device_id)
        assert d is not None
        return d


@pytest.mark.parametrize(
    "payload",
    [b"", b"{not json", b"\xff\xfe\x00", b"[]", b"null", b'"string"', b"12345", b"{}"],
)
def test_garbage_payloads_are_counted_and_never_raise(env: Env, ingest: UplinkIngest, payload: bytes) -> None:
    before = device(env).last_seen_at
    ingest.handle(HB_TOPIC, payload)  # must not raise: a bad message cannot kill the MQTT network thread
    assert failures(env, "heartbeat", "invalid_payload") == 1
    assert device(env).last_seen_at == before  # no state change from an invalid message


def test_schema_violations_are_rejected(env: Env, ingest: UplinkIngest) -> None:
    good = json.loads(heartbeat().model_dump_json())
    cases = {
        "missing field": {k: v for k, v in good.items() if k != "boot_id"},
        "unknown field": {**good, "surprise": 1},
        "wrong type": {**good, "cpu_percent": "high"},
        "negative value": {**good, "memory_mb": -5},
        "bad device id": {**good, "device_id": "../etc/passwd"},
        "oversized sample list": {**good, "model_stats": "x"},
    }
    for body in cases.values():
        ingest.handle(HB_TOPIC, json.dumps(body).encode())
    assert failures(env, "heartbeat", "invalid_payload") == len(cases)
    assert set(device(env).last_heartbeat) == {"state"}  # unchanged: still the fixture's original heartbeat


def test_oversized_message_is_dropped_before_parsing(env: Env, ingest: UplinkIngest) -> None:
    ingest.handle(HB_TOPIC, b" " * (300 * 1024))
    assert failures(env, "heartbeat", "too_large") == 1
    big = predictions(1).model_dump_json().encode() + b" " * (260 * 1024)
    ingest.handle(PRED_TOPIC, big)
    assert failures(env, "predictions", "too_large") == 1


@pytest.mark.parametrize(
    "topic",
    [
        "fleetops/devices/edge-01/commands",
        "fleetops/devices/edge-01/unknown",
        "other/topic",
        "fleetops/devices/../etc/heartbeat",
        "fleetops/devices/edge-01",
    ],
)
def test_unknown_or_downlink_topics_are_rejected(env: Env, ingest: UplinkIngest, topic: str) -> None:
    ingest.handle(topic, heartbeat().model_dump_json().encode())
    assert failures(env, "unknown", "bad_topic") == 1


def test_payload_device_must_match_topic_device(env: Env, ingest: UplinkIngest) -> None:
    ingest.handle(HB_TOPIC, heartbeat(device="edge-02").model_dump_json().encode())  # edge-01 pretending to be edge-02
    assert failures(env, "heartbeat", "device_mismatch") == 1
    assert device(env, "edge-02").last_heartbeat != json.loads(heartbeat(device="edge-02").model_dump_json())
    ingest.handle(PRED_TOPIC, predictions(device="edge-02").model_dump_json().encode())
    assert failures(env, "predictions", "device_mismatch") == 1
    with session_scope(env.sessions) as db:
        assert db.scalar(select(func.count()).select_from(FeatureSample)) == 0


def test_unregistered_device_is_rejected_and_not_created(env: Env, ingest: UplinkIngest) -> None:
    ingest.handle("fleetops/devices/ghost-01/heartbeat", heartbeat(device="ghost-01").model_dump_json().encode())
    assert failures(env, "heartbeat", "unregistered") == 1
    ingest.handle("fleetops/devices/ghost-01/predictions", predictions(device="ghost-01").model_dump_json().encode())
    with session_scope(env.sessions) as db:
        assert db.get(Device, "ghost-01") is None
        assert db.scalar(select(func.count()).select_from(FeatureSample)) == 0  # no samples from unknown devices


def test_recoverable_processing_failure_is_contained_and_next_message_works(
    env: Env, ingest: UplinkIngest, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    real = env.engine.observe_heartbeat

    def flaky(hb: Heartbeat) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database is locked")
        real(hb)

    monkeypatch.setattr(env.engine, "observe_heartbeat", flaky)
    ingest.handle(HB_TOPIC, heartbeat().model_dump_json().encode())  # engine blows up: contained
    assert failures(env, "heartbeat", "exception") == 1
    ingest.handle(HB_TOPIC, heartbeat().model_dump_json().encode())  # the very next message is processed normally
    assert calls["n"] == 2 and env.engine._metrics.mqtt_messages.labels("heartbeat")._value.get() == 1  # type: ignore[attr-defined]
    assert device(env).status == "ONLINE"


def test_valid_messages_still_work_after_a_burst_of_bad_ones(env: Env, ingest: UplinkIngest) -> None:
    for payload in (b"x", b"{}", b"\x00" * 10, b"[1,2"):
        ingest.handle(HB_TOPIC, payload)
        ingest.handle(PRED_TOPIC, payload)
        ingest.handle(STATUS_TOPIC, payload)
    ingest.handle(PRED_TOPIC, predictions(4).model_dump_json().encode())
    with session_scope(env.sessions) as db:
        assert db.scalar(select(func.count()).select_from(FeatureSample)) == 4


def test_sample_retention_is_bounded(env: Env, ingest: UplinkIngest) -> None:
    for _ in range(5):
        ingest.handle(PRED_TOPIC, predictions(4).model_dump_json().encode())
    with session_scope(env.sessions) as db:
        assert db.scalar(select(func.count()).select_from(FeatureSample)) <= 10 + 4  # retention 10, pruned per batch


def test_status_online_triggers_reconciliation_and_offline_will_marks_device(env: Env, ingest: UplinkIngest) -> None:
    online = StatusMessage(kind="online", device_id="edge-01", sent_at=utcnow(), capabilities=CAPS)
    ingest.handle(STATUS_TOPIC, online.model_dump_json().encode())
    assert device(env).status == "ONLINE"
    will = StatusMessage(kind="offline", device_id="edge-01", sent_at=utcnow())
    ingest.handle(STATUS_TOPIC, will.model_dump_json().encode())
    assert device(env).status == "OFFLINE"
    with session_scope(env.sessions) as db:
        ev = db.scalars(
            select(AuditEvent).where(AuditEvent.action == "device.offline", AuditEvent.device_id == "edge-01")
        ).all()
        assert len(ev) == 1 and ev[0].details["reason"] == "last will"


def test_acks_for_unknown_deployments_are_ignored_safely(env: Env, ingest: UplinkIngest) -> None:
    ack = {
        "kind": "ack",
        "device_id": "edge-01",
        "sent_at": utcnow().isoformat(),
        "command_id": "c" * 8,
        "command_type": "ACTIVATE",
        "deployment_id": "does-not-exist",
        "generation": 1,
        "outcome": "ACTIVE",
    }
    ingest.handle(STATUS_TOPIC, json.dumps(ack).encode())  # no exception, no failure counted, no side effects
    assert failures(env, "status", "exception") == 0


def test_clock_skewed_heartbeat_is_logged_with_numbers(
    env: Env, ingest: UplinkIngest, caplog: pytest.LogCaptureFixture
) -> None:
    skewed = heartbeat(sent_at=utcnow() - timedelta(seconds=400))
    with caplog.at_level(logging.WARNING, logger="fleetops.controlplane.ingest"):
        ingest.handle(HB_TOPIC, skewed.model_dump_json().encode())
    rec = next(r for r in caplog.records if getattr(r, "reason", "") == "clock_skew")
    assert 390 < rec.skew_seconds < 410 and rec.device_id == "edge-01"  # type: ignore[attr-defined]
    caplog.clear()
    ingest.handle(HB_TOPIC, heartbeat().model_dump_json().encode())
    assert not [r for r in caplog.records if getattr(r, "reason", "") == "clock_skew"]


def test_clock_skew_gauge_is_exported_per_device(env: Env, tmp_path: Path) -> None:
    from prometheus_client import generate_latest  # noqa: PLC0415

    from fleetops.monitoring.metrics import ControlPlaneMetrics  # noqa: PLC0415

    metrics = ControlPlaneMetrics(env.sessions)
    ingest = UplinkIngest(env.sessions, env.engine, metrics, sample_retention=10)
    ingest.handle(HB_TOPIC, heartbeat(sent_at=utcnow() - timedelta(seconds=120)).model_dump_json().encode())
    line = next(
        ln
        for ln in generate_latest(metrics.registry).decode().splitlines()
        if ln.startswith("fleetops_device_clock_skew_seconds{")
    )
    assert 'device_id="edge-01"' in line and 115 < float(line.rsplit(" ", 1)[1]) < 130
