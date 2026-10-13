"""Device inventory operations (registration, liveness, authentication)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetops.deployment.engine import ConflictError
from fleetops.domain.messages import Heartbeat, StatusMessage
from fleetops.domain.models import utcnow
from fleetops.persistence.audit import record_audit
from fleetops.persistence.tables import Device
from fleetops.security.device_identity import derive_device_secret, verify_device_secret


def register_device(
    db: Session, device_id: str, group: str, description: str, labels: dict[str, Any], actor: str, master: str
) -> tuple[Device, str]:
    if db.get(Device, device_id) is not None:
        raise ConflictError(f"device {device_id} is already registered")
    dev = Device(
        device_id=device_id,
        group_name=group,
        description=description,
        labels=labels,
        status="REGISTERED",
        key_epoch=1,
        registered_at=utcnow(),
        command_generation=0,
    )
    db.add(dev)
    record_audit(db, actor, "device.registered", "device", device_id, device_id=device_id, group=group)
    return dev, derive_device_secret(master, device_id, dev.key_epoch)


def authenticate_device(db: Session, device_id: str, presented: str, master: str) -> Device | None:
    dev = db.get(Device, device_id)
    if dev is None or dev.status == "REVOKED":
        # still burn the same work to avoid an existence oracle via timing
        verify_device_secret(master, device_id, 1, presented)
        return None
    return dev if verify_device_secret(master, device_id, dev.key_epoch, presented) else None


def _mark_online(db: Session, dev: Device, now: datetime) -> None:
    if dev.status != "ONLINE":
        record_audit(
            db,
            f"device:{dev.device_id}",
            "device.online",
            "device",
            dev.device_id,
            device_id=dev.device_id,
            previous=dev.status,
        )
    dev.status, dev.last_seen_at = "ONLINE", now


def apply_heartbeat(db: Session, hb: Heartbeat) -> Device | None:
    dev = db.get(Device, hb.device_id)
    if dev is None or dev.status == "REVOKED":
        return None
    _mark_online(db, dev, utcnow())
    dev.last_heartbeat = hb.model_dump(mode="json")
    dev.capabilities = hb.capabilities.model_dump(mode="json")
    active = hb.state.active
    dev.active_model_name = active.name if active else None
    dev.active_model_version = active.version if active else None
    return dev


def apply_status(db: Session, msg: StatusMessage) -> Device | None:
    dev = db.get(Device, msg.device_id)
    if dev is None or dev.status == "REVOKED":
        return None
    if msg.kind == "online":
        _mark_online(db, dev, utcnow())
        if msg.capabilities:
            dev.capabilities = msg.capabilities.model_dump(mode="json")
    elif msg.kind == "offline" and dev.status == "ONLINE":
        dev.status = "OFFLINE"
        record_audit(
            db,
            f"device:{dev.device_id}",
            "device.offline",
            "device",
            dev.device_id,
            device_id=dev.device_id,
            reason="last will",
        )
    elif msg.kind == "ack":
        dev.last_seen_at = utcnow()
        if msg.state is not None:
            # An ack carries the device's authoritative model state right after a switch. Apply it now so
            # eligibility and rollback-destination capture never rely on a heartbeat up to one interval old.
            hb = dict(dev.last_heartbeat or {})
            hb["state"] = msg.state.model_dump(mode="json")
            dev.last_heartbeat = hb
            active = msg.state.active
            dev.active_model_name = active.name if active else None
            dev.active_model_version = active.version if active else None
    return dev


def mark_stale(db: Session, offline_after_seconds: float, now: datetime | None = None) -> list[str]:
    now = now or utcnow()
    cutoff = now - timedelta(seconds=offline_after_seconds)
    stale = db.scalars(select(Device).where(Device.status == "ONLINE", Device.last_seen_at < cutoff)).all()
    for dev in stale:
        dev.status = "OFFLINE"
        record_audit(
            db,
            "system",
            "device.offline",
            "device",
            dev.device_id,
            device_id=dev.device_id,
            reason=f"no heartbeat for {offline_after_seconds}s",
        )
    return [d.device_id for d in stale]
