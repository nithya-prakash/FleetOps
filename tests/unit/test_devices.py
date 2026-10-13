from __future__ import annotations

from pathlib import Path

from fleetops.controlplane.devices import apply_status
from fleetops.domain.messages import AckOutcome, CommandType, DeviceModelState, StatusMessage
from fleetops.domain.models import ModelIdentity, utcnow
from fleetops.persistence.db import make_engine, make_session_factory, session_scope, upgrade_database
from fleetops.persistence.tables import Device


def test_ack_state_updates_inventory_immediately(tmp_path: Path) -> None:
    """Regression: after a rollback the inventory must not wait for the next heartbeat (stale v2 -> device skipped)."""
    url = f"sqlite:///{tmp_path / 'd.db'}"
    upgrade_database(url)
    sessions = make_session_factory(make_engine(url))
    v1 = ModelIdentity(name="anomaly", version=1, sha256="1" * 64)
    with session_scope(sessions) as db:
        db.add(
            Device(
                device_id="edge-01",
                group_name="line-a",
                status="ONLINE",
                registered_at=utcnow(),
                active_model_name="anomaly",
                active_model_version=2,
                last_heartbeat={"state": {"active": {"name": "anomaly", "version": 2, "sha256": "2" * 64}}},
            )
        )
    ack = StatusMessage(
        kind="ack",
        device_id="edge-01",
        sent_at=utcnow(),
        command_id="c" * 8,
        command_type=CommandType.ROLLBACK,
        outcome=AckOutcome.ROLLED_BACK,
        state=DeviceModelState(active=v1, last_generation=5),
    )
    with session_scope(sessions) as db:
        apply_status(db, ack)
    with session_scope(sessions) as db:
        dev = db.get(Device, "edge-01")
        assert dev is not None
        assert (dev.active_model_name, dev.active_model_version) == ("anomaly", 1)
        assert dev.last_heartbeat["state"]["active"]["version"] == 1  # rollback destination capture is fresh too
