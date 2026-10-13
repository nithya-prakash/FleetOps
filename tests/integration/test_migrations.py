from __future__ import annotations

from pathlib import Path

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from fleetops.persistence.db import make_engine, upgrade_database
from fleetops.persistence.tables import Base


def test_migrations_produce_exactly_the_orm_schema(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    upgrade_database(url)
    with make_engine(url).connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == [], f"ORM models and Alembic migrations disagree: {diff}"


def test_upgrade_is_idempotent_and_preserves_existing_rows(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    upgrade_database(url)
    engine = make_engine(url)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO devices (device_id, group_name, description, labels, status, key_epoch, registered_at, "
            "command_generation) VALUES ('edge-01', 'line-a', '', '{}', 'ONLINE', 1, '2026-01-01 00:00:00', 0)"
        )
    upgrade_database(url)  # second run is a no-op
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT device_id FROM devices").scalars().all() == ["edge-01"]
