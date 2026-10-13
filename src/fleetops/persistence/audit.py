"""Append-only audit trail."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from fleetops.domain.models import utcnow
from fleetops.persistence.tables import AuditEvent


def record_audit(
    db: Session,
    actor: str,
    action: str,
    entity_type: str,
    entity_id: str,
    *,
    device_id: str | None = None,
    deployment_id: str | None = None,
    correlation_id: str | None = None,
    **details: Any,
) -> None:
    db.add(
        AuditEvent(
            ts=utcnow(),
            actor=actor,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            device_id=device_id,
            deployment_id=deployment_id,
            correlation_id=correlation_id or deployment_id,
            details=details,
        )
    )
