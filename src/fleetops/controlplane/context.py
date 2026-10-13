"""Wiring of the control plane's collaborators (constructed once, injected via app.state)."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from fleetops.config.settings import ControlPlaneSettings
from fleetops.controlplane.ingest import UplinkIngest
from fleetops.deployment.engine import RolloutEngine
from fleetops.messaging.bus import MessageBus
from fleetops.monitoring.drift_service import DriftService
from fleetops.monitoring.metrics import ControlPlaneMetrics
from fleetops.registry.artifact_store import ArtifactStore
from fleetops.registry.mlflow_registry import MlflowRegistry
from fleetops.registry.service import ModelRegistrationService
from fleetops.security.auth import ApiKeyAuthenticator
from fleetops.security.ratelimit import TokenBucketLimiter


@dataclass
class AppContext:
    settings: ControlPlaneSettings
    db_engine: Engine
    sessions: sessionmaker[Session]
    store: ArtifactStore
    registry: MlflowRegistry
    registration: ModelRegistrationService
    engine: RolloutEngine
    metrics: ControlPlaneMetrics
    ingest: UplinkIngest
    drift: DriftService
    auth: ApiKeyAuthenticator
    bus: MessageBus | None
    limiter: TokenBucketLimiter
    auth_failures: TokenBucketLimiter
