from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from fleetops.config.settings import ControlPlaneSettings
from fleetops.persistence.db import (
    make_engine,
    make_session_factory,
    session_scope,
    upgrade_database,
)
from fleetops.persistence.tables import ModelVersion
from fleetops.registry.artifact_store import ArtifactStore
from fleetops.registry.mlflow_registry import MlflowRegistry, RegistryUnavailableError
from fleetops.registry.modelformat import load_bundle
from fleetops.registry.service import ModelRegistrationService
from fleetops.security.artifact import sha256_file
from fleetops.training.pipeline import TrainParams, train_model


def test_train_register_roundtrip(cp_settings: ControlPlaneSettings, tmp_path: Path) -> None:
    upgrade_database(cp_settings.database_url)
    sessions = make_session_factory(make_engine(cp_settings.database_url))
    store = ArtifactStore(cp_settings.artifact_root)
    registry = MlflowRegistry(
        cp_settings.mlflow_tracking_uri,
        cp_settings.mlflow_experiment,
        artifact_location=(tmp_path / "mlruns").as_uri(),
    )
    service = ModelRegistrationService(sessions, store, registry)

    v1 = service.register_training_result("anomaly-detector", train_model(TrainParams(n_samples=600, epochs=3)))
    v2 = service.register_training_result("anomaly-detector", train_model(TrainParams(n_samples=600, epochs=3, seed=5)))
    assert (v1.version, v2.version) == (1, 2)

    path = store.get("anomaly-detector", 2)
    assert sha256_file(path) == v2.sha256
    load_bundle(path).smoke_test()

    listed = {v["version"]: v for v in registry.list_versions("anomaly-detector")}
    assert set(listed) == {1, 2}
    assert listed[2]["tags"]["sha256"] == v2.sha256

    with session_scope(sessions) as db:
        rows = db.scalars(select(ModelVersion).order_by(ModelVersion.version)).all()
        assert [r.version for r in rows] == [1, 2]
        assert rows[0].input_schema["features"][0]["name"] == "pressure_bar"
        assert "test_f1" in rows[0].metrics


def test_registry_failure_leaves_nothing_registered(
    cp_settings: ControlPlaneSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    upgrade_database(cp_settings.database_url)
    sessions = make_session_factory(make_engine(cp_settings.database_url))
    store = ArtifactStore(cp_settings.artifact_root)
    registry = MlflowRegistry("http://127.0.0.1:1", "x")  # unreachable
    service = ModelRegistrationService(sessions, store, registry)
    with pytest.raises(RegistryUnavailableError):
        service.register_training_result("anomaly-detector", train_model(TrainParams(n_samples=300, epochs=1)))
    with session_scope(sessions) as db:
        assert db.scalars(select(ModelVersion)).all() == []
