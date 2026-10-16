"""Train → register pipeline.

Order matters: MLflow assigns the version number, so we log + register first, then copy the
bundle into the artifact store and write the immutable ``model_versions`` row. If anything after
MLflow registration fails, the registry version is deleted and the file removed so nothing is
half-registered.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from fleetops.domain.models import FRAMEWORK, MODEL_FORMAT, MODEL_FORMAT_VERSION, Compatibility, utcnow
from fleetops.persistence.db import session_scope
from fleetops.persistence.tables import ModelVersion
from fleetops.registry.artifact_store import ArtifactStore
from fleetops.registry.mlflow_registry import MlflowRegistry
from fleetops.registry.modelformat import load_bundle, save_bundle
from fleetops.security.artifact import sha256_file
from fleetops.training.pipeline import TrainingResult

logger = logging.getLogger(__name__)


class ModelRegistrationService:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        store: ArtifactStore,
        registry: MlflowRegistry,
    ) -> None:
        self._sessions = sessions
        self._store = store
        self._registry = registry

    def register_training_result(self, model_name: str, result: TrainingResult) -> ModelVersion:
        compat = Compatibility(format=MODEL_FORMAT, format_version=MODEL_FORMAT_VERSION)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "model.safetensors"
            save_bundle(bundle, result.tensors, result.meta)
            load_bundle(bundle).smoke_test()  # never register a bundle we could not load ourselves
            sha = sha256_file(bundle)
            schema = result.meta.input_schema.model_dump(mode="json")
            registered = self._registry.register(
                model_name=model_name,
                bundle=bundle,
                params={
                    **result.params,
                    "dataset_sha256": result.dataset_sha256,
                    **{f"eval_{k}": v for k, v in result.evaluation.items() if k != "note"},
                },
                metrics=result.metrics,
                tags={
                    "sha256": sha,
                    "format": compat.format,
                    "framework": FRAMEWORK,
                    "dataset": "uci-hydraulic-rig-v1",
                    "features": ",".join(result.meta.input_schema.names),
                    "note": "real lab-rig data (UCI hydraulic, CC BY 4.0); not evidence of production readiness",
                },
                extra_json={
                    "input_schema.json": schema,
                    "reference_stats.json": result.reference_stats,
                    "evaluation.json": result.evaluation,
                },
            )
            try:
                stored = self._store.put(model_name, registered.version, bundle)
                with session_scope(self._sessions) as db:
                    row = ModelVersion(
                        name=model_name,
                        version=registered.version,
                        sha256=stored.sha256,
                        size_bytes=stored.size_bytes,
                        framework=FRAMEWORK,
                        compatibility=compat.model_dump(mode="json"),
                        input_schema=schema,
                        metrics=result.metrics,
                        params={**result.params, "dataset_sha256": result.dataset_sha256},
                        reference_stats=result.reference_stats,
                        evaluation=result.evaluation,
                        mlflow_run_id=registered.run_id,
                        created_at=utcnow(),
                    )
                    db.add(row)
                    db.flush()
                    db.expunge(row)
                logger.info("registered model", extra={"model": model_name, "version": registered.version})
                return row
            except Exception:
                logger.exception("post-registration step failed; compensating")
                self._store.delete(model_name, registered.version)
                self._registry.delete_version(model_name, registered.version)
                raise
