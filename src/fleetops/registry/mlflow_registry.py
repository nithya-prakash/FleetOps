"""MLflow tracking + registry adapter (the system of record for experiments and lineage)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)


def write_mlflow_credentials(username: str, password: str, home: Path | None = None) -> Path:
    """Store the MLflow client credentials in ``~/.mlflow/credentials`` (mode 0600) instead of the process environment.

    MLflow reads this file when ``MLFLOW_TRACKING_USERNAME/PASSWORD`` are unset, so the secret does not appear in
    ``/proc/<pid>/environ`` or in inherited environments. In containers ``HOME`` is a private tmpfs.
    """
    directory = (home or Path.home()) / ".mlflow"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "credentials"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(f"[mlflow]\nmlflow_tracking_username = {username}\nmlflow_tracking_password = {password}\n")
    path.chmod(0o600)
    return path


class RegistryUnavailableError(Exception):
    """MLflow could not be reached or rejected the request."""


@dataclass(frozen=True)
class RegisteredVersion:
    name: str
    version: int
    run_id: str


class MlflowRegistry:
    def __init__(self, tracking_uri: str, experiment: str, artifact_location: str | None = None) -> None:
        # Bounded retries/timeouts: an unavailable MLflow must fail fast (API returns 502).
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
        self._client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        self._experiment_name = experiment
        self._artifact_location = artifact_location
        self._experiment_id: str | None = None

    def _experiment(self) -> str:
        if self._experiment_id is None:
            exp = self._client.get_experiment_by_name(self._experiment_name)
            self._experiment_id = (
                exp.experiment_id
                if exp is not None
                else self._client.create_experiment(self._experiment_name, artifact_location=self._artifact_location)
            )
        return self._experiment_id

    def register(
        self,
        *,
        model_name: str,
        bundle: Path,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str],
        extra_json: dict[str, Any],
    ) -> RegisteredVersion:
        """Log a run with its artifacts and create a registry version. Compensates on failure."""
        run_id: str | None = None
        try:
            run = self._client.create_run(self._experiment(), tags={"fleetops.model": model_name})
            run_id = run.info.run_id
            for key, value in params.items():
                self._client.log_param(run_id, key, json.dumps(value) if isinstance(value, list) else value)
            for key, metric in metrics.items():
                self._client.log_metric(run_id, key, float(metric))
            self._client.log_artifact(run_id, str(bundle), artifact_path="model")
            with tempfile.TemporaryDirectory() as tmp:
                for fname, payload in extra_json.items():
                    path = Path(tmp) / fname
                    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
                    self._client.log_artifact(run_id, str(path), artifact_path="metadata")
            self._client.set_terminated(run_id, "FINISHED")
            try:
                self._client.create_registered_model(model_name)
            except MlflowException as exc:
                if "already exists" not in str(exc).lower() and "RESOURCE_ALREADY_EXISTS" not in str(exc):
                    raise
            mv = self._client.create_model_version(
                model_name,
                source=f"{run.info.artifact_uri}/model",
                run_id=run_id,
                tags=tags,
            )
            return RegisteredVersion(name=model_name, version=int(mv.version), run_id=run_id)
        except Exception as exc:
            if run_id is not None:
                try:
                    self._client.set_terminated(run_id, "FAILED")
                except Exception:
                    logger.warning("could not mark run %s failed", run_id)
            raise RegistryUnavailableError(f"MLflow registration failed: {exc}") from exc

    def delete_version(self, name: str, version: int) -> None:
        try:
            self._client.delete_model_version(name, str(version))
        except Exception:
            logger.exception("compensating delete of %s:%s failed", name, version)

    def list_versions(self, name: str) -> list[dict[str, Any]]:
        try:
            versions = self._client.search_model_versions(f"name='{name}'")
        except MlflowException as exc:
            raise RegistryUnavailableError(str(exc)) from exc
        return [{"version": int(v.version), "run_id": v.run_id, "tags": dict(v.tags)} for v in versions]

    def ping(self) -> bool:
        try:
            self._client.search_experiments(max_results=1)
        except Exception:
            return False
        return True
