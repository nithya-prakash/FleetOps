from __future__ import annotations

from pathlib import Path

import pytest

from fleetops.domain.models import Manifest, ModelIdentity
from fleetops.training.pipeline import TrainingResult, TrainParams, train_model
from tests.helpers import make_manifest, write_bundle


class Bundle:
    """A real trained model bundle on disk plus the manifest/identity a deployment would carry."""

    def __init__(self, tmp: Path, version: int, result: TrainingResult) -> None:
        self.version = version
        self.path = tmp / f"v{version}.safetensors"
        self.sha, self.size = write_bundle(self.path, result)
        self.meta = result.meta
        self.ident = ModelIdentity(name="anomaly-detector", version=version, sha256=self.sha)

    def manifest(self, deployment_id: str = "dep-00000001") -> Manifest:
        return make_manifest("anomaly-detector", self.version, self.sha, self.size, self.meta, deployment_id)


@pytest.fixture(scope="module")
def bundles(tmp_path_factory: pytest.TempPathFactory) -> dict[int, Bundle]:
    tmp = tmp_path_factory.mktemp("bundles")
    return {
        1: Bundle(tmp, 1, train_model(TrainParams(n_samples=800, epochs=5, seed=1))),
        2: Bundle(tmp, 2, train_model(TrainParams(n_samples=800, epochs=5, seed=2))),
    }
