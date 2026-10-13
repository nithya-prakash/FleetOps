"""``mlp-v1`` bundle: safetensors weights + JSON metadata, evaluated with NumPy.

Security: safetensors is a data-only format (no pickle, no code execution). Everything read
from the file is validated (shapes, finiteness, metadata schema, size) before use. This module is
imported by the edge agent, so it must not import torch.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, ValidationError, model_validator
from safetensors import safe_open
from safetensors.numpy import save_file

from fleetops.domain.models import MODEL_FORMAT, FeatureSpec, InputSchema, StrictModel

MAX_BUNDLE_BYTES = 64 * 1024 * 1024
META_KEY = "fleetops"
HARD_BOUND_MARGIN = 1.0  # accept values within [min - 1*span, max + 1*span]

Array = NDArray[np.float32]


class ModelFormatError(Exception):
    """The file is not a valid, safe ``mlp-v1`` bundle."""


class InputRejectedError(ValueError):
    """The inference input violates the model's input schema."""


class BundleMeta(StrictModel):
    format: str = MODEL_FORMAT
    features: list[FeatureSpec] = Field(min_length=1, max_length=256)
    hidden_sizes: list[int] = Field(min_length=1, max_length=8)
    threshold: float = Field(ge=0, le=1)
    created_with: str = "pytorch"

    @model_validator(mode="after")
    def _check_format(self) -> BundleMeta:
        if self.format != MODEL_FORMAT:
            raise ValueError(f"unsupported model format {self.format!r}")
        return self

    @property
    def input_schema(self) -> InputSchema:
        return InputSchema(features=self.features)


@dataclass(frozen=True)
class Prediction:
    score: float
    anomaly: bool


class MlpModel:
    """Standardise → [Linear → ReLU]* → Linear → sigmoid."""

    def __init__(self, meta: BundleMeta, tensors: Mapping[str, Array]) -> None:
        self.meta = meta
        n_features = len(meta.features)
        sizes = [n_features, *meta.hidden_sizes, 1]
        try:
            self._mean = tensors["norm.mean"]
            self._std = tensors["norm.std"]
            self._weights = [tensors[f"layers.{i}.weight"] for i in range(len(sizes) - 1)]
            self._biases = [tensors[f"layers.{i}.bias"] for i in range(len(sizes) - 1)]
        except KeyError as exc:
            raise ModelFormatError(f"missing tensor {exc}") from exc
        if self._mean.shape != (n_features,) or self._std.shape != (n_features,):
            raise ModelFormatError("normalisation tensors do not match feature count")
        for i, (w, b) in enumerate(zip(self._weights, self._biases, strict=True)):
            if w.shape != (sizes[i + 1], sizes[i]) or b.shape != (sizes[i + 1],):
                raise ModelFormatError(f"layer {i} has unexpected shape {w.shape}/{b.shape}")
        for t in (self._mean, self._std, *self._weights, *self._biases):
            if not np.all(np.isfinite(t)):
                raise ModelFormatError("non-finite value in tensors")
        if np.any(self._std <= 0):
            raise ModelFormatError("non-positive std")
        lows = np.array([f.min for f in meta.features], dtype=np.float64)
        highs = np.array([f.max for f in meta.features], dtype=np.float64)
        span = highs - lows
        self._lo = lows - HARD_BOUND_MARGIN * span
        self._hi = highs + HARD_BOUND_MARGIN * span

    @property
    def n_features(self) -> int:
        return len(self.meta.features)

    def validate_input(self, values: Sequence[float]) -> Array:
        if len(values) != self.n_features:
            raise InputRejectedError(f"expected {self.n_features} features, got {len(values)}")
        arr = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise InputRejectedError("input contains NaN or infinity")
        bad = np.flatnonzero((arr < self._lo) | (arr > self._hi))
        if bad.size:
            name = self.meta.features[int(bad[0])].name
            raise InputRejectedError(f"feature {name!r} outside plausible range")
        return arr.astype(np.float32)

    def scores(self, x: Array) -> Array:
        h = (np.atleast_2d(x) - self._mean) / self._std
        for w, b in zip(self._weights[:-1], self._biases[:-1], strict=True):
            h = np.maximum(h @ w.T + b, 0.0)
        logits = h @ self._weights[-1].T + self._biases[-1]
        out: Array = (1.0 / (1.0 + np.exp(-logits[:, 0]))).astype(np.float32)
        return out

    def predict(self, values: Sequence[float]) -> Prediction:
        score = float(self.scores(self.validate_input(values))[0])
        if not math.isfinite(score):
            raise InputRejectedError("model produced a non-finite score")
        return Prediction(score=score, anomaly=score >= self.meta.threshold)

    def smoke_test(self) -> None:
        """Run the centre of the training distribution through the model; must be finite."""
        centre = [(f.min + f.max) / 2 for f in self.meta.features]
        self.predict(centre)


def save_bundle(path: Path, tensors: Mapping[str, Array], meta: BundleMeta) -> None:
    payload = json.dumps(meta.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    save_file(
        {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in tensors.items()},
        str(path),
        metadata={META_KEY: payload},
    )


def load_bundle(path: Path) -> MlpModel:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ModelFormatError(f"cannot stat bundle: {exc}") from exc
    if size > MAX_BUNDLE_BYTES:
        raise ModelFormatError(f"bundle too large ({size} bytes)")
    try:
        with safe_open(str(path), framework="numpy") as fh:
            raw = (fh.metadata() or {}).get(META_KEY)
            if raw is None:
                raise ModelFormatError("bundle has no fleetops metadata")
            meta = BundleMeta.model_validate_json(raw)
            tensors = {k: fh.get_tensor(k).astype(np.float32, copy=False) for k in fh.keys()}  # noqa: SIM118
    except ModelFormatError:
        raise
    except (ValidationError, OSError, ValueError, RuntimeError, Exception) as exc:
        # safetensors raises library-specific exceptions for malformed files; treat all as invalid.
        raise ModelFormatError(f"invalid bundle: {exc}") from exc
    return MlpModel(meta, tensors)
