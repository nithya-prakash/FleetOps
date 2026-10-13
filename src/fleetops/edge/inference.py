"""Thread-safe local inference holder with per-activation statistics."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fleetops.domain.messages import ModelStats
from fleetops.domain.models import FaultConfig, ModelIdentity
from fleetops.registry.modelformat import InputRejectedError, MlpModel, Prediction

LATENCY_WINDOW = 500


class NoModelError(Exception):
    """No active model is loaded (inference unavailable)."""


class InjectedFaultError(Exception):
    """Deterministic failure injected through FaultConfig (simulation)."""


@dataclass(frozen=True)
class InferenceResult:
    prediction: Prediction
    latency_ms: float
    model: ModelIdentity


class InferenceEngine:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: MlpModel | None = None
        self._ident: ModelIdentity | None = None
        self.faults = FaultConfig()
        self._reset_stats()
        self.inputs_rejected = 0

    def _reset_stats(self) -> None:
        self._count = 0
        self._errors = 0
        self._latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._since = time.monotonic()

    @property
    def active(self) -> ModelIdentity | None:
        return self._ident

    def swap(self, model: MlpModel | None, ident: ModelIdentity | None) -> None:
        with self._lock:
            self._model, self._ident = model, ident
            self._reset_stats()

    def predict(self, values: Sequence[float]) -> InferenceResult:
        with self._lock:
            model, ident, faults = self._model, self._ident, self.faults
            if model is None or ident is None:
                raise NoModelError("no active model")
        start = time.perf_counter()
        try:
            if ident.version in faults.fail_inference_versions:
                raise InjectedFaultError(f"injected inference failure for version {ident.version}")
            pred = model.predict(values)
        except InputRejectedError:
            with self._lock:
                self.inputs_rejected += 1  # bad *input* is not a model failure
            raise
        except Exception:
            self._record(ident, (time.perf_counter() - start) * 1000, failed=True)
            raise
        extra = faults.extra_latency_ms.get(str(ident.version), 0.0)
        latency = (time.perf_counter() - start) * 1000 + extra
        self._record(ident, latency, failed=False)
        return InferenceResult(pred, latency, ident)

    def _record(self, ident: ModelIdentity, latency_ms: float, failed: bool) -> None:
        with self._lock:
            if self._ident != ident:  # model swapped mid-call; do not pollute new stats
                return
            self._count += 1
            self._errors += int(failed)
            self._latencies.append(latency_ms)

    def stats(self) -> ModelStats | None:
        with self._lock:
            if self._ident is None:
                return None
            lat = np.array(self._latencies) if self._latencies else np.zeros(1)
            return ModelStats(
                name=self._ident.name,
                version=self._ident.version,
                inference_count=self._count,
                inference_errors=self._errors,
                latency_p50_ms=float(np.percentile(lat, 50)),
                latency_p95_ms=float(np.percentile(lat, 95)),
                active_seconds=time.monotonic() - self._since,
            )
