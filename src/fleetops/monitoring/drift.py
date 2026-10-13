"""Transparent drift statistics: Population Stability Index + chi-square goodness of fit.

Reference: per feature, 10 quantile bins from the training data (9 inner edges) and the training
proportions in each bin. Current data is bucketed with the same edges.

* PSI = sum((cur_i - ref_i) * ln(cur_i / ref_i))   (proportions floored at 1e-4)
* chi-square p-value: observed bin counts vs. expected counts ref_i·n (scipy).

Large windows make the chi-square test reject for tiny, harmless shifts, so a feature is flagged
only when *both* the effect size (PSI) and the test (p < alpha) agree:
    ALERT: PSI >= alert and p < alpha,   WARN: PSI >= warn and p < alpha,   else OK.
Drift means the *inputs* changed; it does not mean the model is inaccurate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

N_BINS = 10
EPS = 1e-4

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DriftThresholds:
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    p_value: float = 0.01
    min_samples: int = 100


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    p_value: float
    status: str  # OK / WARN / ALERT
    reference_mean: float
    current_mean: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_reference(feature_names: Sequence[str], x: FloatArray, n_bins: int = N_BINS) -> dict[str, Any]:
    if x.ndim != 2 or x.shape[1] != len(feature_names):
        raise ValueError("x must be (n_samples, n_features)")
    out: dict[str, Any] = {"n_samples": int(x.shape[0]), "features": {}}
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    for i, name in enumerate(feature_names):
        col = x[:, i]
        edges = np.unique(np.quantile(col, qs))
        counts = _bin_counts(col, edges)
        out["features"][name] = {
            "bin_edges": [float(e) for e in edges],
            "bin_proportions": [float(c) / len(col) for c in counts],
            "mean": float(col.mean()),
            "std": float(col.std()),
        }
    return out


def _bin_counts(values: FloatArray, edges: FloatArray) -> NDArray[np.int64]:
    idx = np.searchsorted(edges, values, side="right")
    return np.bincount(idx, minlength=len(edges) + 1).astype(np.int64)


def population_stability_index(ref: Sequence[float] | FloatArray, cur: Sequence[float] | FloatArray) -> float:
    r = np.clip(np.asarray(ref, dtype=np.float64), EPS, None)
    c = np.clip(np.asarray(cur, dtype=np.float64), EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def evaluate_feature(name: str, ref: dict[str, Any], values: FloatArray, th: DriftThresholds) -> FeatureDrift:
    edges = np.asarray(ref["bin_edges"], dtype=np.float64)
    counts = _bin_counts(values, edges)
    n = int(counts.sum())
    ref_props = np.asarray(ref["bin_proportions"], dtype=np.float64)
    ref_props = ref_props / ref_props.sum()
    psi = population_stability_index(ref_props, counts / n)
    expected = np.clip(ref_props, EPS, None)
    expected = expected / expected.sum() * n
    p = float(stats.chisquare(f_obs=counts, f_exp=expected).pvalue)
    if p < th.p_value and psi >= th.psi_alert:
        status = "ALERT"
    elif p < th.p_value and psi >= th.psi_warn:
        status = "WARN"
    else:
        status = "OK"
    return FeatureDrift(name, psi, p, status, float(ref["mean"]), float(values.mean()))


def evaluate_drift(
    reference: dict[str, Any], feature_names: Sequence[str], x: FloatArray, th: DriftThresholds
) -> tuple[str, list[FeatureDrift]]:
    """Return ``(overall_status, per_feature)``; INSUFFICIENT_DATA below ``th.min_samples``."""
    if x.shape[0] < th.min_samples:
        return "INSUFFICIENT_DATA", []
    results = [evaluate_feature(name, reference["features"][name], x[:, i], th) for i, name in enumerate(feature_names)]
    statuses = {r.status for r in results}
    overall = "ALERT" if "ALERT" in statuses else "WARN" if "WARN" in statuses else "OK"
    return overall, results


def labelled_quality(anomaly: Sequence[bool], label: Sequence[bool | None]) -> dict[str, Any]:
    """Precision/recall on samples with ground truth; explicit about when it is unmeasurable."""
    pairs = [(a, bool(lbl)) for a, lbl in zip(anomaly, label, strict=True) if lbl is not None]
    if not pairs:
        return {"measured": False, "reason": "no ground-truth labels received in window"}
    tp = sum(1 for a, lbl in pairs if a and lbl)
    fp = sum(1 for a, lbl in pairs if a and not lbl)
    fn = sum(1 for a, lbl in pairs if not a and lbl)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "measured": True,
        "labelled_samples": len(pairs),
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
