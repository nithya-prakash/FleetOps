"""Synthetic factory-sensor dataset (clearly synthetic; see docs/architecture.md §7).

Six correlated process sensors for a motor/pump station. Anomalies are three fault modes
(bearing wear, overheating, pressure leak) injected as 2-4 sigma shifts, so classes overlap and
metrics are not trivially perfect. ``scenario="drift"`` shifts the *healthy* distribution to
exercise drift monitoring without changing labels. Nothing here demonstrates real-world
detection performance.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pandera.pandas as pa

from fleetops.training.sensors import FEATURE_NAMES, SENSORS, Scenario, generate_arrays

__all__ = ["FEATURE_NAMES", "SENSORS", "dataset_fingerprint", "generate_samples", "validate_dataset"]

DATASET_SCHEMA = pa.DataFrameSchema(
    {
        **{s.name: pa.Column(float, pa.Check.in_range(s.mean - 12 * s.std, s.mean + 12 * s.std)) for s in SENSORS},
        "label": pa.Column(int, pa.Check.isin([0, 1])),
        "fault_mode": pa.Column(str),
    },
    strict=True,
    coerce=False,
)


def generate_samples(n: int, anomaly_fraction: float, seed: int, scenario: Scenario = "normal") -> pd.DataFrame:
    if n < 10:
        raise ValueError("n must be >= 10")
    x, labels, modes = generate_arrays(n, anomaly_fraction, seed, scenario)
    df = pd.DataFrame(x, columns=FEATURE_NAMES)
    df["label"] = labels
    df["fault_mode"] = modes
    return df


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return DATASET_SCHEMA.validate(df)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of the dataset content for lineage in MLflow."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()
