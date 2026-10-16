"""Training dataset: real hydraulic test-rig cycles (see ``fleetops.training.hydraulic``).

The fixed benchmark cycles are excluded here, so a model is never trained on what the quality gate scores it on.
The synthetic generator in ``fleetops.training.sensors`` is kept only as a small, fast fixture for unit tests.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pandera.pandas as pa

from fleetops.training.hydraulic import FEATURE_NAMES, load_cycles, training_arrays

__all__ = ["FEATURE_NAMES", "dataset_fingerprint", "load_training_samples", "validate_dataset"]


def _schema() -> pa.DataFrameSchema:
    features = load_cycles().features
    columns = {}
    for i, name in enumerate(FEATURE_NAMES):
        low, high = float(features[:, i].min()), float(features[:, i].max())
        margin = max(high - low, 1e-6)  # generous: catches unit mix-ups and NaN-like garbage, not outliers
        columns[name] = pa.Column(float, pa.Check.in_range(low - margin, high + margin))
    columns["label"] = pa.Column(int, pa.Check.isin([0, 1]))
    columns["fault_mode"] = pa.Column(str)
    return pa.DataFrameSchema(columns, strict=True, coerce=False)


def load_training_samples() -> pd.DataFrame:
    x, y, mode = training_arrays()
    df = pd.DataFrame(x, columns=FEATURE_NAMES)
    df["label"] = y
    df["fault_mode"] = [str(m) for m in mode]
    return df


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return _schema().validate(df)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of the dataset content for lineage in MLflow."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()
