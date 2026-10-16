"""Small synthetic sensor generator, kept ONLY as a fast fixture for unit tests of pure maths (drift statistics).

Nothing in the product path uses it: training, the fixed benchmark and the edge replay all use the real UCI
hydraulic-rig cycles (``fleetops.training.hydraulic``). Six invented sensors, three fault modes injected as
shifts of 2-4 sigma; ``scenario="drift"`` shifts the healthy distribution without changing labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Scenario = Literal["normal", "drift"]


@dataclass(frozen=True)
class SensorSpec:
    name: str
    mean: float
    std: float


SENSORS: tuple[SensorSpec, ...] = (
    SensorSpec("vibration_rms_mm_s", 2.0, 0.40),
    SensorSpec("temperature_c", 65.0, 4.0),
    SensorSpec("pressure_bar", 6.0, 0.30),
    SensorSpec("motor_current_a", 12.0, 1.0),
    SensorSpec("rotational_speed_rpm", 1480.0, 25.0),
    SensorSpec("acoustic_level_db", 70.0, 3.0),
)
FEATURE_NAMES = [s.name for s in SENSORS]
_IDX = {s.name: i for i, s in enumerate(SENSORS)}

# Correlation between sensors under healthy operation.
_CORR = np.eye(len(SENSORS))
for a, b, r in [
    ("temperature_c", "motor_current_a", 0.5),
    ("vibration_rms_mm_s", "acoustic_level_db", 0.6),
    ("rotational_speed_rpm", "motor_current_a", 0.3),
]:
    _CORR[_IDX[a], _IDX[b]] = _CORR[_IDX[b], _IDX[a]] = r

# fault mode -> {sensor: shift in sigmas}
FAULT_MODES: dict[str, dict[str, float]] = {
    "bearing_wear": {"vibration_rms_mm_s": 1.0, "acoustic_level_db": 0.8},
    "overheating": {"temperature_c": 1.0, "motor_current_a": 0.5},
    "pressure_leak": {"pressure_bar": -1.0, "rotational_speed_rpm": -0.4},
}
# Sigma shift applied to healthy data in the "drift" scenario.
DRIFT_SHIFT_SIGMAS = {"temperature_c": 1.2, "vibration_rms_mm_s": 0.8}


def generate_arrays(
    n: int, anomaly_fraction: float, seed: int, scenario: Scenario = "normal"
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0.0 <= anomaly_fraction <= 0.5:
        raise ValueError("anomaly_fraction must be within [0, 0.5]")
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(_CORR)
    z = rng.standard_normal((n, len(SENSORS))) @ chol.T
    labels = (rng.random(n) < anomaly_fraction).astype(np.int64)
    modes = ["none"] * n
    mode_names = list(FAULT_MODES)
    for i in np.flatnonzero(labels):
        mode = mode_names[int(rng.integers(len(mode_names)))]
        magnitude = rng.uniform(2.0, 4.0)
        modes[i] = mode
        for sensor, factor in FAULT_MODES[mode].items():
            z[i, _IDX[sensor]] += factor * magnitude
    if scenario == "drift":
        for sensor, shift in DRIFT_SHIFT_SIGMAS.items():
            z[:, _IDX[sensor]] += shift
    means = np.array([s.mean for s in SENSORS])
    stds = np.array([s.std for s in SENSORS])
    return z * stds + means, labels, modes


def generate_samples(n: int, anomaly_fraction: float, seed: int, scenario: Scenario = "normal"):  # type: ignore[no-untyped-def]
    x, labels, modes = generate_arrays(n, anomaly_fraction, seed, scenario)
    df = pd.DataFrame(x, columns=FEATURE_NAMES)
    df["label"] = labels
    df["fault_mode"] = modes
    return df
