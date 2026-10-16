"""Real sensor data for training, the fixed benchmark and the edge replay.

Source: "Condition monitoring of hydraulic systems" (Helwig, Pignanelli and Schuetze, ZeMA gGmbH / Saarland
University, UCI Machine Learning Repository, https://doi.org/10.24432/C5CW21, CC BY 4.0). A hydraulic test rig
repeated a 60 s load cycle 2,205 times while four components were set to known conditions. The packaged file
``resources/hydraulic_cycles.csv`` holds one row per cycle (the mean of five physical sensors over the cycle)
with the four condition labels; ``scripts/prepare_hydraulic_data.py`` shows how it was derived.

This module uses only the standard library and numpy, because edge agents (which replay these cycles) run a
lean image without pandas.

What this data is and is not:
* It is real recordings from one lab rig, not a factory fleet. Faults were applied deliberately, so the fault
  rate (about a third of cycles) says nothing about a production line.
* "Anomaly" here means one of two severe faults: a severe internal pump leak, or a valve close to total
  failure. Cooler and accumulator conditions vary freely and act as realistic nuisance variation.
* The benchmark is a fixed, stratified 30% of the cycles that training never sees, chosen with a reserved seed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Scenario = Literal["normal", "drift"]

RESOURCE = Path(__file__).parent / "resources" / "hydraulic_cycles.csv"
FEATURE_NAMES = ["pressure_bar", "temperature_c", "flow_l_min", "vibration_mm_s", "motor_power_w"]

PUMP_SEVERE_LEAK = 2  # pump_leakage: 0 none, 1 weak, 2 severe
VALVE_NEAR_FAILURE = 73  # valve_condition %: 100 optimal ... 73 close to total failure
COOLER_NEAR_FAILURE = 3  # cooler_condition %: 100 full efficiency, 20 reduced, 3 close to total failure

BENCHMARK_SEED = 20_240_601
BENCHMARK_FRACTION = 0.30


@dataclass(frozen=True)
class Cycles:
    """All recorded cycles as parallel arrays (row i is cycle i)."""

    features: NDArray[np.float64]  # (n, 5), columns as FEATURE_NAMES
    cooler: NDArray[np.int64]
    valve: NDArray[np.int64]
    pump: NDArray[np.int64]
    accumulator: NDArray[np.int64]
    label: NDArray[np.int64]  # 1 = severe pump leak or valve near total failure
    fault_mode: NDArray[np.str_]

    def __len__(self) -> int:
        return len(self.label)


@lru_cache(maxsize=1)
def load_cycles() -> Cycles:
    with RESOURCE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    features = np.array([[float(r[name]) for name in FEATURE_NAMES] for r in rows], dtype=np.float64)

    def col(name: str) -> NDArray[np.int64]:
        return np.array([int(r[name]) for r in rows], dtype=np.int64)

    cooler, valve = col("cooler_condition"), col("valve_condition")
    pump, accumulator = col("pump_leakage"), col("accumulator_pressure")
    pump_bad, valve_bad = pump == PUMP_SEVERE_LEAK, valve == VALVE_NEAR_FAILURE
    mode = np.full(len(rows), "none", dtype="<U24")
    mode[pump_bad & ~valve_bad] = "pump_leak"
    mode[valve_bad & ~pump_bad] = "valve_failure"
    mode[pump_bad & valve_bad] = "pump_leak+valve_failure"
    return Cycles(features, cooler, valve, pump, accumulator, (pump_bad | valve_bad).astype(np.int64), mode)


@lru_cache(maxsize=1)
def benchmark_mask() -> NDArray[np.bool_]:
    """Fixed stratified hold-out: BENCHMARK_FRACTION of anomalies and of normal cycles, never trained on."""
    cycles = load_cycles()
    rng = np.random.default_rng(BENCHMARK_SEED)
    mask = np.zeros(len(cycles), dtype=bool)
    for value in (0, 1):
        idx = np.flatnonzero(cycles.label == value)
        chosen = rng.choice(idx, size=round(len(idx) * BENCHMARK_FRACTION), replace=False)
        mask[chosen] = True
    return mask


def benchmark_arrays() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    cycles, mask = load_cycles(), benchmark_mask()
    return cycles.features[mask], cycles.label[mask]


def training_arrays() -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.str_]]:
    """Cycles available for training, validation and test splits (the benchmark cycles are excluded)."""
    cycles, keep = load_cycles(), ~benchmark_mask()
    return cycles.features[keep], cycles.label[keep], cycles.fault_mode[keep]


def stream_pool(scenario: Scenario) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Cycles an edge device replays.

    ``normal``: every cycle, so the stream resembles the training data. ``drift``: only cycles where the cooler
    was close to total failure. The recorded cooler state shifts every sensor, which is a real change in
    operating regime, while the anomaly labelling is untouched.
    """
    cycles = load_cycles()
    keep = cycles.cooler == COOLER_NEAR_FAILURE if scenario == "drift" else np.ones(len(cycles), dtype=bool)
    return cycles.features[keep], cycles.label[keep]
