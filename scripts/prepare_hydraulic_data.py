"""Rebuilds src/fleetops/training/resources/hydraulic_cycles.csv from the original UCI download.

    python scripts/prepare_hydraulic_data.py [--zip path/to/zip]

Source: "Condition monitoring of hydraulic systems", Helwig, Pignanelli and Schuetze (ZeMA gGmbH /
Saarland University), UCI Machine Learning Repository, https://doi.org/10.24432/C5CW21, licence CC BY 4.0.
A hydraulic test rig repeats a 60 s load cycle 2,205 times while four components (cooler, valve, pump,
accumulator) are set to known conditions. The committed CSV is a derived, much smaller file (one row per
cycle: the mean of each physical sensor over the cycle, plus the four condition labels); this script is
how it was made, so the numbers can be checked against the original.

Only the five physical sensors are kept (pressure PS1, temperature TS1, flow FS1, vibration VS1, motor
power EPS1). The dataset's "virtual" sensors (cooling efficiency, cooling power, efficiency factor) are
computed from other sensors and would not exist on a plain edge device, so they are left out.
"""

import argparse
import csv
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
SHA256 = "24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2"
OUT = Path(__file__).resolve().parent.parent / "src/fleetops/training/resources/hydraulic_cycles.csv"

SENSORS = {
    "pressure_bar": "PS1.txt",
    "temperature_c": "TS1.txt",
    "flow_l_min": "FS1.txt",
    "vibration_mm_s": "VS1.txt",
    "motor_power_w": "EPS1.txt",
}
LABELS = ["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure", "stable_flag"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, help="use a local copy of the UCI zip instead of downloading")
    args = parser.parse_args()
    if args.zip:
        data = args.zip.read_bytes()
    else:
        request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 (fixed https URL)
            data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"checksum mismatch: expected {SHA256}, got {digest}")
    archive = zipfile.ZipFile(io.BytesIO(data))
    columns = {
        name: np.loadtxt(io.BytesIO(archive.read(member)), delimiter="\t").mean(axis=1)
        for name, member in SENSORS.items()
    }
    labels = np.loadtxt(io.BytesIO(archive.read("profile.txt")), delimiter="\t")
    n = len(labels)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cycle", *SENSORS, *LABELS])
        for i in range(n):
            writer.writerow([i, *(f"{columns[name][i]:.4f}" for name in SENSORS), *(int(v) for v in labels[i])])
    print(f"wrote {n} cycles to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
