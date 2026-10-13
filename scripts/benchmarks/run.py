"""Benchmark every runtime in its own process on identical inputs and print a Markdown comparison.

python -m scripts.benchmarks.build_models && python -m scripts.benchmarks.run [--repeat 3]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import psutil

RUNTIMES = ["numpy", "onnx_fp32", "onnx_int8_dynamic", "onnx_int8_static"]


def run_worker(runtime: str, models: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.benchmarks.bench_worker", "--runtime", runtime, "--models", str(models)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"{runtime} failed:\n{proc.stderr[-2000:]}")
    return dict(json.loads(proc.stdout.strip().splitlines()[-1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=Path("benchmark_results/models"))
    ap.add_argument(
        "--repeat", type=int, default=3, help="runs per runtime; the median run (by p50 latency) is reported"
    )
    args = ap.parse_args()
    first = run_worker("numpy", args.models)  # writes scores_numpy.npy, the numerical reference for the others
    np.save(args.models / "reference_scores.npy", np.load(args.models / "scores_numpy.npy"))
    results: dict[str, dict[str, Any]] = {}
    for runtime in RUNTIMES:
        runs = [run_worker(runtime, args.models) for _ in range(args.repeat)]
        runs.sort(key=lambda r: float(r["latency_ms"]["p50"]))
        results[runtime] = runs[len(runs) // 2]
    _ = first
    env = {
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "cpu_count": psutil.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "onnxruntime": onnxruntime.__version__,
        "threads": "1 (OMP/OpenBLAS/ORT intra-op)",
        "repeat": args.repeat,
    }
    out = Path("benchmark_results")
    out.mkdir(exist_ok=True)
    (out / "inference.json").write_text(json.dumps({"environment": env, "results": results}, indent=2))
    meta = json.loads((args.models / "meta.json").read_text())
    print(
        f"environment: {env}\nmodel: MLP hidden {meta['hidden_sizes']}, {meta['parameters']} parameters, 6 features\n"
    )
    print(
        "| runtime | artifact KB | import s | load ms | cold 1st ms | p50 ms | p95 ms | p99 ms | single/s | batch256/s | peak RSS MB | cpu ms/inf | max |dscore| vs numpy | decisions agree | precision | recall | F1 |"
    )
    print("|" + "---|" * 18)
    for rt, r in results.items():
        lat, m = r["latency_ms"], r["metrics"]
        print(
            f"| {rt} | {r['artifact_bytes'] / 1024:.1f} | {r['import_s']} | {r['load_ms']} | {r['first_inference_ms']} | {lat['p50']} | {lat['p95']} | {lat['p99']} | "
            f"{r['single_sample_per_s']:.0f} | {r['batch256_samples_per_s']:.0f} | {r['peak_rss_mb']} | {r['cpu_ms_per_inference']} | "
            f"{r.get('max_abs_score_diff_vs_numpy', 0):.2e} | {r.get('decision_agreement_vs_numpy', 1):.4f} | {m['precision']} | {m['recall']} | {m['f1']} |"
        )


if __name__ == "__main__":
    main()
