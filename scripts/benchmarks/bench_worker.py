"""One runtime, one process: cold start, load time, latency distribution, throughput, memory, CPU, accuracy.

Invoked by scripts/benchmarks/run.py; prints a single JSON object. Single-threaded by construction so runtimes are
compared under the same constraint an edge CPU slot would impose.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

LAT_N, WARM = 5000, 200


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def prf(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp, fp, fn = float((pred & (y == 1)).sum()), float((pred & (y == 0)).sum()), float((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True, choices=["numpy", "onnx_fp32", "onnx_int8_dynamic", "onnx_int8_static"])
    ap.add_argument("--models", type=Path, required=True)
    a = ap.parse_args()
    meta = json.loads((a.models / "meta.json").read_text())
    threshold = float(meta["threshold"])
    rss_before_import = rss_mb()

    t_import = time.perf_counter()
    if a.runtime == "numpy":
        from fleetops.registry.modelformat import load_bundle  # noqa: PLC0415

        artifact = a.models / "model.safetensors"
    else:
        import onnxruntime as ort  # noqa: PLC0415

        artifact = (
            a.models
            / {
                "onnx_fp32": "model_fp32.onnx",
                "onnx_int8_dynamic": "model_int8_dynamic.onnx",
                "onnx_int8_static": "model_int8_static.onnx",
            }[a.runtime]
        )
    import_s = time.perf_counter() - t_import

    t_load = time.perf_counter()
    if a.runtime == "numpy":
        model = load_bundle(artifact)

        def infer(x: np.ndarray) -> np.ndarray:
            return model.scores(x)

        def infer_one(row: list[float]) -> float:
            return float(
                model.scores(np.asarray([row], dtype=np.float32))[0]
            )  # raw compute; no input validation, like the ONNX paths
    else:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(artifact), opts, providers=["CPUExecutionProvider"])

        def infer(x: np.ndarray) -> np.ndarray:
            return np.asarray(sess.run(None, {"input": x.astype(np.float32)})[0])[:, 0]

        def infer_one(row: list[float]) -> float:
            return float(sess.run(None, {"input": np.asarray([row], dtype=np.float32)})[0][0, 0])

    load_s = time.perf_counter() - t_load

    x, y = np.load(a.models / "eval_x.npy"), np.load(a.models / "eval_y.npy")
    rows = [r.tolist() for r in x]
    t0 = time.perf_counter()
    infer_one(rows[0])
    first_ms = (time.perf_counter() - t0) * 1000  # cold first inference
    for r in rows[:WARM]:
        infer_one(r)
    cpu0, wall0 = time.process_time(), time.perf_counter()
    lat = []
    for i in range(LAT_N):
        s = time.perf_counter()
        infer_one(rows[i % len(rows)])
        lat.append((time.perf_counter() - s) * 1000)
    cpu_s, wall_s = time.process_time() - cpu0, time.perf_counter() - wall0
    t1 = time.perf_counter()
    for _ in range(20):
        scores = infer(x.astype(np.float32))
    batch_rps = (20 * len(x)) / (time.perf_counter() - t1)
    scores = np.asarray(scores, dtype=np.float64)

    reference = np.load(a.models / "reference_scores.npy") if (a.models / "reference_scores.npy").exists() else None
    out = {
        "runtime": a.runtime,
        "artifact_bytes": artifact.stat().st_size,
        "import_s": round(import_s, 4),
        "load_ms": round(load_s * 1000, 3),
        "first_inference_ms": round(first_ms, 4),
        "latency_ms": {f"p{q}": round(float(np.percentile(lat, q)), 5) for q in (50, 95, 99)}
        | {"mean": round(float(np.mean(lat)), 5)},
        "single_sample_per_s": round(LAT_N / wall_s, 0),
        "cpu_ms_per_inference": round(cpu_s / LAT_N * 1000, 5),
        "batch256_samples_per_s": round(batch_rps, 0),
        "peak_rss_mb": round(rss_mb(), 1),
        "rss_mb_at_start": round(rss_before_import, 1),
        "metrics": {k: round(v, 4) for k, v in prf(y, scores >= threshold).items()},
        "scores_sha": None,
    }
    np.save(a.models / f"scores_{a.runtime}.npy", scores)
    if reference is not None:
        out["max_abs_score_diff_vs_numpy"] = float(np.max(np.abs(scores - reference)))
        out["decision_agreement_vs_numpy"] = float(np.mean((scores >= threshold) == (reference >= threshold)))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
