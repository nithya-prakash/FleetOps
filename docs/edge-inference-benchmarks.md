# Edge inference runtime benchmark: NumPy vs ONNX Runtime

**Question**: should the edge agent keep its NumPy/safetensors inference runtime, or move to ONNX Runtime (fp32 or INT8)?
**Answer from these measurements: keep NumPy.** For this model (an MLP with hidden layers 16-8, 257 parameters, 6 features) ONNX Runtime
is not meaningfully faster, loads ~10-15x slower, adds a ~76 MB dependency, and INT8 changes outputs without improving anything.
This conclusion applies to *this tiny model on this machine*; a larger model would need to be re-measured.

## Method
* `python -m scripts.benchmarks.build_models` trains the baseline (seed 1, 40 epochs) and writes: the safetensors bundle (current runtime),
  an ONNX fp32 graph built **directly from the trained weights** (Sub/Div/MatMul/Add/Relu/Sigmoid, opset 17, checked with `onnx.checker`, not via a
  framework exporter), ONNX **dynamic INT8**, and ONNX **static INT8** (QDQ, calibrated on 500 training-distribution samples from a separate seed).
* `python -m scripts.benchmarks.run --repeat 5` runs each runtime in its **own process**, single-threaded (OMP/OpenBLAS/ORT intra-op = 1), on identical
  inputs: the repository's fixed evaluation benchmark (4000 samples, seed `EVAL_SEED`), and reports the median of 5 runs (by p50 latency).
* Per run: 200 warm-up calls, then 5000 timed single-sample calls (p50/p95/p99, throughput, process CPU time per call), 20 passes of a 4000-row batch,
  cold first-call latency, import and load time, peak RSS of the process, and precision/recall/F1 at the model's threshold.
* Single-sample timing measures raw model compute only: the NumPy path calls `MlpModel.scores` (without the agent's input validation) so all runtimes
  do the same work. Agent HTTP/MQTT overhead is excluded (see `docs/load-test-results.md`: server-side inference ~0.05 ms inside an HTTP request).
* Accuracy columns compare against the NumPy outputs on the same 4000 inputs.

## Environment
Apple M4 (arm64, 10 cores), macOS 24.6.0, Python 3.12.14, NumPy 2.5.4, onnxruntime 1.31.0 (CPUExecutionProvider), onnx 1.23.2. Not run on x86_64 or on any
physical edge device; relative results on a Raspberry Pi-class CPU may differ.

## Results (median of 5 runs)
| runtime | artifact KB | import s | load ms | cold 1st call ms | p50 ms | p95 ms | p99 ms | single-sample/s | batch/s (4000-row passes) | peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|---|
| NumPy + safetensors (current) | 2.4 | 0.230 | 0.56 | 0.115 | 0.0131 | 0.0138 | 0.0192 | 74.8k | 16.5M | 61.7 |
| ONNX fp32 | 1.5 | 0.044 | 8.56 | 0.336 | 0.0110 | 0.0113 | 0.0117 | 90.0k | 13.4M | 64.8 |
| ONNX INT8 dynamic | 2.5 | 0.044 | 7.18 | 0.306 | 0.0148 | 0.0153 | 0.0165 | 68.2k | 4.1M | 62.0 |
| ONNX INT8 static (QDQ) | 4.4 | 0.041 | 5.51 | 0.300 | 0.0131 | 0.0136 | 0.0143 | 75.3k | 9.6M | 70.3 |

| runtime | max abs score diff vs NumPy | decisions identical to NumPy | precision | recall | F1 |
|---|---|---|---|---|---|
| NumPy | 0 | 100% | 0.668 | 0.647 | 0.658 |
| ONNX fp32 | 1.2e-7 | 100% | 0.668 | 0.647 | 0.658 |
| ONNX INT8 dynamic | 2.3e-2 | 99.95% | 0.675 | 0.647 | 0.661 |
| ONNX INT8 static | 6.8e-2 | 99.90% | 0.668 | 0.647 | 0.658 |

(The import column for NumPy includes `fleetops`' model-format code and pydantic; ONNX's includes `onnxruntime`. Peak RSS is for a process that has
imported only the runtime and the evaluation arrays. Detection quality is not the point here; it is equality across runtimes. NOTE: these latency and accuracy-parity numbers were measured on 2026-10-13 with the earlier 6-feature model and a 4,000-row synthetic benchmark; they were not re-run after the switch to real 5-feature hydraulic-rig data (662 benchmark rows).)

## Interpretation
* **Latency**: ONNX fp32 is ~16% faster per single sample (0.011 vs 0.013 ms), a difference of two microseconds, far below the ~1.7 ms client-observed latency of the agent's HTTP API. For batches NumPy is faster. Neither matters for a 6-feature, 1 Hz-to-kHz sensor workload.
* **INT8 is not worth it here**: no latency gain (dynamic is slower; static equal), 4x lower batch throughput for dynamic, a larger static artifact (QDQ nodes), and scores that deviate by up to 0.07 and flip 2 (dynamic) and 4 (static) decisions out of 4000. Quantisation pays off on large matrix workloads, not on 257 parameters.
* **Cost of ONNX**: ~76 MB of installed `onnxruntime` (NumPy is 22 MB and already required), a ~10-15x slower model load and cold first call, a new native-code dependency in the image to scan and patch, and a second model format to keep in the signed/verified supply chain. fp32 outputs would be numerically equivalent (max difference 1.2e-7, identical decisions), so switching would be safe but unnecessary.
* **When to revisit**: a model with orders of magnitude more parameters (e.g. convolutional vision models, which the optional vision extension would need), a target CPU where ORT's kernels are decisively faster, or a requirement to ship models authored in other frameworks. Then re-run these scripts on the target hardware.

## Reproduce
```bash
uv sync --extra dev --extra control --extra train --extra bench
python -m scripts.benchmarks.build_models
python -m scripts.benchmarks.run --repeat 5      # writes benchmark_results/inference.json (git-ignored)
```
