"""Train the baseline, export it to every runtime format under test and verify the exports numerically.

    python -m scripts.benchmarks.build_models --out benchmark_results/models

Formats: safetensors (current NumPy runtime), ONNX fp32 (graph built directly from the trained weights, not via a
framework exporter), ONNX dynamic INT8 (weights quantised, activations quantised at run time) and ONNX static INT8
(QDQ, calibrated on training-distribution data). Nothing is assumed to export or quantise correctly: each artifact is
checked against the PyTorch-equivalent NumPy reference before it is benchmarked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static

from fleetops.registry.modelformat import MlpModel, load_bundle, save_bundle
from fleetops.training.dataset import load_training_samples
from fleetops.training.hydraulic import FEATURE_NAMES, benchmark_arrays
from fleetops.training.pipeline import TrainParams, train_model


def build_onnx(model: MlpModel, path: Path) -> None:
    """((x - mean) / std) -> [MatMul + Add + Relu]* -> MatMul + Add -> Sigmoid, with MatMul/Add (not Gemm) so that
    dynamic INT8 quantisation applies."""
    nodes, inits = [], []
    n_features = model.n_features
    inits += [
        numpy_helper.from_array(model._mean.astype(np.float32), "mean"),
        numpy_helper.from_array(model._std.astype(np.float32), "std"),
    ]
    nodes += [
        helper.make_node("Sub", ["input", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "std"], ["h0"]),
    ]
    last = "h0"
    for i, (w, b) in enumerate(zip(model._weights, model._biases, strict=True)):
        inits += [
            numpy_helper.from_array(np.ascontiguousarray(w.T.astype(np.float32)), f"w{i}"),
            numpy_helper.from_array(b.astype(np.float32), f"b{i}"),
        ]
        nodes += [
            helper.make_node("MatMul", [last, f"w{i}"], [f"mm{i}"]),
            helper.make_node("Add", [f"mm{i}", f"b{i}"], [f"z{i}"]),
        ]
        last = f"z{i}"
        if i < len(model._weights) - 1:
            nodes.append(helper.make_node("Relu", [last], [f"h{i + 1}"]))
            last = f"h{i + 1}"
    nodes.append(helper.make_node("Sigmoid", [last], ["score"]))
    graph = helper.make_graph(
        nodes,
        "fleetops_mlp",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", n_features])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, ["batch", 1])],
        inits,
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_model.ir_version = 9
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(path))


class Calibration(CalibrationDataReader):  # type: ignore[misc]
    def __init__(self, x: np.ndarray) -> None:
        self._it = iter([{"input": x[i : i + 1].astype(np.float32)} for i in range(len(x))])

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._it, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("benchmark_results/models"))
    ap.add_argument("--hidden", default="16,8", help="hidden layer sizes of the model to benchmark")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    result = train_model(TrainParams(seed=1, epochs=args.epochs, hidden_sizes=[int(h) for h in args.hidden.split(",")]))
    bundle = out / "model.safetensors"
    save_bundle(bundle, result.tensors, result.meta)
    model = load_bundle(bundle)
    build_onnx(model, out / "model_fp32.onnx")
    calib_x = (
        load_training_samples()[FEATURE_NAMES].sample(n=500, random_state=4242).to_numpy(np.float32)
    )  # calibration data: real training cycles, never the benchmark
    quantize_dynamic(str(out / "model_fp32.onnx"), str(out / "model_int8_dynamic.onnx"), weight_type=QuantType.QInt8)
    quantize_static(
        str(out / "model_fp32.onnx"),
        str(out / "model_int8_static.onnx"),
        Calibration(calib_x),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    ex, ey = benchmark_arrays()  # the fixed benchmark cycles, saved so workers need no training code
    np.save(out / "eval_x.npy", ex.astype(np.float32))
    np.save(out / "eval_y.npy", ey)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "features": FEATURE_NAMES,
                "threshold": model.meta.threshold,
                "hidden_sizes": model.meta.hidden_sizes,
                "parameters": int(sum(w.size + b.size for w, b in zip(model._weights, model._biases, strict=True))),
            },
            indent=2,
        )
    )
    print(f"wrote {sorted(p.name for p in out.iterdir())}")


if __name__ == "__main__":
    main()
