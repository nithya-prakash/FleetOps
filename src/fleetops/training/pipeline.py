"""PyTorch training and evaluation of the sensor-anomaly MLP; exports an ``mlp-v1`` bundle."""

from __future__ import annotations

import hashlib
import platform
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from torch import nn

from fleetops.domain.models import FeatureSpec
from fleetops.domain.quality import EVAL_SUITE, METRIC_DEFINITIONS_VERSION
from fleetops.monitoring.drift import build_reference
from fleetops.registry.modelformat import BundleMeta, MlpModel
from fleetops.training.dataset import (
    FEATURE_NAMES,
    dataset_fingerprint,
    load_training_samples,
    validate_dataset,
)
from fleetops.training.hydraulic import BENCHMARK_SEED, benchmark_arrays

Array = NDArray[np.float32]

# Fixed benchmark used to compare *any* two models (training seeds differ, so their own test splits do not).
# It is a stratified 30% of the real hydraulic-rig cycles that training never sees (fleetops.training.hydraulic).
EVAL_SEED = BENCHMARK_SEED
EVAL_LATENCY_SAMPLES = 500
THRESHOLD_POLICY = "model-own-threshold"


class TrainParams(BaseModel):
    # Cap on the training cycles used (a random subset of the ~1,540 real cycles outside the benchmark).
    # Values above the pool size use all of them; small values make deliberately weaker models.
    n_samples: int = Field(default=6000, ge=200, le=200_000)
    hidden_sizes: list[int] = Field(default_factory=lambda: [16, 8], min_length=1, max_length=4)
    epochs: int = Field(default=40, ge=1, le=200)
    learning_rate: float = Field(default=0.01, gt=0, le=1)
    batch_size: int = Field(default=128, ge=8, le=4096)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

    def hidden_ok(self) -> bool:
        return all(1 <= h <= 256 for h in self.hidden_sizes)


@dataclass(frozen=True)
class TrainingResult:
    tensors: dict[str, Array]
    meta: BundleMeta
    metrics: dict[str, float]
    params: dict[str, Any]
    reference_stats: dict[str, Any]
    dataset_sha256: str
    evaluation: dict[str, Any]


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _build_net(n_in: int, hidden: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def _roc_auc(y: NDArray[np.int_], s: NDArray[np.floating[Any]]) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _prf(y: NDArray[np.int_], pred: NDArray[np.bool_]) -> tuple[float, float, float]:
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def train_model(params: TrainParams) -> TrainingResult:
    if not params.hidden_ok():
        raise ValueError("hidden layer sizes must be within [1, 256]")
    if params.seed == EVAL_SEED:
        raise ValueError("seed is reserved for the evaluation benchmark")
    _set_determinism(params.seed)
    df = validate_dataset(load_training_samples())
    if params.n_samples < len(df):
        df = df.sample(n=params.n_samples, random_state=params.seed).reset_index(drop=True)
    x_all = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y_all = df["label"].to_numpy(dtype=np.int64)

    perm = np.random.default_rng(params.seed).permutation(len(df))
    n_train, n_val = int(0.7 * len(df)), int(0.15 * len(df))
    tr, va, te = np.split(perm, [n_train, n_train + n_val])

    mean = x_all[tr].mean(axis=0)
    std = np.maximum(x_all[tr].std(axis=0), 1e-6)

    def norm(a: Array) -> torch.Tensor:
        return torch.from_numpy((a - mean) / std)

    net = _build_net(len(FEATURE_NAMES), params.hidden_sizes)
    pos_weight = torch.tensor([(1 - y_all[tr].mean()) / max(y_all[tr].mean(), 1e-6)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=params.learning_rate)
    x_tr, y_tr = norm(x_all[tr]), torch.from_numpy(y_all[tr]).float().unsqueeze(1)
    gen = torch.Generator().manual_seed(params.seed)
    last_loss = float("nan")
    for _ in range(params.epochs):
        net.train()
        order = torch.randperm(len(x_tr), generator=gen)
        for i in range(0, len(order), params.batch_size):
            idx = order[i : i + params.batch_size]
            opt.zero_grad()
            loss = loss_fn(net(x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

    net.eval()

    def scores(idx: NDArray[np.int_]) -> NDArray[np.float32]:
        with torch.no_grad():
            out: NDArray[np.float32] = torch.sigmoid(net(norm(x_all[idx]))).numpy()[:, 0]
        return out

    s_val, s_te = scores(va), scores(te)
    grid = np.linspace(0.05, 0.99, 95)
    f1s = [_prf(y_all[va], s_val >= t)[2] for t in grid]
    threshold = float(grid[int(np.argmax(f1s))])
    p, r, f1 = _prf(y_all[te], s_te >= threshold)
    metrics = {
        "test_precision": p,
        "test_recall": r,
        "test_f1": f1,
        "test_roc_auc": _roc_auc(y_all[te], s_te),
        "val_f1": float(max(f1s)),
        "final_train_loss": last_loss,
        "threshold": threshold,
    }

    tensors: dict[str, Array] = {"norm.mean": mean.astype(np.float32), "norm.std": std.astype(np.float32)}
    linear = [m for m in net if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linear):
        tensors[f"layers.{i}.weight"] = layer.weight.detach().numpy().astype(np.float32)
        tensors[f"layers.{i}.bias"] = layer.bias.detach().numpy().astype(np.float32)

    features = [
        FeatureSpec(name=n, min=float(x_all[:, i].min()), max=float(x_all[:, i].max()))
        for i, n in enumerate(FEATURE_NAMES)
    ]
    meta = BundleMeta(features=features, hidden_sizes=params.hidden_sizes, threshold=threshold)
    reference = build_reference(FEATURE_NAMES, x_all[tr].astype(np.float64))
    bench_metrics, evaluation = run_benchmark(MlpModel(meta, tensors))
    metrics.update(bench_metrics)
    return TrainingResult(
        tensors=tensors,
        meta=meta,
        metrics=metrics,
        params=params.model_dump(),
        reference_stats=reference,
        dataset_sha256=dataset_fingerprint(df),
        evaluation=evaluation,
    )


def latency_environment() -> str:
    major, minor, _ = platform.python_version_tuple()
    return f"{platform.system().lower()}-{platform.machine().lower()}-py{major}.{minor}-np{np.__version__}"


def run_benchmark(model: MlpModel) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate on the fixed benchmark. Returns ``(bench_* metrics, evaluation metadata)``.

    Quality metrics use the model's own decision threshold (``THRESHOLD_POLICY``). Latency is the p95 of
    single-sample ``predict`` calls (validation included) on *this* host, so it is only comparable between
    models benchmarked in the same environment (``latency_environment``).
    """
    x, y = benchmark_arrays()
    scores = model.scores(x.astype(np.float32))
    pred = scores >= model.meta.threshold
    precision, recall, f1 = _prf(y, pred)
    negatives = float(np.sum(y == 0))
    fpr = float(np.sum(pred & (y == 0))) / negatives if negatives else 0.0
    sample = [x[i].tolist() for i in range(EVAL_LATENCY_SAMPLES)]
    for row in sample[:50]:  # warm-up
        model.predict(row)
    timings = []
    for row in sample:
        start = time.perf_counter()
        model.predict(row)
        timings.append((time.perf_counter() - start) * 1000)
    metrics = {
        "bench_precision": precision,
        "bench_recall": recall,
        "bench_f1": f1,
        "bench_roc_auc": _roc_auc(y, scores),
        "bench_false_positive_rate": fpr,
        "bench_p95_latency_ms": float(np.percentile(timings, 95)),
    }
    evaluation = {
        "suite": EVAL_SUITE,
        "metric_definitions_version": METRIC_DEFINITIONS_VERSION,
        "threshold_policy": THRESHOLD_POLICY,
        "dataset_fingerprint": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
        "n_samples": len(y),
        "anomaly_fraction": round(float(y.mean()), 4),
        "eval_seed": EVAL_SEED,
        "latency_environment": latency_environment(),
        "note": (
            "fixed hold-out of real hydraulic test-rig cycles (UCI, CC BY 4.0); one lab rig with deliberately "
            "applied faults, not evidence of production performance"
        ),
    }
    return metrics, evaluation
