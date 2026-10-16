from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandera.errors
import pytest
import torch

from fleetops.monitoring.drift import (
    DriftThresholds,
    build_reference,
    evaluate_drift,
    labelled_quality,
    population_stability_index,
)
from fleetops.registry.artifact_store import ArtifactStore, ArtifactStoreError
from fleetops.registry.mlflow_registry import MlflowRegistry
from fleetops.registry.modelformat import (
    InputRejectedError,
    ModelFormatError,
    load_bundle,
    save_bundle,
)
from fleetops.security.artifact import sha256_file
from fleetops.training.dataset import FEATURE_NAMES, load_training_samples, validate_dataset
from fleetops.training.hydraulic import benchmark_arrays, load_cycles, stream_pool
from fleetops.training.pipeline import TrainingResult, TrainParams, train_model
from tests.synthetic_sensors import FEATURE_NAMES as SYNTH_FEATURES
from tests.synthetic_sensors import generate_samples


@pytest.fixture(scope="module")
def trained() -> TrainingResult:
    return train_model(TrainParams(n_samples=2000, epochs=15))


@pytest.fixture
def bundle(tmp_path: Path, trained: TrainingResult) -> Path:
    p = tmp_path / "model.safetensors"
    save_bundle(p, trained.tensors, trained.meta)
    return p


def test_real_dataset_labels_match_the_recorded_conditions() -> None:
    cycles = load_cycles()
    assert len(cycles) == 2205
    pump, valve = cycles.pump == 2, cycles.valve == 73
    assert int(pump.sum()) == 492 and int(valve.sum()) == 360  # counts published with the UCI dataset
    assert (cycles.label == (pump | valve)).all()
    assert 0.3 < cycles.label.mean() < 0.4
    assert set(cycles.fault_mode) == {"none", "pump_leak", "valve_failure", "pump_leak+valve_failure"}


def test_benchmark_is_held_out_fixed_and_stratified() -> None:
    train = load_training_samples()
    bx, by = benchmark_arrays()
    assert len(train) + len(by) == 2205
    # no benchmark cycle may appear in training: compare rows of all five sensors
    train_rows = {tuple(r) for r in train[FEATURE_NAMES].round(4).to_numpy()}
    assert not any(tuple(np.round(r, 4)) in train_rows for r in bx)
    assert abs(by.mean() - load_cycles().label.mean()) < 0.01
    x2, y2 = benchmark_arrays()
    assert np.array_equal(bx, x2) and np.array_equal(by, y2)


def test_training_set_is_deterministic_and_valid() -> None:
    a, b = load_training_samples(), load_training_samples()
    assert a.equals(b)
    validate_dataset(a)


def test_dataset_schema_rejects_bad_data() -> None:
    df = load_training_samples().head(100).copy()
    df.loc[0, "temperature_c"] = 1e6
    with pytest.raises(pandera.errors.SchemaError):
        validate_dataset(df)


def test_numpy_runtime_matches_torch(trained: TrainingResult, bundle: Path) -> None:
    model = load_bundle(bundle)
    df = load_training_samples().head(200)
    x = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    # rebuild the torch net from the same tensors and compare outputs
    t = trained.tensors
    h = torch.from_numpy((x - t["norm.mean"]) / t["norm.std"])
    n_layers = len(trained.meta.hidden_sizes) + 1
    for i in range(n_layers):
        h = h @ torch.from_numpy(t[f"layers.{i}.weight"]).T + torch.from_numpy(t[f"layers.{i}.bias"])
        if i < n_layers - 1:
            h = torch.relu(h)
    expected = torch.sigmoid(h)[:, 0].numpy()
    np.testing.assert_allclose(model.scores(x), expected, rtol=1e-5, atol=1e-6)


def test_training_is_reproducible(tmp_path: Path) -> None:
    hashes = []
    for i in range(2):
        r = train_model(TrainParams(n_samples=800, epochs=5, seed=7))
        p = tmp_path / f"{i}.safetensors"
        save_bundle(p, r.tensors, r.meta)
        hashes.append(sha256_file(p))
    assert hashes[0] == hashes[1]


def test_model_beats_chance(trained: TrainingResult) -> None:
    assert trained.metrics["test_roc_auc"] > 0.85
    assert trained.metrics["bench_roc_auc"] > 0.85  # on the fixed held-out real cycles


def test_prediction_and_input_validation(bundle: Path) -> None:
    model = load_bundle(bundle)
    x, y = benchmark_arrays()
    healthy = x[y == 0][0].tolist()
    faulty = x[y == 1][0].tolist()
    assert not model.predict(healthy).anomaly
    assert model.predict(faulty).anomaly
    assert model.predict(faulty).score > model.predict(healthy).score
    with pytest.raises(InputRejectedError, match="expected 5"):
        model.predict([1.0])
    with pytest.raises(InputRejectedError, match="NaN"):
        model.predict([float("nan")] * 5)
    bad = list(healthy)
    bad[1] = 5000.0
    with pytest.raises(InputRejectedError, match="plausible"):
        model.predict(bad)
    model.smoke_test()


def test_load_rejects_garbage(tmp_path: Path, bundle: Path) -> None:
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file at all" * 10)
    with pytest.raises(ModelFormatError):
        load_bundle(junk)
    pickle_like = tmp_path / "p.safetensors"
    pickle_like.write_bytes(b"\x80\x04\x95" + b"\x00" * 64)
    with pytest.raises(ModelFormatError):
        load_bundle(pickle_like)
    truncated = tmp_path / "t.safetensors"
    truncated.write_bytes(bundle.read_bytes()[:-50])
    with pytest.raises(ModelFormatError):
        load_bundle(truncated)
    with pytest.raises(ModelFormatError):
        load_bundle(tmp_path / "missing.safetensors")


def test_artifact_store_immutable_and_safe(tmp_path: Path, bundle: Path) -> None:
    store = ArtifactStore(tmp_path / "store")
    stored = store.put("anomaly", 1, bundle)
    assert stored.sha256 == sha256_file(bundle)
    assert store.put("anomaly", 1, bundle).sha256 == stored.sha256  # idempotent
    other = tmp_path / "other.bin"
    other.write_bytes(b"different")
    with pytest.raises(ArtifactStoreError, match="different content"):
        store.put("anomaly", 1, other)
    assert store.get("anomaly", 1).read_bytes() == bundle.read_bytes()
    for name, version in [("../x", 1), ("anomaly", 0), ("A", 1)]:
        with pytest.raises(ArtifactStoreError):
            store.path_for(name, version)
    with pytest.raises(ArtifactStoreError, match="not found"):
        store.get("anomaly", 2)


# ---- drift maths ----


def test_psi_properties() -> None:
    same = [0.1] * 10
    assert population_stability_index(same, same) == pytest.approx(0.0, abs=1e-9)
    shifted = [0.3, 0.3, 0.1, 0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.03]
    assert population_stability_index(same, shifted) > 0.25
    assert population_stability_index(same, shifted) == pytest.approx(
        population_stability_index(shifted, same), rel=0.5
    )


def test_drift_ok_on_same_distribution_alert_on_shift() -> None:
    ref_df = generate_samples(5000, 0.06, seed=1)
    reference = build_reference(SYNTH_FEATURES, ref_df[SYNTH_FEATURES].to_numpy(dtype=np.float64))
    th = DriftThresholds(min_samples=100)

    same = generate_samples(1000, 0.06, seed=2)[SYNTH_FEATURES].to_numpy(dtype=np.float64)
    status, per_feature = evaluate_drift(reference, SYNTH_FEATURES, same, th)
    assert status == "OK", [f.as_dict() for f in per_feature if f.status != "OK"]

    drifted = generate_samples(1000, 0.06, seed=3, scenario="drift")[SYNTH_FEATURES].to_numpy(dtype=np.float64)
    status, per_feature = evaluate_drift(reference, SYNTH_FEATURES, drifted, th)
    assert status == "ALERT"
    flagged = {f.feature for f in per_feature if f.status == "ALERT"}
    assert "temperature_c" in flagged
    assert "pressure_bar" not in flagged  # untouched feature must not be flagged


def test_drift_insufficient_data() -> None:
    ref_df = generate_samples(2000, 0.06, seed=1)
    reference = build_reference(SYNTH_FEATURES, ref_df[SYNTH_FEATURES].to_numpy(dtype=np.float64))
    x = generate_samples(20, 0.06, seed=2)[SYNTH_FEATURES].to_numpy(dtype=np.float64)
    assert evaluate_drift(reference, SYNTH_FEATURES, x, DriftThresholds(min_samples=100)) == (
        "INSUFFICIENT_DATA",
        [],
    )


def test_real_regime_shift_is_flagged_and_normal_replay_is_not() -> None:
    reference = build_reference(FEATURE_NAMES, load_training_samples()[FEATURE_NAMES].to_numpy(dtype=np.float64))
    th = DriftThresholds(min_samples=100)
    rng = np.random.default_rng(0)
    for scenario, expected in (("normal", "OK"), ("drift", "ALERT")):
        x, _ = stream_pool(scenario)  # type: ignore[arg-type]
        window = x[rng.choice(len(x), size=300, replace=False)]
        status, per_feature = evaluate_drift(reference, FEATURE_NAMES, window, th)
        assert status == expected, [f.as_dict() for f in per_feature]


def test_labelled_quality() -> None:
    assert labelled_quality([True, False], [None, None])["measured"] is False
    q = labelled_quality([True, True, False, False], [True, False, True, None])
    assert q["measured"] and q["labelled_samples"] == 3
    assert q["precision"] == 0.5 and q["recall"] == 0.5


def test_mlflow_http_defaults_are_valid_for_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLflow parses these as ints; a bad default only fails on the real HTTP path (found in Compose)."""
    for key in ("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "MLFLOW_HTTP_REQUEST_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    MlflowRegistry("http://127.0.0.1:1", "x")
    for key in ("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "MLFLOW_HTTP_REQUEST_TIMEOUT"):
        int(os.environ[key])
