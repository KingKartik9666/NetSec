"""Synthetic end-to-end checks for the cascade v3 TabM / TabPack adapters (CUDA fit, CPU replay)."""
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before any CUDA initialisation

import numpy as np
import pytest
import torch

from framework.training.csv_v3 import adapters_tabm as A
from framework.training.csv_v3.contract import TaskData

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for adapter fit")

CLASSES = ["benign", "attack_a", "attack_b", "attack_c"]
NAMES = ["flow_rate", "packet_count", "network_ports_dst__present_80", "mean_size", "ratio", "noise"]
FAST = {
    "tabm_ne": dict(k=2, width=16, depth=1, n_bins=4, batch_size=64, epochs=2, patience=2, runtime_seconds=60),
    "tabm_hier": dict(k=2, width=16, depth=1, n_bins=4, batch_size=64, epochs=2, patience=2, runtime_seconds=60),
    "tabm_hier_ne": dict(k=2, width=16, depth=1, n_bins=4, batch_size=64, epochs=2, patience=2,
                         runtime_seconds=60),
    "tabpack": dict(batch_size=64, epochs=2, patience=2, runtime_seconds=60),
}


def _rows(rng, n):
    y = np.concatenate([np.arange(4), rng.integers(0, 4, n - 4)]).astype(np.int64)
    X = rng.normal(size=(n, len(NAMES))).astype(np.float32)
    X[:, 0] += 2.0 * y
    X[:, 1] = np.abs(X[:, 1]) * (1 + y) * 10
    X[:, 2] = (y % 2 == 1).astype(np.float32)
    X[:, 3] += np.where(y == 3, 3.0, 0.0)
    X[::17, 0] = np.nan
    X[::11, 2] = np.nan
    X[::13, 4] = np.nan
    return X.astype(np.float32), y


def _data(seed=0, task="84"):
    rng = np.random.default_rng(seed)
    X_fit, y_fit = _rows(rng, 400)
    X_tune, y_tune = _rows(rng, 150)
    return TaskData(task=task, X_fit=X_fit, y_fit=y_fit, X_tune=X_tune, y_tune=y_tune, classes=list(CLASSES),
                    feature_names=list(NAMES), benign_index=0)


def _config(family):
    config = A.default_config(family, "84")
    config.update(FAST[family])
    return config


@pytest.mark.parametrize("family", A.FAMILIES)
def test_fit_then_cpu_replay(family, tmp_path):
    data = _data()
    out = tmp_path / family
    record = A.fit(family, data, _config(family), seed=7, out_dir=out)

    json.dumps(record)
    for key in ("family", "task", "seed", "checkpoint", "config_used", "training_seconds", "device",
                "stop_reason", "tune_accuracy", "tune_macro_f1", "n_features", "n_classes", "source_recipe"):
        assert key in record, key
    assert record["family"] == family and record["task"] == "84" and record["seed"] == 7
    assert record["n_features"] == len(NAMES) and record["n_classes"] == len(CLASSES)
    assert record["device"].startswith("cuda") and record["trainer_device"] == "cuda"
    assert (out / record["checkpoint"]).exists()
    assert record["source_recipe"]["family"] == family
    assert record["source_recipe"]["selected_json"].endswith("selected.json")
    assert 0.0 <= record["tune_accuracy"] <= 1.0 and 0.0 <= record["tune_macro_f1"] <= 1.0
    assert record["epochs"] <= 2
    if family in A.EMBEDDED:
        assert record["n_embedded_features"] > 0

    X = np.concatenate([data.X_tune, data.X_fit[:37]])
    p = A.predict_cpu(family, out, X)
    assert p.dtype == np.float32 and p.shape == (len(X), len(CLASSES))
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1.0, atol=2e-5)

    device_p = np.load(out / record["tune_probabilities"])
    assert np.array_equal(p[:len(data.y_tune)].argmax(1), device_p.argmax(1))
    assert record["cpu_tune_argmax_agreement"] == 1.0
    assert record["tune_accuracy"] == pytest.approx(float(np.mean(device_p.argmax(1) == data.y_tune)))

    overflow = data.X_tune.astype(np.float64); overflow[0, 0] = 1e300   # finite in float64, inf in float32
    with pytest.raises(ValueError, match="inf"):
        A.predict_cpu(family, out, overflow)

    with pytest.raises(FileExistsError):
        A.fit(family, data, _config(family), seed=7, out_dir=out)


def test_default_config_provenance_and_task8():
    for family in A.FAMILIES:
        config = A.default_config(family, "77")
        source = config[A.SOURCE_KEY]
        frozen = {k: v for k, v in config.items() if k != A.SOURCE_KEY}
        assert frozen == source["frozen_config"] and len(source["selected_json_sha256"]) == 64
        with pytest.raises(NotImplementedError):
            A.default_config(family, "8")
    assert A.default_config("tabm_ne", "84")["runtime_seconds"] == 1500
    assert A.default_config("tabm_hier", "84")["runtime_seconds"] == 3600
    assert A.default_config("tabpack", "84")["runtime_seconds"] == 3600
    assert A.default_config("tabm_ne", "84")[A.SOURCE_KEY]["trial_number"] == 2


def test_refusals(tmp_path):
    data = _data()
    bad = data.X_fit.copy(); bad[0, 2] = 0.5
    bad_data = TaskData(task="84", X_fit=bad, y_fit=data.y_fit, X_tune=data.X_tune, y_tune=data.y_tune,
                        classes=list(CLASSES), feature_names=list(NAMES), benign_index=0)
    with pytest.raises(ValueError, match="indicator"):
        A.fit("tabm_ne", bad_data, _config("tabm_ne"), seed=1, out_dir=tmp_path / "a")
    no_source = {k: v for k, v in _config("tabm_ne").items() if k != A.SOURCE_KEY}
    with pytest.raises(ValueError, match="_source"):
        A.fit("tabm_ne", data, no_source, seed=1, out_dir=tmp_path / "b")
    with pytest.raises(FileNotFoundError):
        A.predict_cpu("tabm_ne", tmp_path / "missing", data.X_tune)
