"""Smoke tests for the cascade v3 neural adapters (kan, hem) on tiny synthetic data."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before any CUDA context exists

import numpy as np
import pytest
import torch

from framework.training.csv_v3 import adapters_neural as A
from framework.training.csv_v3.contract import TaskData

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for kan/hem fit")

CLASSES = ["benign", "attack_a", "attack_b", "attack_c"]
FEATURES = ["flow_bytes", "flow_packets", "network_ports_dst__present_80", "syn_ratio", "duration", "noise"]


def _rows(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.arange(n, dtype=np.int64) % len(CLASSES)
    rng.shuffle(y)
    X = rng.normal(size=(n, len(FEATURES))).astype(np.float32)
    X[:, 0] += 2.0 * y
    X[:, 1] = np.abs(X[:, 1]) * (1 + y) * 10
    X[:, 2] = (y == 2) | (rng.random(n) < 0.1)
    X[:, 3] -= 1.5 * (y == 3)
    X[rng.random((n, len(FEATURES))) < 0.03] = np.nan
    X[:, 2] = np.nan_to_num(X[:, 2], nan=0.0)  # presence flag stays 0/1
    return X.astype(np.float32), y


def _data(original_rows: int | None = None) -> TaskData:
    rng = np.random.default_rng(7)
    X_fit, y_fit = _rows(rng, 400)
    X_tune, y_tune = _rows(rng, 150)
    extras = {} if original_rows is None else {"original_training_rows": original_rows}
    data = TaskData("84", X_fit, y_fit, X_tune, y_tune, CLASSES, FEATURES, 0, extras)
    assert np.isnan(X_fit).any() and set(np.unique(X_fit[:, 2])) <= {0.0, 1.0}
    return data


def _fast(family: str) -> dict:
    config = A.default_config(family, "84")
    config.update(epochs=3, patience=2, batch_size=64, runtime_seconds=120)
    if family == "kan":
        config.update(hidden=[16], grid_size=5)
    else:
        config.update(hidden=[32, 16], binary_hidden=[32, 16], pca_components=4)
    return config


def test_default_config_is_frozen_recipe():
    kan, hem = A.default_config("kan", "77"), A.default_config("hem", "84")
    assert kan["runtime_seconds"] == 1500 and kan["source_recipe"]["trial_number"] == 140
    assert kan["grid_size"] == 9 and kan["hidden"] == [128, 64, 32]
    assert hem["runtime_seconds"] == 3300 and hem["source_recipe"]["recipe"] == "hem_strict_c1"
    assert (hem["pca_components"], hem["knn_k"], hem["knn_max_rows"]) == (64, 3, 480405)
    for family in A.FAMILIES:
        with pytest.raises(NotImplementedError):
            A.default_config(family, "8")
    # The returned config must not alias the recorded provenance copy.
    kan["hidden"].append(1)
    assert kan["source_recipe"]["recorded_config"]["hidden"] == [128, 64, 32]


def test_fit_raises_without_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        A.fit("kan", _data(), _fast("kan"), 42, tmp_path / "nocuda")
    assert not (tmp_path / "nocuda").exists()


_FORBIDDEN = ("framework.training.day_worker", "framework.training.hybrid_protocol",
              "experiment.train_verified_clean", "experiment.train_tabm")
_CPU_CHILD = r"""
import json, sys
import numpy as np
from framework.training.csv_v3 import adapters_neural as A
family, out, x_path, p_path = sys.argv[1:]
np.save(p_path, A.predict_cpu(family, out, np.load(x_path)))
import torch
import framework.training.neural, framework.training.research_losses
import framework.training.csv_monitoring, framework.training.tracking
print(json.dumps({"cuda_initialized": torch.cuda.is_initialized(),
                  "forbidden": [m for m in %r if m in sys.modules]}))
""" % (_FORBIDDEN,)


def _cpu_child(family: str, out: Path, X: np.ndarray, tmp_path: Path) -> tuple[np.ndarray, dict]:
    """Replay in a fresh process with CUDA hidden, as the runner does."""
    x_path, p_path = tmp_path / f"{family}_X.npy", tmp_path / f"{family}_P.npy"
    np.save(x_path, X)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    root = Path(A.ROOT)
    done = subprocess.run([sys.executable, "-c", _CPU_CHILD, family, str(out), str(x_path), str(p_path)],
                          cwd=root, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    return np.load(p_path), json.loads(done.stdout.strip().splitlines()[-1])


@needs_cuda
@pytest.mark.parametrize("family", A.FAMILIES)
def test_fit_then_cpu_replay(family, tmp_path):
    real_rows = 360 if family == "hem" else None
    data = _data(real_rows)
    out = tmp_path / family
    record = A.fit(family, data, _fast(family), 42, out)
    json.dumps(record)
    for key in ("family", "task", "seed", "checkpoint", "config_used", "training_seconds", "device",
                "stop_reason", "tune_accuracy", "tune_macro_f1", "n_features", "n_classes", "source_recipe"):
        assert key in record, key
    assert record["device"].startswith("cuda") and record["n_features"] == 6 and record["n_classes"] == 4
    assert "predictions" not in record and "history" not in record
    assert (out / record["checkpoint"]).is_file()
    assert {c["key"] for c in record["config_changes_vs_recorded"]} >= {"epochs", "runtime_seconds"}
    assert record["cpu_reload_check"]["agreeing_rows"] == record["cpu_reload_check"]["rows"]
    assert record["cpu_reload_check"]["matches"] is True

    proba = A.predict_cpu(family, out, data.X_tune)
    assert proba.dtype == np.float32 and proba.shape == (150, 4)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(1), 1.0, atol=1e-5)
    if family == "hem":  # documented: one-hot of the hard vote, not probabilities
        assert set(np.unique(proba)) <= {0.0, 1.0}
        with np.load(out / "knn_references.npz") as refs:
            assert len(refs["train_indices"]) == real_rows and refs["train_indices"].max() < real_rows
    fit_time = np.load(out / record["tune_predictions"])
    np.testing.assert_array_equal(proba.argmax(1), fit_time)

    child, info = _cpu_child(family, out, data.X_tune, tmp_path)
    assert info == {"cuda_initialized": False, "forbidden": []}
    assert child.dtype == np.float32
    np.testing.assert_array_equal(child, proba)

    bare = tmp_path / f"{family}_bare"  # checkpoint without fit_record.json is refused
    bare.mkdir()
    for name in ("model.pt", "preprocessing.joblib", "knn_references.npz"):
        if (out / name).exists():
            (bare / name).write_bytes((out / name).read_bytes())
    with pytest.raises(FileNotFoundError):
        A.predict_cpu(family, bare, data.X_tune)

    with pytest.raises(ValueError):
        A.predict_cpu(family, out, np.full((2, 6), np.inf, dtype=np.float32))
    with pytest.raises(ValueError):
        A.predict_cpu(family, out, np.zeros((2, 5), dtype=np.float32))

    digest = hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        A.fit(family, data, _fast(family), 42, out)
    assert hashlib.sha256((out / "model.pt").read_bytes()).hexdigest() == digest


@needs_cuda
def test_fit_rejects_task_8_and_bad_labels(tmp_path):
    data = _data()
    task8 = TaskData("8", data.X_fit, data.y_fit, data.X_tune, data.y_tune, CLASSES, FEATURES, None)
    with pytest.raises(NotImplementedError):
        A.fit("kan", task8, _fast("kan"), 42, tmp_path / "t8")
    bad = TaskData("84", data.X_fit, data.y_fit % 3, data.X_tune, data.y_tune, CLASSES, FEATURES, 0)
    with pytest.raises(ValueError):
        A.fit("hem", bad, _fast("hem"), 42, tmp_path / "bad")
    assert not (tmp_path / "bad" / "model.pt").exists()
