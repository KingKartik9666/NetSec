"""Cascade v3 tree adapters: fit on CUDA (RF control on CPU), CPU-only reload, refusal to reuse a directory."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for adapter fits")

from framework.training.csv_v3 import adapters_trees as trees  # noqa: E402
from framework.training.csv_v3.contract import TaskData  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAST_XGB = dict(n_estimators=8, max_depth=3, max_bin=32, early_stopping_rounds=4, runtime_seconds=60)


def synthetic(task="84", n_fit=400, n_tune=150, seed=0):
    rng = np.random.default_rng(seed)
    names = ["network_ports_dst__present_80", "network_bytes_per_packet", "network_syn_per_packet",
             "network_window_seconds", "log_failed_logins", "network_dst_ip_count"]
    classes = ["benign", "scan", "flood", "brute"] if task != "8" else ["recon_a", "recon_b", "recon_c", "recon_d"]
    n = n_fit + n_tune
    y = np.arange(n) % 4
    rng.shuffle(y)
    X = rng.normal(size=(n, len(names))).astype(np.float32)
    X[:, 0] = (y == 1) | (rng.random(n) < .1)          # 0/1 port-presence flag, mostly class 1
    X[:, 1] += 2.0 * (y == 2)
    X[:, 2] += 2.0 * (y == 3)
    X[rng.random(X.shape) < .05] = np.nan               # a few missing values
    X[:, 0] = np.nan_to_num(X[:, 0])                    # keep the flag strictly 0/1
    return TaskData(task=task, X_fit=X[:n_fit], y_fit=y[:n_fit].astype(np.int64),
                    X_tune=X[n_fit:], y_tune=y[n_fit:].astype(np.int64), classes=classes,
                    feature_names=names, benign_index=None if task == "8" else 0)


def fast_config(family, task):
    config = trees.default_config(family, task)
    if family == "xgboost":
        config.update(FAST_XGB)
    return config                                       # the RF recipe is frozen and already fast


def cpu_subprocess(family, out_dir, X, tmp_path):
    """Replay in a fresh process with CUDA hidden, as the runner does."""
    np.save(tmp_path / "x.npy", X)
    code = ("import sys, numpy as np; sys.path.insert(0, sys.argv[1]);"
            "from framework.training.csv_v3.adapters_trees import predict_cpu;"
            "np.save(sys.argv[4], predict_cpu(sys.argv[2], sys.argv[3], np.load(sys.argv[5])))")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, "-c", code, str(ROOT), family, str(out_dir), str(tmp_path / "p.npy"),
                    str(tmp_path / "x.npy")], check=True, env=env, timeout=120)
    return np.load(tmp_path / "p.npy")


def test_default_config_is_frozen_recipe_with_only_the_xgboost_cap_changed():
    selected = json.loads((ROOT / trees.SELECTED_RELATIVE).read_text(encoding="utf-8"))
    xgb = trees.default_config("xgboost", "84")
    source = xgb.pop("_source_recipe")
    assert xgb == dict(selected["xgboost"]["config"], runtime_seconds=3600)
    assert source["job"] == "xgboost_trial0000" and source["trial_number"] == 0
    assert source["changes"]["runtime_seconds"] == {"from": 1500, "to": 3600, "reason": source["changes"]["runtime_seconds"]["reason"]}
    assert trees.default_config("xgboost", "8")["runtime_seconds"] == 3600
    rf = trees.default_config("random_forest", "77")
    assert rf.pop("_source_recipe")["recipe"] == "random_forest_full_c0"
    assert rf == selected["random_forest"]["config"]
    with pytest.raises(NotImplementedError):
        trees.default_config("random_forest", "8")


@pytest.mark.parametrize("family,task", [("xgboost", "84"), ("xgboost", "8"), ("random_forest", "77")])
def test_fit_then_cpu_replay(family, task, tmp_path):
    data = synthetic(task)
    out = tmp_path / "run"
    record = trees.fit(family, data, fast_config(family, task), 42, out)

    json.dumps(record)
    for key in ("family", "task", "seed", "checkpoint", "config_used", "training_seconds", "device", "stop_reason",
                "tune_accuracy", "tune_macro_f1", "n_features", "n_classes", "source_recipe"):
        assert key in record
    assert "probabilities" not in record["trainer"] and "predictions" not in record["trainer"]
    assert (out / record["checkpoint"]).is_file() and not Path(record["checkpoint"]).is_absolute()
    assert record["n_features"] == 6 and record["n_classes"] == 4 and record["seed"] == 42
    assert record["device"].startswith("cuda") if family == "xgboost" else record["device"] == "cpu"
    assert "_source_recipe" not in record["config_used"]
    if family == "xgboost":
        assert record["config_used"]["runtime_seconds"] == 60
        assert set(record["source_recipe"]["overrides_vs_default"]) == set(FAST_XGB)
        assert record["trainer_benign_index"] == (-1 if task == "8" else 0)
    assert record["tune_accuracy"] > .5                   # the synthetic classes are separable

    probabilities = trees.predict_cpu(family, out, data.X_tune)
    assert probabilities.dtype == np.float32 and probabilities.shape == (150, 4)
    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1, atol=1e-5)
    fit_predictions = np.load(out / "tune_predictions.npy")
    assert np.array_equal(probabilities.argmax(1), fit_predictions)
    assert np.isclose(np.mean(fit_predictions == data.y_tune), record["tune_accuracy"])

    hidden = cpu_subprocess(family, out, data.X_tune, tmp_path)
    assert np.array_equal(hidden, probabilities)

    with pytest.raises(ValueError):
        trees.predict_cpu(family, out, data.X_tune[:, :5])
    bad = data.X_tune.copy()
    bad[0, 1] = np.inf
    with pytest.raises(ValueError):
        trees.predict_cpu(family, out, bad)
    with pytest.raises(FileExistsError):
        trees.fit(family, data, fast_config(family, task), 42, out)


def test_tune_macro_f1_averages_present_classes_only():
    y = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)       # class 3 has no tune rows
    pred = np.array([0, 1, 1, 1, 2, 3], dtype=np.int64)
    scores = trees._tune_scores(y, pred, 4)
    # per-class F1: c0 2*1/(2+1)=2/3, c1 2*2/(2+3)=4/5, c2 2*1/(2+1)=2/3; class 3 excluded
    assert np.isclose(scores["tune_macro_f1"], (2 / 3 + 4 / 5 + 2 / 3) / 3)
    assert np.isclose(scores["tune_accuracy"], 4 / 6)
    assert scores["tune_classes_present"] == 3 and scores["tune_classes_absent"] == [3]
    assert trees._jsonable({"a": np.float64("nan"), "b": np.float32(1.5)}) == {"a": None, "b": 1.5}


def test_refusals_without_training(tmp_path):
    data = synthetic("8")
    with pytest.raises(NotImplementedError):
        trees.fit("random_forest", data, {}, 42, tmp_path / "rf8")
    with pytest.raises(ValueError):
        trees.fit("random_forest", synthetic("77"), {}, 43, tmp_path / "rf43")
    with pytest.raises(ValueError):
        trees.fit("random_forest", synthetic("77"), {"n_estimators": 10}, 42, tmp_path / "rf10")
    missing = synthetic("84")
    y = missing.y_fit.copy()
    y[y == 3] = 2
    with pytest.raises(ValueError):                     # a class absent from y_fit
        trees.fit("xgboost", TaskData(**{**missing.__dict__, "y_fit": y}), fast_config("xgboost", "84"), 42,
                  tmp_path / "missing")
    with pytest.raises(ValueError):
        trees.fit("tabm", data, {}, 42, tmp_path / "x")
