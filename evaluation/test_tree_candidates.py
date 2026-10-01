"""Candidate correctness checks; optional CUDA checks use synthetic fixtures only."""

import json

import numpy as np
import pytest

from framework.training.trees import (
    _macro_f1, _parameters, _require_cuda, class_weights, predict_tree, train_tree,
)


def test_frequency_weights_are_train_only_capped_and_reduce_imbalance():
    labels = np.array([0] * 998 + [1] + [2])
    uniform = class_weights(labels, 3, "none")
    balanced = class_weights(labels, 3, "balanced")
    square_root = class_weights(labels, 3, "sqrt_balanced")
    np.testing.assert_equal(uniform, [1, 1, 1])
    assert balanced[1] == balanced[2] == 20
    assert balanced[0] < square_root[0] < 1
    assert square_root[1] > 1
    with pytest.raises(ValueError, match="Each encoded class"):
        class_weights(np.array([0, 0]), 2, "none")


def test_standalone_forest_cannot_accidentally_become_boosted():
    params, rounds = _parameters("random_forest", 3, {"num_parallel_tree": 7}, 42)
    assert rounds == 1 and params["num_parallel_tree"] == 7
    assert params["learning_rate"] == 1 and params["device"] == "cuda:0"
    for bad in ({"n_estimators": 10}, {"num_boost_round": 2}, {"subsample": 1}):
        with pytest.raises(ValueError):
            _parameters("random_forest", 3, bad, 42)


def test_cpu_fallback_is_rejected():
    class FakeBooster:
        def save_config(self):
            return json.dumps({"learner": {"generic_param": {"device": "cpu"}}})

    with pytest.raises(RuntimeError, match="GPU execution required"):
        _require_cuda(FakeBooster())
    with pytest.raises(ValueError, match="requires device"):
        _parameters("xgboost", 3, {"device": "cpu"}, 42)


def test_macro_f1_preserves_unpredicted_classes():
    assert _macro_f1(np.array([0, 0, 1, 2]), np.array([0, 0, 0, 0]), 3) == pytest.approx(2 / 9)


def test_xgboost_search_applies_split_penalty_and_regularization():
    params, rounds = _parameters("xgboost", 3, {
        "gamma": .25, "reg_alpha": .2, "reg_lambda": 5,
        "min_child_weight": 4, "n_estimators": 1500,
    }, 42)
    assert params["gamma"] == .25
    assert params["reg_alpha"] == .2 and params["reg_lambda"] == 5
    assert params["min_child_weight"] == 4 and rounds == 1500


@pytest.fixture(scope="module")
def cuda_xgboost():
    xgb = pytest.importorskip("xgboost")
    if not xgb.build_info().get("USE_CUDA", False):
        pytest.skip("Optional CUDA XGBoost not installed")
    # Device selection is checked before allocating a training fixture.
    probe = xgb.Booster(params={"device": "cuda:0", "num_feature": 4})
    try:
        _require_cuda(probe)
    except (RuntimeError, xgb.core.XGBoostError) as error:
        pytest.skip(f"CUDA device unavailable: {error}")
    return xgb


@pytest.mark.parametrize("candidate", ["random_forest", "xgboost"])
def test_cuda_training_and_checkpoint_reload(candidate, cuda_xgboost, tmp_path):
    rng = np.random.default_rng(21)
    X = rng.normal(size=(180, 4)).astype(np.float32)
    y = np.tile(np.arange(3), 60).astype(np.int32)
    X[:, 0] += y * 4
    X[::20, 1] = np.nan
    config = {"max_depth": 2, "max_bin": 32, "weight_strategy": "sqrt_balanced"}
    config.update({"num_parallel_tree": 3} if candidate == "random_forest" else {"n_estimators": 6, "early_stopping_rounds": 2})
    result = train_tree(candidate, X[:120], y[:120], X[120:], y[120:], 3, config, 42, tmp_path)
    assert result["device"].startswith("cuda")
    assert result["training_seconds"] > 0
    assert len(result["predictions"]) == 60
    np.testing.assert_array_equal(predict_tree(result["checkpoint_path"], X[120:]), result["predictions"])
    booster = cuda_xgboost.Booster(model_file=result["checkpoint_path"])
    if candidate == "random_forest":
        assert booster.num_boosted_rounds() == 1
        assert len(booster.get_dump()) == 3 * 3  # classes * parallel trees
    else:
        assert booster.num_boosted_rounds() == result["best_iteration"] + 1
    with pytest.raises(FileExistsError):
        train_tree(candidate, X[:120], y[:120], X[120:], y[120:], 3, config, 42, tmp_path)
