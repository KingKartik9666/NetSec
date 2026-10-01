"""Scientific correctness checks for the optional CUDA research backends."""
from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

from framework.training.neural import ROOT, class_weights, hem_closed_set_vote


def test_class_weights_use_training_frequencies_and_cap():
    y = np.array([0] * 90 + [1] * 9 + [2])
    np.testing.assert_allclose(class_weights(y, 3, "none"), np.ones(3))
    weights = class_weights(y, 3, "sqrt_balanced")
    assert weights[2] > weights[1] > weights[0]
    assert np.mean(weights[y]) == pytest.approx(1.0)
    assert class_weights(y, 3, "balanced").max() <= 20
    with pytest.raises(ValueError, match="Every requested class"):
        class_weights(np.array([0, 0]), 2, "balanced")


def test_vote_preserves_arbitrary_benign_index():
    multi, binary, knn = np.array([[2, 0, 2], [1, 0, 2], [2, 1, 1], [0, 1, 2]]).T
    np.testing.assert_array_equal(hem_closed_set_vote(multi, binary, knn, 2), [2, 2, 1, 0])


def test_closed_set_vote_matches_original_hem_for_all_combinations(monkeypatch):
    pytest.importorskip("torch")
    pd = pytest.importorskip("pandas")
    source = ROOT / "external/HEM/model_evaluation"
    monkeypatch.syspath_prepend(str(source))
    spec = importlib.util.spec_from_file_location("_imdai_test_full_hem", source / "full_model.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    multi, binary, knn = np.array(list(itertools.product(range(3), range(2), range(3)))).T
    model = upstream.DecisionModel.__new__(upstream.DecisionModel)
    model.test_set = pd.DataFrame({"label": np.zeros(len(multi))})
    original = model.voting_ensemble((multi, None, binary, None, knn, None, None), alpha=0, include_novel=False)
    np.testing.assert_array_equal(hem_closed_set_vote(multi, binary, knn, 0), original)


def test_no_cuda_has_no_silent_cpu_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    from framework.training.neural import _require_cuda
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        _require_cuda()


@pytest.mark.parametrize("candidate", ["kan", "hem"])
def test_cuda_training_and_saved_best_checkpoint_reload(candidate, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Optional CUDA research smoke test")
    from framework.training.neural import predict_neural, train_neural
    rng = np.random.default_rng(123)
    X = rng.normal(size=(120, 6)).astype(np.float32)
    y = np.tile(np.arange(3), 40)
    X[:, 0] += y * 3
    config = {"epochs": 2, "batch_size": 32, "lr": 0.01, "hidden": [8], "binary_hidden": [8], "pca_components": 4, "knn_k": 3, "knn_max_rows": 60, "weight_strategy": "sqrt_balanced"}
    result = train_neural(candidate, X[:90], y[:90], X[90:], y[90:], 3, 0, config, 42, tmp_path)
    assert result["device_evidence"]["parameter_devices"] == ["cuda:0"]
    assert result["peak_vram_bytes"] > 0
    checkpoint = torch.load(result["checkpoint_path"], weights_only=True)
    assert checkpoint["best_epoch"] == result["best_epoch"]
    assert result["best_epoch"] == max(result["history"], key=lambda item: item["validation_macro_f1"])["epoch"]
    reloaded = predict_neural(Path(result["checkpoint_path"]), X[90:])
    np.testing.assert_array_equal(reloaded, result["predictions"])
    if candidate == "kan":
        assert any("spline_weight" in name for name in checkpoint["state_dicts"]["kan"])


def test_cuda_knn_matches_sklearn_away_from_distance_ties():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Optional CUDA KNN equivalence check")
    from sklearn.neighbors import KNeighborsClassifier
    from framework.training.neural import cuda_knn_predict
    rng = np.random.default_rng(99)
    refs, queries = rng.normal(size=(30, 5)).astype(np.float32), rng.normal(size=(9, 5)).astype(np.float32)
    labels = np.tile(np.arange(3), 10)
    expected = KNeighborsClassifier(n_neighbors=3).fit(refs, labels).predict(queries)
    np.testing.assert_array_equal(cuda_knn_predict(queries, refs, labels, 3, 3, batch_size=4), expected)


def test_single_neighbor_ties_use_first_reference_across_query_batches():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Optional CUDA KNN equivalence check")
    from framework.training.neural import cuda_knn_predict, _reference_indices
    refs = np.array([[0., 0.], [0., 0.], [2., 0.]], dtype=np.float32)
    labels = np.array([2, 1, 0])
    queries = np.array([[0., 0.], [1., 0.], [2., 0.]], dtype=np.float32)
    np.testing.assert_array_equal(_reference_indices(labels, 3, 42), [0, 1, 2])
    np.testing.assert_array_equal(cuda_knn_predict(queries, refs, labels, 3, 1, batch_size=1), [2, 2, 0])
