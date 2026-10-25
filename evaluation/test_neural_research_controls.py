"""CPU checks for explicit research controls; no CUDA fit is started here."""
from pathlib import Path
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from framework.training import neural


def _losses():
    from framework.training import research_losses
    return research_losses


@pytest.mark.parametrize("binary", [False, True])
def test_native_and_gamma_zero_preserve_weighted_loss(binary):
    losses = _losses()
    targets = torch.tensor([0, 1, 1], dtype=torch.long)
    weights = torch.tensor([.4, 1.6], dtype=torch.double)
    if binary:
        logits = torch.tensor([-.2, .5, -.8], dtype=torch.double, requires_grad=True)
        expected = (torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.double(), reduction="none") * weights[targets]).mean()
        function = losses.binary_loss
    else:
        logits = torch.tensor([[.8, -.2], [-.1, .5], [.4, -.8]], dtype=torch.double, requires_grad=True)
        expected = torch.nn.functional.cross_entropy(logits, targets, weight=weights)
        function = losses.multiclass_loss
    assert torch.equal(function(logits, targets, weights), expected)
    assert torch.allclose(function(logits, targets, weights, loss="focal", gamma=0), expected, atol=1e-12)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("loss", ["label_smoothed_ce", "focal"])
def test_research_losses_have_finite_difference_gradients(binary, loss):
    function = _losses().binary_loss if binary else _losses().multiclass_loss
    targets = torch.tensor([0, 1, 1], dtype=torch.long)
    weights = torch.tensor([.4, 1.6], dtype=torch.double)
    logits = torch.tensor([-.3, .7, -1.1] if binary else [[.7, -.3], [-.2, .8], [1.1, -.4]], dtype=torch.double, requires_grad=True)
    value = function(logits, targets, weights, loss=loss)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(logits.grad).all()
    numeric = torch.zeros_like(logits)
    for index in range(logits.numel()):
        plus, minus = logits.detach().clone(), logits.detach().clone()
        plus.reshape(-1)[index] += 1e-5
        minus.reshape(-1)[index] -= 1e-5
        numeric.reshape(-1)[index] = (function(plus, targets, weights, loss=loss) - function(minus, targets, weights, loss=loss)) / 2e-5
    assert torch.allclose(logits.grad, numeric, atol=1e-8, rtol=1e-5)


def test_smoothing_and_focal_change_easy_example_loss():
    losses = _losses()
    logits, targets = torch.tensor([[4., -4.]]), torch.tensor([0])
    native = losses.multiclass_loss(logits, targets)
    assert losses.multiclass_loss(logits, targets, loss="label_smoothed_ce") > native
    assert 0 < losses.multiclass_loss(logits, targets, loss="focal") < native


@pytest.mark.parametrize("config", [{"loss": "wrong"}, {"loss": "focal", "gamma": -1}, {"loss": "label_smoothed_ce", "smoothing": 1.1}])
def test_invalid_loss_settings_rejected(config):
    with pytest.raises(ValueError):
        _losses().multiclass_loss(torch.zeros(2, 2), torch.tensor([0, 1]), **config)


def test_reference_selection_excludes_synthetics_and_keeps_rare_classes():
    labels = np.array([0] * 12 + [1] * 5 + [2] + [2] * 30)
    selected = neural._reference_indices(labels, 7, 42, original_training_rows=18)
    assert len(selected) == 7 and selected.max() < 18
    assert set(labels[selected]) == {0, 1, 2}
    np.testing.assert_array_equal(selected, neural._reference_indices(labels, 7, 42, original_training_rows=18))
    with pytest.raises(ValueError, match="original"):
        neural._reference_indices(labels, 7, 42, original_training_rows=12)


def test_training_monitor_is_fixed_bounded_and_authentic():
    first = neural._training_monitor_indices(12000, 8000, 42)
    assert len(first) == 4096 and first.max() < 8000
    np.testing.assert_array_equal(first, neural._training_monitor_indices(12000, 8000, 42))
    np.testing.assert_array_equal(neural._training_monitor_indices(10, 6, 42), np.arange(6))


def _cpu_runtime(monkeypatch):
    """Replace only CUDA boundary hooks; optimization/model/loss/save stay real."""
    monkeypatch.setattr(neural, "_require_cuda", lambda: torch)
    as_tensor = torch.as_tensor
    def cpu_tensor(*args, **kwargs):
        assert str(kwargs.get("device", "cpu")).startswith("cpu"), "CPU control tests must never allocate CUDA tensors"
        return as_tensor(*args, **kwargs)
    monkeypatch.setattr(torch, "as_tensor", cpu_tensor)
    for name in ("empty_cache", "reset_peak_memory_stats", "synchronize", "manual_seed_all"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "isolated CPU test")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    models = neural._models
    monkeypatch.setattr(neural, "_models", lambda candidate, size, classes, config, module: models(candidate, size, classes, config, module, device="cpu"))
    knn = neural.cuda_knn_predict
    monkeypatch.setattr(neural, "cuda_knn_predict", lambda *args, **kwargs: knn(*args, **kwargs, device="cpu"))


def _small_fit(tmp_path, config, candidate="kan"):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 4)).astype(np.float32)
    y = np.tile(np.arange(3), 10)
    settings = {"epochs": 5, "batch_size": 6, "lr": .01, "hidden": [4], "binary_hidden": [4], "pca_components": 2,
                "knn_max_rows": 15, "knn_k": 1, "selection_metric": "accuracy", "original_training_rows": 18, "patience": 10, **config}
    return neural.train_neural(candidate, X[:24], y[:24], X[24:], y[24:], 3, 0, settings, 42, tmp_path)


@pytest.mark.parametrize("candidate", ["kan", "hem"])
def test_prune_preserves_validated_best_and_authentic_telemetry(monkeypatch, tmp_path, candidate):
    _cpu_runtime(monkeypatch)
    (tmp_path / "PRUNE").write_text("controller plateau decision")
    result = _small_fit(tmp_path, {"scheduler": "plateau", "loss": "focal"}, candidate)
    assert result["pruned"] is True and result["stop_reason"] == "pruned"
    assert len(result["history"]) == 1 and result["best_epoch"] == 1
    assert Path(result["checkpoint_path"]).is_file()
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert 0 <= progress["training_accuracy"] <= 1 and progress["training_monitor_rows"] == 18
    if candidate == "hem":
        with np.load(tmp_path / "knn_references.npz") as reference:
            assert reference["train_indices"].max() < 18
            assert set(reference["y"]) == {0, 1, 2}


def test_plateau_scheduler_steps_on_validation_metric():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    scheduler = neural._make_scheduler(optimizer, "plateau", 6, torch)
    for _ in range(4):
        scheduler.step(.75)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.05)


def test_deadline_before_first_checkpoint_is_informative(monkeypatch, tmp_path):
    _cpu_runtime(monkeypatch)
    with pytest.raises(TimeoutError, match="validated checkpoint"):
        _small_fit(tmp_path, {"runtime_seconds": 1e-9})


def test_mid_epoch_deadline_returns_prior_best(monkeypatch, tmp_path):
    _cpu_runtime(monkeypatch)
    original_step = torch.optim.AdamW.step
    steps = 0
    def step(*args, **kwargs):
        nonlocal steps
        result = original_step(*args, **kwargs)
        steps += 1
        return result
    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    monkeypatch.setattr(neural.time, "perf_counter", lambda: 100. if steps >= 5 else 0.)
    result = _small_fit(tmp_path, {"runtime_seconds": 50.})
    assert result["timed_out"] is True and result["best_epoch"] == 1
    assert len(result["history"]) == 1 and result["predictions"] is not None
    assert steps == 5


def test_deadline_after_final_batch_does_not_start_unbudgeted_validation(monkeypatch, tmp_path):
    _cpu_runtime(monkeypatch)
    original_step = torch.optim.AdamW.step
    steps = 0
    def step(*args, **kwargs):
        nonlocal steps
        result = original_step(*args, **kwargs)
        steps += 1
        return result
    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    monkeypatch.setattr(neural.time, "perf_counter", lambda: 100. if steps >= 4 else 0.)
    with pytest.raises(TimeoutError, match="validated checkpoint"):
        _small_fit(tmp_path, {"runtime_seconds": 50.})
