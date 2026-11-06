"""Checks for cascade v3 post-hoc calibration and its adoption guard (no training)."""
import numpy as np
import pytest

from framework.training.csv_v3 import calibrate, prep, runner

needs_data = pytest.mark.skipif(not (prep.DEFAULT_OUT / "manifest.json").exists(), reason="v3 dataset not prepared")


def _overconfident(n=4000, k=5, temperature=3.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    z = rng.normal(0, 1, (n, k))
    z[np.arange(n), y] += 1.5
    return calibrate.softmax(z * temperature), y


def test_temperature_recovers_overconfidence_without_changing_predictions():
    p, y = _overconfident()
    cfg = calibrate.fit_temperature(p, y)
    # True posterior is softmax(1.5 z) (mean shift 1.5, unit noise); the model emits 3 z, so the optimum is T = 2.
    assert cfg["temperature"] == pytest.approx(2.0, rel=0.1)
    q = calibrate.apply(cfg, p)
    assert np.array_equal(q.argmax(1), p.argmax(1))
    assert calibrate.ece(q, y) < calibrate.ece(p, y)
    assert calibrate.nll(q, y) < calibrate.nll(p, y)


def test_smoothing_is_fitted_for_exact_zeros_instead_of_a_fixed_floor():
    p, y = _overconfident(temperature=1.0)
    hard = np.where(p < 0.05, 0.0, p)                       # random-forest-like exact zeros
    hard /= hard.sum(1, keepdims=True)
    cfg = calibrate.fit_temperature(hard, y)
    assert cfg["alpha"] > 0
    q = calibrate.apply(cfg, hard)
    assert np.array_equal(q.argmax(1), hard.argmax(1))
    assert calibrate.quality(q, y)["zero_true_prob_rate"] == 0.0 < calibrate.quality(hard, y)["zero_true_prob_rate"]


def test_vector_bias_corrects_a_prior_shift_without_pushing_unsupported_classes_down():
    p, y = _overconfident(temperature=1.0)
    shifted = p.copy()
    shifted[:, 0] *= 4.0                                    # model over-predicts class 0
    shifted /= shifted.sum(1, keepdims=True)
    keep = y != 4                                           # class 4 has no calibration rows
    cfg = calibrate.fit_vector(shifted[keep], y[keep], 5, lam=1e-3)
    bias = np.array(cfg["bias"])
    assert cfg["free_classes"] == 4 and bias[4] == 0.0
    assert bias[:4].mean() == pytest.approx(0.0, abs=1e-9)
    assert bias[0] < bias[1:4].min()
    fixed = calibrate.apply(cfg, shifted)
    assert np.mean(fixed.argmax(1) == y) > np.mean(shifted.argmax(1) == y)
    four = y == 4
    assert np.mean(fixed[four].argmax(1) == 4) >= np.mean(shifted[four].argmax(1) == 4) - 0.05


def test_one_hot_outputs_are_not_calibratable():
    assert calibrate.is_one_hot(np.eye(4, dtype=np.float32)[[0, 2, 3]])
    assert not calibrate.is_one_hot(np.full((2, 4), 0.25))


def test_quality_metrics_on_perfect_predictions():
    p = np.eye(3)[[0, 1, 2, 1]]
    q = calibrate.quality(p, np.array([0, 1, 2, 1]))
    assert q["accuracy"] == 1.0 and q["ece"] == pytest.approx(0.0) and q["brier"] == pytest.approx(0.0)


@needs_data
@pytest.mark.parametrize("task", ["77", "8"])
def test_calibration_rows_split_roles_and_disjoint_scenarios(task):
    r = runner.calibration_rows(task)
    rows = set(runner.task_arrays(task)["val_rows"])
    scen = runner._arr("scenario_validation")
    for name in ("tuning", "calibration", "assessment"):
        assert set(r[name]) <= rows
    assert not set(scen[r["tuning_disjoint"]]) & set(scen[r["calibration"]])
    assert not set(scen[r["assessment_disjoint"]]) & set(scen[r["calibration"]])
    assert set(r["tuning_disjoint"]) <= set(r["tuning"]) and set(r["assessment_disjoint"]) <= set(r["assessment"])


@needs_data
@pytest.mark.parametrize("task", ["77", "8"])
def test_vector_adoption_ignores_assessment_rows(task):
    a = runner.task_arrays(task)
    k, y = len(a["classes"]), a["y_val"]
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, (len(y), k))
    z[np.arange(len(y)), np.where(y >= 0, y, 0)] += 3.0
    z[:, 0] += 2.5                                           # prior shift a per-class bias can undo
    p = calibrate.softmax(z).astype(np.float32)
    info, _ = runner.calibrate_proba(task, p)
    assert info["method"] == "vector" and info["decisions_changed"]
    r = runner.calibration_rows(task)
    p2 = p.copy()
    p2[r["assessment"]] = rng.dirichlet(np.ones(k), size=len(r["assessment"]))
    info2, _ = runner.calibrate_proba(task, p2)
    assert (info2["method"], info2["config"].get("lam"), info2["selection_value"]) == \
           (info["method"], info["config"].get("lam"), info["selection_value"])


@needs_data
def test_vector_gain_only_on_calibration_scenarios_is_not_adopted():
    """A decision change that ties on calibration-disjoint tuning scenarios must be rejected."""
    a = runner.task_arrays("77")
    y = a["y_val"]
    r = runner.calibration_rows("77")
    rng = np.random.default_rng(1)
    z = rng.normal(0, 0.3, (len(y), 77))
    z[np.arange(len(y)), y] += 8.0                           # disjoint rows: correct with a wide margin
    shared = np.setdiff1d(r["tuning"], r["tuning_disjoint"])
    shift_rows = np.concatenate([r["calibration"], shared])
    z[shift_rows, 0] += 8.5                                  # class-0 shift only in scenarios shared with calibration
    p = calibrate.softmax(z).astype(np.float32)
    info, _ = runner.calibrate_proba("77", p)
    assert not info["decisions_changed"] and info["selection_value"] == info["raw_tune_value"]


@needs_data
def test_one_hot_hem_vote_is_reported_not_calibratable():
    rng = np.random.default_rng(0)
    hem = np.eye(77, dtype=np.float32)[rng.integers(0, 77, 30000)]
    info, p = runner.calibrate_proba("77", hem)
    assert info["status"] == "not_calibratable" and p is None
