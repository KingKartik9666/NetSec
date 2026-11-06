"""Checks for cascade v3 evaluation, search spaces and task assembly (no training)."""
import numpy as np
import optuna
import pytest

from framework.training.csv_v3 import evaluate, prep, runner, search

pytestmark = pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")


def test_metrics_supported_vs_declared_and_rates():
    y = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 1, 2])
    m = evaluate.metrics(y, pred, n_classes=4, benign_index=0)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["supported_classes"] == 3
    assert m["macro_f1_supported"] > m["macro_f1_declared"]   # class 3 has no rows: 0 only in the declared mean
    assert m["benign_false_alarm_rate"] == pytest.approx(0.5)
    assert m["attack_missed_as_benign_rate"] == 0.0
    assert "benign_false_alarm_rate" not in evaluate.metrics(y, pred, 4, None)


def test_scenario_bootstrap_brackets_accuracy():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 500)
    pred = np.where(rng.random(500) < 0.8, y, (y + 1) % 3)
    lo, hi = evaluate.scenario_bootstrap_ci(y, pred, np.repeat(np.arange(50), 10))
    assert lo < np.mean(y == pred) < hi


@pytest.mark.parametrize("family", ["xgboost", "kan", "hem", "tabm_ne", "tabm_hier", "tabm_hier_ne"])
def test_search_space_samples_known_keys(family):
    default = {"runtime_seconds": 1500, "selection_metric": "accuracy", "epochs": 60}
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    config = search.suggest(family, "77", study.ask(), default)
    assert config["selection_metric"] == "accuracy"           # unsearched recipe keys are preserved
    assert "rows_policy" not in config


def test_specialist_space_has_rows_policy_and_fixed_trial_replay():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    config = search.suggest("xgboost", "8", trial, {})
    replay = search.suggest("xgboost", "8", optuna.trial.FixedTrial(trial.params), {})
    assert config == replay and config["rows_policy"] in ("all", "targeted_only")


@pytest.mark.skipif(not (prep.DEFAULT_OUT / "manifest.json").exists(), reason="v3 dataset not prepared")
@pytest.mark.parametrize("task,n", [("77", 77), ("84", 84), ("8", 8)])
def test_task_arrays_contiguous_labels(task, n):
    a = runner.task_arrays(task)
    assert len(a["classes"]) == n
    fit_labels = a["y_train"][a["fit"]]
    assert np.array_equal(np.unique(fit_labels), np.arange(n))
    assert (a["y_val"][a["tune_rows"]] >= 0).all()
    if task == "8":
        assert a["benign"] is None and len(a["val_rows"]) == 5261
    else:
        assert a["classes"][0] == "benign" and len(a["val_rows"]) == 30000
