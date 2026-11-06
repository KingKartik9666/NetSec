"""77-class TabM-NE feature-selection test: feature scope, paired arm rules, seen/unseen split, frozen files (CPU)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from framework.training.csv_v3 import arms, live77, prep

ROOT = Path(__file__).resolve().parents[1]
needs_v3 = pytest.mark.skipif(not (prep.DEFAULT_OUT / "manifest.json").exists(), reason="v3 dataset not prepared")


def _v3_names() -> list[str]:
    return list(live77._names())


# ------------------------------------------------------------------------------------------ feature scope
@needs_v3
def test_csv_scope_allows_log_and_artifact_live_still_refuses():
    from framework.training.csv_v3 import runner
    names = _v3_names()
    g = live77.groups()
    listed = [g["artifact"][0], g["log"][0], g["core"][0]]
    with pytest.raises(ValueError, match="cannot exist live"):
        runner.feature_columns({"feature_list": listed})
    with pytest.raises(ValueError, match="cannot exist live"):
        runner.feature_columns({"feature_list": listed, "feature_scope": "live"})
    cols, out = runner.feature_columns({"feature_list": listed, "feature_scope": "csv"})
    assert list(cols) == sorted(names.index(n) for n in listed)          # v3 order, not the given order
    assert out == [names[i] for i in cols] and set(out) == set(listed)
    with pytest.raises(ValueError):
        runner.feature_columns({"feature_list": listed, "feature_scope": "everything"})
    with pytest.raises(ValueError):
        runner.feature_columns({"feature_list": listed + [listed[0]], "feature_scope": "csv"})
    with pytest.raises(ValueError):
        runner.feature_columns({"feature_list": ["no_such_feature"], "feature_scope": "csv"})
    cols, out = runner.feature_columns({"feature_list": names, "feature_scope": "csv"})
    assert list(cols) == list(range(len(names))) and out == names
    assert runner.feature_columns({"feature_scope": "csv"})[0] is None


# --------------------------------------------------------------------------------------- paired rules
def _job(seed, tune, oof, unseen=0.9, seen=0.98, oof_f1=0.8, tune_f1=0.85):
    return {"seed": seed, "tune_acc": tune, "oof_acc": oof, "tune_unseen_acc": unseen, "tune_seen_acc": seen,
            "oof_f1": oof_f1, "tune_f1": tune_f1}


def test_paired_gain_sd_and_seen_unseen():
    control = [_job(s, 0.95, 0.94) for s in (142, 143, 144)]
    runs = [_job(142, 0.955, 0.945, unseen=0.91, seen=0.97), _job(143, 0.957, 0.943, unseen=0.90),
            _job(144, 0.953, 0.947, unseen=0.92, oof_f1=0.79)]
    s = arms.paired(control, runs)
    gains = [0.005, 0.005, 0.005]
    assert s["gains"] == pytest.approx(gains) and s["gain"] == pytest.approx(0.005)
    assert s["gain_sd"] == pytest.approx(0.0, abs=1e-12)
    assert s["unseen_gain"] == pytest.approx(0.01) and s["seen_gain"] == pytest.approx(-0.01 / 3)
    assert s["oof_f1_change"] == pytest.approx(-0.01 / 3)
    assert arms.J(runs[0]) == pytest.approx(0.95)


def _stats(gains, unseen=0.0, f1=0.0):
    g = np.array(gains, float)
    return {"gain": float(g.mean()), "gain_sd": float(g.std(ddof=1)) if len(g) > 1 else 0.0, "gains": list(g),
            "unseen_gain": unseen, "oof_f1_change": f1}


def test_admission_rule():
    assert arms.admitted(_stats([0.002, 0.002, 0.002]))
    assert not arms.admitted(_stats([0.0005, 0.001, 0.0015]))                    # mean 0.1 pp but not > 2 SD
    assert not arms.admitted(_stats([0.0009, 0.0009, 0.0009]))                   # below +0.1 pp
    assert not arms.admitted(_stats([0.002, 0.002, 0.002], unseen=-1e-6))        # unseen gain must be >= 0
    assert arms.admitted(_stats([0.002, 0.002, 0.002], f1=-0.005))               # boundary: -0.005 allowed
    assert not arms.admitted(_stats([0.002, 0.002, 0.002], f1=-0.0051))
    assert not arms.admitted(_stats([0.002, 0.002]))                             # needs all 3 seeds
    assert arms.admitted(_stats([0.002]), n_seeds=1)


def test_early_drop_rule():
    assert arms.early_drop([-0.001]) and arms.early_drop([-0.005])              # seed 1: <= -0.1 pp
    assert not arms.early_drop([-0.0009]) and not arms.early_drop([0.0])
    assert arms.early_drop([0.0005, -0.001]) and not arms.early_drop([0.001, -0.001])   # seed 2: mean < 0
    assert not arms.early_drop([-0.01, -0.01, -0.01])                           # never after the last seed


def test_scenario_diff_ci_brackets_mean():
    rng = np.random.default_rng(0)
    scen = np.repeat(np.arange(40), 25)
    a, c = (rng.random(1000) < 0.9).astype(float), (rng.random(1000) < 0.85).astype(float)
    lo, hi = arms.scenario_diff_ci(a, c, scen)
    assert lo < (a - c).mean() < hi
    assert arms.scenario_diff_ci(a, a, scen) == [0.0, 0.0]


# ----------------------------------------------------------------------------------- seen / unseen split
def test_role_split_uses_training_scenarios():
    roles = np.array([0, 0, 1, 0, 2, 0, -1])
    scen_val = np.array([5, 7, 5, 9, 7, 5, 9])
    rows, seen, unseen = arms.role_split(roles, scen_val, np.array([1, 5, 5, 6]))
    assert list(rows) == [0, 1, 3, 5] and list(seen) == [0, 5] and list(unseen) == [1, 3]


@needs_v3
def test_role_split_matches_prepared_seen_flags():
    from framework.training.csv_v3 import runner
    roles = runner._arr("validation_roles")
    rows, seen, unseen = arms.role_split(roles, runner._arr("scenario_validation"), runner._arr("scenario_train"))
    stored = runner._arr("validation_scenario_seen").astype(bool)
    assert np.array_equal(np.isin(rows, seen), stored[rows])
    assert len(seen) + len(unseen) == len(rows) == int((roles == 0).sum())


# ------------------------------------------------------------------------------------------ frozen files
def test_freeze_is_write_once_with_sha(tmp_path):
    path = tmp_path / "rankings.json"
    value = {"P": ["a", "b"], "n": 2}
    assert arms.freeze(path, value) == value
    assert not list(tmp_path.glob("*.tmp"))          # written atomically, no temp file left behind
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert set(stored) == {"value", "sha256", "frozen"}
    assert arms.freeze(path, value) == value                                     # same value: fine
    assert arms.freeze(path, lambda: pytest.fail("must not recompute")) == value
    with pytest.raises(RuntimeError, match="frozen"):
        arms.freeze(path, {"P": ["b", "a"], "n": 2})
    stored["value"]["n"] = 3                                                     # edited by hand
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(RuntimeError, match="sha256"):
        arms.freeze(path, lambda: value)
    calls = []
    assert arms.freeze(tmp_path / "lazy.json", lambda: calls.append(1) or [1, 2]) == [1, 2] and calls == [1]


# ------------------------------------------------------------------------------------------ feature lists
def test_ranked_ties_keep_given_order():
    names = ["a", "b", "c", "d", "e"]
    assert arms.ranked(names, {"c": 2.0, "e": 1.0}) == ["c", "e", "a", "b", "d"]          # zero gain: v3 order
    assert arms.ranked(names, {"a": 1.0, "b": 1.0, "d": 0.5}, {"b": 0.2}) == ["b", "a", "d", "c", "e"]


@needs_v3
def test_arm_lists_sizes_order_and_subsets():
    names = _v3_names()
    rng = np.random.default_rng(3)
    rankings = {k: list(rng.permutation(names)) for k in ("P", "G", "FP")}
    lists = arms.arm_lists(names, rankings)
    assert {k: len(v) for k, v in lists.items()} == {"C0": 445, "P300": 300, "P200": 200, "G300": 300, "G200": 200,
                                                     "FP20": 425}
    pos = {n: i for i, n in enumerate(names)}
    for arm, listed in lists.items():
        assert set(listed) <= set(names) and len(set(listed)) == len(listed)
        assert [pos[n] for n in listed] == sorted(pos[n] for n in listed), arm
    assert set(lists["P200"]) == set(rankings["P"][:200]) and set(lists["P200"]) <= set(lists["P300"])
    assert set(lists["G300"]) == set(rankings["G"][:300])
    assert set(names) - set(lists["FP20"]) == set(rankings["FP"][:20])
    combo = arms.drop_top(lists["P300"], rankings["FP"], 20)
    assert set(combo) == set(lists["P300"]) - set(rankings["FP"][:20])
    with pytest.raises(ValueError):
        arms.arm_lists(names, {**rankings, "G": rankings["G"][:-1]})


def test_fingerprint_score_prefers_scenario_features():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 600)
    scen = np.concatenate([rng.integers(10 * c, 10 * c + 3, 600) for c in range(3)])
    X = np.stack([y + rng.normal(0, 0.1, len(y)),                  # class signal only
                  scen + rng.normal(0, 0.1, len(y)),               # scenario fingerprint
                  rng.normal(size=len(y))], axis=1)                # noise
    s = arms.fingerprint_scores(X, y, scen, ["cls", "scen", "noise"])
    assert s["scen"] > s["cls"] + 0.5
    pick = arms.per_class_sample(np.arange(len(y)) + 100, y, 50, seed=0)
    assert len(pick) == 150 and np.all(np.diff(pick) > 0)


# ------------------------------------------------------------------------------------------ driver guards
def _import_driver(env_update: dict) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASCADE_")} | env_update
    code = "import experiment.train_fs77_tabm as d; print(d.STUDY.name)"
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)


@needs_v3
def test_driver_uses_its_own_study_and_refuses_finished_ones():
    ok = _import_driver({})
    assert ok.returncode == 0 and ok.stdout.strip() == "fs77_tabm_ne_20261101_v1"
    smoke = _import_driver({"CASCADE_SMOKE": "1"})
    assert smoke.returncode == 0 and smoke.stdout.strip() == "fs77_tabm_ne_smoke"
    # Smoke and real results must never share a folder; inherited live feature groups must not override feature_list.
    for env in ({"CASCADE_STUDY": "artifacts/training/fs77_tabm_ne_smoke"},
                {"CASCADE_SMOKE": "1", "CASCADE_SMOKE_STUDY": "artifacts/training/fs77_tabm_ne_20261101_v1"},
                {"CASCADE_STUDY": "artifacts/training/all_models_day_smoke_20261020"}):
        bad = _import_driver(env)
        assert bad.returncode != 0 and "refusing" in bad.stderr
    groups = _import_driver({"CASCADE_FEATURE_GROUPS": '["core"]',
                             "CASCADE_STUDY": "artifacts/training/fs77_tabm_ne_20261101_v1"})
    assert groups.returncode != 0 and "CASCADE_FEATURE_GROUPS" in groups.stderr
    for study in ("artifacts/training/cascade_v3_20261030_v1", "artifacts/training/live77_upgrades_20261031_v1"):
        bad = _import_driver({"CASCADE_STUDY": study})
        assert bad.returncode != 0 and "refusing" in bad.stderr
    data = _import_driver({"CASCADE_DATA": "dataset/experiments/csv_v3_live77_derived_20261031_v1"})
    assert data.returncode != 0 and "csv_v3 data" in data.stderr
