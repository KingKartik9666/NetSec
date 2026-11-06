"""Upgrade round 2 rules (plan: ml_pipeline/notes_20261101/upgrades2_plan.md): U3/U4/U5 decisions, the class merge,
the queue gate and the U2 skip. CPU only; drivers that set environment variables are imported in subprocesses."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from framework.training.csv_v3 import arms, class_merge, fix77, upgrades2 as U

ROOT = Path(__file__).resolve().parents[1]
CLASSES = json.loads((ROOT / "artifacts/application/native_models_v1/catalog.json").read_text(encoding="utf-8"))["classes"]


# ------------------------------------------------------------------------------------------------ U3
def _u3(**kw):
    s = {"gains": [0.002] * 3, "gain": 0.002, "gain_sd": 0.0009, "unseen_gain": 0.0, "oof_f1_change": -0.005,
         "tune_acc_gain": 0.0}
    s.update(kw)
    return s


def test_u3_rule_edges():
    assert U.u3_admitted(_u3())
    assert not U.u3_admitted(_u3(tune_acc_gain=-1e-9))           # role-0 accuracy must not fall
    assert not U.u3_admitted(_u3(gain_sd=0.001))                  # gain must exceed 2 x SD
    assert not U.u3_admitted(_u3(gain=0.00099, gains=[0.00099] * 3, gain_sd=0.0))
    assert not U.u3_admitted(_u3(unseen_gain=-1e-9))
    assert not U.u3_admitted(_u3(oof_f1_change=-0.0051))
    assert not U.u3_admitted(_u3(gains=[0.002] * 2))              # every confirmation seed run


def test_paired_summary_adds_role0_gain():
    run = lambda seed, acc, oof: {"seed": seed, "tune_acc": acc, "oof_acc": oof, "tune_seen_acc": acc,
                                  "tune_unseen_acc": acc, "tune_f1": 0.9, "oof_f1": 0.9}
    control = [run(s, 0.90, 0.90) for s in (151, 152, 153)]
    better = [run(151, 0.903, 0.903), run(152, 0.902, 0.902), run(153, 0.9025, 0.9025)]
    s = U.paired_summary(control, better, 3)
    assert s["tune_acc_gain"] == pytest.approx(0.0025) and s["admitted"]
    worse_role0 = [run(s_, 0.899, 0.905) for s_ in (151, 152, 153)]   # J up 0.2 pp but role 0 down
    assert not U.paired_summary(control, worse_role0, 3)["admitted"]


# ------------------------------------------------------------------------------------------------ U4
def _curve(accs, f1s=None, start=10):
    f1s = f1s or [0.9] * len(accs)
    return [{"rounds": start + 10 * i, "tune_acc": a, "tune_f1": f} for i, (a, f) in enumerate(zip(accs, f1s))]


def test_round_curve_keeps_shared_10_round_truncations_and_averages():
    p75 = [{"rounds": r, "tune_acc": 0.9, "tune_f1": 0.8} for r in (10, 20, 30, 40, 50, 60, 70, 75)]
    p150 = [{"rounds": r, "tune_acc": 0.92, "tune_f1": 0.82} for r in range(10, 151, 10)]
    c = U.round_curve([p75, p150])
    assert [p["rounds"] for p in c] == list(range(10, 151, 10))   # 75 is not a 10-round truncation
    assert c[0]["tune_acc"] == pytest.approx(0.91) and c[0]["n_probes"] == 2 and c[-1]["n_probes"] == 1


def test_rounds_ties_choose_fewer_rounds():
    accs = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.95, 0.95, 0.95, 0.95]       # 10..100, plateau from 60
    out = U.choose_rounds(_curve(accs), base=50)
    assert out["best"] == 70                       # first maximum of the smoothed curve (fewest rounds)
    assert out["adopted"] and out["chosen"] == 70  # smoothed gain at 70 vs 50 = 0.95 - 0.94 = +1 pp


def test_rounds_threshold_and_f1_guard():
    base = [0.95] * 6                               # 10..60 flat (50's smoothing window stays flat)
    # 70..90 raised by exactly 0.1 pp -> smoothed gain at 80 vs 50 is exactly +0.1 pp: adopted
    up = U.choose_rounds(_curve(base + [0.951, 0.951, 0.951, 0.95]), base=50)
    assert up["best"] == 80 and up["adopted"] and up["smoothed_gain"] == pytest.approx(0.001)
    small = U.choose_rounds(_curve(base + [0.9509, 0.9509, 0.9509, 0.95]), base=50)
    assert small["best"] == 80 and not small["adopted"] and small["chosen"] == 50
    f1 = [0.9] * 6 + [0.894, 0.894, 0.894, 0.9]      # 80: F1 falls by 0.006
    guard = U.choose_rounds(_curve(base + [0.96, 0.96, 0.96, 0.95], f1), base=50)
    assert guard["best"] == 80 and not guard["adopted"] and guard["chosen"] == 50
    f1_ok = [0.9] * 6 + [0.895, 0.895, 0.895, 0.9]   # exactly -0.005 is allowed
    assert U.choose_rounds(_curve(base + [0.96, 0.96, 0.96, 0.95], f1_ok), base=50)["adopted"]


def test_rounds_never_reduce_below_the_frozen_count():
    out = U.choose_rounds(_curve([0.94, 0.97, 0.95, 0.93, 0.93, 0.92, 0.92]), base=50)
    assert out["best"] == 20 and not out["adopted"] and out["chosen"] == 50
    assert not U.choose_rounds(_curve([0.9, 0.91]), base=50)["adopted"]   # 50 not on the curve (smoke)


# ------------------------------------------------------------------------------------------------ U5
def _oof(seed, n=4000, k=4):
    """Labels drawn with a skewed prior; probabilities over-favour the frequent class (adjustment helps)."""
    rng = np.random.default_rng(seed)
    prior = np.array([0.7, 0.1, 0.1, 0.1])
    y = rng.choice(k, size=n, p=prior)
    logits = rng.normal(0, 1.0, (n, k))
    logits[np.arange(n), y] += 1.2
    logits += 1.5 * np.log(prior)                 # model over-weights the prior
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return p, y, prior


def test_select_tau_prefers_adjustment_and_ties_go_to_smaller_tau():
    sel = U.select_tau([_oof(s) for s in (1, 2, 3)])
    assert sel["tau"] > 0 and sel["gain"] > 0.001
    p = np.array([[0.6, 0.4], [0.3, 0.7]])
    y = np.array([0, 1])
    flat = U.select_tau([(p, y, np.array([0.5, 0.5]))])  # uniform prior: every tau gives the same accuracy
    assert flat["tau"] == 0.0 and flat["gain"] == 0.0


def test_u5_adoption_rule():
    sel = {"tau": 0.25, "gain": 0.001}
    ok = {"acc_gain": 0.0, "unseen_gain": 0.0, "f1_change": -0.005}
    assert U.u5_adopted(sel, ok)
    assert not U.u5_adopted({"tau": 0.25, "gain": 0.0009}, ok)
    assert not U.u5_adopted(sel, {**ok, "acc_gain": -1e-9})
    assert not U.u5_adopted(sel, {**ok, "unseen_gain": -1e-9})
    assert not U.u5_adopted(sel, {**ok, "f1_change": -0.0051})
    assert not U.u5_adopted({"tau": 0.0, "gain": 0.0}, ok)


@pytest.mark.parametrize("temperature", [0.3, 0.4726, 1.0, 2.5])
def test_temperature_never_changes_the_adjusted_decision(temperature):
    rng = np.random.default_rng(7)
    p = rng.dirichlet(np.full(77, 0.3), size=500)
    p[:5] = 0.0
    p[:5, 3] = 1.0                                  # exact zeros hit the log floor
    prior = rng.dirichlet(np.ones(77))
    for tau in U.TAU_GRID:
        z = U.adjusted_logits(p, prior, tau)
        shown = U.display_proba(z, temperature)
        assert np.array_equal(shown.argmax(1), z.argmax(1))
        assert np.allclose(shown.sum(1), 1.0)
    with pytest.raises(ValueError):
        U.display_proba(z, 0.0)


def test_class_prior():
    y = np.array([0, 0, 1, 2, 2, 2])
    assert np.allclose(U.class_prior(y, np.array([0, 1, 2, 3]), 3), [0.5, 0.25, 0.25])


# ------------------------------------------------------------------------------------------------ U1
def test_class_merge_invariants():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(77), size=200).astype(np.float32)
    src, dst = class_merge.indices(CLASSES)
    q = class_merge.merge_proba(p, CLASSES)
    others = [c for c in range(77) if c not in (src, dst)]
    assert np.allclose(q.sum(1), 1.0, atol=1e-6)
    assert (q[:, src] == 0).all()
    assert np.array_equal(q[:, others], p[:, others].astype(np.float64))
    assert np.allclose(q[:, dst], p[:, dst].astype(np.float64) + p[:, src])
    assert np.array_equal(class_merge.merge_proba(p[0], CLASSES), q[0])            # one vector works too
    assert p[0, src] != 0                                                         # input not modified
    y = rng.integers(0, 77, 200)
    y[:3] = src
    fq, fy = fix77.ping_merge(p, y, CLASSES)
    assert np.array_equal(fq, q) and np.array_equal(fy, class_merge.merge_labels(y, CLASSES))
    assert (class_merge.merge_labels(y, CLASSES)[:3] == dst).all()
    with pytest.raises(ValueError):
        class_merge.indices(CLASSES[:76])


def test_golden_rows_come_from_the_tuning_role_only():
    rng = np.random.default_rng(1)
    roles = rng.integers(0, 3, 500)
    y = rng.integers(0, 77, 500)
    src, _ = class_merge.indices(CLASSES)
    y[:40] = src
    probas = [rng.dirichlet(np.ones(77), size=500) for _ in range(2)]
    rows = class_merge.golden_rows(roles, y, probas, CLASSES)
    assert len(rows) == 32 and len(set(rows)) == 32 and (roles[rows] == 0).all()
    assert np.sum(y[rows] == src) >= 8
    assert np.array_equal(rows, class_merge.golden_rows(roles, y, probas, CLASSES))   # deterministic


# --------------------------------------------------------------------------------------- queue + U2 gate
def test_fix77_gate():
    assert U.fix77_gate(True, [], {"a": "done", "b": "done"})[0] == "wait"
    assert U.fix77_gate(False, [123], {"a": "done", "b": "done"})[0] == "wait"
    assert U.fix77_gate(False, [], {"a": "done", "b": "done"})[0] == "go"
    for bad in ("error", "stopped", "running", None):
        assert U.fix77_gate(False, [], {"a": "done", "b": bad})[0] == "blocked"


def test_queue_refuses_when_a_fix77_study_is_in_error(tmp_path):
    from experiment import upgrades2_queue as Q
    studies = {}
    for name, phase in (("tabm", "done"), ("live", "error")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "state.json").write_text(json.dumps({"phase": phase}), encoding="utf-8")
        studies[name] = tmp_path / name
    queue = tmp_path / "queue"
    ok = Q.wait_for_fix77(studies, queue, poll=0, alive=lambda: False, busy=lambda: [])
    assert ok is False and "queue blocked" in (queue / "BLOCKED.md").read_text(encoding="utf-8")
    assert json.loads((queue / "queue_state.json").read_text(encoding="utf-8"))["phase"] == "blocked"
    assert "error" in (queue / "BLOCKED.md").read_text(encoding="utf-8")
    # both done -> proceeds; chain alive -> keeps waiting (STOP ends the wait)
    (tmp_path / "live" / "state.json").write_text(json.dumps({"phase": "done"}), encoding="utf-8")
    assert Q.wait_for_fix77(studies, tmp_path / "q2", poll=0, alive=lambda: False, busy=lambda: [])
    (tmp_path / "q3").mkdir()
    (tmp_path / "q3" / "STOP").write_text("x")
    assert Q.wait_for_fix77(studies, tmp_path / "q3", poll=0, alive=lambda: True, busy=lambda: []) is False


def test_queue_is_not_a_controller_marker():
    assert not any("upgrades2_queue" in m for m in U.CONTROLLER_MARKERS + tuple(fix77.CONTROLLER_MARKERS))
    assert set(fix77.CONTROLLER_MARKERS) <= set(U.CONTROLLER_MARKERS)
    assert {"experiment.train_upgrades2_live", "experiment.train_fix77_ext"} <= set(U.CONTROLLER_MARKERS)
    # both directions (fix77 chain finished 2026-11-02): the fix77 drivers' own guard lists this round's drivers
    assert set(U.DRIVER_MARKERS) <= set(fix77.CONTROLLER_MARKERS)
    assert len(U.CONTROLLER_MARKERS) == len(set(U.CONTROLLER_MARKERS))


def test_fix77_guard_sees_upgrades2_drivers_but_not_the_queue(monkeypatch):
    """A fix77 driver started by hand must refuse while an upgrades2 driver is alive (also between its jobs, when no
    runner job exists); the waiting queue and short status/stop commands are not GPU jobs."""
    import psutil
    from types import SimpleNamespace
    base = 10 ** 8
    procs = [("experiment.train_upgrades2_live", "run"), ("experiment.train_fix77_ext", "run"),
             ("experiment.upgrades2_queue", "_watch"), ("experiment.train_upgrades2_live", "status"),
             ("experiment.train_fix77_ext", "stop"), ("experiment.train_cascade_v3", "fit")]
    fake = [SimpleNamespace(info={"pid": base + i, "cmdline": ["python.exe", "-m", mod, arg]})
            for i, (mod, arg) in enumerate(procs)]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: iter(fake))
    want = [base + 0, base + 1, base + 5]                 # both drivers + a runner job (train_cascade_v3 CLI)
    assert fix77.other_controllers() == want
    assert U.other_controllers() == want


def test_ext_skip_reasons():
    ok = {"admitted": ["ROB"], "chosen": "ROB", "confirmed": True, "smoke_forced": False}
    assert U.ext_skip_reason(ok, "done") is None
    assert "not 'done'" in U.ext_skip_reason(ok, "error")
    assert "missing" in U.ext_skip_reason(None, "done")
    assert "no fix77 arm" in U.ext_skip_reason({"admitted": [], "chosen": None, "confirmed": False}, "done")
    assert "confirmation" in U.ext_skip_reason({**ok, "confirmed": False}, "done")
    assert "smoke" in U.ext_skip_reason({**ok, "smoke_forced": True}, "done")


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASCADE_")}
    env.update(extra)
    return env


def test_ext_driver_skips_when_nothing_admitted(tmp_path):
    fix = tmp_path / "fix77"
    fix.mkdir()
    (fix / "state.json").write_text(json.dumps({"phase": "done"}), encoding="utf-8")
    arms.freeze(fix / "decision.json", {"admitted": [], "chosen": None, "confirmed": False, "smoke_forced": False})
    out = tmp_path / "ext"
    code = ("from pathlib import Path; from experiment import train_fix77_ext as E; "
            f"r, d = E.skip_reason(Path(r'{fix}')); print(r); E.write_skipped(Path(r'{out}'), r)")
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=_env(), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "no fix77 arm" in proc.stdout
    assert json.loads((out / "fix77_ext.json").read_text(encoding="utf-8"))["skipped"] is True
    assert "Skipped" in (out / "REPORT_fix77_ext.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("module", ["experiment.train_fix77_ext", "experiment.train_upgrades2_live"])
def test_drivers_refuse_feature_groups_and_pin_their_study(module):
    proc = subprocess.run([sys.executable, "-c", f"import {module}"], cwd=ROOT,
                          env=_env(CASCADE_FEATURE_GROUPS='["core"]'), capture_output=True, text=True)
    assert proc.returncode != 0 and "refusing" in proc.stderr
    code = f"import {module} as m; print(m.STUDY.name, m.runner.DATA.name)"
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                          env=_env(CASCADE_STUDY="artifacts/training/elsewhere"), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split()[0] in ("fix77_ext_20261101_v1", "upgrades2_live_20261101_v1")


# ------------------------------------------------------------------------- adversarial-review fixes (round 2)
def _run_script(tmp_path, body: str, **env) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter (the drivers set CASCADE_* variables at import)."""
    script = tmp_path / "script.py"
    script.write_text(f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n" + body, encoding="utf-8")
    return subprocess.run([sys.executable, str(script)], cwd=ROOT, env=_env(**env), capture_output=True, text=True)


_SUMMARIZE = """
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from experiment import train_fix77_ext as E
from framework.training.csv_v3 import arms
tmp = Path(sys.argv[0]).parent
n = 200
R = SimpleNamespace(tune=np.arange(n), y_val=np.zeros(n, int), scen=np.arange(n) % 10)
def run(seed, acc, oof, tag):
    work = tmp / f"{tag}_{seed}"
    work.mkdir()
    p = np.zeros((n, 2)); p[:, 0] = 1.0
    np.save(work / "val_proba.npy", p)
    return {"seed": seed, "tune_acc": acc, "oof_acc": oof, "tune_seen_acc": acc, "tune_unseen_acc": 0.9,
            "tune_f1": 0.9, "oof_f1": 0.9, "n_features": 3, "work": str(work), "focus_role0": {"a": 0.1},
            "focus_oof": {"a": 0.1}}
c0 = [run(s, 0.90, 0.90, "c0") for s in (142, 143, 144)]
worse = [run(s, 0.898, 0.905 + 1e-5 * i, "worse") for i, s in enumerate((142, 143, 144))]
better = [run(s, 0.902, 0.903 + 1e-5 * i, "better") for i, s in enumerate((142, 143, 144))]
s_w, s_b = E.summarize(R, c0, worse, None, 3), E.summarize(R, c0, better, None, 3)
print(json.dumps({"bare_fix77_rule": bool(arms.admitted(arms.paired(c0, worse), 3)), "gain": s_w["gain"],
                  "worse": s_w["admitted"], "worse_acc_gain": s_w["tune_acc_gain"], "better": s_b["admitted"],
                  "dropped": E.summarize(R, c0, better, 2, 3)["admitted"]}))
"""


def test_u2_admission_and_confirmation_need_role0_accuracy_not_lower(tmp_path):
    """J = 0.5 role 0 + 0.5 OOF: OOF +0.5 pp with role 0 -0.2 pp gives J +0.15 pp, which the bare fix77 rule admits.
    The U2 summarize() (used for admission and for confirmation) must refuse it."""
    proc = _run_script(tmp_path, _SUMMARIZE)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["bare_fix77_rule"] and out["gain"] == pytest.approx(0.0015, abs=1e-4)   # the gap the fix closes
    assert out["worse"] is False and out["worse_acc_gain"] == pytest.approx(-0.002)
    assert out["better"] is True and out["dropped"] is False


_REFIT_GATE = """
import json
from experiment import train_fix77_ext as E
rep = lambda a: {"report": {"tuning": {"accuracy": 0.5}}, "calibration": {"report": {"tuning": {"accuracy": a}}}}
print(json.dumps([E.refit_gate(rep(0.951), rep(0.95))["passes"], E.refit_gate(rep(0.95), rep(0.95))["passes"],
                  E.refit_gate(rep(0.9499), rep(0.95))["passes"], E.refit_gate(rep(0.9), rep(0.95), True)["passes"],
                  E.tuning_acc({"report": {"tuning": {"accuracy": 0.7}}})]))
"""


def test_u2_refit_gate_keeps_accuracy_not_lower(tmp_path):
    proc = _run_script(tmp_path, _REFIT_GATE)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == [True, True, False, True, 0.7]


_BASE_CONFIGS = """
import json
from pathlib import Path
from experiment import train_fix77_ext as E
tmp = Path(sys.argv[0]).parent
E.STUDY = tmp / "study"; E.STUDY.mkdir()
export = json.loads(E.EXPORT.read_text(encoding="utf-8"))           # read-only: the real file is never written
E.EXPORT = tmp / "export_report.json"
E.EXPORT.write_text(json.dumps(export), encoding="utf-8")
first = E.base_configs()
export["oracle"] = {"changed": True}; export["study_replay"] = "re-exported"
E.EXPORT.write_text(json.dumps(export), encoding="utf-8")
second = E.base_configs()
export["models"]["xgboost"]["job"] = export["models"]["kan"]["job"]
E.EXPORT.write_text(json.dumps(export), encoding="utf-8")
try:
    E.base_configs(); raised = False
except RuntimeError:
    raised = True
print(json.dumps({"same": first == second, "raised": raised, "keys": sorted(first),
                  "shipped": sorted(first["shipped"]), "info": (E.STUDY / "export_seen.json").exists()}))
"""


def test_u2_base_configs_freeze_ignores_unrelated_export_changes(tmp_path):
    """A re-export that keeps every family/hybrid mapping must not block a restart; a changed mapping must."""
    proc = _run_script(tmp_path, _BASE_CONFIGS)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["same"] and out["raised"] and out["info"]
    assert out["keys"] == ["families", "shipped", "smoke"]          # no whole-file export sha in the freeze
    assert {"tabm_ne", "tabm_xgboost", "tabm_xgboost_kan", "tabpack_xgboost", "xgboost"} <= set(out["shipped"])


def test_hybrid_job_name_changes_with_members_and_stale_cache_is_refused():
    shipped = {"tabm_ne": {"job_id": "77_fix_final_ROB"}, "xgboost": {"job_id": "77_tune_xgboost_t005"}}
    adopted = {**shipped, "xgboost": {"job_id": "77_ext_xgboost_final_ROB"}}
    folds = {"tabm_ne": [{"work_dir": "a/f0"}, {"work_dir": "a/f1"}], "xgboost": [{"work_dir": "b/f0"}]}
    n1, n2 = U.hybrid_job_name("ext_tabm_xgboost", shipped, folds), U.hybrid_job_name("ext_tabm_xgboost", adopted, folds)
    assert n1 != n2 and n1.startswith("ext_tabm_xgboost_")
    assert n1 == U.hybrid_job_name("ext_tabm_xgboost", dict(reversed(list(shipped.items()))), folds)   # order-free
    assert n1 != U.hybrid_job_name("ext_tabm_xgboost", shipped, {**folds, "xgboost": [{"work_dir": "c/f0"}]})
    cached = {"job_id": "77_hybrid_x", "status": "complete", "branch_jobs": {m: b["job_id"] for m, b in shipped.items()}}
    U.check_cached_hybrid(cached, shipped)
    U.check_cached_hybrid(None, adopted)
    with pytest.raises(RuntimeError, match="built from"):
        U.check_cached_hybrid(cached, adopted)


def test_rounds_ulp_ties_still_choose_fewer_rounds():
    """Equal windows summed in another order can differ by one ulp; the first maximum must still win."""
    accs = [0.5, 0.85605, 0.9594, 0.949, 0.85605, 0.9594, 0.5]          # rounds 10..70 (review example)
    out = U.choose_rounds(_curve(accs), base=10)
    assert out["best"] == 30 and out["chosen"] == 30
    rng = np.random.default_rng(3)
    for _ in range(2000):                        # (a,b,c) plateau repeated: the tie must go to the earliest window
        a, b, c = rng.integers(15000, 20000, 3) / 20000
        vals = [0.1, a, b, c, a, b, c, 0.1]
        smooth = U.smooth3(vals)
        first = int(np.flatnonzero(smooth >= smooth.max() - 1e-12)[0])
        assert U.choose_rounds(_curve(vals), base=10)["best"] == 10 + 10 * first


def test_u4_frozen_record_is_deterministic_and_has_no_latency(tmp_path):
    curve = _curve([0.9, 0.91, 0.92, 0.93, 0.94])
    choice = U.choose_rounds(curve, base=50)
    probes = [{"job": "p75", "probe_rounds": 75, "curve": curve}]
    rec = U.u4_record("C0", 50, curve, choice, probes, False)
    assert not any("latency" in k for k in rec)
    path = tmp_path / "u4_decision.json"
    first = arms.freeze(path, rec)
    again = arms.freeze(path, U.u4_record("C0", 50, curve, choice, probes, False))   # restart: no "frozen and differs"
    assert first == again and again["rounds"] == choice["chosen"]


def test_proba_scale_and_final_display():
    p = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
    sc = U.proba_scale(p)
    assert sc["mean_max_proba"] == pytest.approx(0.55) and sc["rows"] == 2
    assert sc["mean_log_margin"] == pytest.approx(np.mean([np.log(3.5), np.log(0.4 / 0.35)]))
    from framework.training.csv_v3 import calibrate
    cfg = {"kind": "temperature", "temperature": 0.47, "alpha": 0.22}
    rng = np.random.default_rng(5)
    raw = rng.dirichlet(np.full(77, 0.4), size=300)
    off = {"adopted": False}
    assert np.allclose(U.final_display(raw, None, off, cfg), calibrate.apply(cfg, raw), atol=1e-6)
    assert np.array_equal(U.final_display(raw, None, off, {"kind": "identity"}), raw)
    prior = rng.dirichlet(np.ones(77))
    on = {"adopted": True, "tau": 0.25, "temperature": 0.47}
    assert np.allclose(U.final_display(raw, prior, on, cfg), U.display_proba(U.adjusted_logits(raw, prior, 0.25), 0.47))
    # the ping merge sums two entries, so its decision depends on the transform (why U1's OOF must match role 0)
    src, dst = class_merge.indices(CLASSES)
    other = next(c for c in range(77) if c not in (src, dst))
    q = np.full((1, 77), 1e-4)
    q[0, src], q[0, dst], q[0, other] = 0.3, 0.3, 0.4
    q /= q.sum()
    sharp = U.display_proba(np.log(q), 0.2)
    assert class_merge.decision(q, CLASSES)[0] == dst and class_merge.decision(sharp, CLASSES)[0] == other


class _FakeProc:
    def __init__(self, created=None, cmd=None, deny_create=False, deny_cmd=False):
        self.created, self.cmd, self.deny_create, self.deny_cmd = created, cmd, deny_create, deny_cmd

    def create_time(self):
        import psutil
        if self.deny_create:
            raise psutil.AccessDenied(1)
        return self.created

    def cmdline(self):
        import psutil
        if self.deny_cmd:
            raise psutil.AccessDenied(1)
        return self.cmd


def test_chain_alive_checks_create_time_and_boot(tmp_path, monkeypatch):
    import psutil
    from experiment import upgrades2_queue as Q
    pid_file = tmp_path / "chain_pid.json"
    pid_file.write_text(json.dumps({"pid": 4242, "started": 1000.0}), encoding="utf-8")
    marker = ["python", "-m", Q.CHAIN_MARKER, "_chain"]
    monkeypatch.setattr(psutil, "boot_time", lambda: 500.0)
    cases = [(_FakeProc(1000.01, marker), True),                      # the chain itself
             (_FakeProc(1000.01, deny_cmd=True), True),               # create time matches: still the chain
             (_FakeProc(5000.0, deny_cmd=True), False),               # reused pid of a protected process
             (_FakeProc(deny_create=True, deny_cmd=True), False),     # not ours, so not the chain
             (_FakeProc(5000.0, marker), False)]                      # same module but another process
    for proc, want in cases:
        monkeypatch.setattr(psutil, "Process", lambda pid, _p=proc: _p)
        assert Q.chain_alive(pid_file) is want
    monkeypatch.setattr(psutil, "Process", lambda pid: _FakeProc(1000.0, marker))
    monkeypatch.setattr(psutil, "boot_time", lambda: 2000.0)          # rebooted after the chain started
    assert Q.chain_alive(pid_file) is False
    pid_file.write_text(json.dumps({"pid": 4242}), encoding="utf-8")  # legacy file without 'started'
    monkeypatch.setattr(psutil, "Process", lambda pid: _FakeProc(deny_cmd=True))
    assert Q.chain_alive(pid_file) is True
    pid_file.write_text('{"pid": 42', encoding="utf-8")               # cut short by a reboot
    with pytest.raises(ValueError):
        Q.chain_alive(pid_file)
    ok = Q.wait_for_fix77({}, tmp_path / "queue", poll=0, alive=lambda: Q.chain_alive(pid_file), busy=lambda: [])
    assert ok is False and "cannot read" in (tmp_path / "queue" / "BLOCKED.md").read_text(encoding="utf-8")


class _FakeCtl:
    def __init__(self, cached=(), stop=False):
        self.cached, self.stop, self.calls = set(cached), stop, []

    def result(self, job_id):
        return {"status": "complete"} if job_id in self.cached else None

    def stopped(self):
        return self.stop

    def run_job(self, job_id, *args, **kwargs):
        self.calls.append(job_id)
        return {"job_id": job_id}


def test_guard_run_job_waits_for_other_controllers_before_each_fit():
    states = iter([[999], [999], []])
    sleeps = []
    ctl = U.guard_run_job(_FakeCtl(cached={"done"}), log=lambda m: None, busy=lambda: next(states), poll=0,
                          sleep=sleeps.append)
    assert ctl.run_job("new", "77", "xgboost", {}) == {"job_id": "new"}
    assert len(sleeps) == 2 and ctl.calls == ["new"]
    ctl.run_job("done", "77", "xgboost", {})                          # cached: no controller check at all
    assert ctl.calls == ["new", "done"]
    stopping = U.guard_run_job(_FakeCtl(stop=True), log=lambda m: None, busy=lambda: [7], poll=0, sleep=lambda s: None)
    with pytest.raises(KeyboardInterrupt):
        stopping.run_job("x", "77", "xgboost", {})
    assert stopping.calls == []


def test_queue_treats_an_honoured_stop_as_stopped_not_blocked(tmp_path, monkeypatch):
    from experiment import upgrades2_queue as Q
    steps = (("a", "mod.a", tmp_path / "a_smoke", tmp_path / "a_real"),
             ("b", "mod.b", tmp_path / "b_smoke", tmp_path / "b_real"))
    monkeypatch.setattr(Q, "STEPS", steps)
    monkeypatch.setattr(Q, "ROOT", tmp_path)
    monkeypatch.setattr(Q, "wait_for_fix77", lambda queue: True)
    monkeypatch.setattr(Q, "wait_ready", lambda queue: True)
    calls = []

    def driver(module, smoke, queue, phase="stopped"):
        study = [s for _, m, a, b in steps if m == module for s in (a, b)][0 if smoke else 1]
        study.mkdir(parents=True, exist_ok=True)
        (study / "state.json").write_text(json.dumps({"phase": phase}), encoding="utf-8")
        calls.append((module, smoke))
        return 0

    monkeypatch.setattr(Q, "run_driver", driver)
    queue = tmp_path / "queue"
    assert Q.watch(queue) == 1
    assert not (queue / "BLOCKED.md").exists()
    assert json.loads((queue / "queue_state.json").read_text(encoding="utf-8"))["phase"] == "stopped"
    calls.clear()

    def done_then_stop(module, smoke, q):     # a STOP that lands while a step runs is honoured right after it
        driver(module, smoke, q, phase="done")
        (q / "STOP").write_text("stop requested\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(Q, "run_driver", done_then_stop)
    q2 = tmp_path / "q2"
    assert Q.watch(q2) == 1 and calls == [("mod.a", True)] and (q2 / "STOP").exists()   # watch() never deletes it
    assert not (q2 / "BLOCKED.md").exists()


def test_queued_drivers_keep_their_stop_and_manual_runs_clear_it(tmp_path):
    from experiment import upgrades2_queue as Q
    (tmp_path / "STOP").write_text("stop requested\n", encoding="utf-8")
    assert U.clear_stop_for_run(tmp_path, {U.QUEUED_ENV: "1"}) is False and (tmp_path / "STOP").exists()
    assert U.clear_stop_for_run(tmp_path, {}) is True and not (tmp_path / "STOP").exists()
    assert Q.driver_env(True)[U.QUEUED_ENV] == "1" and Q.driver_env(True)["CASCADE_SMOKE"] == "1"
    assert U.QUEUED_ENV not in Q.driver_env(False, queued=False)


def test_unreadable_state_files_never_crash_the_queue(tmp_path):
    from experiment import upgrades2_queue as Q
    (tmp_path / "queue_state.json").write_text('{"phase": "runn', encoding="utf-8")   # cut short by a reboot
    Q.set_state(tmp_path, phase="waiting_for_fix77")
    state = json.loads((tmp_path / "queue_state.json").read_text(encoding="utf-8"))
    assert state["phase"] == "waiting_for_fix77" and "previous_state_unreadable" in state
    assert not (tmp_path / "queue_state.json.tmp").exists()
    (tmp_path / "queue_pid.json").write_text("{", encoding="utf-8")
    assert Q._watch_alive(tmp_path) is None
    study = tmp_path / "study"
    for job, text in (("ok", json.dumps({"status": "complete"})), ("cut", '{"status": "compl'),
                      ("bad", json.dumps({"status": "failed"}))):
        (study / "jobs" / job).mkdir(parents=True)
        (study / "jobs" / job / "result.json").write_text(text, encoding="utf-8")
    U.quarantine_failed(study, log=lambda m: None)
    assert sorted(p.name for p in (study / "jobs").iterdir()) == ["ok"]
    assert len(list((study / "failed_jobs").iterdir())) == 2


def test_cpu_only_reason_rules():
    assert U.cpu_only_reason({}, busy=lambda: []) is None
    assert "pids [12]" in U.cpu_only_reason({}, busy=lambda: [12])
    assert U.cpu_only_reason({U.CPU_TESTS_ENV: "1"}, busy=lambda: []) == f"{U.CPU_TESTS_ENV}=1"

    def broken():
        raise OSError("psutil failed")

    assert "check failed" in U.cpu_only_reason({}, busy=broken)       # cannot tell: stay off the GPU


def test_suite_hides_cuda_while_a_controller_runs():
    conf = [m for m in list(sys.modules.values()) if hasattr(m, "CPU_ONLY_REASON") and hasattr(m, "pytest_configure")]
    assert conf, "evaluation/conftest.py was not loaded"
    if conf[0].CPU_ONLY_REASON is None:
        pytest.skip("no controller alive at session start: CUDA tests are allowed")
    import torch
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and torch.cuda.is_available() is False


# ---------------------------------------------------------------- U5 amendment: tau on final-spec, final-round models
def test_tau_selection_jobs_use_final_spec_and_final_rounds():
    base = {s: {"n_estimators": 18, "early_stopping_rounds": 19, "lr": 0.03} for s in (151, 152, 153)}
    snapshot = json.dumps(base, sort_keys=True)
    wc = U.tau_selection_jobs(base, (151, 152, 153), "WC", 100, {"kind": "scenario", "rho": 1.0})
    assert sorted(wc) == [151, 152, 153]
    for s, (job, cfg) in wc.items():
        assert job == f"77_u2_tau_WC_r100_s{s}"
        assert cfg["n_estimators"] == 100 and cfg["early_stopping_rounds"] == 101     # exactly the final count
        assert cfg["row_weight"] == {"kind": "scenario", "rho": 1.0} and cfg["lr"] == 0.03
    c0 = U.tau_selection_jobs(base, (151, 152, 153), "C0", 50, None)
    assert all("row_weight" not in cfg and cfg["n_estimators"] == 50 for _, cfg in c0.values())
    assert {j for j, _ in c0.values()}.isdisjoint({j for j, _ in wc.values()})        # no cache reuse across specs
    assert json.dumps(base, sort_keys=True) == snapshot                               # inputs never mutated
    with pytest.raises(ValueError):
        U.tau_selection_jobs(base, (151,), "WC", 100, None)
    with pytest.raises(ValueError):
        U.tau_selection_jobs(base, (151,), "C0", 100, {"kind": "scenario", "rho": 1.0})
    with pytest.raises(ValueError):
        U.tau_selection_jobs(base, (151,), "C0", 0, None)


_U5_DRIVER = """
import inspect, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from experiment import train_upgrades2_live as D
from framework.training.csv_v3 import upgrades2 as U
tmp = Path(sys.argv[0]).parent
D.STUDY = tmp / "study"; (D.STUDY / "jobs").mkdir(parents=True)
D.job_metrics = lambda R, work: {}
k, n, m = 4, 4000, 1000
rng = np.random.default_rng(0)
prior = np.array([0.7, 0.1, 0.1, 0.1])
y_train, y_val = rng.choice(k, size=n, p=prior), rng.choice(k, size=m, p=prior)
D.runner.folds = lambda task: {1: {"fit": np.arange(n)}}
def proba(y, over):
    # logits = noise + 2 * onehot + over * log prior: the Bayes decision is argmax(logit - (over - 1/2) log prior),
    # so over = 1.5 favours the largest grid tau and over = -1 favours tau = 0.
    lg = rng.normal(0, 1.0, (len(y), k)); lg[np.arange(len(y)), y] += 2.0; lg += over * np.log(prior)
    return np.exp(lg) / np.exp(lg).sum(1, keepdims=True)
def write(job, over):
    work = D.STUDY / "jobs" / job / "work"; work.mkdir(parents=True, exist_ok=True)
    np.save(work / "val_proba.npy", proba(y_val, over)); np.save(work / "oof_index.npy", np.arange(n))
    np.save(work / "oof_proba.npy", proba(y_train, over))
    return work
class Ctl:
    def __init__(self, saved=None):
        self.calls, self.saved = [], saved
    def run_job(self, job_id, task, family, config, seed=42, rows="full"):
        self.calls.append([job_id, task, family, config["n_estimators"], config["early_stopping_rounds"],
                           config.get("row_weight"), seed, rows])
        work = write(job_id, 1.5)
        saved = config["n_estimators"] if self.saved is None else self.saved
        return {"status": "complete", "job_id": job_id, "work_dir": str(work.relative_to(D.STUDY)),
                "fit": {"trainer": {"saved_rounds": saved}}}
R = SimpleNamespace(y_train=y_train, y_val=y_val, tune=np.arange(m), tune_unseen=np.arange(m // 2), n_classes=k,
                    fit_all=np.arange(n), classes=[f"c{i}" for i in range(k)])
configs = {s: {"n_estimators": 18, "early_stopping_rounds": 19, "lr": 0.03} for s in D.CONFIRM_SEEDS}
ctl = Ctl()
sel = D.tau_models(ctl, R, configs, "WC", 100)
fold = [{"job_id": f"77_u2_WC_s{s}", "seed": s, "work": str(write(f"77_u2_WC_s{s}", -1.0))} for s in D.CONFIRM_SEEDS]
fin = {"raw": proba(y_val, 1.5), "cal": None, "calibration": {"kind": "identity"}, "job": "final"}
fin["cal"] = fin["raw"]
d = D.u5(R, sel, fin, fold)
fold_tau = U.select_tau([(np.load(Path(r["work"]) / "oof_proba.npy"), y_train, prior) for r in fold])["tau"]
try:
    D.tau_models(Ctl(saved=18), R, configs, "C0", 100); raised = False
except RuntimeError:
    raised = True
src = inspect.getsource(D.run)
order = [src.index("u4(ctl"), src.index("tau_models(ctl"), src.index("u5(R, selection_models")]
print(json.dumps({"calls": ctl.calls, "tau": d["selection"]["tau"], "fold_tau": fold_tau,
                  "selection_models": d["selection_models"], "rounds": d["selection_rounds"],
                  "fold_ref": d["fold_models_18_round_report_only"], "raised": raised, "order": order,
                  "uses_final_rounds": "tau_models(ctl, R, configs, spec, u4d[" in src}))
"""


def test_u5_tau_is_selected_on_final_spec_final_round_fold_models(tmp_path):
    """Plan U5 amendment: the three selection jobs run with U4's round count and the final spec on seeds 151-153, and
    tau comes from THEIR fold-1 OOF, not from the 18-round U3 fold models (constructed here to prefer tau = 0)."""
    proc = _run_script(tmp_path, _U5_DRIVER)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["calls"] == [[f"77_u2_tau_WC_r100_s{s}", "77", "xgboost", 100, 101, {"kind": "scenario", "rho": 1.0},
                             s, "fold1"] for s in (151, 152, 153)]
    assert out["tau"] == 0.5 and out["fold_tau"] == 0.0          # selection really reads the new models
    assert out["selection_models"] == [f"77_u2_tau_WC_r100_s{s}" for s in (151, 152, 153)]
    assert out["rounds"] == [100] and out["fold_ref"] == [f"77_u2_WC_s{s}" for s in (151, 152, 153)]
    assert out["raised"]                                         # a fit that did not keep the final count is refused
    assert out["order"] == sorted(out["order"]) and out["uses_final_rounds"]   # after U4, with U4's rounds


# ------------------------------------------------------------- per-class no-harm (plan amendment 2026-11-02)
def _labels(counts):
    return np.repeat(np.arange(len(counts)), counts)


def _moved(y, cls, rows, to):
    """Copy of ``y`` with the first ``rows`` rows of class ``cls`` predicted as ``to``."""
    p = y.copy()
    p[np.flatnonzero(y == cls)[:rows]] = to
    return p


def test_no_harm_passes_at_exactly_2pp_and_names_classes():
    y = _labels([100, 100, 100])
    out = U.per_class_no_harm(y, y.copy(), _moved(y, 1, 2, 0), 3, names=["a", "b", "c"])
    assert out["passed"] and out["breaches"] == [] and out["checked"] == ["a", "b", "c"] and out["n_pairs"] == 1
    assert out["largest_recall_drop"]["class"] == "b" and out["largest_recall_drop"]["drop"] == pytest.approx(0.02)
    assert out["largest_precision_drop"]["class"] == "a"
    assert out["largest_precision_drop"]["drop"] == pytest.approx(1 - 100 / 102)
    b = out["per_class"][1]
    assert b["recall_before"] == 1.0 and b["recall_after"] == pytest.approx(0.98) and b["precision_checked"]


def test_no_harm_recall_breach():
    y = _labels([1000, 100, 100])                       # moved rows land in a big class: its precision barely moves
    out = U.per_class_no_harm(y, y.copy(), _moved(y, 1, 3, 0), 3, names=["a", "b", "c"])
    assert not out["passed"]
    assert out["breaches"] == [{"class": "b", "metric": "recall", "drop": pytest.approx(0.03)}]


def test_no_harm_precision_breach_and_never_predicted_class():
    y = _labels([100, 1000])
    out = U.per_class_no_harm(y, y.copy(), _moved(y, 1, 3, 0), 2)
    assert out["breaches"] == [{"class": 0, "metric": "precision", "drop": pytest.approx(1 - 100 / 103)}]
    y = _labels([100, 100, 100])
    gone = U.per_class_no_harm(y, y.copy(), np.where(y == 2, 0, y), 3)   # candidate never predicts class 2
    c2 = gone["per_class"][2]
    assert c2["precision_checked"] and c2["precision_after"] == 0.0 and c2["precision_drop"] == pytest.approx(1.0)
    assert {(b["class"], b["metric"]) for b in gone["breaches"]} == {(2, "recall"), (2, "precision"), (0, "precision")}
    # the control never predicts class 2: its precision is not checked (recall still is)
    y = _labels([100, 100, 40])
    ctl = np.where(y == 2, 0, y)
    quiet = U.per_class_no_harm(y, ctl, ctl.copy(), 3)
    assert quiet["passed"] and not quiet["per_class"][2]["precision_checked"]
    assert quiet["per_class"][2]["precision_before"] is None and quiet["per_class"][2]["recall_drop"] == 0.0


def test_no_harm_ignores_classes_under_30_rows():
    y = _labels([2000, 29])                             # big class 0: absorbing the rows costs it < 2 pp precision
    out = U.per_class_no_harm(y, y.copy(), np.zeros_like(y), 2)         # class 1 wiped out, but only 29 rows
    assert out["passed"] and out["checked"] == [0]
    y = _labels([2000, 30])
    edge = U.per_class_no_harm(y, y.copy(), np.zeros_like(y), 2)
    assert not edge["passed"] and edge["checked"] == [0, 1]
    assert edge["breaches"] == [{"class": 1, "metric": "recall", "drop": 1.0},
                                {"class": 1, "metric": "precision", "drop": 1.0}]     # never predicted: precision 0


def test_no_harm_exempt_classes():
    y = _labels([100, 100, 100])
    cand = np.where(y == 1, 2, y)                       # class 1 merged into class 2 (U1-style)
    names = ["benign", "recon_ping-sweep", "recon_scan"]
    ok = U.per_class_no_harm(y, y.copy(), cand, 3, exempt={1, 2}, names=names)
    assert ok["passed"] and ok["checked"] == ["benign"] and ok["exempt"] == ["recon_ping-sweep", "recon_scan"]
    only_src = U.per_class_no_harm(y, y.copy(), cand, 3, exempt=[1], names=names)
    assert not only_src["passed"] and only_src["breaches"][0]["class"] == "recon_scan"   # its precision halves


def test_no_harm_multi_seed_uses_the_mean_paired_drop():
    y = _labels([100, 100])
    ctl = [y.copy()] * 3
    mixed = [_moved(y, 1, 4, 0), _moved(y, 1, 1, 0), y.copy()]        # recall drops 4, 1, 0 pp: mean 1.67 pp
    out = U.per_class_no_harm(y, ctl, mixed, 2)
    assert out["passed"] and out["n_pairs"] == 3
    assert out["per_class"][1]["recall_drop"] == pytest.approx(0.05 / 3)
    assert not U.per_class_no_harm(y, y, mixed[0], 2)["passed"]       # that one seed alone would fail
    bad = U.per_class_no_harm(y, ctl, [_moved(y, 1, 3, 0)] * 3, 2)     # 3 pp on every seed
    assert not bad["passed"] and {b["metric"] for b in bad["breaches"]} == {"recall", "precision"}
    assert U.per_class_no_harm([y, y, y], ctl, mixed, 2) == out        # one label array per pair (OOF) works too
    with pytest.raises(ValueError):
        U.per_class_no_harm(y, ctl, mixed[:2], 2)
    with pytest.raises(ValueError):
        U.per_class_no_harm(y, y, y[:-1], 2)


def test_no_harm_adopted_needs_every_check():
    ok, fail = {"passed": True}, {"passed": False}
    assert U.no_harm_adopted(True, ok, ok)
    assert not U.no_harm_adopted(True, ok, fail)
    assert not U.no_harm_adopted(False, ok)
    assert not U.no_harm_adopted(True, None)            # a skipped check never adopts
    assert not U.no_harm_adopted(True)


def test_u4_record_applies_the_no_harm_check(tmp_path):
    curve = _curve([0.95] * 6 + [0.96, 0.96, 0.96, 0.95])
    choice = U.choose_rounds(curve, base=50)
    assert choice["adopted"] and choice["chosen"] == 80
    with pytest.raises(ValueError):
        U.u4_record("C0", 50, curve, choice, [], False)
    y = _labels([100, 100])
    fail = U.per_class_no_harm(y, y, _moved(y, 1, 5, 0), 2)
    refused = U.u4_record("C0", 50, curve, choice, [], False, fail)
    assert not refused["adopted"] and refused["rounds"] == 50 and refused["no_harm_role0"]["breaches"]
    passed = U.u4_record("C0", 50, curve, choice, [], False, U.per_class_no_harm(y, y, y, 2))
    assert passed["adopted"] and passed["rounds"] == 80
    first = arms.freeze(tmp_path / "u4.json", refused)
    assert arms.freeze(tmp_path / "u4.json", U.u4_record("C0", 50, curve, choice, [], False, fail)) == first


_U3_NO_HARM = """
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from experiment import train_upgrades2_live as D
tmp = Path(sys.argv[0]).parent
k, m, n = 3, 600, 600
y_val, y_train = np.arange(m) % k, np.arange(n) % k
R = SimpleNamespace(y_val=y_val, y_train=y_train, tune=np.arange(m), scen=np.arange(m) % 20, n_classes=k,
                    classes=["benign", "ddos", "scan"])
def onehot(pred):
    p = np.full((len(pred), k), 0.01); p[np.arange(len(pred)), pred] = 0.98; return p
def write(work, pv, po):
    work.mkdir(parents=True, exist_ok=True)
    np.save(work / "val_proba.npy", onehot(pv)); np.save(work / "oof_index.npy", np.arange(n))
    np.save(work / "oof_proba.npy", onehot(po))
def hurt(y, cls, rows):
    p = y.copy(); p[np.flatnonzero(y == cls)[:rows]] = (cls + 1) % k; return p
metrics = lambda acc: {"tune_acc": acc, "oof_acc": acc, "tune_seen_acc": acc, "tune_unseen_acc": acc,
                       "tune_f1": 0.9, "oof_f1": 0.9}
WC_ACC = {151: 0.903, 152: 0.902, 153: 0.9025}                 # passes the paired U3 rule (J +0.25 pp, SD small)
def case(name, pv, po, repeat=1):
    D.STUDY = tmp / name; (D.STUDY / "jobs").mkdir(parents=True, exist_ok=True)
    seed_of, control = {}, []
    for s in D.CONFIRM_SEEDS:
        w = D.STUDY / "c0" / str(s); write(w, y_val, y_train)
        control.append({"job_id": f"77_up_C0_s{s}", "seed": s, "work": str(w), **metrics(0.90)})
    class Ctl:
        def run_job(self, job_id, task, family, config, seed=42, rows="full"):
            w = D.STUDY / "jobs" / job_id / "work"; write(w, pv, po); seed_of[str(w)] = seed
            return {"status": "complete", "job_id": job_id, "work_dir": str(w.relative_to(D.STUDY))}
    D.job_metrics = lambda R, work: metrics(WC_ACC[seed_of[str(work)]])
    for _ in range(repeat):                                      # a restart re-derives the same frozen decision
        d, _ = D.u3(Ctl(), R, control, {s: {} for s in D.CONFIRM_SEEDS})
    h = d["no_harm"]
    return {"rule": d["summary"]["admitted"], "adopted": d["adopted"], "spec": d["spec"], "passed": h["passed"],
            "role0_breaches": h["role0"]["breaches"], "oof_breaches": h["oof"]["breaches"],
            "checked": h["role0"]["checked"], "largest": h["role0"]["largest_recall_drop"]}
print(json.dumps({"clean": case("clean", y_val, y_train, repeat=2), "role0": case("role0", hurt(y_val, 1, 10), y_train),
                  "oof": case("oof", y_val, hurt(y_train, 2, 10))}))
"""


def test_u3_adoption_refused_when_no_harm_fails(tmp_path):
    """WC passes the paired U3 rule in every case; a class losing 5 pp recall on role 0 OR on fold-1 OOF blocks it."""
    proc = _run_script(tmp_path, _U3_NO_HARM)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert all(out[c]["rule"] for c in out)
    assert out["clean"]["adopted"] and out["clean"]["spec"] == "WC" and out["clean"]["checked"] == ["benign", "ddos", "scan"]
    assert not out["role0"]["adopted"] and out["role0"]["spec"] == "C0" and out["role0"]["oof_breaches"] == []
    assert {"class": "ddos", "metric": "recall", "drop": pytest.approx(0.05)} in out["role0"]["role0_breaches"]
    assert out["role0"]["largest"] == {"class": "ddos", "drop": pytest.approx(0.05)}
    assert not out["oof"]["adopted"] and out["oof"]["role0_breaches"] == []
    assert {"class": "scan", "metric": "recall", "drop": pytest.approx(0.05)} in out["oof"]["oof_breaches"]


_U4_NO_HARM = """
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import xgboost
from experiment import train_upgrades2_live as D
tmp = Path(sys.argv[0]).parent
k, m = 4, 1000
y = np.arange(m) % k
R = SimpleNamespace(y_val=y, tune=np.arange(m), n_classes=k, classes=["a", "b", "c", "d"])
D.runner.DATA = tmp
np.save(tmp / "X_validation.npy", np.zeros((m, 3), np.float32))
D.runner.feature_columns = lambda config: (np.arange(3), [])
D.L._latency = lambda path, rounds, X: 0.5
class Booster:                                     # no model is trained or loaded in this test
    def __init__(self, model_file=None): pass
    def num_boosted_rounds(self): return 150
xgboost.Booster = Booster
def make_pred(harm):
    def pred(r):
        p = y.copy()
        if r <= 50:
            p[np.flatnonzero(y == 0)[:100]] = 1    # class-0 errors up to 50 rounds: more rounds help a lot
        elif harm:
            p[np.flatnonzero(y == 1)[:10]] = 2     # ... but class b loses 4 pp recall after 50 rounds
        return p
    return pred
def case(name, harm):
    pred = make_pred(harm)
    D.STUDY = tmp / name; (D.STUDY / "jobs").mkdir(parents=True)
    D.runner.job_dir = lambda job: D.STUDY / "jobs" / job
    tops = {}
    class Ctl:
        def run_job(self, job_id, task, family, config, seed=42, rows="full"):
            w = D.STUDY / "jobs" / job_id / "work"; w.mkdir(parents=True, exist_ok=True)
            tops[str(w)] = config["n_estimators"]
            r = {"status": "complete", "job_id": job_id, "work_dir": str(w.relative_to(D.STUDY))}
            (D.STUDY / "jobs" / job_id / "result.json").write_text(json.dumps(r)); return r
    def curve(R_, work, cols, dm, yy, X_train):
        top = tops[str(work)]
        return [{"rounds": r, "tune_acc": float(np.mean(pred(r) == yy)), "tune_f1": D.arms.macro_f1(yy, pred(r), k)}
                for r in list(range(10, top, 10)) + [top]]
    D.L._curve = curve
    D._truncation_preds = lambda work, dm, keep, n: {int(r): pred(int(r)) for r in keep}
    d = D.u4(Ctl(), R, {"n_estimators": 50}, "C0", 50)
    h = d["no_harm_role0"]
    return {"rule": d["choice"]["adopted"], "chosen_by_rule": d["choice"]["chosen"], "adopted": d["adopted"],
            "rounds": d["rounds"], "probes": h["probes"], "pairs": h["n_pairs"],
            "breaches": sorted((b["class"], b["metric"]) for b in h["breaches"])}
print(json.dumps({"clean": case("clean", False), "harm": case("harm", True)}))
"""


def test_u4_adoption_refused_when_no_harm_fails(tmp_path):
    """More rounds pass the smoothed-accuracy rule (+6.7 pp) in both cases; when class b loses 4 pp recall (and c
    4 pp precision) at the chosen count vs the 50-round truncation, the frozen count is kept."""
    proc = _run_script(tmp_path, _U4_NO_HARM)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    for c in ("clean", "harm"):
        assert out[c]["rule"] and out[c]["chosen_by_rule"] == 70 and out[c]["pairs"] == 3   # all probes reach 70
    assert out["clean"]["adopted"] and out["clean"]["rounds"] == 70 and out["clean"]["breaches"] == []
    assert not out["harm"]["adopted"] and out["harm"]["rounds"] == 50
    assert out["harm"]["breaches"] == [["b", "recall"], ["c", "precision"]]


_U5_NO_HARM = _U5_DRIVER.split("fold_tau = ")[0] + """
h = d["no_harm"]
print(json.dumps({"tau": d["selection"]["tau"], "rule": d["rule_passed"], "adopted": d["adopted"],
                  "passed": h["passed"], "pairs": h["oof"]["n_pairs"], "role0_breaches": h["role0"]["breaches"],
                  "oof_breaches": h["oof"]["breaches"], "frozen_tau": d["tau"]}))
"""


def test_u5_adoption_refused_when_no_harm_fails(tmp_path):
    """Strong prior over-weighting: tau 0.5 passes every U5 accuracy rule, but shifting decisions towards the minority
    classes costs their precision far more than 2 pp, so the adjustment is not adopted (frozen tau 0)."""
    proc = _run_script(tmp_path, _U5_NO_HARM)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["tau"] == 0.5 and out["rule"] is True
    assert out["adopted"] is False and out["frozen_tau"] == 0.0 and out["passed"] is False and out["pairs"] == 3
    for key in ("role0_breaches", "oof_breaches"):
        hit = {(b["class"], b["metric"]) for b in out[key]}
        assert hit and {c for c, _ in hit} <= {"c1", "c2", "c3"} and {m for _, m in hit} == {"precision"}


_U1_NO_HARM = """
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from experiment import train_upgrades2_live as D
from framework.training.csv_v3 import class_merge
tmp = Path(sys.argv[0]).parent
D.STUDY = tmp / "study"; D.STUDY.mkdir()
classes = json.loads((D.runner.ROOT / "artifacts/application/native_models_v1/catalog.json").read_text(encoding="utf-8"))["classes"]
src, dst = class_merge.indices(classes)
other = next(c for c in range(77) if c not in (src, dst))
y = np.concatenate([np.full(60, src), np.full(60, dst), np.full(60, other)])
def proba(close):
    p = np.full((len(y), 77), 1e-6); p[np.arange(len(y)), y] = 0.9
    ping = np.flatnonzero(y == src); p[ping, src], p[ping, dst] = 0.4, 0.45   # ping rows read as scan: merge helps
    if close:                                    # 'other' rows: other 0.4 vs ping 0.3 + scan 0.3 -> merged to scan
        rows = np.flatnonzero(y == other)[:6]; p[rows] = 1e-6; p[rows, other], p[rows, src], p[rows, dst] = 0.4, 0.3, 0.3
    return p / p.sum(1, keepdims=True)
def case(name, close):
    D.STUDY = tmp / name; D.STUDY.mkdir()
    p = proba(close)
    work = D.STUDY / "fold"; work.mkdir()
    np.save(work / "oof_index.npy", np.arange(len(y))); np.save(work / "oof_proba.npy", p)
    R = SimpleNamespace(y_val=y, y_train=y, tune=np.arange(len(y)), n_classes=77, classes=classes)
    d = D.u1(R, [{"job_id": "f", "work": str(work)}], {"cal": p, "raw": p, "calibration": {"kind": "identity"}},
             {"adopted": False, "prior_fold1_for_selection": [1 / 77] * 77})
    return {"acc_rule": d["tuning_after"] >= d["tuning_before"], "adopted": d["adopted"],
            "exempt": d["no_harm"]["role0"]["exempt"], "breaches": d["no_harm"]["role0"]["breaches"],
            "oof_passed": d["no_harm"]["oof"]["passed"], "md": D.no_harm_md("role 0", d["no_harm"]["role0"]),
            "absent": D.no_harm_md("x", None, "not needed"), "other": classes[other]}
print(json.dumps({"clean": case("clean", False), "close": case("close", True)}))
"""


def test_u1_no_harm_exempts_ping_and_scan_and_report_lines(tmp_path):
    """U1: ping-sweep (removed) and recon_scan (receives its rows) are exempt; a third class whose rows the merge
    moves to recon_scan is checked, and its 10 pp recall loss blocks the merge even though accuracy passes."""
    proc = _run_script(tmp_path, _U1_NO_HARM)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert sorted(out["clean"]["exempt"]) == ["recon_ping-sweep", "recon_scan"]
    assert out["clean"]["acc_rule"] and out["clean"]["adopted"] and out["clean"]["breaches"] == []
    close = out["close"]
    assert close["acc_rule"] and not close["adopted"] and not close["oof_passed"]   # 60 hits gained, 6 lost
    assert close["breaches"] == [{"class": close["other"], "metric": "recall", "drop": pytest.approx(0.1)}]
    assert "FAILED" in close["md"][0] and f"{close['other']} recall -10.00 pp" in close["md"][0]
    assert "largest recall drop " + close["other"] in close["md"][0] and close["other"] in close["md"][1]
    assert "passed" in out["clean"]["md"][0] and out["clean"]["absent"] == ["- No-harm, x: not needed."]
