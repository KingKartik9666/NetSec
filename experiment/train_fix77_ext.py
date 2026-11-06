"""Upgrade round 2, U2: the fix77 arm admitted for TabM-NE, tested on every other CSV family (pre-registered 2026-11-01).

    python -m experiment.train_fix77_ext start      # detached (log: <study>/controller.log)
    python -m experiment.train_fix77_ext run        # foreground
    python -m experiment.train_fix77_ext status
    python -m experiment.train_fix77_ext stop       # STOP file; halts between jobs
    CASCADE_SMOKE=1 python -m experiment.train_fix77_ext run     # wiring check (own folder, tiny rows, one seed)

Plan: ml_pipeline/notes_20261101/upgrades2_plan.md (U2) with the rules of fix77_plan.md. Runs only if fix77 stage 1
(``artifacts/training/fix77_tabm_ne_20261101_v1``, read-only) finished with an admitted TabM-NE arm that passed
confirmation; otherwise it writes a 'skipped' report and exits 0.

Per family (xgboost, kan, hem, tabm_hier, tabm_hier_ne, random_forest, tabpack): the shipped tuned/baseline config
(``artifacts/application/native_models_v1/export_report.json`` maps family -> cascade job; both read-only) is held
fixed; C0 = the 445 v3 features, the arm = the frozen fix77 feature list of the chosen arm. Fold 1, seeds 142/143/144
paired with that family's C0; early drop; admission (fix77 rule plus the round-wide "accuracy not lower": mean
role-0 accuracy gain >= 0); confirmation on 151/152/153 with the same rule; all-row refit (seed 42) calibrated on
role 1, kept only if its tuning accuracy is not below the shipped model's; assessment report-only. Families whose trainer cannot take row weights (kan, hem,
random_forest, tabpack) run the ROB features without weights, labelled as such. The fixed RandomForest control trains
only with seed 42 (adapter rule), so it gets one paired seed, report-only, and cannot be admitted (the rule needs
three seeds). Hybrids containing an adopted member (TabM-NE from stage 1 included) are rebuilt with the existing
combiner search (``runner.Controller.hybrid``) on the tuning role from fold-0/fold-1 OOF of the adopted configs and
the shipped OOF of the others; a rebuilt hybrid replaces the shipped one only if its tuning value is >= the shipped one.
Adoption into the app is a separate owner decision. The held-out test is never read.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Set explicitly (not setdefault): a queued or inherited environment may carry another study's folders.
os.environ["CASCADE_STUDY"] = "artifacts/training/fix77_ext_20261101_v1"
os.environ["CASCADE_SMOKE_STUDY"] = "artifacts/training/fix77_ext_smoke"
os.environ["CASCADE_DATA"] = "dataset/experiments/csv_v3_fix77_20261101_v1"

import numpy as np  # noqa: E402

from framework.training.csv_v3 import arms, fix77, prep, runner, upgrades2 as U  # noqa: E402

STUDY = runner.STUDY
REAL_STUDY = runner.ROOT / "artifacts/training/fix77_ext_20261101_v1"
SMOKE_STUDY = runner.ROOT / "artifacts/training/fix77_ext_smoke"
FIX77 = runner.ROOT / "artifacts/training/fix77_tabm_ne_20261101_v1"         # read-only (stage-1 decision)
CASCADE = runner.ROOT / "artifacts/training/cascade_v3_20261030_v1"         # read-only (shipped jobs)
EXPORT = runner.ROOT / "artifacts/application/native_models_v1/export_report.json"   # read-only
PLAN = runner.ROOT / "ml_pipeline/notes_20261101/upgrades2_plan.md"
if STUDY.resolve() != (SMOKE_STUDY if runner.SMOKE else REAL_STUDY).resolve():
    raise RuntimeError(f"refusing to run in {STUDY}: fix77 ext uses {SMOKE_STUDY if runner.SMOKE else REAL_STUDY}")
if runner.DATA.resolve() != fix77.OUT.resolve():
    raise RuntimeError(f"refusing data {runner.DATA}: the plan uses the fix77 data ({fix77.OUT})")
if runner.FEATURE_GROUPS is not None:
    raise RuntimeError("refusing to run with CASCADE_FEATURE_GROUPS set")

FAMILIES = ("xgboost", "kan", "hem", "tabm_hier", "tabm_hier_ne", "random_forest", "tabpack")
WEIGHTED = ("xgboost", "tabm_ne", "tabm_hier", "tabm_hier_ne")      # trainers that accept runner row weights
SEED42_ONLY = ("random_forest",)                                    # adapters_trees: RF only trains with seed 42
SEEDS = (142, 143, 144)
CONFIRM_SEEDS = (151, 152, 153)
FINAL_SEED = 42
SMOKE_FORCE = ("xgboost", "kan")      # smoke pushes one weighted and one unweighted family through every stage
if runner.SMOKE:
    SEEDS, CONFIRM_SEEDS = (142,), (151,)


def _log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


class Rows:
    def __init__(self):
        a = runner.task_arrays("77")
        self.y_val, self.y_train = a["y_val"], a["y_train"]
        self.classes = list(a["classes"])
        self.scen = runner._arr("scenario_validation")
        roles = runner._arr("validation_roles")
        self.tune, self.tune_seen, self.tune_unseen = arms.role_split(roles, self.scen, runner._arr("scenario_train"))
        stored = runner._arr("validation_scenario_seen").astype(bool)[self.tune]
        if not np.array_equal(stored, np.isin(self.tune, self.tune_seen)):
            raise RuntimeError("role-0 seen split differs from validation_scenario_seen.npy")
        m = runner.manifest()
        v3 = json.loads((prep.DEFAULT_OUT / "manifest.json").read_text(encoding="utf-8"))["feature_names"]
        if m["feature_names"] != v3 + fix77.NAMES or m.get("fix77_groups") != fix77.GROUPS:
            raise RuntimeError("fix77 data folder does not match fix77.py")
        self.v3 = v3


# --------------------------------------------------------------------------------------------- the gate
def skip_reason(fix77_study: Path = FIX77) -> tuple[str | None, dict | None]:
    """(None, decision) when U2 must run; (reason, decision-or-None) when it is skipped. Read-only."""
    path = Path(fix77_study) / "decision.json"
    decision = U.read_frozen(path) if path.exists() else None
    return U.ext_skip_reason(decision, U.read_phase(fix77_study)), decision


def write_skipped(study: Path, reason: str) -> None:
    study = Path(study)
    study.mkdir(parents=True, exist_ok=True)
    (study / "fix77_ext.json").write_text(json.dumps({"skipped": True, "reason": reason, "smoke": runner.SMOKE},
                                                     indent=1), encoding="utf-8")
    (study / "REPORT_fix77_ext.md").write_text(
        "# Upgrade round 2, U2: fix77 arm on every CSV family\n\n"
        f"**Skipped: {reason}.** Plan U2 runs only if fix77 stage 1 admitted and confirmed an arm for TabM-NE; every "
        "shipped CSV model and hybrid stays as it is.\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ configs
SHIPPED_MODELS = tuple(dict.fromkeys(FAMILIES + tuple(m for ms in runner.HYBRIDS.values() for m in ms)
                                     + tuple(runner.HYBRIDS)))


def base_configs() -> dict:
    """Freeze the inputs that matter: the shipped job of every family, hybrid member and hybrid (with the sha of
    its result.json) and the family configs. The whole export_report.json sha is recorded for information only
    (export_seen.json, not frozen): a re-export that keeps every mapping must not block a restart, while a changed
    mapping or a changed shipped result still fails the freeze. Later steps read the frozen mapping, never the
    live export."""
    export = json.loads(EXPORT.read_text(encoding="utf-8"))
    jobs = U.shipped_jobs(export, SHIPPED_MODELS)
    missing = [m for m in FAMILIES + ("tabm_ne",) if m not in jobs]
    if missing:
        raise RuntimeError(f"export_report.json has no shipped job for {missing}")
    out, shipped = {}, {}
    for name, job in jobs.items():
        path = CASCADE / "jobs" / job / "result.json"
        shipped[name] = {"job": job, "result_json_sha256": _sha256(path)}
        if name not in FAMILIES:
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        if r.get("status") != "complete" or r["family"] != name or r["task"] != "77":
            raise RuntimeError(f"{job} is not a complete 77-class {name} job")
        if any(k in r["config"] for k in ("feature_groups", "feature_list", "row_weight")):
            raise RuntimeError(f"{job}: shipped config already selects features or weights")
        out[name] = {"job": job, "result_json_sha256": shipped[name]["result_json_sha256"], "config": r["config"]}
    U.write_json_atomic(STUDY / "export_seen.json", {"export_report_sha256": _sha256(EXPORT),
                                                     "read": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                     "note": "information only; base_configs.json freezes the mapping"})
    return arms.freeze(STUDY / "base_configs.json", {"families": out, "shipped": shipped, "smoke": runner.SMOKE})


def make_config(config: dict, spec: dict, family: str) -> tuple[dict, str | None]:
    c = copy.deepcopy(config)
    c.update(feature_list=list(spec["names"]), feature_scope="csv")
    label = None
    if spec.get("row_weight") is not None:
        if family in WEIGHTED:
            c["row_weight"] = dict(spec["row_weight"])
            label = "scenario-balanced weights rho 1.0"
        else:
            label = "ROB features without weights (trainer cannot take row weights)"
    return c, label


def fold_job(ctl, R: Rows, job_id: str, family: str, config: dict, seed: int) -> dict:
    r = ctl.run_job(job_id, "77", family, config, seed=seed, rows="fold1")
    if r is None or r.get("status") != "complete":
        raise RuntimeError(f"{job_id} failed")
    work = STUDY / r["work_dir"]
    return {"job_id": job_id, "seed": seed, "work": str(work), "n_features": len(config["feature_list"]),
            **fix77.fold_metrics(work, R.y_val, R.y_train, R.tune, R.tune_seen, R.tune_unseen, R.classes)}


def _hits(R: Rows, runs: list[dict]) -> np.ndarray:
    return np.mean([np.load(Path(r["work"]) / "val_proba.npy")[R.tune].argmax(1) == R.y_val[R.tune] for r in runs], axis=0)


ADMISSION_RULE = ("every seed run; mean J gain >= 0.1 pp and > 2 x SD; mean role-0 unseen gain >= 0; mean OOF macro-F1 "
                  "change >= -0.005; mean role-0 accuracy gain >= 0 (upgrades2_plan.md: 'accuracy not lower')")


def summarize(R: Rows, control: list[dict], runs: list[dict], dropped, n_seeds: int) -> dict:
    """Paired summary with the round-wide accuracy floor (``U.paired_summary`` = fix77 rule AND mean role-0 accuracy
    gain >= 0), used for both admission and confirmation."""
    by = {r["seed"]: r for r in control}
    s = {**U.paired_summary(control, runs, n_seeds), "n_features": runs[0]["n_features"],
         "dropped_after_seed": dropped, "runs": runs,
         "focus_role0": fix77.mean_focus(runs, "focus_role0"), "focus_oof": fix77.mean_focus(runs, "focus_oof")}
    s["admitted"] = bool(dropped is None and s["admitted"])
    s["role0_diff_ci95"] = arms.scenario_diff_ci(_hits(R, runs), _hits(R, [by[r["seed"]] for r in runs]), R.scen[R.tune])
    return s


# ------------------------------------------------------------------------------------------- per family
def family_test(ctl, R: Rows, fam: str, base: dict, arm: str, lists: dict) -> dict:
    seeds = (FINAL_SEED,) if fam in SEED42_ONLY else SEEDS
    c0cfg, _ = make_config(base["config"], lists["C0"], fam)
    armcfg, weights = make_config(base["config"], lists[arm], fam)
    ctl.state(stage=f"{fam}: arms")
    control = [fold_job(ctl, R, f"77_ext_{fam}_C0_s{s}", fam, c0cfg, s) for s in seeds]
    by = {r["seed"]: r for r in control}
    runs, dropped = [], None
    for s in seeds:
        if ctl.stopped():
            raise KeyboardInterrupt("STOP present")
        runs.append(fold_job(ctl, R, f"77_ext_{fam}_{arm}_s{s}", fam, armcfg, s))
        if arms.early_drop([arms.J(r) - arms.J(by[r["seed"]]) for r in runs]):
            dropped = len(runs)
            break
    s = summarize(R, control, runs, dropped, 3)      # the plan rule always needs three seeds
    forced = runner.SMOKE and fam in SMOKE_FORCE
    _log(f"{fam} {arm}: J gain {100 * s['gain']:+.3f} pp, admitted {s['admitted']}")
    confirmation = None
    if (s["admitted"] or forced) and fam not in SEED42_ONLY:
        ctl.state(stage=f"{fam}: confirmation")
        c0 = [fold_job(ctl, R, f"77_ext_{fam}_C0_s{x}", fam, c0cfg, x) for x in CONFIRM_SEEDS]
        cr = [fold_job(ctl, R, f"77_ext_{fam}_{arm}_s{x}", fam, armcfg, x) for x in CONFIRM_SEEDS]
        confirmation = summarize(R, c0, cr, None, len(CONFIRM_SEEDS))
        _log(f"{fam} confirmation: J gain {100 * confirmation['gain']:+.3f} pp, passes {confirmation['admitted']}")
    confirmed = confirmation is not None and (confirmation["admitted"] or forced)
    decision = arms.freeze(STUDY / f"decision_{fam}.json", {
        "family": fam, "arm": arm, "shipped_job": base["job"], "weights": weights, "admitted": bool(s["admitted"]),
        "confirmed": bool(confirmed), "smoke_forced": forced,
        "note": "fixed RandomForest control: seed 42 only, report-only, cannot be admitted" if fam in SEED42_ONLY else None,
        "tune_acc_gain": s["tune_acc_gain"],
        "confirmation_tune_acc_gain": None if confirmation is None else confirmation["tune_acc_gain"],
        "rule": ADMISSION_RULE + "; then the same rule on seeds 151-153",
        "summary": {k: v for k, v in s.items() if k != "runs"},
        "confirmation": None if confirmation is None else {k: v for k, v in confirmation.items() if k != "runs"},
        "smoke": runner.SMOKE})
    fin, gate = None, None
    if decision["confirmed"]:
        ctl.state(stage=f"{fam}: final")
        fin = ctl.run_job(f"77_ext_{fam}_final_{arm}", "77", fam, armcfg, seed=FINAL_SEED, rows="full")
        if fin is None or fin.get("status") != "complete":
            raise RuntimeError(f"{fam} final refit failed")
        shipped = json.loads((CASCADE / "jobs" / base["job"] / "result.json").read_text(encoding="utf-8"))
        gate = refit_gate(fin, shipped, forced)
        _log(f"{fam} refit tuning accuracy {100 * gate['refit_tuning_acc']:.3f}% vs shipped "
             f"{100 * gate['shipped_tuning_acc']:.3f}%: passes {gate['passes']}")
    return {"decision": decision, "control": control, "summary": s, "confirmation": confirmation,
            "config": armcfg, "final": fin, "refit_gate": gate}


def tuning_acc(r: dict) -> float:
    """Role-0 accuracy of a full-row job as the app would use it (calibrated report when calibrated)."""
    return float((r.get("calibration") or {}).get("report", r["report"])["tuning"]["accuracy"])


def refit_gate(fin: dict, shipped: dict, forced: bool = False) -> dict:
    """The round-wide accuracy floor on the refitted model: its tuning accuracy must not be below the shipped
    model's (both calibrated as shipped). A family that fails stays shipped and is not used in hybrid rebuilds.
    Smoke-forced families pass for wiring only."""
    new, old = tuning_acc(fin), tuning_acc(shipped)
    return {"refit_tuning_acc": new, "shipped_tuning_acc": old, "gain": new - old,
            "passes": bool(new >= old - U.EPS or forced), "smoke_forced": bool(forced)}


# ------------------------------------------------------------------------------------------------ hybrids
def _abs(study: Path, r: dict) -> str:
    return str((study / r["work_dir"]).resolve())


def rebuild_hybrids(ctl, adopted: dict, shipped_map: dict) -> dict:
    """``adopted``: family -> {"config", "final" (result), "study"}; ``shipped_map``: the frozen model -> {"job"} map
    of base_configs.json. Hybrids with an adopted member are rebuilt. The hybrid job name carries a hash of its member
    jobs, so a restart whose adopted set changed never reuses a hybrid built from another set."""
    export = shipped_map
    out = {}
    for name, members in runner.HYBRIDS.items():
        if name not in export or not any(m in adopted for m in members):
            continue
        ctl.state(stage=f"hybrid {name}")
        branches, fold_jobs = {}, {}
        for m in members:
            if m in adopted:
                folds = [ctl.run_job(f"77_ext_oof_{m}_fold{k}", "77", m, adopted[m]["config"], seed=FINAL_SEED,
                                     rows=f"fold{k}") for k in (0, 1)]
                if any(f is None or f.get("status") != "complete" for f in folds):
                    raise RuntimeError(f"OOF jobs for {m} failed")
                fold_jobs[m] = [{"work_dir": _abs(STUDY, f)} for f in folds]
                fin = adopted[m]["final"]
                branches[m] = {"job_id": fin["job_id"], "work_dir": _abs(adopted[m]["study"], fin)}
            else:
                job = export[m]["job"]
                r = json.loads((CASCADE / "jobs" / job / "result.json").read_text(encoding="utf-8"))
                branches[m] = {"job_id": job, "work_dir": _abs(CASCADE, r)}
                fold_jobs[m] = []
                for k in (0, 1):
                    f = json.loads((CASCADE / "jobs" / f"77_oof_tuned_{m}_fold{k}" / "result.json").read_text(encoding="utf-8"))
                    if f.get("status") != "complete" or f.get("config") != r.get("config"):
                        raise RuntimeError(f"shipped OOF job 77_oof_tuned_{m}_fold{k} does not match {job}")
                    fold_jobs[m].append({"work_dir": _abs(CASCADE, f)})
        hname = U.hybrid_job_name(f"ext_{name}", branches, fold_jobs)
        U.check_cached_hybrid(ctl.result(f"77_hybrid_{hname}"), branches)
        new = ctl.hybrid("77", hname, branches, fold_jobs)
        U.check_cached_hybrid(new, branches)
        shipped = json.loads((CASCADE / "jobs" / export[name]["job"] / "result.json").read_text(encoding="utf-8"))
        v_new, v_old = runner.selection_value(new), runner.selection_value(shipped)
        out[name] = {"job": new["job_id"], "members": {m: ("adopted" if m in adopted else "shipped") for m in members},
                     "branch_jobs": new.get("branch_jobs"),
                     "combiner": (new.get("combiner") or {}).get("label"), "tuning_value": v_new,
                     "shipped_job": shipped["job_id"], "shipped_tuning_value": v_old, "replaces_shipped": v_new >= v_old}
        _log(f"hybrid {name}: tuning value {v_new:.5f} vs shipped {v_old:.5f}; replaces {v_new >= v_old}")
    return out


# ------------------------------------------------------------------------------------------------ main
def stage1_tabm_ne(decision: dict, lists: dict) -> dict | None:
    """The stage-1 TabM-NE final as an adopted hybrid member (config rebuilt exactly as train_fix77_tabm did)."""
    chosen = decision["chosen"]
    path = FIX77 / "jobs" / f"77_fix_final_{chosen}" / "result.json"
    if not path.exists():
        return None
    fin = json.loads(path.read_text(encoding="utf-8"))
    base = U.read_frozen(FIX77 / "base_config.json")
    config, _ = make_config(base["config"], lists[chosen], "tabm_ne")
    if fin.get("status") != "complete" or fin.get("config") != config:
        raise RuntimeError("stage-1 TabM-NE final does not match its frozen config")
    return {"config": config, "final": fin, "study": FIX77}


def run() -> None:
    busy = U.other_controllers()
    if busy:
        raise RuntimeError(f"another training controller or job is running (pids {busy}); one GPU job at a time")
    (STUDY / "controller_pid.json").write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    U.quarantine_failed(STUDY, _log)
    ctl = U.guard_run_job(runner.Controller(log=_log), _log)     # re-check other controllers before every fit
    ctl.state(phase="running", smoke=runner.SMOKE)
    try:
        reason, decision = skip_reason()
        if reason is not None:
            write_skipped(STUDY, reason)
            _log(f"U2 skipped: {reason}")
            ctl.state(phase="done", stage="skipped", reason=reason)
            return
        frozen_arms = U.read_frozen(FIX77 / "arms.json")
        if frozen_arms.get("smoke"):
            raise RuntimeError("fix77 arms.json is a smoke freeze")
        lists, arm = frozen_arms["arms"], decision["chosen"]
        R = Rows()
        if lists["C0"]["names"] != R.v3:
            raise RuntimeError("fix77 C0 list is not the v3 feature list")
        bases = base_configs()
        if bases["smoke"] != runner.SMOKE:
            raise RuntimeError("base_configs.json was frozen in the other (smoke/real) mode")
        arms.freeze(STUDY / "inputs.json", {"plan_sha256": _sha256(PLAN), "fix77_decision": decision, "arm": arm,
                                            "arm_spec": lists[arm], "data_manifest_sha256": _sha256(runner.DATA / "manifest.json"),
                                            "smoke": runner.SMOKE})
        results, failed, adopted = {}, {}, {}
        for fam in FAMILIES:
            try:
                results[fam] = family_test(ctl, R, fam, bases["families"][fam], arm, lists)
            except RuntimeError as exc:
                failed[fam] = str(exc)
                _log(f"{fam}: failed ({exc}); the shipped model stays")
                continue
            if results[fam]["final"] is not None and results[fam]["refit_gate"]["passes"]:
                adopted[fam] = {"config": results[fam]["config"], "final": results[fam]["final"], "study": STUDY}
        tabm = stage1_tabm_ne(decision, lists)
        if tabm is not None:
            adopted["tabm_ne"] = tabm
        hybrids = rebuild_hybrids(ctl, adopted, bases["shipped"]) if adopted else {}
        out = {"arm": arm, "fix77_decision": decision, "families": {f: {k: v for k, v in r.items() if k != "config"}
                                                                    for f, r in results.items()},
               "failed": failed, "adopted": sorted(adopted), "hybrids": hybrids}
        (STUDY / "fix77_ext.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        write_report(R, out, bases)
        ctl.state(phase="done", stage="report", failed_families=sorted(failed))
    except KeyboardInterrupt:
        ctl.state(phase="stopped")
        _log("stopped (STOP file); restart resumes from cached jobs")
    except Exception:
        ctl.state(phase="error")
        raise


# ------------------------------------------------------------------------------------------------ report
def _proba(job_dir: Path, r: dict, study: Path) -> np.ndarray:
    cal = job_dir / "val_proba_cal.npy"
    return np.load(cal) if cal.exists() else np.load(study / r["work_dir"] / "val_proba.npy")


def _model_row(label: str, r: dict) -> str:
    pct = lambda x: "-" if x is None else f"{100 * x:.2f}%"
    rep = (r.get("calibration") or {}).get("report", r["report"])
    a = rep["assessment"]
    ci = a["accuracy_ci95_scenario_bootstrap"]
    return (f"| {label} | {pct(rep['tuning']['accuracy'])} | {pct(a['accuracy'])} ({100 * ci[0]:.1f}-{100 * ci[1]:.1f}) | "
            f"{a['macro_f1_supported']:.4f} | {pct(a['benign_false_alarm_rate'])} | {(r.get('calibration') or {}).get('method')} |")


def write_report(R: Rows, o: dict, bases: dict) -> None:
    lines = ["# Upgrade round 2, U2: fix77 arm on every CSV family", "",
             f"Plan: ml_pipeline/notes_20261101/upgrades2_plan.md (U2) with fix77_plan.md rules. Arm: **{o['arm']}** "
             "(admitted and confirmed for TabM-NE in stage 1). Fold 1, seeds " + ", ".join(map(str, SEEDS)) +
             " vs each family's C0 (shipped config fixed); J = 0.5 x role-0 acc + 0.5 x fold-1 OOF acc. Validation "
             "only; assessment report-only; held-out test untouched.", ""]
    if runner.SMOKE:
        lines += ["**SMOKE RUN: wiring check only (tiny rows, 1 epoch / 4 rounds, one seed). No number here is a "
                  "result.**", ""]
    lines += ["| family | shipped job | weights | seeds | J gain (pp) | SD (pp) | unseen gain (pp) | OOF macro-F1 change "
              "| admitted | confirmation J gain (pp) | adopted |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam in FAMILIES:
        if fam in o["failed"]:
            lines.append(f"| {fam} | {bases['families'][fam]['job']} | | | | | | | failed: {o['failed'][fam]} | | no |")
            continue
        d = o["families"][fam]["decision"]
        s, c = d["summary"], d["confirmation"]
        note = f" (dropped after seed {s['dropped_after_seed']})" if s["dropped_after_seed"] else ""
        conf = "-" if c is None else f"{100 * c['gain']:+.3f}"
        lines.append(f"| {fam} | {d['shipped_job']} | {d['weights'] or '-'} | {s['seeds_run']} | {100 * s['gain']:+.3f} | "
                     f"{100 * s['gain_sd']:.3f} | {100 * s['unseen_gain']:+.3f} | {s['oof_f1_change']:+.4f} | "
                     f"{'yes' if d['admitted'] else 'no'}{note} | {conf} | "
                     f"{'yes' if fam in o['adopted'] else 'no'}{' (smoke-forced)' if d['smoke_forced'] else ''} |")
    lines += ["", "Admission (all must hold): mean J gain >= 0.1 pp and > 2 x SD; mean role-0 unseen gain >= 0; mean OOF "
                  "macro-F1 change >= -0.005; mean role-0 accuracy gain >= 0; every seed run; then the same rule on "
                  "seeds 151-153; then the refitted model's tuning accuracy must not be below the shipped model's. "
                  "RandomForest: the fixed control trains only with seed 42, so it is report-only.", ""]
    gates = [(f, r["refit_gate"]) for f, r in o["families"].items() if r.get("refit_gate")]
    if gates:
        lines += ["| family | refit tuning acc | shipped tuning acc | passes accuracy floor |", "|---|---|---|---|"]
        lines += [f"| {f} | {100 * g['refit_tuning_acc']:.3f}% | {100 * g['shipped_tuning_acc']:.3f}% | "
                  f"{'yes' if g['passes'] else 'no'}{' (smoke-forced)' if g['smoke_forced'] else ''} |" for f, g in gates]
        lines.append("")
    export = bases["shipped"]
    roles = runner._arr("validation_roles")
    a_rows = np.flatnonzero(roles == 2)
    finals = [(f, r["final"]) for f, r in o["families"].items() if r.get("final")]
    if finals:
        lines += ["## Refitted models vs shipped (all fit77 rows, calibration on role 1, assessment report-only)", "",
                  "| model | tuning acc | assessment acc (CI) | macro-F1 | benign FAR | calibration |", "|---|---|---|---|---|---|"]
        for fam, fin in finals:
            shipped = json.loads((CASCADE / "jobs" / export[fam]["job"] / "result.json").read_text(encoding="utf-8"))
            lines += [_model_row(f"{fam} shipped ({shipped['job_id']})", shipped), _model_row(f"{fam} {fin['job_id']}", fin)]
        lines.append("")
        for fam, fin in finals:
            shipped = json.loads((CASCADE / "jobs" / export[fam]["job"] / "result.json").read_text(encoding="utf-8"))
            p_new = _proba(runner.job_dir(fin["job_id"]), fin, STUDY)
            p_old = _proba(CASCADE / "jobs" / shipped["job_id"], shipped, CASCADE)
            diff = (p_new[a_rows].argmax(1) == R.y_val[a_rows]).astype(float) - (p_old[a_rows].argmax(1) == R.y_val[a_rows])
            ci = arms.scenario_diff_ci(diff, np.zeros_like(diff), R.scen[a_rows])
            fm_new, fm_old = (fix77.focus_metrics(p[R.tune], R.y_val[R.tune], R.classes) for p in (p_new, p_old))
            lines.append(f"- {fam}: paired assessment difference {100 * diff.mean():+.2f} pp (CI {100 * ci[0]:+.2f} to "
                         f"{100 * ci[1]:+.2f}, report-only); role 0 with ping merged {100 * fm_old['acc_ping_merged']:.2f}% "
                         f"-> {100 * fm_new['acc_ping_merged']:.2f}%.")
    if o["hybrids"]:
        lines += ["", "## Rebuilt hybrids (combiner search on the tuning role; replace only if >= shipped)", "",
                  "| hybrid | members | combiner | tuning value | shipped | replaces shipped |", "|---|---|---|---|---|---|"]
        for name, h in o["hybrids"].items():
            lines.append(f"| {name} | {', '.join(f'{m}: {v}' for m, v in h['members'].items())} | {h['combiner']} | "
                         f"{h['tuning_value']:.5f} | {h['shipped_tuning_value']:.5f} | {'yes' if h['replaces_shipped'] else 'no'} |")
    lines += ["", "Adoption of any refitted model or hybrid into the app is a separate owner decision (app chat).", ""]
    (STUDY / "REPORT_fix77_ext.md").write_text("\n".join(lines), encoding="utf-8")
    _log("report written")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "start", "status", "stop"])
    args = parser.parse_args(argv)
    STUDY.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        U.clear_stop_for_run(STUDY)      # kept when the queue launched this run (its stop must not be erased)
        run()
    elif args.command == "start":
        busy = U.other_controllers()
        if busy:
            print(f"not started: another training controller or job is running (pids {busy})")
            return 1
        (STUDY / "STOP").unlink(missing_ok=True)
        log = open(STUDY / "controller.log", "a", encoding="utf-8")
        flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0   # DETACHED_PROCESS | NEW_PROCESS_GROUP
        proc = subprocess.Popen([sys.executable, "-m", "experiment.train_fix77_ext", "run"], cwd=runner.ROOT,
                                stdout=log, stderr=subprocess.STDOUT, creationflags=flags, env=os.environ.copy())
        print(f"controller started, pid {proc.pid}; log: {STUDY / 'controller.log'}")
    elif args.command == "stop":
        (STUDY / "STOP").write_text("stop requested\n", encoding="utf-8")
        print("STOP written")
    else:
        state = STUDY / "state.json"
        print(state.read_text(encoding="utf-8") if state.exists() else "no state yet")
        print(f"{len(list((STUDY / 'jobs').glob('*/result.json')))} finished jobs; "
              f"other controllers/jobs: {U.other_controllers()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
