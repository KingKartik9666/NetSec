"""Upgrade round 2, live 77-class XGBoost: U3 (WC confirmation), U4 (more rounds), U5 (logit adjustment), U1 (ping merge).

    python -m experiment.train_upgrades2_live start     # detached (log: <study>/controller.log)
    python -m experiment.train_upgrades2_live run       # foreground
    python -m experiment.train_upgrades2_live status
    python -m experiment.train_upgrades2_live stop      # STOP file; halts between jobs
    CASCADE_SMOKE=1 python -m experiment.train_upgrades2_live run    # wiring check (own folder, tiny rows, one seed)

Plan (pre-registered 2026-11-01, not changed here): ml_pipeline/notes_20261101/upgrades2_plan.md. Owner 2026-11-01:
"finish every suggested upgrade that benefits positively"; "this live screening cant score less acc".
Selection reads role 0 (tuning) and training fold-1 OOF rows only; role 1 only calibrates a refitted model; role 2 is
report-only; the held-out test is never read. One GPU job at a time (runner subprocesses, controller guard).

Control reuse for U3 (why C0 is not re-run). The plan pairs WC with the existing C0 jobs ``77_up_C0_s151..153`` of
``artifacts/training/live77_upgrades_20261031_v1``. That study is finished; opening a runner Controller on it would
rewrite its state.json and append to its ledger, i.e. change existing files of a finished study, which is not allowed.
So this driver never runs a controller there: it only READS the three C0 results (result.json, val_proba.npy,
oof_index.npy, oof_proba.npy; hashed into the frozen ``controls.json``) and trains WC in its own study with each C0
config copied verbatim plus only ``row_weight = {"kind": "scenario", "rho": 1.0}``. That is exactly how the original
WC_s142..144 were built (checked at start: their config equals C0_s142..144's plus row_weight), on the same data
folder (CASCADE_DATA below), the same fold rows, the same 18 rounds and the same seeds; only the study folder differs,
and the runner's job folder does not enter training. In smoke mode C0 is re-run inside the smoke study (the reused
C0 were trained on all fit rows and would not pair with smoke fits).

Order (plan): U3 -> U4 -> final refit if U3 or U4 changed the model (seed 42, all fit rows, calibration on role 1)
-> U5 -> U1 -> report. The refit is done before U5 because it depends only on U3/U4, and U5's role-0 checks must use
the final model's raw probabilities. U5 (plan amendment 2026-11-01) trains three short fold-1 jobs with the final
spec and U4's final round count on seeds 151-153 (``77_u2_tau_<spec>_r<rounds>_s<seed>``) and selects tau on their
fold-1 OOF, not on the 18-round U3 fold models, whose logit scale differs from the final model's. U1 is an output
rule and needs no training (its OOF report stays on the U3 fold models of the chosen spec, as the plan states).

Per-class no-harm (plan amendment 2026-11-02, owner "yes then", before any U3-U5 result; ``U.per_class_no_harm``):
adoption also requires that no class with >= 30 evaluated rows loses more than 2 pp recall or precision against the
control. U3: role 0 and fold-1 OOF, WC vs C0 paired by seed, mean paired drop. U4: role 0, the chosen larger count vs
the frozen-count truncation within the probes that reach it (probe mean). U5: role 0 (final raw, tau vs 0) and the
tau-selection jobs' fold-1 OOF. U1: role 0 and fold-1 OOF with ping-sweep and recon_scan exempt. The frozen decision
files and the report list the classes checked, the largest drops (with class names) and any breach.
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
os.environ["CASCADE_STUDY"] = "artifacts/training/upgrades2_live_20261101_v1"
os.environ["CASCADE_SMOKE_STUDY"] = "artifacts/training/upgrades2_live_smoke"
os.environ["CASCADE_DATA"] = "dataset/experiments/csv_v3_live77_derived_20261031_v1"

import numpy as np  # noqa: E402

from experiment import train_live77 as L  # noqa: E402
from framework.training.csv_v3 import arms, class_merge, evaluate, prep, runner, upgrades2 as U  # noqa: E402

STUDY = runner.STUDY
REAL_STUDY = runner.ROOT / "artifacts/training/upgrades2_live_20261101_v1"
SMOKE_STUDY = runner.ROOT / "artifacts/training/upgrades2_live_smoke"
DATA = runner.ROOT / "dataset/experiments/csv_v3_live77_derived_20261031_v1"
UPG = runner.ROOT / "artifacts/training/live77_upgrades_20261031_v1"       # read-only (C0 controls, WC s142-144)
MAIN = runner.ROOT / "artifacts/training/live77_20261031_v1"               # read-only (shipped live final)
PLAN = runner.ROOT / "ml_pipeline/notes_20261101/upgrades2_plan.md"
if STUDY.resolve() != (SMOKE_STUDY if runner.SMOKE else REAL_STUDY).resolve():
    raise RuntimeError(f"refusing to run in {STUDY}: upgrades2 live uses {SMOKE_STUDY if runner.SMOKE else REAL_STUDY}")
if runner.DATA.resolve() != DATA.resolve():
    raise RuntimeError(f"refusing data {runner.DATA}: the C0 controls were trained on {DATA}")
if runner.FEATURE_GROUPS is not None:
    raise RuntimeError("refusing to run with CASCADE_FEATURE_GROUPS set")

CONFIRM_SEEDS = (151, 152, 153)
FINAL_SEED = 42
WC = {"kind": "scenario", "rho": 1.0}
PROBES = (75, 100, 150)
if runner.SMOKE:
    CONFIRM_SEEDS = (151,)
_log = L._log


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
        self.n_classes = len(self.classes)
        self.scen = runner._arr("scenario_validation")
        self.roles = runner._arr("validation_roles")
        self.tune, self.tune_seen, self.tune_unseen = arms.role_split(self.roles, self.scen, runner._arr("scenario_train"))
        stored = runner._arr("validation_scenario_seen").astype(bool)[self.tune]
        if not np.array_equal(stored, np.isin(self.tune, self.tune_seen)):
            raise RuntimeError("role-0 seen split differs from validation_scenario_seen.npy")
        self.fit_all = np.flatnonzero(a["fit"])


def job_metrics(R: Rows, work: Path) -> dict:
    """Role-0 and fold OOF numbers of one fold job (roles 1/2 never read)."""
    pv = np.load(work / "val_proba.npy")
    idx, po = np.load(work / "oof_index.npy"), np.load(work / "oof_proba.npy")
    return {**arms.role0_metrics(pv, R.y_val, R.tune, R.tune_seen, R.tune_unseen, R.n_classes),
            **arms.oof_metrics(po, R.y_train[idx], R.n_classes)}


# ------------------------------------------------------------------------------------------------ controls
def _read_result(study: Path, job_id: str) -> dict:
    r = json.loads((study / "jobs" / job_id / "result.json").read_text(encoding="utf-8"))
    if r.get("status") != "complete":
        raise RuntimeError(f"{study.name}/{job_id} is not complete")
    return r


def check_wc_construction() -> dict:
    """The original WC jobs were C0's config plus only row_weight WC (so copying C0 + WC reproduces the arm)."""
    out = {}
    for s in (142, 143, 144):
        c0 = _read_result(UPG, f"77_up_C0_s{s}")["config"]
        wc = dict(_read_result(UPG, f"77_up_WC_scenario_weight_1.0_s{s}")["config"])
        if wc.pop("row_weight", None) != WC or wc != c0:
            raise RuntimeError(f"WC_s{s} is not C0_s{s} + row_weight {WC}: the reuse argument does not hold")
        out[s] = True
    return out


def controls(ctl, R: Rows) -> tuple[list[dict], dict[int, dict]]:
    """C0 fold-1 runs on the confirmation seeds and their configs (read from the finished study; re-run in smoke)."""
    configs = {s: _read_result(UPG, f"77_up_C0_s{s}")["config"] for s in CONFIRM_SEEDS}
    if runner.SMOKE:
        runs = [fold_job(ctl, R, f"77_u2_C0_s{s}", configs[s], s) for s in CONFIRM_SEEDS]
        return runs, configs
    runs, record = [], {}
    for s in CONFIRM_SEEDS:
        r = _read_result(UPG, f"77_up_C0_s{s}")
        if r["seed"] != s or r["rows"] != "fold1" or r["family"] != "xgboost":
            raise RuntimeError(f"77_up_C0_s{s} is not a fold-1 xgboost job with seed {s}")
        work = UPG / r["work_dir"]
        files = [UPG / "jobs" / f"77_up_C0_s{s}" / "result.json"] + [work / f for f in
                                                                    ("val_proba.npy", "oof_index.npy", "oof_proba.npy")]
        record[f"77_up_C0_s{s}"] = {"work_dir": str(work.relative_to(runner.ROOT).as_posix()),
                                    "sha256": {f.name: _sha256(f) for f in files}}
        runs.append({"job_id": f"77_up_C0_s{s}", "seed": s, "work": str(work), **job_metrics(R, work)})
    arms.freeze(STUDY / "controls.json", {"study": UPG.relative_to(runner.ROOT).as_posix(), "jobs": record,
                                           "data_manifest_sha256": _sha256(runner.DATA / "manifest.json"),
                                           "wc_construction_checked": {str(k): v for k, v in check_wc_construction().items()}})
    return runs, configs


def fold_job(ctl, R: Rows, job_id: str, config: dict, seed: int) -> dict:
    r = ctl.run_job(job_id, "77", "xgboost", config, seed=seed, rows="fold1")
    if r is None or r.get("status") != "complete":
        raise RuntimeError(f"{job_id} failed")
    work = STUDY / r["work_dir"]
    return {"job_id": job_id, "seed": seed, "work": str(work), **job_metrics(R, work)}


def with_weight(config: dict, weighted: bool) -> dict:
    c = copy.deepcopy(config)
    if weighted:
        c["row_weight"] = dict(WC)
    return c


def with_rounds(config: dict, rounds: int) -> dict:
    c = copy.deepcopy(config)
    c.update(n_estimators=int(rounds), early_stopping_rounds=int(rounds) + 1)
    return c


def _hits(R: Rows, runs: list[dict]) -> np.ndarray:
    return np.mean([np.load(Path(r["work"]) / "val_proba.npy")[R.tune].argmax(1) == R.y_val[R.tune] for r in runs], axis=0)


def _role0_pred(R: Rows, run: dict) -> np.ndarray:
    return np.load(Path(run["work"]) / "val_proba.npy")[R.tune].argmax(1)


def _oof_pred(R: Rows, run: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fold-1 OOF labels and argmax predictions of one fold job."""
    work = Path(run["work"])
    return R.y_train[np.load(work / "oof_index.npy")], np.load(work / "oof_proba.npy").argmax(1)


def _no_harm(R: Rows, y, control, candidate, exempt=()) -> dict:
    """``U.per_class_no_harm`` with this study's class names (frozen files and the report name the classes)."""
    return U.per_class_no_harm(y, control, candidate, R.n_classes, exempt=exempt, names=R.classes)


def u3_no_harm(R: Rows, control: list[dict], runs: list[dict]) -> dict:
    """Plan amendment 2026-11-02: WC vs C0 paired by seed (151-153), mean paired drop per class, on role 0 AND on
    fold-1 OOF."""
    by = {r["seed"]: r for r in control}
    pairs = [(by[r["seed"]], r) for r in runs]
    role0 = _no_harm(R, R.y_val[R.tune], [_role0_pred(R, c) for c, _ in pairs], [_role0_pred(R, w) for _, w in pairs])
    oc, ow = [_oof_pred(R, c) for c, _ in pairs], [_oof_pred(R, w) for _, w in pairs]
    for (yc, _), (yw, _) in zip(oc, ow):
        if not np.array_equal(yc, yw):
            raise RuntimeError("paired C0 and WC fold-1 OOF rows differ")
    oof = _no_harm(R, [y for y, _ in oc], [p for _, p in oc], [p for _, p in ow])
    return {"role0": role0, "oof": oof, "passed": bool(role0["passed"] and oof["passed"])}


# ---------------------------------------------------------------------------------------------- U3 / U4
def u3(ctl, R: Rows, control: list[dict], configs: dict) -> tuple[dict, list[dict]]:
    runs = [fold_job(ctl, R, f"77_u2_WC_s{s}", with_weight(configs[s], True), s) for s in CONFIRM_SEEDS]
    s = U.paired_summary(control, runs, len(CONFIRM_SEEDS))
    s["role0_diff_ci95"] = arms.scenario_diff_ci(_hits(R, runs), _hits(R, control), R.scen[R.tune])
    harm = u3_no_harm(R, control, runs)
    adopted = U.no_harm_adopted(s["admitted"], harm["role0"], harm["oof"])
    forced = runner.SMOKE and not adopted
    decision = arms.freeze(STUDY / "u3_decision.json", {
        "summary": s, "no_harm": harm, "adopted": adopted, "smoke_forced": forced,
        "spec": "WC" if adopted or forced else "C0",
        "rule": "mean J gain >= 0.1 pp and > 2 x SD; mean role-0 unseen gain >= 0; mean OOF macro-F1 change >= -0.005; "
                "mean role-0 accuracy gain >= 0 (seeds 151-153, paired with C0 at identical config and rounds); and "
                "the per-class no-harm check (mean of paired seeds) on role 0 and on fold-1 OOF",
        "smoke": runner.SMOKE})
    _log(f"U3 WC: J gain {100 * s['gain']:+.3f} pp (SD {100 * s['gain_sd']:.3f}), rule {s['admitted']}, no-harm "
         f"{harm['passed']}, adopted {decision['adopted']}")
    return decision, runs


def _truncation_preds(work: Path, dm, keep, n_classes: int, step: int = L.TRUNCATION_STEP) -> dict[int, np.ndarray]:
    """Role-0 argmax predictions of one probe at the truncations in ``keep``, accumulated in exactly the chunks
    ``train_live77._curve`` uses (same arithmetic, so they reproduce the curve's accuracies at those points)."""
    import xgboost as xgb
    booster = xgb.Booster(model_file=str(work / "model" / "model.ubj"))
    booster.set_param({"device": "cpu", "nthread": 4})
    rounds, keep = booster.num_boosted_rounds(), {int(k) for k in keep}
    margin, out = np.zeros((dm.num_row(), n_classes)), {}          # float64 accumulator, as in _curve
    edges = list(range(0, rounds, step)) + [rounds]
    for a, b in zip(edges[:-1], edges[1:]):
        margin += booster.predict(dm, iteration_range=(a, b), output_margin=True)
        if b in keep:
            out[b] = margin.argmax(1)
    if set(out) != keep:
        raise RuntimeError(f"{work}: truncations {sorted(keep - set(out))} are not on this probe's curve")
    return out


def u4_no_harm(R: Rows, probes: list[dict], works: list[Path], dm, base_rounds: int, chosen: int,
               preds=None) -> dict:
    """Plan amendment 2026-11-02 for U4 (role 0 only): the chosen larger count vs the frozen-count truncation, paired
    within every probe that reaches the chosen count (the curve's probe mean), mean paired drop per class. Each
    prediction is checked against the accuracy recorded on the curve."""
    preds = preds or _truncation_preds
    y = R.y_val[R.tune]
    control, candidate, used = [], [], []
    for p, work in zip(probes, works):
        acc = {int(q["rounds"]): q["tune_acc"] for q in p["curve"]}
        if chosen not in acc or base_rounds not in acc:
            continue
        got = preds(work, dm, (base_rounds, chosen), R.n_classes)
        for k in (base_rounds, chosen):
            if abs(float(np.mean(got[k] == y)) - acc[k]) > 1e-9:
                raise RuntimeError(f"{p['job']}: truncation {k} predictions do not reproduce the curve accuracy")
        control.append(got[base_rounds]), candidate.append(got[chosen]), used.append(p["job"])
    if not used:
        raise RuntimeError(f"no probe reaches both {base_rounds} and {chosen} rounds")
    return {**_no_harm(R, y, control, candidate), "probes": used, "control_rounds": int(base_rounds),
            "candidate_rounds": int(chosen)}


def u4(ctl, R: Rows, spec_config: dict, spec: str, base_rounds: int) -> dict:
    import xgboost as xgb
    cols, _ = runner.feature_columns(spec_config)
    X = np.asarray(np.load(runner.DATA / "X_validation.npy", mmap_mode="r")[R.tune])[:, cols]
    dm = xgb.DMatrix(X, nthread=4)
    probes, curves, works = [], [], []
    for rounds in PROBES:
        r = ctl.run_job(f"77_u2_probe_{spec}_r{rounds}", "77", "xgboost", with_rounds(spec_config, rounds), FINAL_SEED,
                        rows="full")
        if r is None or r.get("status") != "complete":
            raise RuntimeError(f"U4 probe {rounds} failed")
        pts = L._curve(R, STUDY / r["work_dir"], cols, dm, R.y_val[R.tune], None)
        curves.append(pts)
        works.append(STUDY / r["work_dir"])
        probes.append({"job": r["job_id"], "probe_rounds": rounds, "curve": pts})
    lo = 1 if runner.SMOKE else U.ROUND_LO       # smoke fits have 4 rounds: keep the wiring, no decision
    curve = U.round_curve(curves, lo=lo, step=1 if runner.SMOKE else U.ROUND_STEP)
    choice = U.choose_rounds(curve, base_rounds)
    # The no-harm check is needed only when the rule would adopt a larger count (otherwise the frozen count stays).
    harm = u4_no_harm(R, probes, works, dm, base_rounds, choice["chosen"]) if choice["adopted"] else None
    model = STUDY / json.loads((runner.job_dir(probes[-1]["job"]) / "result.json").read_text(encoding="utf-8"))["work_dir"] \
        / "model" / "model.ubj"
    n_max = xgb.Booster(model_file=str(model)).num_boosted_rounds()
    # Only deterministic decision fields are frozen; the wall-clock latency (different on every measurement) goes to
    # an unfrozen file, so a restart after U4 re-measures it without failing the freeze.
    decision = arms.freeze(STUDY / "u4_decision.json",
                           U.u4_record(spec, base_rounds, curve, choice, probes, runner.SMOKE, harm))
    latency = {str(k): L._latency(model, min(k, n_max), X) for k in sorted({base_rounds, decision["rounds"]})}
    U.write_json_atomic(STUDY / "u4_latency.json", {"latency_ms_per_window_python_1thread": latency,
                                                    "measured": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                    "note": "report only; re-measured on every run, not frozen"})
    _log(f"U4: best {choice.get('best')} rounds, rule {choice['adopted']}, no-harm "
         f"{None if harm is None else harm['passed']}, chosen {decision['rounds']} (adopted {decision['adopted']})")
    return {**decision, "latency_ms_per_window_python_1thread": latency}


# ------------------------------------------------------------------------------------------------ final
def final_model(ctl, spec_config: dict, spec: str, rounds: int, refit: bool) -> dict:
    """The model the output rules apply to: a new all-row refit if U3/U4 changed anything, else the shipped final."""
    if refit:
        job = f"77_u2_final_{spec}_r{rounds}"
        fin = ctl.run_job(job, "77", "xgboost", with_rounds(spec_config, rounds), seed=FINAL_SEED, rows="full")
        if fin is None or fin.get("status") != "complete":
            raise RuntimeError("final refit failed")
        saved = fin["fit"].get("trainer", {}).get("saved_rounds")
        if saved != rounds and not runner.SMOKE:
            raise RuntimeError(f"final refit saved {saved} rounds, expected {rounds}")
        study, jdir = STUDY, runner.job_dir(job)
    else:
        fin = _read_result(MAIN, "77_live_final")
        study, jdir = MAIN, MAIN / "jobs" / "77_live_final"
    raw = np.load(study / fin["work_dir"] / "val_proba.npy")
    cal = np.load(jdir / "val_proba_cal.npy") if (jdir / "val_proba_cal.npy").exists() else raw
    calib = fin.get("calibration") or {}
    cfg = calib.get("config") or {"kind": "identity"}
    return {"job": fin["job_id"], "study": study.relative_to(runner.ROOT).as_posix(), "refit": refit, "raw": raw,
            "cal": cal, "calibration": cfg, "report": calib.get("report", fin["report"])}


# ---------------------------------------------------------------------------------------------- U5 / U1
def _role0(R: Rows, pred: np.ndarray) -> dict:
    t, u = R.tune, R.tune_unseen
    return {"acc": float(np.mean(pred[t] == R.y_val[t])), "unseen": float(np.mean(pred[u] == R.y_val[u])),
            "f1": arms.macro_f1(R.y_val[t], pred[t], R.n_classes)}


def tau_models(ctl, R: Rows, configs: dict, spec: str, rounds: int) -> list[dict]:
    """Plan U5 amendment: fold-1 models with the final spec and final round count on seeds 151-153 (three short
    XGBoost jobs, run after U4 has decided the rounds); tau is selected on their fold-1 OOF."""
    jobs = U.tau_selection_jobs(configs, CONFIRM_SEEDS, spec, rounds, dict(WC) if spec == "WC" else None)
    runs = []
    for s, (job_id, config) in jobs.items():
        r = ctl.run_job(job_id, "77", "xgboost", config, seed=s, rows="fold1")
        if r is None or r.get("status") != "complete":
            raise RuntimeError(f"{job_id} failed")
        saved = (r.get("fit") or {}).get("trainer", {}).get("saved_rounds")
        if saved != int(rounds) and not runner.SMOKE:      # smoke fits are capped at a few rounds
            raise RuntimeError(f"{job_id} saved {saved} rounds, expected the final count {rounds}")
        work = STUDY / r["work_dir"]
        runs.append({"job_id": job_id, "seed": s, "rounds": int(rounds), "saved_rounds": saved, "work": str(work),
                     **job_metrics(R, work)})
    return runs


def u5(R: Rows, selection_models: list[dict], fin: dict, fold_models: list[dict]) -> dict:
    """Tau by fold-1 OOF of ``selection_models`` (final spec + final rounds; ``tau_models``). ``fold_models`` (the
    18-round U3 fold models) enter only the report-only scale check."""
    fold_fit = runner.folds("77")[1]["fit"]
    prior_fold = U.class_prior(R.y_train, fold_fit, R.n_classes)
    oof = []
    for r in selection_models:
        work = Path(r["work"])
        idx = np.load(work / "oof_index.npy")
        oof.append((np.load(work / "oof_proba.npy"), R.y_train[idx], prior_fold))
    sel = U.select_tau(oof)
    prior = U.class_prior(R.y_train, R.fit_all, R.n_classes)
    base = _role0(R, U.adjusted_logits(fin["raw"], prior, 0.0).argmax(1))
    at = _role0(R, U.adjusted_logits(fin["raw"], prior, sel["tau"]).argmax(1))
    role0 = {"acc_gain": at["acc"] - base["acc"], "unseen_gain": at["unseen"] - base["unseen"],
             "f1_change": at["f1"] - base["f1"], "tau0": base, "tau": at}
    kind = fin["calibration"].get("kind")
    temperature = {"temperature": fin["calibration"].get("temperature"), "identity": 1.0}.get(kind)
    # Plan amendment 2026-11-02: per-class no-harm of tau vs tau = 0 on role 0 (final model's raw probabilities) AND
    # on the fold-1 OOF of the tau-selection jobs (mean of paired seeds). At tau = 0 nothing changes (not adopted).
    harm = None
    if sel["tau"] > 0:
        t = R.tune
        h0 = _no_harm(R, R.y_val[t], U.adjusted_logits(fin["raw"][t], prior, 0.0).argmax(1),
                      U.adjusted_logits(fin["raw"][t], prior, sel["tau"]).argmax(1))
        ho = _no_harm(R, [y for _, y, _ in oof], [U.adjusted_logits(p, pr, 0.0).argmax(1) for p, _, pr in oof],
                      [U.adjusted_logits(p, pr, sel["tau"]).argmax(1) for p, _, pr in oof])
        harm = {"role0": h0, "oof": ho, "passed": bool(h0["passed"] and ho["passed"])}
    rule_passed = U.u5_adopted(sel, role0)
    adopted = U.no_harm_adopted(rule_passed, *((harm["role0"], harm["oof"]) if harm else (None,)))
    note = None
    if adopted and temperature is None:
        adopted, note = False, f"final calibration '{kind}' changes decisions; the adjustment cannot be combined with it"
    # Report-only (nothing is re-chosen from it): tau is selected on fold models with the final spec and rounds (plan
    # amendment), so their logit scale should match the final model's; the 18-round U3 fold models are shown as the
    # reference the amendment replaced. Role 0 at every grid tau on the final's raw probabilities shows where the
    # selected tau sits.
    role0_of = lambda runs: [np.load(Path(r["work"]) / "val_proba.npy")[R.tune] for r in runs]
    mean_scale = lambda items: {k: float(np.mean([U.proba_scale(p)[k] for p in items]))
                                for k in ("mean_max_proba", "mean_log_margin")}
    scale_check = {
        "role0_final_raw_by_tau": {str(t): _role0(R, U.adjusted_logits(fin["raw"], prior, t).argmax(1))
                                   for t in U.TAU_GRID},
        "scale_role0": {"selection_models_mean": mean_scale(role0_of(selection_models)),
                        "fold_models_18_round_mean": mean_scale(role0_of(fold_models)),
                        "final": mean_scale([fin["raw"][R.tune]])},
        "scale_oof_selection_models_mean": mean_scale([p for p, _, _ in oof]),
        "note": "report only; the frozen selection rule (fold-1 OOF of the final-spec, final-round fold models) is "
                "unchanged and nothing is re-chosen from this"}
    quality = None
    if temperature is not None:     # report-only: confidence quality of the display on role 0 (no decision uses it)
        from framework.training.csv_v3 import calibrate
        shown = U.display_proba(U.adjusted_logits(fin["raw"], prior, sel["tau"]), temperature)
        quality = {"display_at_tau": calibrate.quality(shown[R.tune], R.y_val[R.tune]),
                   "final_calibrated": calibrate.quality(np.asarray(fin["cal"], np.float64)[R.tune], R.y_val[R.tune])}
    decision = arms.freeze(STUDY / "u5_decision.json", {
        "display_quality_role0_report_only": quality, "tau_scale_check_report_only": scale_check,
        "display_note": "plan rule: display = softmax(z / T) on unsmoothed log p; the final calibrator's smoothing "
                        "alpha is not applied to z, so confidence may differ from the calibrated output (see quality)",
        "selection": sel, "role0_final_raw": role0, "rule_passed": bool(rule_passed), "no_harm": harm,
        "adopted": bool(adopted), "tau": sel["tau"] if adopted else 0.0,
        "prior_final": prior.tolist(), "prior_fold1_for_selection": prior_fold.tolist(), "temperature": temperature,
        "log_floor": U.LOG_FLOOR, "final_job": fin["job"], "note": note,
        "selection_models": [r["job_id"] for r in selection_models],
        "selection_rounds": sorted({int(r["rounds"]) for r in selection_models}),
        "selection_saved_rounds": [r.get("saved_rounds") for r in selection_models],
        "fold_models_18_round_report_only": [r["job_id"] for r in fold_models],
        "rule": "tau in {0,.125,.25,.375,.5} by mean fold-1 OOF accuracy of fold-1 models trained with the final spec "
                "and the final round count (seeds 151-153; plan amendment 2026-11-01; ties -> smaller); adopt if OOF "
                "gain >= 0.1 pp and role-0 acc gain >= 0, unseen gain >= 0, macro-F1 change >= -0.005 (final raw); "
                "z = log p_raw - tau log prior (prior = the model's fit-row class frequencies); decision argmax z; "
                "display softmax(z / T) with the final model's temperature; and the per-class no-harm check (tau vs "
                "0) on role 0 (final raw) and on the selection jobs' fold-1 OOF (mean of paired seeds)",
        "smoke": runner.SMOKE})
    _log(f"U5: tau {sel['tau']} (OOF gain {100 * sel['gain']:+.3f} pp), rule {rule_passed}, no-harm "
         f"{None if harm is None else harm['passed']}, adopted {decision['adopted']}")
    return decision


def display(fin: dict, u5d: dict) -> np.ndarray:
    """Displayed probabilities of the final pipeline before the ping merge."""
    if u5d["adopted"]:
        z = U.adjusted_logits(fin["raw"], np.array(u5d["prior_final"]), u5d["tau"])
        return U.display_proba(z, u5d["temperature"])
    return np.asarray(fin["cal"], np.float64)


def u1(R: Rows, fold_models: list[dict], fin: dict, u5d: dict) -> dict:
    p = display(fin, u5d)
    t = R.tune
    before = float(np.mean(p[t].argmax(1) == R.y_val[t]))
    after = float(np.mean(class_merge.decision(p[t], R.classes) == class_merge.merge_labels(R.y_val[t], R.classes)))
    # Per-class no-harm (amendment 2026-11-02), original labels, unmerged argmax vs the merged decision; ping-sweep
    # (removed by the owner's choice) and recon_scan (receives its rows) are exempt for U1 only.
    exempt = class_merge.indices(R.classes)
    harm0 = _no_harm(R, R.y_val[t], p[t].argmax(1), class_merge.decision(p[t], R.classes), exempt)
    oof, ys, ctrl, cand = [], [], [], []
    prior_fold = np.array(u5d["prior_fold1_for_selection"])
    for r in fold_models:
        work = Path(r["work"])
        idx, po = np.load(work / "oof_index.npy"), np.load(work / "oof_proba.npy")
        # Same display transform as role 0 (the merge sums two probabilities, which depends on T and smoothing):
        # softmax(z / T_final) with the fold prior when U5 is adopted, else the final model's calibrator.
        po = U.final_display(po, prior_fold, u5d, fin["calibration"])
        y = R.y_train[idx]
        dec = class_merge.decision(po, R.classes)
        ys.append(y), ctrl.append(po.argmax(1)), cand.append(dec)
        oof.append({"job": r["job_id"], "before": float(np.mean(po.argmax(1) == y)),
                    "after": float(np.mean(dec == class_merge.merge_labels(y, R.classes)))})
    harm_oof = _no_harm(R, ys, ctrl, cand, exempt)
    harm = {"role0": harm0, "oof": harm_oof, "passed": bool(harm0["passed"] and harm_oof["passed"])}
    decision = arms.freeze(STUDY / "u1_decision.json", {
        "tuning_before": before, "tuning_after": after, "no_harm": harm,
        "adopted": U.no_harm_adopted(after >= before, harm0, harm_oof), "oof": oof,
        "oof_mean_change": float(np.mean([o["after"] - o["before"] for o in oof])), "spec": class_merge.SPEC,
        "oof_transform": ("softmax((log p - tau log prior_fold1) / T_final)" if u5d["adopted"] else
                          f"final calibrator ({fin['calibration'].get('kind', 'identity')}) applied to raw OOF"),
        "rule": "adopted for live if role-0 accuracy does not fall (merged labels after vs original labels before) and "
                "the per-class no-harm check passes on role 0 and fold-1 OOF (ping-sweep and recon_scan exempt); "
                "OOF of the fold-1 models (seeds 151-153) reported, with the same display transform as role 0",
        "smoke": runner.SMOKE})
    _log(f"U1: role-0 {100 * before:.2f}% -> {100 * after:.2f}%, adopted {decision['adopted']}")
    return decision


# ------------------------------------------------------------------------------------------------ report
def _score(R: Rows, proba: np.ndarray, merged: bool) -> dict:
    """``runner.score`` with the labels mapped when the ping merge is on (assessment numbers: report-only)."""
    y = class_merge.merge_labels(R.y_val, R.classes) if merged else R.y_val
    focus = [R.classes.index(prep.MERGED), R.classes.index(class_merge.SOURCE)]
    return evaluate.report(y, proba, R.roles, runner._arr("validation_scenario_seen"), R.scen, R.n_classes, 0, focus)


def pipeline(fin: dict, u5d: dict, u1d: dict, R: Rows) -> np.ndarray:
    p = display(fin, u5d)
    return class_merge.merge_proba(p, R.classes) if u1d["adopted"] else p


def run() -> None:
    busy = U.other_controllers()
    if busy:
        raise RuntimeError(f"another training controller or job is running (pids {busy}); one GPU job at a time")
    (STUDY / "controller_pid.json").write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    U.quarantine_failed(STUDY, _log)
    ctl = U.guard_run_job(runner.Controller(log=_log), _log)     # re-check other controllers before every fit
    ctl.state(phase="running", smoke=runner.SMOKE)
    try:
        R = Rows()
        base_rounds = int(json.loads((MAIN / "final_rounds.json").read_text(encoding="utf-8"))["chosen_rounds"])
        meta = arms.freeze(STUDY / "inputs.json", {
            "plan_sha256": _sha256(PLAN), "data_manifest_sha256": _sha256(runner.DATA / "manifest.json"),
            "main_final_rounds": base_rounds, "main_final_rounds_sha256": _sha256(MAIN / "final_rounds.json"),
            "smoke": runner.SMOKE})
        if meta["smoke"] != runner.SMOKE:
            raise RuntimeError("inputs.json was frozen in the other (smoke/real) mode")
        ctl.state(stage="U3")
        control, configs = controls(ctl, R)
        u3d, wc_runs = u3(ctl, R, control, configs)
        weighted = u3d["spec"] == "WC"
        spec = u3d["spec"]
        spec_config = with_weight(configs[CONFIRM_SEEDS[0]], weighted)
        fold_models = wc_runs if weighted else control
        ctl.state(stage="U4")
        u4d = u4(ctl, R, spec_config, spec, base_rounds)
        refit = bool(u3d["adopted"] or u4d["adopted"] or runner.SMOKE)
        ctl.state(stage="final" if refit else "U5")
        fin = final_model(ctl, spec_config, spec, u4d["rounds"], refit)
        ctl.state(stage="U5")
        selection_models = tau_models(ctl, R, configs, spec, u4d["rounds"])     # after U4 decided the rounds
        u5d = u5(R, selection_models, fin, fold_models)
        ctl.state(stage="U1")
        u1d = u1(R, fold_models, fin, u5d)
        rules = arms.freeze(STUDY / "live_output_rules.json", {
            "final_job": fin["job"], "final_study": fin["study"], "spec": spec, "rounds": u4d["rounds"],
            "calibration": {k: v for k, v in fin["calibration"].items() if k != "bias"},
            "logit_adjustment": None if not u5d["adopted"] else {
                "tau": u5d["tau"], "prior": u5d["prior_final"], "log_floor": U.LOG_FLOOR,
                "temperature": u5d["temperature"],
                "decision": "argmax(log(max(p_raw, 1e-12)) - tau * log(max(prior, 1e-12)))",
                "display": "softmax(z / temperature)"},
            "class_merge": class_merge.SPEC if u1d["adopted"] else None, "smoke": runner.SMOKE})
        out = {"u3": u3d, "u4": {k: v for k, v in u4d.items() if k != "probes"}, "u5": u5d, "u1": u1d, "rules": rules,
               "control": control, "wc_runs": wc_runs, "tau_selection_runs": selection_models,
               "final": {k: fin[k] for k in ("job", "study", "refit", "report")}}
        write_report(R, out, fin, u5d, u1d)
        ctl.state(phase="done", stage="report")
    except KeyboardInterrupt:
        ctl.state(phase="stopped")
        _log("stopped (STOP file); restart resumes from cached jobs")
    except Exception:
        ctl.state(phase="error")
        raise


def no_harm_md(label: str, check: dict | None, absent: str = "not run") -> list[str]:
    """Report lines of one per-class no-harm check: every class checked, the largest drops (with class names) and
    the breaches."""
    if check is None:
        return [f"- No-harm, {label}: {absent}."]
    big = lambda d: "-" if d is None else f"{d['class']} {100 * d['drop']:+.2f} pp"
    breaches = "; ".join(f"{b['class']} {b['metric']} -{100 * b['drop']:.2f} pp" for b in check["breaches"]) or "none"
    extra = f" (exempt: {', '.join(map(str, check['exempt']))})" if check["exempt"] else ""
    return [f"- No-harm, {label}: {'passed' if check['passed'] else 'FAILED'}; {len(check['checked'])} classes checked "
            f"(>= {check['min_rows']} rows, {check['n_pairs']} paired run(s)){extra}; largest recall drop "
            f"{big(check['largest_recall_drop'])}; largest precision drop {big(check['largest_precision_drop'])}; "
            f"breaches (> {100 * check['max_drop']:.0f} pp): {breaches}.",
            f"  Classes checked: {', '.join(map(str, check['checked'])) or 'none'}."]


def write_report(R: Rows, o: dict, fin: dict, u5d: dict, u1d: dict) -> None:
    pct = lambda x: "-" if x is None else f"{100 * x:.2f}%"
    shipped_job = MAIN / "jobs" / "77_live_final"
    shipped = np.load(shipped_job / "val_proba_cal.npy") if (shipped_job / "val_proba_cal.npy").exists() else \
        np.load(MAIN / _read_result(MAIN, "77_live_final")["work_dir"] / "val_proba.npy")
    new = pipeline(fin, u5d, u1d, R)
    merged = bool(u1d["adopted"])
    rep_new, rep_old = _score(R, new, merged), _score(R, shipped, False)
    rep_old_m = _score(R, class_merge.merge_proba(shipped, R.classes), True)
    a_rows = np.flatnonzero(R.roles == 2)
    y_cmp = class_merge.merge_labels(R.y_val, R.classes) if merged else R.y_val
    old_cmp = class_merge.merge_proba(shipped, R.classes) if merged else shipped
    diff = (new[a_rows].argmax(1) == y_cmp[a_rows]).astype(float) - (old_cmp[a_rows].argmax(1) == y_cmp[a_rows])
    ci = arms.scenario_diff_ci(diff, np.zeros_like(diff), R.scen[a_rows])
    o["assessment_report_only"] = {"new": rep_new["assessment"], "shipped": rep_old["assessment"],
                                   "shipped_with_merge": rep_old_m["assessment"],
                                   "paired_diff": {"mean": float(diff.mean()), "ci95": ci, "labels_merged": merged}}
    (STUDY / "upgrades2_live.json").write_text(json.dumps(o, indent=1, default=str), encoding="utf-8")
    s, c4 = o["u3"]["summary"], o["u4"]["choice"]
    lines = ["# Upgrade round 2: live 77-class XGBoost (U3, U4, U5, U1)", "",
             "Plan: ml_pipeline/notes_20261101/upgrades2_plan.md (pre-registered 2026-11-01). Selection: role 0 + "
             "training fold-1 OOF; calibration: role 1; assessment: report-only; held-out test untouched.", ""]
    if runner.SMOKE:
        lines += ["**SMOKE RUN: wiring check only (tiny rows, 4 rounds, one seed). No number here is a result.**", ""]
    lines += ["## U3: scenario weight 1.0 (WC) vs C0, seeds " + ", ".join(map(str, CONFIRM_SEEDS)), "",
              "| J gain (pp) | SD (pp) | role-0 acc gain (pp) | unseen gain (pp) | OOF macro-F1 change | role-0 diff CI (pp) | adopted |",
              "|---|---|---|---|---|---|---|",
              f"| {100 * s['gain']:+.3f} | {100 * s['gain_sd']:.3f} | {100 * s['tune_acc_gain']:+.3f} | "
              f"{100 * s['unseen_gain']:+.3f} | {s['oof_f1_change']:+.4f} | {100 * s['role0_diff_ci95'][0]:+.2f} to "
              f"{100 * s['role0_diff_ci95'][1]:+.2f} | {'yes' if o['u3']['adopted'] else 'no'}"
              f"{' (smoke-forced)' if o['u3']['smoke_forced'] else ''} |", "",
              f"Paired rule passed: {'yes' if s['admitted'] else 'no'}; per-class no-harm (mean of paired seeds):", ""]
    lines += no_harm_md("role 0", o["u3"]["no_harm"]["role0"]) + no_harm_md("fold-1 OOF", o["u3"]["no_harm"]["oof"])
    lines += ["", f"C0 controls: {', '.join(r['job_id'] for r in o['control'])} "
              f"({'re-run in the smoke study' if runner.SMOKE else 'read-only from live77_upgrades_20261031_v1'}).", "",
              f"## U4: boosting rounds (spec {o['u4']['spec']}, probes {', '.join(map(str, PROBES))}, seed 42, all rows)", "",
              "| rounds | role-0 acc (probe mean) | smoothed | role-0 macro-F1 |", "|---|---|---|---|"]
    sm = c4.get("smoothed") or []
    for i, p in enumerate(o["u4"]["curve"]):
        lines.append(f"| {p['rounds']} | {pct(p['tune_acc'])} | {pct(sm[i]) if i < len(sm) else '-'} | {p['tune_f1']:.4f} |")
    lines += ["", f"Best smoothed: {c4.get('best')} rounds; chosen {o['u4']['rounds']} (frozen count "
                  f"{o['u4']['base_rounds']}); {c4['reason']}. Latency (ms/window, Python, 1 thread): "
                  f"{json.dumps(o['u4']['latency_ms_per_window_python_1thread'])}.", ""]
    lines += no_harm_md(f"role 0 ({c4['chosen']} vs {o['u4']['base_rounds']} rounds, probe mean)",
                        o["u4"].get("no_harm_role0"), "not needed (the rule kept the frozen count)")
    lines += ["", f"U4 adopted: {'yes' if o['u4']['adopted'] else 'no'}.", "",
              f"Final model: {fin['job']} ({'new all-row refit, calibrated on role 1' if fin['refit'] else 'shipped, unchanged'}).", "",
              "## U5: logit (prior) adjustment", "",
              f"Tau selected on fold-1 OOF of {', '.join(u5d['selection_models'])} (final spec, "
              f"{', '.join(map(str, u5d['selection_rounds']))} rounds; plan amendment 2026-11-01).", "",
              "| tau | mean fold-1 OOF acc |", "|---|---|"]
    for t, a in u5d["selection"]["oof_acc"].items():
        lines.append(f"| {t} | {pct(a)} |")
    r0 = u5d["role0_final_raw"]
    lines += ["", f"Selected tau {u5d['selection']['tau']} (OOF gain {100 * u5d['selection']['gain']:+.3f} pp). Role 0, "
                  f"final raw: acc {100 * r0['acc_gain']:+.3f} pp, unseen {100 * r0['unseen_gain']:+.3f} pp, macro-F1 "
                  f"{r0['f1_change']:+.4f}. Rule passed: {'yes' if u5d['rule_passed'] else 'no'}. Adopted: "
                  f"{'yes' if u5d['adopted'] else 'no'}{' (' + u5d['note'] + ')' if u5d.get('note') else ''}.", ""]
    h5 = u5d.get("no_harm") or {}
    lines += no_harm_md("role 0 (final raw, tau vs 0)", h5.get("role0"), "not needed (selected tau is 0)")
    lines += no_harm_md("fold-1 OOF of the selection jobs (mean of paired seeds)", h5.get("oof"),
                        "not needed (selected tau is 0)") + [""]
    sc = u5d.get("tau_scale_check_report_only")
    if sc:
        lines += ["Report only (nothing re-chosen): role 0 on the final model's raw probabilities at every grid tau, and "
                  "the probability scale of the selection fold models (final spec and rounds, where tau was selected), "
                  "the 18-round U3 fold models and the final model.", "",
                  "| tau | role-0 acc | role-0 unseen | role-0 macro-F1 |", "|---|---|---|---|"]
        lines += [f"| {t} | {pct(v['acc'])} | {pct(v['unseen'])} | {v['f1']:.4f} |"
                  for t, v in sc["role0_final_raw_by_tau"].items()]
        sel_, f_, n_ = (sc["scale_role0"][k] for k in ("selection_models_mean", "fold_models_18_round_mean", "final"))
        lines += ["", f"Role-0 scale (mean max-prob, log margin): selection models {sel_['mean_max_proba']:.4f}, "
                      f"{sel_['mean_log_margin']:.3f}; 18-round fold models {f_['mean_max_proba']:.4f}, "
                      f"{f_['mean_log_margin']:.3f}; final {n_['mean_max_proba']:.4f}, {n_['mean_log_margin']:.3f}.", ""]
    lines += ["## U1: ping-sweep merged into recon_scan", "",
              f"Role 0: {pct(u1d['tuning_before'])} -> {pct(u1d['tuning_after'])}; fold-1 OOF mean change "
              f"{100 * u1d['oof_mean_change']:+.3f} pp (OOF transform: {u1d.get('oof_transform')}). "
              f"Adopted: {'yes' if u1d['adopted'] else 'no'}.", ""]
    lines += no_harm_md("role 0", u1d["no_harm"]["role0"]) + no_harm_md("fold-1 OOF", u1d["no_harm"]["oof"])
    lines += ["", "## Final pipeline vs shipped live model (assessment: report-only)", "",
              "| model | tuning acc | assessment acc (CI) | assessment macro-F1 | benign FPR (assess) |", "|---|---|---|---|---|"]
    for label, rep in (("shipped live (as shipped)", rep_old), ("shipped live + ping merge", rep_old_m),
                       ("upgrades2 pipeline", rep_new)):
        a = rep["assessment"]
        lines.append(f"| {label} | {pct(rep['tuning']['accuracy'])} | {pct(a['accuracy'])} "
                     f"({100 * a['accuracy_ci95_scenario_bootstrap'][0]:.1f}-{100 * a['accuracy_ci95_scenario_bootstrap'][1]:.1f}) "
                     f"| {a['macro_f1_supported']:.4f} | {pct(a['benign_false_alarm_rate'])} |")
    lines += ["", f"Paired assessment difference (pipeline - shipped{', both with merged labels' if merged else ''}): "
                  f"{100 * diff.mean():+.2f} pp, scenario-bootstrap CI {100 * ci[0]:+.2f} to {100 * ci[1]:+.2f} pp "
                  "(report-only; nothing is chosen on it).", "",
              "Output rules for the app: live_output_rules.json. Adoption into the app (Rust decision(), catalog) is a "
              "separate owner decision for the app chat. Every live score stays 'dataset-trained, live accuracy unverified'.", ""]
    (STUDY / "REPORT_upgrades2_live.md").write_text("\n".join(lines), encoding="utf-8")
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
        proc = subprocess.Popen([sys.executable, "-m", "experiment.train_upgrades2_live", "run"], cwd=runner.ROOT,
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
