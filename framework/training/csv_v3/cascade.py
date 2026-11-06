"""Compose the 77-class primary and the 8-class recon specialist into 84-class predictions and report them.

Stage 1 is the 77-class predictor with the best calibrated tuning value; stage 2 is the specialist with the best
calibrated tuning value. Both are chosen on the validation tuning role only (seed repeats and OOF fold models are
not candidates). Decisions follow the deployed cascade exactly: when stage 1 predicts ``recon_scan`` the specialist
picks one of the eight recon subtypes, otherwise the stage-1 class is mapped back to its 84-class label. The
specialist is run on every validation row stage 1 sends to it, including rows whose true class is not a recon
probe, so its errors on those rows count. Owner decision 2026-10-31: no direct 84-class retrain; one 84-class
XGBoost baseline is kept as a control.
"""
from __future__ import annotations

import json

import numpy as np

from framework.training.csv_v3 import calibrate, prep, report, runner


def _candidates(task: str) -> list[dict]:
    out = []
    for path in sorted((runner.STUDY / "jobs").glob(f"{task}_*/result.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        # no calibration block = a result the current controller skipped (e.g. a hybrid deduped as a duplicate
        # member set); it would otherwise compete on its raw value
        if r.get("status") == "complete" and "report" in r and "calibration" in r \
                and "_repeat_" not in r["job_id"] and "_oof_" not in r["job_id"]:
            out.append(r)
    return out


def best(task: str) -> dict:
    """Highest calibrated tuning value; ties keep the earlier job id."""
    cands = _candidates(task)
    if not cands:
        raise RuntimeError(f"no complete {task}-class predictors")
    return max(cands, key=runner.selection_value)


def _stage1_proba(r: dict) -> np.ndarray:
    cal = runner.job_dir(r["job_id"]) / "val_proba_cal.npy"
    return np.load(cal if cal.exists() else runner.STUDY / r["work_dir"] / "val_proba.npy")


def _stage2_proba(r: dict, rows: np.ndarray) -> np.ndarray:
    x = np.asarray(runner._arr("X_validation", mmap=True)[rows], dtype=np.float32)
    p = np.asarray(runner.adapter(r["family"]).predict_cpu(r["family"], runner.STUDY / r["work_dir"] / "model", x),
                   dtype=np.float32)
    cal = r.get("calibration") or {}
    if cal.get("status") == "complete" and cal["method"] != "identity":
        p = calibrate.apply(cal["config"], p)
    return p


def compose(stage1: dict, stage2: dict) -> tuple[np.ndarray, dict]:
    m = runner.manifest()
    to77, to8 = np.array(m["map84_to77"]), np.array(m["map84_to8"])
    classes77 = m["classes77"]
    recon_scan = classes77.index(prep.MERGED)
    # 77 -> 84 for every non-merged class (one-to-one); 8 -> 84 for the specialist's classes
    from77 = {int(to77[j]): j for j in range(len(to77)) if to8[j] < 0}
    from8 = {int(to8[j]): j for j in range(len(to8)) if to8[j] >= 0}
    p77 = _stage1_proba(stage1)
    pred77 = p77.argmax(1)
    routed = np.flatnonzero(pred77 == recon_scan)
    pred84 = np.array([from77.get(int(c), -1) for c in pred77])
    p8 = _stage2_proba(stage2, routed)
    pred84[routed] = [from8[int(k)] for k in p8.argmax(1)]
    if (pred84 < 0).any():
        raise RuntimeError("unmapped stage-1 class")
    info = {"stage1_job": stage1["job_id"], "stage1_selection_value": runner.selection_value(stage1),
            "stage2_job": stage2["job_id"], "stage2_selection_value": runner.selection_value(stage2),
            "rows_routed_to_specialist": int(len(routed))}
    return np.eye(len(m["classes84"]), dtype=np.float32)[pred84], info


def write() -> dict:
    stage1, stage2 = best("77"), best("8")
    proba, info = compose(stage1, stage2)
    rep = runner.score("84", proba)
    y84 = runner.task_arrays("84")["y_val"]
    recon8 = np.isin(y84, [runner.task_arrays("84")["classes"].index(c) for c in prep.RECON8])
    pred = proba.argmax(1)
    roles = runner._arr("validation_roles")
    assess = roles == 2
    info["recon8_rows_assessment"] = int((recon8 & assess).sum())
    info["recon8_subtype_accuracy_assessment"] = float(np.mean(pred[recon8 & assess] == y84[recon8 & assess]))
    rows = report.frozen_rows("84")
    rows.append(report._row("cascade v3 (77 + specialist)", "cascade",
                            {"status": "complete", "job_id": f"{info['stage1_job']} -> {info['stage2_job']}",
                             "report": rep}))
    control = runner.job_dir("84_baseline_xgboost") / "result.json"
    if control.exists():
        r = report._row("xgboost 84 direct (v3 features)", "control", json.loads(control.read_text(encoding="utf-8")))
        if r:
            rows.append(r)
    (runner.STUDY / "REPORT_cascade84.json").write_text(json.dumps({"cascade": info, "rows": rows}, indent=2),
                                                       encoding="utf-8")
    pct = lambda v: "-" if v is None else f"{100 * v:.2f}"
    lines = ["# Cascade v3 — combined 84-class result", "",
             f"Stage 1: `{info['stage1_job']}`; stage 2 (recon specialist): `{info['stage2_job']}`; both chosen on the "
             "tuning role by calibrated value. The specialist decides every row stage 1 predicts as `recon_scan` "
             f"({info['rows_routed_to_specialist']} validation rows). Recon-subtype accuracy on the "
             f"{info['recon8_rows_assessment']} assessment recon-probe rows: "
             f"{pct(info['recon8_subtype_accuracy_assessment'])}% (chance 12.5%).", "",
             "| model | stage | final assess acc | 95% CI | macro-F1 (supported) | seen | unseen | benign FAR |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: -r["final_assessment_accuracy"]):
        ci = r["final_assessment_ci95"]
        lines.append(f"| {r['model']} | {r['stage']} | {pct(r['final_assessment_accuracy'])} | "
                     f"{'-' if not ci else f'{100*ci[0]:.1f}–{100*ci[1]:.1f}'} | "
                     f"{pct(r['assessment_macro_f1_supported'])} | {pct(r['assessment_seen_accuracy'])} | "
                     f"{pct(r['assessment_unseen_accuracy'])} | {pct(r['benign_false_alarm_rate'])} |")
    (runner.STUDY / "REPORT_cascade84.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return info
