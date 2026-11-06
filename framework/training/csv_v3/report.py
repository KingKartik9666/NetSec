"""Write ``REPORT_<task>.md`` / ``.json`` for a finished cascade v3 task, with frozen 655-feature comparators."""
from __future__ import annotations

import json

import numpy as np

from framework.training.csv_v3 import runner

FROZEN = runner.ROOT / "artifacts/training/all_models_wrapup_20261023_v1/assessment/selected"


def final_report(r: dict) -> dict:
    """Report of the adopted predictor: calibrated only when the calibrator changed decisions."""
    cal = r.get("calibration") or {}
    return cal["report"] if cal.get("status") == "complete" and cal.get("decisions_changed") else r["report"]


def _row(name: str, stage: str, r: dict | None) -> dict | None:
    if not r or r.get("status") != "complete" or "report" not in r:
        return None
    a = r["report"]["assessment"]          # raw, kept as its own column
    rep = final_report(r)                  # every other column describes the adopted predictor
    cal = r.get("calibration") or {}
    calibrated = cal.get("status") == "complete"
    q = (cal.get("quality") or {}).get("assessment", {})
    final = rep["assessment"]
    return {
        "model": name, "stage": stage, "job": r["job_id"],
        "calibration": cal.get("method") if calibrated else cal.get("status", "none"),
        "calibration_changed_decisions": cal.get("decisions_changed") if calibrated else None,
        "selection_value": cal.get("selection_value") if calibrated else None,
        "final_assessment_accuracy": final["accuracy"],
        "final_assessment_ci95": final.get("accuracy_ci95_scenario_bootstrap"),
        "assessment_ece": {k: v["ece"] for k, v in q.items()} if q else None,
        "assessment_nll": {k: v["nll"] for k, v in q.items()} if q else None,
        "assessment_disjoint_accuracy": cal.get("assessment_disjoint_accuracy") if calibrated else None,
        "tuning_accuracy": rep["tuning"]["accuracy"],
        "assessment_accuracy": a["accuracy"], "assessment_ci95": a.get("accuracy_ci95_scenario_bootstrap"),
        "assessment_macro_f1_supported": final["macro_f1_supported"],
        "assessment_seen_accuracy": rep["assessment_seen_scenarios"]["accuracy"],
        "assessment_unseen_accuracy": rep["assessment_unseen_scenarios"]["accuracy"],
        "full_validation_accuracy": rep["full_validation"]["accuracy"],
        "benign_false_alarm_rate": final.get("benign_false_alarm_rate"),
        "focus_class_recall": rep.get("focus_class_recall"),
    }


def frozen_rows(task: str) -> list[dict]:
    """Frozen 655-feature standalone models scored on the same rows (84 predictions mapped to 77)."""
    m = runner.manifest()
    if task == "8":
        # Frozen 84-class predictions restricted to recon8 rows; a non-recon prediction counts as an error.
        to8 = np.array(m["map84_to8"])
        y8 = runner.task_arrays("8")["y_val"]
        rows = []
        for path in sorted(FROZEN.glob("*/validation_predictions.npy")):
            pred = np.load(path)
            pred = to8[pred.argmax(1) if pred.ndim > 1 else pred]
            pred = np.where(pred >= 0, pred, (np.maximum(y8, 0) + 1) % 8)
            proba = np.eye(8, dtype=np.float32)[pred]
            rows.append(_row(f"frozen655:{path.parent.name}", "frozen", {
                "status": "complete", "job_id": str(path.relative_to(runner.ROOT)), "report": runner.score("8", proba)}))
        return [r for r in rows if r]
    to77 = np.array(m["map84_to77"])
    n = len(m[f"classes{task}"])
    rows = []
    for path in sorted(FROZEN.glob("*/validation_predictions.npy")):
        pred = np.load(path)
        pred = pred.argmax(1) if pred.ndim > 1 else pred
        if task == "77":
            pred = to77[pred]
        proba = np.eye(n, dtype=np.float32)[pred]
        rows.append(_row(f"frozen655:{path.parent.name}", "frozen", {
            "status": "complete", "job_id": str(path.relative_to(runner.ROOT)), "report": runner.score(task, proba)}))
    return [r for r in rows if r]


def write(task: str, base: dict, hybrids0: dict, tuned: dict, hybrids1: dict, repeats: dict) -> None:
    rows = frozen_rows(task)
    rows += [r for name, res in base.items() if (r := _row(name, "baseline", res))]
    rows += [r for name, res in hybrids0.items() if (r := _row(name, "baseline_hybrid", res))]
    rows += [r for name, res in tuned.items()
             if (r := _row(name, "tuned", res)) and res["job_id"] != (base.get(name) or {}).get("job_id")]
    rows += [r for name, res in hybrids1.items() if (r := _row(name, "tuned_hybrid", res))]
    rep_summary = {}
    for name, results in repeats.items():
        acc = [final_report(r)["assessment"]["accuracy"] for r in results
               if r and r.get("status") == "complete" and "report" in r]
        if acc:
            rep_summary[name] = {"seeds": len(acc), "assessment_accuracy_mean": float(np.mean(acc)),
                                 "assessment_accuracy_sd": float(np.std(acc, ddof=1)) if len(acc) > 1 else None}
    (runner.STUDY / f"REPORT_{task}.json").write_text(json.dumps({"rows": rows, "repeats": rep_summary}, indent=2),
                                                      encoding="utf-8")
    pct = lambda v: "-" if v is None else f"{100 * v:.2f}"
    pair = lambda d, f: "-" if not d else f"{f(d.get('raw'))} → {f(d.get('calibrated'))}"
    num = lambda v: "-" if v is None else f"{v:.3f}"
    lines = [f"# Cascade v3 — task {task}", "",
             "Scored on the original 30,000 validation rows with the frozen positional roles; selection used the "
             "tuning role only. Assessment CI resamples whole scenarios. Frozen rows are the 655-feature models' "
             "saved predictions on the same rows" + (" mapped to 77 classes." if task == "77" else "."), "",
             "Calibration is fitted on the calibration role. `temperature` never changes a prediction; `vector` "
             "(per-class bias) is adopted only when it raises tuning accuracy by >= 0.1 pp on all tuning rows AND "
             "by >= 0.1 pp on tuning scenarios absent from the calibration role. 'Final' is the calibrated accuracy when the "
             "calibrator changed decisions, otherwise the raw accuracy. Every column except 'raw assess' describes "
             "that adopted predictor, as do the seed-repeat means. `identity` means the raw probabilities were "
             "kept because temperature scaling did not lower tuning-role ECE. 'Disjoint' is assessment accuracy "
             "on scenarios never seen by the calibrator (leakage control, reported only).", "",
             "| model | stage | tuning acc | raw assess | final assess | 95% CI | calibrator | ECE | NLL | disjoint acc | "
             "macro-F1 (supported) | seen | unseen | benign FAR |",
             "|---|---|---:|---:|---:|---|---|---|---|---|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: -r["final_assessment_accuracy"]):
        ci = r["final_assessment_ci95"]
        cal = r["calibration"] + (" (decisions)" if r["calibration_changed_decisions"] else "")
        lines.append(f"| {r['model']} | {r['stage']} | {pct(r['tuning_accuracy'])} | {pct(r['assessment_accuracy'])} | "
                     f"{pct(r['final_assessment_accuracy'])} | "
                     f"{'-' if not ci else f'{100*ci[0]:.1f}–{100*ci[1]:.1f}'} | {cal} | "
                     f"{pair(r['assessment_ece'], num)} | {pair(r['assessment_nll'], num)} | "
                     f"{pair(r['assessment_disjoint_accuracy'], pct)} | {pct(r['assessment_macro_f1_supported'])} | "
                     f"{pct(r['assessment_seen_accuracy'])} | {pct(r['assessment_unseen_accuracy'])} | "
                     f"{pct(r['benign_false_alarm_rate'])} |")
    if rep_summary:
        lines += ["", "## Seed repeats (43, 44)", ""]
        lines += [f"- {k}: mean assessment {100*v['assessment_accuracy_mean']:.2f}%"
                  + (f" ± {100*v['assessment_accuracy_sd']:.2f}" if v["assessment_accuracy_sd"] is not None else "")
                  for k, v in rep_summary.items()]
    (runner.STUDY / f"REPORT_{task}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
