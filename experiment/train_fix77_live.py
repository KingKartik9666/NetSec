"""77-class fix test, stage 1: live XGBoost arms C0 vs CTX-live (pre-registered 2026-11-01).

    python -m experiment.train_fix77_live run        # foreground (``train_fix77_tabm start`` chains it after TabM-NE)
    python -m experiment.train_fix77_live status
    CASCADE_SMOKE=1 python -m experiment.train_fix77_live run    # wiring check (own folder, tiny rows, one seed)

Plan: ml_pipeline/notes_20261101/fix77_plan.md (stage 1). C0 = the live model's 252 exact features with the main
study's tuned params at their fold-1 round count; CTX = the same plus the ``ctx_live`` columns (context means of
exact live columns only). Fold 1, seeds 142/143/144, J = 0.5 x role-0 acc + 0.5 x fold-1 OOF acc, early drop and
admission as in the plan; a chosen arm must pass again on seeds 151/152/153; then the all-row refit at the main
study's final round count, calibrated on role 1. Role 2 is report-only; the held-out test is never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# Set explicitly (not setdefault): a chained or inherited environment may carry another study's folders.
os.environ["CASCADE_STUDY"] = "artifacts/training/fix77_live_20261101_v1"
os.environ["CASCADE_SMOKE_STUDY"] = "artifacts/training/fix77_live_smoke"
os.environ["CASCADE_DATA"] = "dataset/experiments/csv_v3_fix77_20261101_v1"

import numpy as np  # noqa: E402

from experiment import train_live77 as L  # noqa: E402
from framework.training.csv_v3 import arms, fix77, runner  # noqa: E402

STUDY = runner.STUDY
MAIN = runner.ROOT / "artifacts/training/live77_20261031_v1"            # read-only input
REAL_STUDY = runner.ROOT / "artifacts/training/fix77_live_20261101_v1"
SMOKE_STUDY = runner.ROOT / "artifacts/training/fix77_live_smoke"
PLAN = runner.ROOT / "ml_pipeline/notes_20261101/fix77_plan.md"
if STUDY.resolve() != (SMOKE_STUDY if runner.SMOKE else REAL_STUDY).resolve():
    raise RuntimeError(f"refusing to run in {STUDY}")
if runner.DATA.resolve() != fix77.OUT.resolve():
    raise RuntimeError(f"the plan uses the fix77 data ({fix77.OUT}), not {runner.DATA}")
if runner.FEATURE_GROUPS is not None:
    raise RuntimeError("refusing to run with CASCADE_FEATURE_GROUPS set")

SEEDS = (142, 143, 144)
CONFIRM_SEEDS = (151, 152, 153)
FINAL_SEED = 42
if runner.SMOKE:
    SEEDS, CONFIRM_SEEDS = (142,), (151,)
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
        self.scen = runner._arr("scenario_validation")
        roles = runner._arr("validation_roles")
        self.tune, self.tune_seen, self.tune_unseen = arms.role_split(roles, self.scen, runner._arr("scenario_train"))
        stored = runner._arr("validation_scenario_seen").astype(bool)[self.tune]
        if not np.array_equal(stored, np.isin(self.tune, self.tune_seen)):
            raise RuntimeError("role-0 seen split differs from validation_scenario_seen.npy")
        if runner.manifest().get("fix77_groups") != fix77.GROUPS:
            raise RuntimeError("fix77 data folder does not match fix77.py (rebuild it)")


def main_artifacts() -> dict:
    sel = json.loads((MAIN / "selection.json").read_text(encoding="utf-8"))
    size = json.loads((MAIN / "size.json").read_text(encoding="utf-8"))
    rounds = json.loads((MAIN / "final_rounds.json").read_text(encoding="utf-8"))
    final = json.loads((MAIN / "jobs" / "77_live_final" / "result.json").read_text(encoding="utf-8"))
    return arms.freeze(STUDY / "base_config.json", {
        "names": sel["feature_names"], "params": size["chosen"]["params"], "fold_rounds": size["chosen"]["rounds"],
        "final_rounds": rounds["chosen_rounds"], "final_job": "77_live_final",
        "sha256": {p.name: _sha256(p) for p in (MAIN / "selection.json", MAIN / "size.json", MAIN / "final_rounds.json")},
        "final_report": final.get("calibration", {}).get("report", final["report"]), "smoke": runner.SMOKE})


def make_config(base: dict, names: list[str], rounds: int) -> dict:
    c = L.base_config()
    c.update({k: v for k, v in base["params"].items() if k in L.PARAMS})
    c.update(n_estimators=int(rounds), early_stopping_rounds=int(rounds) + 1, runtime_seconds=10800,
             checkpoint_rule="last", feature_list=list(names))
    return c


def fold_job(ctl, R: Rows, job_id: str, config: dict, seed: int) -> dict:
    r = ctl.run_job(job_id, "77", "xgboost", config, seed=seed, rows="fold1")
    if r is None or r.get("status") != "complete":
        raise RuntimeError(f"{job_id} failed")
    work = STUDY / r["work_dir"]
    return {"job_id": job_id, "seed": seed, "work": str(work), "n_features": len(config["feature_list"]),
            **fix77.fold_metrics(work, R.y_val, R.y_train, R.tune, R.tune_seen, R.tune_unseen, R.classes)}


def run_arm(ctl, R, base, arm: str, names: list[str], seeds) -> list[dict]:
    return [fold_job(ctl, R, f"77_fixlive_{arm}_s{s}", make_config(base, names, base["fold_rounds"]), s) for s in seeds]


def _hits(R: Rows, runs: list[dict]) -> np.ndarray:
    return np.mean([np.load(Path(r["work"]) / "val_proba.npy")[R.tune].argmax(1) == R.y_val[R.tune] for r in runs], axis=0)


def summarize(R: Rows, control: list[dict], runs: list[dict], dropped, n_seeds: int) -> dict:
    by = {r["seed"]: r for r in control}
    s = {**arms.paired(control, runs), "n_features": runs[0]["n_features"], "seeds_run": len(runs),
         "dropped_after_seed": dropped, "runs": runs,
         "focus_role0": fix77.mean_focus(runs, "focus_role0"), "focus_oof": fix77.mean_focus(runs, "focus_oof")}
    s["admitted"] = dropped is None and arms.admitted(s, n_seeds)
    s["role0_diff_ci95"] = arms.scenario_diff_ci(_hits(R, runs), _hits(R, [by[r["seed"]] for r in runs]), R.scen[R.tune])
    return s


def quarantine_failed() -> None:
    for res in sorted((STUDY / "jobs").glob("*/result.json")):
        if json.loads(res.read_text(encoding="utf-8")).get("status") == "failed":
            dest = STUDY / "failed_jobs" / f"{res.parent.name}_{time.strftime('%Y%m%d_%H%M%S')}"
            dest.parent.mkdir(exist_ok=True)
            os.replace(res.parent, dest)
            _log(f"moved failed job {res.parent.name} to {dest.relative_to(STUDY)}; it will be retried")


def run() -> None:
    busy = fix77.other_controllers()
    if busy:
        raise RuntimeError(f"another training controller or job is running (pids {busy}); one GPU job at a time")
    (STUDY / "controller_pid.json").write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    quarantine_failed()
    ctl = runner.Controller(log=_log)
    ctl.state(phase="running", smoke=runner.SMOKE)
    try:
        R = Rows()
        base = main_artifacts()
        if base["smoke"] != runner.SMOKE:
            raise RuntimeError("base_config.json was frozen in the other (smoke/real) mode")
        specs = arms.freeze(STUDY / "arms.json", {
            "arms": fix77.arm_specs(base["names"], {"ctx": fix77.CTX_LIVE}),
            "plan_sha256": _sha256(PLAN), "data_manifest_sha256": _sha256(runner.DATA / "manifest.json"),
            "smoke": runner.SMOKE})
        lists = specs["arms"]
        ctl.state(stage="arms")
        control = run_arm(ctl, R, base, "C0", lists["C0"]["names"], SEEDS)
        by = {r["seed"]: r for r in control}
        runs, dropped = [], None
        for s in SEEDS:
            if ctl.stopped():
                raise KeyboardInterrupt("STOP present")
            runs += run_arm(ctl, R, base, "CTX", lists["CTX"]["names"], (s,))
            if arms.early_drop([arms.J(r) - arms.J(by[r["seed"]]) for r in runs]):
                dropped = len(runs)
                break
        results = {"CTX": summarize(R, control, runs, dropped, len(SEEDS))}
        _log(f"CTX: J gain {100 * results['CTX']['gain']:+.3f} pp, admitted {results['CTX']['admitted']}")
        chosen, forced = fix77.choose(results), False
        if chosen is None and runner.SMOKE:
            chosen, forced = "CTX", True
            _log("smoke: CTX not admitted; forcing it through confirmation and final (wiring check only)")
        confirmation = None
        if chosen is not None:
            ctl.state(stage="confirmation", chosen=chosen)
            c0 = run_arm(ctl, R, base, "C0", lists["C0"]["names"], CONFIRM_SEEDS)
            cr = run_arm(ctl, R, base, chosen, lists[chosen]["names"], CONFIRM_SEEDS)
            confirmation = summarize(R, c0, cr, None, len(CONFIRM_SEEDS))
            _log(f"confirmation {chosen}: J gain {100 * confirmation['gain']:+.3f} pp, passes {confirmation['admitted']}")
        # Smoke always exercises the final refit (one-seed smoke numbers decide nothing).
        confirmed = confirmation is not None and (confirmation["admitted"] or forced or runner.SMOKE)
        decision = arms.freeze(STUDY / "decision.json", {
            "admitted": [a for a in results if results[a]["admitted"]], "chosen": chosen, "smoke_forced": forced,
            "confirmed": confirmed,
            "confirmation": None if confirmation is None else {k: v for k, v in confirmation.items() if k != "runs"},
            "rule": "fold-1 decisions above stand; the all-row refit is calibrated on role 1 and reported on role 2 "
                    "(report-only); nothing is re-chosen on role 2"})
        fin = None
        if decision["confirmed"]:
            ctl.state(stage="final")
            fin = ctl.run_job(f"77_fixlive_final_{chosen}", "77", "xgboost",
                              make_config(base, lists[chosen]["names"], base["final_rounds"]), FINAL_SEED, rows="full")
            if fin is None or fin.get("status") != "complete":
                raise RuntimeError("final refit failed")
        out = {"base": {k: v for k, v in base.items() if k != "final_report"}, "arms": results, "control": control,
               "control_focus_role0": fix77.mean_focus(control, "focus_role0"),
               "control_focus_oof": fix77.mean_focus(control, "focus_oof"), "decision": decision,
               "final_job": None if fin is None else fin["job_id"]}
        (STUDY / "fix77_live.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        write_report(R, out, base, fin)
        ctl.state(phase="done", stage="report")
    except KeyboardInterrupt:
        ctl.state(phase="stopped")
        _log("stopped (STOP file); restart resumes from cached jobs")
    except Exception:
        ctl.state(phase="error")
        raise


def _final_lines(R: Rows, base: dict, fin: dict) -> list[str]:
    pct = lambda x: f"{100 * x:.2f}%"
    rep_new = fin.get("calibration", {}).get("report", fin["report"])
    rep_old = base["final_report"]
    lines = ["| model | tuning acc | assessment acc (CI) | assessment macro-F1 | benign FPR (assess) |", "|---|---|---|---|---|"]
    for label, rep in (("shipped live (C0, 252 features)", rep_old), (f"fix77 final ({fin['job_id']})", rep_new)):
        a = rep["assessment"]
        ci = a["accuracy_ci95_scenario_bootstrap"]
        lines.append(f"| {label} | {pct(rep['tuning']['accuracy'])} | {pct(a['accuracy'])} ({100 * ci[0]:.1f}-"
                     f"{100 * ci[1]:.1f}) | {a['macro_f1_supported']:.4f} | {pct(a['benign_false_alarm_rate'])} |")
    roles = runner._arr("validation_roles")
    a_rows = np.flatnonzero(roles == 2)
    cal = runner.job_dir(fin["job_id"]) / "val_proba_cal.npy"
    p_new = np.load(cal) if cal.exists() else np.load(STUDY / fin["work_dir"] / "val_proba.npy")
    mjob = MAIN / "jobs" / "77_live_final"
    p_old = np.load(mjob / "val_proba_cal.npy") if (mjob / "val_proba_cal.npy").exists() else \
        np.load(MAIN / json.loads((mjob / "result.json").read_text(encoding="utf-8"))["work_dir"] / "val_proba.npy")
    diff = (p_new[a_rows].argmax(1) == R.y_val[a_rows]).astype(float) - (p_old[a_rows].argmax(1) == R.y_val[a_rows])
    ci = arms.scenario_diff_ci(diff, np.zeros_like(diff), R.scen[a_rows])
    lines += ["", f"Paired assessment difference (fix77 final - shipped live): {100 * diff.mean():+.2f} pp, "
                  f"scenario-bootstrap CI {100 * ci[0]:+.2f} to {100 * ci[1]:+.2f} pp (report-only)."]
    for label, p in (("shipped live", p_old), ("fix77 final", p_new)):
        fm = fix77.focus_metrics(p[R.tune], R.y_val[R.tune], R.classes)
        lines.append(f"Fix A on role 0, {label}: accuracy with ping-sweep merged {100 * fm['acc_ping_merged']:.2f}%, "
                     f"ping rows called benign {fm['ping_to_benign']}.")
    return lines


def write_report(R: Rows, o: dict, base: dict, fin: dict | None) -> None:
    d, s = o["decision"], o["arms"]["CTX"]
    lines = ["# 77-class fix test, stage 1: live XGBoost (context arm)", "",
             "Plan: ml_pipeline/notes_20261101/fix77_plan.md (pre-registered 2026-11-01, stage 1). Validation only; "
             f"held-out test untouched. C0 = the live model's {len(base['names'])} exact features, tuned params at "
             f"{base['fold_rounds']} fold-1 rounds; CTX adds {len(fix77.CTX_LIVE)} context columns built from exact live "
             f"columns ({', '.join(fix77.CTX_LIVE)}). Seeds {', '.join(map(str, SEEDS))}; J = 0.5 x role-0 acc + "
             "0.5 x fold-1 OOF acc.", ""]
    if runner.SMOKE:
        lines += ["**SMOKE RUN: wiring check only (300 fit rows per class, 4 rounds, one seed). No number here is a "
                  "result.**", ""]
    c0 = o["control"]
    ci = s["role0_diff_ci95"]
    note = f"no (dropped after seed {s['dropped_after_seed']})" if s["dropped_after_seed"] else ("yes" if s["admitted"] else "no")
    lines += ["| arm | features | seeds | mean J | J gain (pp) | SD (pp) | role-0 seen gain (pp) | role-0 unseen gain (pp) "
              "| role-0 diff CI (pp) | OOF macro-F1 change | admitted |", "|---|---|---|---|---|---|---|---|---|---|---|",
              f"| C0 (control) | {c0[0]['n_features']} | {len(c0)} | {100 * np.mean([arms.J(r) for r in c0]):.2f}% | | | | | | | |",
              f"| CTX | {s['n_features']} | {s['seeds_run']} | {100 * s['J']:.2f}% | {100 * s['gain']:+.3f} | "
              f"{100 * s['gain_sd']:.3f} | {100 * s['seen_gain']:+.3f} | {100 * s['unseen_gain']:+.3f} | "
              f"{100 * ci[0]:+.2f} to {100 * ci[1]:+.2f} | {s['oof_f1_change']:+.4f} | {note} |", "",
              "Admission (all must hold): mean J gain >= 0.1 pp and > 2 x SD; mean role-0 unseen gain >= 0; mean OOF "
              "macro-F1 change >= -0.005; every seed run.", ""]
    lines.append("**Decision: CTX not admitted; the shipped live model stays.**" if not d["admitted"] else
                 f"Admitted: {', '.join(d['admitted'])}. Chosen: {d['chosen']}.")
    if d.get("confirmation"):
        c = d["confirmation"]
        lines += ["", f"Confirmation (seeds {', '.join(map(str, CONFIRM_SEEDS))}): J gain {100 * c['gain']:+.3f} pp "
                      f"(SD {100 * c['gain_sd']:.3f}), unseen {100 * c['unseen_gain']:+.3f} pp; passes: {c['admitted']}."]
    lines += fix77.focus_table_lines(o)
    lines += ["", "Caveats (plan): a device's previous 30 s always has the same label in this dataset, so CTX can look "
                  "better here than live; the attack-start delay cannot be measured; a live CTX model would also need "
                  "the context columns in the Rust extractor. Every live score stays 'dataset-trained, live accuracy "
                  "unverified'.", ""]
    if fin is not None:
        lines += ["## Final model (all fit77 rows, calibration on role 1, assessment report-only)", ""] + \
            _final_lines(R, base, fin) + ["", "Adoption into the app is a separate owner decision.", ""]
    (STUDY / "REPORT_fix77_live.md").write_text("\n".join(lines), encoding="utf-8")
    _log("report written")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "status"])
    args = parser.parse_args(argv)
    STUDY.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        (STUDY / "STOP").unlink(missing_ok=True)
        run()
    else:
        state = STUDY / "state.json"
        print(state.read_text(encoding="utf-8") if state.exists() else "no state yet")
        print(f"{len(list((STUDY / 'jobs').glob('*/result.json')))} finished jobs; "
              f"other controllers/jobs: {fix77.other_controllers()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
