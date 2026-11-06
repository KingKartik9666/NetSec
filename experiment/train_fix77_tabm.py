"""77-class fix test, stage 1: TabM-NE arms C0 / CTX / ROB / CTX+ROB (pre-registered 2026-11-01).

    python -m experiment.train_fix77_tabm start      # detached chain: this driver, then experiment.train_fix77_live
    python -m experiment.train_fix77_tabm run        # this driver only, foreground
    python -m experiment.train_fix77_tabm status     # both stage-1 drivers
    python -m experiment.train_fix77_tabm stop       # STOP file in both studies; halts between jobs
    CASCADE_SMOKE=1 python -m experiment.train_fix77_tabm run    # wiring check (own folder, tiny rows, one seed)

Plan and rules (fixed before any arm runs; not changed here): ml_pipeline/notes_20261101/fix77_plan.md.
Owner 2026-11-01: test past-window context and robustness fixes; "TabM-NE first, then decide".
Arms are fold-1 refits of the tuned cascade config ``77_tune_tabm_ne_t008`` (held fixed, copied verbatim) on the
fix77 data folder (v3 + CTX + ROB columns, ``framework/training/csv_v3/fix77.py``), paired with C0 on seeds
142/143/144; J = 0.5 x role-0 acc + 0.5 x fold-1 OOF acc. A chosen arm must pass again on seeds 151/152/153 before
the all-row refit. Role 1 only calibrates that final model; role 2 is report-only; the held-out test is never read.
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

os.environ.setdefault("CASCADE_STUDY", "artifacts/training/fix77_tabm_ne_20261101_v1")
os.environ.setdefault("CASCADE_SMOKE_STUDY", "artifacts/training/fix77_tabm_ne_smoke")
os.environ["CASCADE_DATA"] = "dataset/experiments/csv_v3_fix77_20261101_v1"

import numpy as np  # noqa: E402

from framework.training.csv_v3 import arms, fix77, prep, runner  # noqa: E402

STUDY = runner.STUDY
CASCADE = runner.ROOT / "artifacts/training/cascade_v3_20261030_v1"     # read-only input
REAL_STUDY = runner.ROOT / "artifacts/training/fix77_tabm_ne_20261101_v1"
SMOKE_STUDY = runner.ROOT / "artifacts/training/fix77_tabm_ne_smoke"
PLAN = runner.ROOT / "ml_pipeline/notes_20261101/fix77_plan.md"
if STUDY.resolve() != (SMOKE_STUDY if runner.SMOKE else REAL_STUDY).resolve():
    raise RuntimeError(f"refusing to run in {STUDY}: fix77 TabM-NE uses {SMOKE_STUDY if runner.SMOKE else REAL_STUDY}")
if runner.DATA.resolve() != fix77.OUT.resolve():
    raise RuntimeError(f"the plan uses the fix77 data ({fix77.OUT}), not {runner.DATA}")
if runner.FEATURE_GROUPS is not None:
    raise RuntimeError("refusing to run with CASCADE_FEATURE_GROUPS set: every arm would train on the same live groups")

BASE_JOB = "77_tune_tabm_ne_t008"
SEEDS = (142, 143, 144)
CONFIRM_SEEDS = (151, 152, 153)
FINAL_SEED = 42
ARM_ORDER = ("CTX", "ROB", "CTXROB")
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
            raise RuntimeError("fix77 data folder does not match fix77.py (rebuild it)")
        self.v3 = v3


def base_config() -> dict:
    path = CASCADE / "jobs" / BASE_JOB / "result.json"
    r = json.loads(path.read_text(encoding="utf-8"))
    if r.get("status") != "complete" or r["family"] != "tabm_ne" or r["task"] != "77":
        raise RuntimeError(f"{BASE_JOB} is not a complete 77-class tabm_ne job")
    return arms.freeze(STUDY / "base_config.json", {
        "job": BASE_JOB, "result_json": path.relative_to(runner.ROOT).as_posix(),
        "result_json_sha256": _sha256(path), "config": r["config"], "smoke": runner.SMOKE})


def make_config(base: dict, spec: dict) -> dict:
    c = copy.deepcopy(base["config"])
    c.update(feature_list=list(spec["names"]), feature_scope="csv")
    if spec.get("row_weight") is not None:
        c["row_weight"] = dict(spec["row_weight"])
    return c


def fold_job(ctl, R: Rows, job_id: str, config: dict, seed: int) -> dict:
    r = ctl.run_job(job_id, "77", "tabm_ne", config, seed=seed, rows="fold1")
    if r is None or r.get("status") != "complete":
        raise RuntimeError(f"{job_id} failed")
    work = STUDY / r["work_dir"]
    out = {"job_id": job_id, "seed": seed, "work": str(work), "n_features": len(config["feature_list"]),
           **fix77.fold_metrics(work, R.y_val, R.y_train, R.tune, R.tune_seen, R.tune_unseen, R.classes)}
    out.update(best_epoch=r["fit"].get("best_epoch"), stop_reason=r["fit"].get("stop_reason"),
               wall_seconds=r.get("wall_seconds"))
    return out


def run_arm(ctl, R: Rows, base: dict, arm: str, spec: dict, seeds) -> list[dict]:
    return [fold_job(ctl, R, f"77_fix_{arm}_s{s}", make_config(base, spec), s) for s in seeds]


def run_arm_adaptive(ctl, R: Rows, base: dict, arm: str, spec: dict, control: list[dict]) -> tuple[list, int | None]:
    by = {r["seed"]: r for r in control}
    runs = []
    for s in SEEDS:
        if ctl.stopped():
            raise KeyboardInterrupt("STOP present")
        runs += run_arm(ctl, R, base, arm, spec, (s,))
        if arms.early_drop([arms.J(r) - arms.J(by[r["seed"]]) for r in runs]):
            return runs, len(runs)
    return runs, None


def _hits(R: Rows, runs: list[dict]) -> np.ndarray:
    return np.mean([np.load(Path(r["work"]) / "val_proba.npy")[R.tune].argmax(1) == R.y_val[R.tune] for r in runs], axis=0)


def summarize(R: Rows, control: list[dict], runs: list[dict], dropped: int | None, n_seeds: int) -> dict:
    by = {r["seed"]: r for r in control}
    s = {**arms.paired(control, runs), "n_features": runs[0]["n_features"], "seeds_run": len(runs),
         "dropped_after_seed": dropped, "runs": runs,
         "focus_role0": fix77.mean_focus(runs, "focus_role0"), "focus_oof": fix77.mean_focus(runs, "focus_oof")}
    s["admitted"] = dropped is None and arms.admitted(s, n_seeds)
    s["role0_diff_ci95"] = arms.scenario_diff_ci(_hits(R, runs), _hits(R, [by[r["seed"]] for r in runs]), R.scen[R.tune])
    return s


# ------------------------------------------------------------------------------------------------ main
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
        base = base_config()
        specs = arms.freeze(STUDY / "arms.json", {
            "arms": fix77.arm_specs(R.v3, {"ctx": fix77.CTX, "rob": fix77.ROB}),
            "plan_sha256": _sha256(PLAN), "data_manifest_sha256": _sha256(runner.DATA / "manifest.json"),
            "smoke": runner.SMOKE})
        if specs["smoke"] != runner.SMOKE:
            raise RuntimeError("arms.json was frozen in the other (smoke/real) mode")
        lists = specs["arms"]
        ctl.state(stage="arms")
        control = run_arm(ctl, R, base, "C0", lists["C0"], SEEDS)
        results = {}
        for arm in ARM_ORDER:
            runs, dropped = run_arm_adaptive(ctl, R, base, arm, lists[arm], control)
            results[arm] = summarize(R, control, runs, dropped, len(SEEDS))
            _log(f"{arm}: J gain {100 * results[arm]['gain']:+.3f} pp, admitted {results[arm]['admitted']}"
                 + (f" (dropped after seed {dropped})" if dropped else ""))
        chosen, forced = fix77.choose(results), False
        if chosen is None and runner.SMOKE:
            chosen, forced = max(results, key=lambda a: results[a]["J"]), True
            _log(f"smoke: no arm admitted; forcing {chosen} through confirmation and final (wiring check only)")
        confirmation = None
        if chosen is not None:
            ctl.state(stage="confirmation", chosen=chosen)
            c0 = run_arm(ctl, R, base, "C0", lists["C0"], CONFIRM_SEEDS)
            runs = run_arm(ctl, R, base, chosen, lists[chosen], CONFIRM_SEEDS)
            confirmation = summarize(R, c0, runs, None, len(CONFIRM_SEEDS))
            _log(f"confirmation {chosen}: J gain {100 * confirmation['gain']:+.3f} pp, passes {confirmation['admitted']}")
        # Smoke always exercises the final refit (one-seed smoke numbers decide nothing).
        confirmed = confirmation is not None and (confirmation["admitted"] or forced or runner.SMOKE)
        decision = arms.freeze(STUDY / "decision.json", {
            "admitted": [a for a in ARM_ORDER if results[a]["admitted"]], "chosen": chosen, "smoke_forced": forced,
            "confirmed": confirmed,
            "confirmation": None if confirmation is None else {k: v for k, v in confirmation.items() if k != "runs"},
            "rule": "fold-1 decisions above stand; the all-row refit is calibrated on role 1 and reported on role 2 "
                    "(report-only); nothing is re-chosen on role 2"})
        fin = None
        if decision["confirmed"]:
            ctl.state(stage="final")
            fin = ctl.run_job(f"77_fix_final_{decision['chosen']}", "77", "tabm_ne",
                              make_config(base, lists[decision["chosen"]]), seed=FINAL_SEED, rows="full")
            if fin is None or fin.get("status") != "complete":
                raise RuntimeError("final refit failed")
        out = {"base": base, "arms_sha256": json.loads((STUDY / "arms.json").read_text(encoding="utf-8"))["sha256"],
               "arms": results, "control": control, "control_focus_role0": fix77.mean_focus(control, "focus_role0"),
               "control_focus_oof": fix77.mean_focus(control, "focus_oof"), "decision": decision,
               "final_job": None if fin is None else fin["job_id"]}
        (STUDY / "fix77.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        write_report(R, out, fin)
        ctl.state(phase="done", stage="report")
    except KeyboardInterrupt:
        ctl.state(phase="stopped")
        _log("stopped (STOP file); restart resumes from cached jobs")
    except Exception:
        ctl.state(phase="error")
        raise


# ---------------------------------------------------------------------------------------------- report
def _proba(job_dir: Path, r: dict, study: Path) -> np.ndarray:
    cal = job_dir / "val_proba_cal.npy"
    return np.load(cal) if cal.exists() else np.load(study / r["work_dir"] / "val_proba.npy")


def _final_rows(R: Rows, fin: dict) -> list[str]:
    from framework.training.csv_v3 import report as reporting
    shipped = json.loads((CASCADE / "jobs" / BASE_JOB / "result.json").read_text(encoding="utf-8"))
    pct = lambda x: "-" if x is None else f"{100 * x:.2f}%"
    lines = ["| model | tuning acc | assessment acc (CI) | macro-F1 | seen / unseen | benign FAR | calibration | "
             "ECE raw -> cal (assess) |", "|---|---|---|---|---|---|---|---|"]
    for label, r in (("shipped TabM-NE (t008, 445 features)", shipped), (f"fix77 final ({fin['job_id']})", fin)):
        row = reporting._row(label, "", r)
        ci, ece = row["final_assessment_ci95"], row["assessment_ece"] or {}
        lines.append(f"| {label} | {pct(row['tuning_accuracy'])} | {pct(row['final_assessment_accuracy'])} "
                     f"({100 * ci[0]:.1f}-{100 * ci[1]:.1f}) | {100 * row['assessment_macro_f1_supported']:.2f} | "
                     f"{pct(row['assessment_seen_accuracy'])} / {pct(row['assessment_unseen_accuracy'])} | "
                     f"{pct(row['benign_false_alarm_rate'])} | {row['calibration']} | "
                     f"{ece.get('raw', float('nan')):.3f} -> {ece.get('calibrated', float('nan')):.3f} |")
    roles = runner._arr("validation_roles")
    a_rows = np.flatnonzero(roles == 2)
    p_new = _proba(runner.job_dir(fin["job_id"]), fin, STUDY)
    p_old = _proba(CASCADE / "jobs" / BASE_JOB, shipped, CASCADE)
    diff = (p_new[a_rows].argmax(1) == R.y_val[a_rows]).astype(float) - (p_old[a_rows].argmax(1) == R.y_val[a_rows])
    ci = arms.scenario_diff_ci(diff, np.zeros_like(diff), R.scen[a_rows])
    lines += ["", f"Paired assessment difference (fix77 final - shipped): {100 * diff.mean():+.2f} pp, scenario-bootstrap "
                  f"CI {100 * ci[0]:+.2f} to {100 * ci[1]:+.2f} pp (report-only)."]
    for label, p in (("shipped", p_old), ("fix77 final", p_new)):
        fm = fix77.focus_metrics(p[R.tune], R.y_val[R.tune], R.classes)
        lines.append(f"Fix A on role 0, {label}: accuracy with ping-sweep merged {100 * fm['acc_ping_merged']:.2f}%, "
                     f"ping rows called benign {fm['ping_to_benign']}.")
    return lines


def write_report(R: Rows, o: dict, fin: dict | None) -> None:
    d = o["decision"]
    lines = ["# 77-class fix test, stage 1: TabM-NE (context and robustness arms)", "",
             "Plan: ml_pipeline/notes_20261101/fix77_plan.md (pre-registered 2026-11-01). Validation only; held-out "
             "test untouched. Arms: fold-1 refits of the tuned config `77_tune_tabm_ne_t008` (fixed), seeds "
             f"{', '.join(map(str, SEEDS))}, paired with C0; J = 0.5 x role-0 acc + 0.5 x fold-1 OOF acc. CTX = "
             f"{len(fix77.CTX)} past-window context columns (same device, previous 30 s, past only); ROB = "
             f"{len(fix77.ROB)} volume-free columns + scenario-balanced weights rho 1.0.", ""]
    if runner.SMOKE:
        lines += ["**SMOKE RUN: wiring check only (300 fit rows per class, 1 epoch, one seed). No number here is a "
                  "result.**", ""]
    lines += ["## Arms (fold 1)", "",
              "| arm | features | seeds | mean J | J gain (pp) | SD (pp) | role-0 seen gain (pp) | role-0 unseen gain (pp) "
              "| role-0 diff CI (pp) | OOF macro-F1 change | admitted |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    c0 = o["control"]
    lines.append(f"| C0 (control) | {c0[0]['n_features']} | {len(c0)} | {100 * np.mean([arms.J(r) for r in c0]):.2f}% "
                 "| | | | | | | |")
    for arm, s in o["arms"].items():
        ci = s["role0_diff_ci95"]
        note = f"no (dropped after seed {s['dropped_after_seed']})" if s["dropped_after_seed"] else \
            ("yes" if s["admitted"] else "no")
        lines.append(f"| {arm} | {s['n_features']} | {s['seeds_run']} | {100 * s['J']:.2f}% | {100 * s['gain']:+.3f} | "
                     f"{100 * s['gain_sd']:.3f} | {100 * s['seen_gain']:+.3f} | {100 * s['unseen_gain']:+.3f} | "
                     f"{100 * ci[0]:+.2f} to {100 * ci[1]:+.2f} | {s['oof_f1_change']:+.4f} | {note} |")
    lines += ["", "Admission (all must hold): mean J gain >= 0.1 pp and > 2 x SD; mean role-0 unseen gain >= 0; "
                  "mean OOF macro-F1 change >= -0.005; every seed run. CTX+ROB is chosen only if it beats the better "
                  "single arm by >= 0.1 pp. The role-0 CI and the tables below are diagnostic only.", ""]
    if not d["admitted"]:
        lines.append("**Decision: no arm admitted; the shipped TabM-NE stays.**"
                     + (" (smoke: an arm was forced through the later stages for wiring only)" if d["smoke_forced"] else ""))
    else:
        lines.append(f"Admitted: {', '.join(d['admitted'])}. Chosen: {d['chosen']}.")
    conf = d.get("confirmation")
    if conf:
        lines += ["", f"Confirmation ({d['chosen']} vs C0, seeds {', '.join(map(str, CONFIRM_SEEDS))}): J gain "
                      f"{100 * conf['gain']:+.3f} pp (SD {100 * conf['gain_sd']:.3f}), unseen {100 * conf['unseen_gain']:+.3f} pp, "
                      f"OOF macro-F1 {conf['oof_f1_change']:+.4f}; passes: {conf['admitted']}."]
    lines += fix77.focus_table_lines(o)
    lines += ["", "Caveats (written in the plan before results): a device's previous 30 s always carries the same "
                  "label in this dataset, so CTX can look better here than in real use, and the attack-start delay "
                  "cannot be measured; the config is held fixed for every arm.", ""]
    if fin is not None:
        lines += ["## Final model (all fit77 rows, early stop on role 0, calibration on role 1, assessment report-only)",
                  ""] + _final_rows(R, fin) + ["", "Adoption into the app is a separate owner decision.", ""]
    (STUDY / "REPORT_fix77_tabm.md").write_text("\n".join(lines), encoding="utf-8")
    _log("report written")


# ------------------------------------------------------------------------------------------- commands
def _clean_env() -> dict:
    env = os.environ.copy()
    for key in ("CASCADE_STUDY", "CASCADE_SMOKE_STUDY", "CASCADE_DATA", "CASCADE_FEATURE_GROUPS"):
        env.pop(key, None)
    return env


def chain() -> int:
    """Stage 1 in order, one process at a time: this driver, then the live driver (never both on the GPU)."""
    for module in ("experiment.train_fix77_tabm", "experiment.train_fix77_live"):
        _log(f"chain: starting {module}")
        code = subprocess.run([sys.executable, "-m", module, "run"], cwd=runner.ROOT, env=_clean_env()).returncode
        _log(f"chain: {module} exited {code}")
        if code != 0:
            return code
        if module.endswith("_tabm"):
            # A STOP ends the TabM-NE run with exit 0 and phase "stopped": the live stage must not start then.
            state = json.loads((STUDY / "state.json").read_text(encoding="utf-8")) if (STUDY / "state.json").exists() else {}
            if state.get("phase") != "done":
                _log(f"chain: TabM-NE phase is {state.get('phase')!r}, not 'done'; live stage not started")
                return 1
    return 0


def _status(study: Path) -> str:
    state = study / "state.json"
    jobs = len(list((study / "jobs").glob("*/result.json")))
    return f"{study.name}: {state.read_text(encoding='utf-8') if state.exists() else 'no state yet'}\n  {jobs} finished jobs"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "start", "status", "stop", "_chain"])
    args = parser.parse_args(argv)
    STUDY.mkdir(parents=True, exist_ok=True)
    live_study = runner.ROOT / ("artifacts/training/fix77_live_smoke" if runner.SMOKE
                                else "artifacts/training/fix77_live_20261101_v1")
    if args.command == "run":
        (STUDY / "STOP").unlink(missing_ok=True)
        run()
    elif args.command == "_chain":
        return chain()
    elif args.command == "start":
        busy = fix77.other_controllers()
        if busy:
            print(f"not started: another training controller or job is running (pids {busy})")
            return 1
        for study in (STUDY, live_study):
            (study / "STOP").unlink(missing_ok=True)
        log = open(STUDY / "chain.log", "a", encoding="utf-8")
        flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0   # DETACHED_PROCESS | NEW_PROCESS_GROUP
        proc = subprocess.Popen([sys.executable, "-m", "experiment.train_fix77_tabm", "_chain"], cwd=runner.ROOT,
                                stdout=log, stderr=subprocess.STDOUT, creationflags=flags, env=os.environ.copy())
        (STUDY / "chain_pid.json").write_text(json.dumps({"pid": proc.pid, "started": time.time()}))
        print(f"chain started, pid {proc.pid}; log: {STUDY / 'chain.log'}")
    elif args.command == "stop":
        for study in (STUDY, live_study):
            study.mkdir(parents=True, exist_ok=True)
            (study / "STOP").write_text("stop requested\n", encoding="utf-8")
        print("STOP written (TabM-NE and live studies)")
    else:
        print(_status(STUDY))
        print(_status(live_study))
        print(f"other controllers/jobs: {fix77.other_controllers()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
