"""77-class TabM-NE feature-selection test on the 445 csv_v3 features (pre-registered 2026-11-01).

    python -m experiment.train_fs77_tabm start     # detached controller (log: <study>/controller.log)
    python -m experiment.train_fs77_tabm run       # foreground
    python -m experiment.train_fs77_tabm status
    python -m experiment.train_fs77_tabm stop      # STOP file; halts between jobs
    CASCADE_SMOKE=1 python -m experiment.train_fs77_tabm run    # wiring check (own folder, tiny rows, one seed)

Plan and rules (fixed before any arm runs; not changed here): ml_pipeline/notes_20261101/fs77_tabm_ne_plan.md.
Owner 2026-10-31 approved the test (top-200/300 by training-fold importance plus a scenario-fingerprint-pruned set,
TabM-NE, seen and unseen reported separately); 2026-11-01 "continue on 77 class csv tabm ne".
Rankings come from training forward fold 0 only and are frozen with sha256 before the first arm job:
P = permutation importance of the fold-0 tuned TabM-NE on fold-0 OOF rows, G = XGBoost total_gain of the fold-0
tuned XGBoost, FP = scenario-fingerprint score on fold-0 fit rows. Arms are fold-1 refits of the tuned cascade config
``77_tune_tabm_ne_t008`` (held fixed, copied verbatim), paired with C0 on seeds 142/143/144; J = 0.5 x role-0 acc +
0.5 x fold-1 OOF acc. A chosen spec must pass the admission rule again on seeds 151/152/153 before the all-row refit.
Role 1 only calibrates that final model; role 2 is report-only; the held-out test partition is never read.
One GPU job at a time (runner subprocesses; the ranking forward passes run in their own subprocess too).
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

os.environ.setdefault("CASCADE_STUDY", "artifacts/training/fs77_tabm_ne_20261101_v1")
os.environ.setdefault("CASCADE_SMOKE_STUDY", "artifacts/training/fs77_tabm_ne_smoke")

import numpy as np  # noqa: E402

from framework.training.csv_v3 import arms, prep, runner  # noqa: E402

STUDY = runner.STUDY
CASCADE = runner.ROOT / "artifacts/training/cascade_v3_20261030_v1"     # read-only input
REAL_STUDY = runner.ROOT / "artifacts/training/fs77_tabm_ne_20261101_v1"
SMOKE_STUDY = runner.ROOT / "artifacts/training/fs77_tabm_ne_smoke"
# Allow-list, not a blacklist: an inherited CASCADE_STUDY must never send fs77 jobs into another study, and smoke
# (1-epoch, tiny-row) results must never land in the real folder or the reverse (smoke settings are not in job configs).
if STUDY.resolve() != (SMOKE_STUDY if runner.SMOKE else REAL_STUDY).resolve():
    raise RuntimeError(f"refusing to run in {STUDY}: fs77 uses {SMOKE_STUDY if runner.SMOKE else REAL_STUDY}")
if runner.DATA != prep.DEFAULT_OUT:
    raise RuntimeError(f"the plan uses the csv_v3 data ({prep.DEFAULT_OUT}), not {runner.DATA}")
if runner.FEATURE_GROUPS is not None:
    # run_job would add feature_groups to every config and feature_columns would ignore each arm's feature_list.
    raise RuntimeError("refusing to run with CASCADE_FEATURE_GROUPS set: every arm would train on the same live groups")

BASE_JOB = "77_tune_tabm_ne_t008"          # tuned TabM-NE config (cascade selection on role 0)
P_JOB = "77_oof_tuned_tabm_ne_fold0"       # ranking P source: same config, fit on fold-0 rows
G_JOB = "77_oof_tuned_xgboost_fold0"       # ranking G source: early-stopped on fold-0 training rows
SEEDS = (142, 143, 144)
CONFIRM_SEEDS = (151, 152, 153)
FINAL_SEED = 42
SIZES, FP_DROP = (300, 200), 20
ARM_ORDER = ("P300", "P200", "G300", "G200", "FP20")
FP_ARM = f"FP{FP_DROP}"
P_PER_CLASS, P_REPEATS, FP_PER_CLASS = 300, (0, 1), 2500
P_BATCH = 4096
if runner.SMOKE:                   # CASCADE_SMOKE=1: wiring check only (runner: 300 fit rows/class, 1 epoch)
    SEEDS, CONFIRM_SEEDS = (142,), (151,)
    P_PER_CLASS, P_REPEATS, FP_PER_CLASS = 20, (0,), 250


def _log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ data
class Rows:
    def __init__(self):
        a = runner.task_arrays("77")
        self.y_val, self.y_train = a["y_val"], a["y_train"]
        self.scen = runner._arr("scenario_validation")
        roles = runner._arr("validation_roles")
        self.tune, self.tune_seen, self.tune_unseen = arms.role_split(roles, self.scen, runner._arr("scenario_train"))
        stored = runner._arr("validation_scenario_seen").astype(bool)[self.tune]
        if not np.array_equal(stored, np.isin(self.tune, self.tune_seen)):
            raise RuntimeError("role-0 seen split differs from validation_scenario_seen.npy")
        self.names = runner.manifest()["feature_names"]
        self.n_classes = len(a["classes"])


def cascade_job(job_id: str, family: str, rows: str) -> dict:
    """A finished cascade v3 job (read-only), checked before anything relies on it."""
    path = CASCADE / "jobs" / job_id / "result.json"
    if not path.exists():
        raise FileNotFoundError(path)
    r = json.loads(path.read_text(encoding="utf-8"))
    if r.get("status") != "complete" or r["family"] != family or r["task"] != "77" or r["rows"] != rows:
        raise RuntimeError(f"{job_id}: not a complete 77-class {family} job on {rows} rows")
    work = CASCADE / r["work_dir"]
    model = work / "model" / ("model.pt" if family == "tabm_ne" else "model.ubj")
    for need in (model, work / "model" / "fit_record.json", work / "predict_done.json"):
        if not need.exists():
            raise FileNotFoundError(need)
    return {"job": job_id, "result": r, "work": work, "model": model,
            "result_json_sha256": _sha256(path), "model_sha256": _sha256(model)}


def base_config() -> dict:
    """The tuned cascade config, copied verbatim from its result.json (frozen with that file's sha256)."""
    path = CASCADE / "jobs" / BASE_JOB / "result.json"
    r = json.loads(path.read_text(encoding="utf-8"))
    if r.get("status") != "complete" or r["family"] != "tabm_ne" or r["task"] != "77":
        raise RuntimeError(f"{BASE_JOB} is not a complete 77-class tabm_ne job")
    return arms.freeze(STUDY / "base_config.json", {
        "job": BASE_JOB, "result_json": path.relative_to(runner.ROOT).as_posix(),
        "result_json_sha256": _sha256(path), "config": r["config"], "smoke": runner.SMOKE})


def make_config(base: dict, names: list[str]) -> dict:
    c = copy.deepcopy(base["config"])
    c.update(feature_list=list(names), feature_scope="csv")
    return c


# -------------------------------------------------------------------------------- rankings (fold 0 only)
def worker_rank_p(out: Path) -> None:
    """Subprocess entry (CUDA): permutation importance of the fold-0 tuned TabM-NE on fold-0 OOF rows.

    The TabM preprocessor maps raw column j to transformed columns j and n+j, row by row, so permuting those two
    transformed columns equals permuting the raw column; the transform runs once.
    """
    from framework.training import tabm_fit
    from framework.training.csv_v3 import adapters_tabm
    torch = adapters_tabm._require_cuda()
    job = cascade_job(P_JOB, "tabm_ne", "fold0")
    record = json.loads((job["work"] / "model" / adapters_tabm.RECORD).read_text(encoding="utf-8"))
    names = runner.manifest()["feature_names"]
    idx = np.load(job["work"] / "oof_index.npy")
    y_all = runner.task_arrays("77")["y_train"]
    pick = arms.per_class_sample(np.arange(len(idx)), y_all[idx], P_PER_CLASS, seed=0)
    rows, y = idx[pick], y_all[idx[pick]]
    X = np.asarray(np.load(runner.DATA / "X_train.npy", mmap_mode="r")[rows], dtype=np.float32)
    model, prep_, saved = tabm_fit.load_checkpoint(job["model"], "cuda")
    if list(saved["feature_names"]) != names:
        raise RuntimeError(f"{P_JOB}: checkpoint features differ from the v3 manifest")
    if record.get("parity_recovered_with_cpu_order"):
        adapters_tabm._cpu_order_on_cuda(model)      # the operation order the job's own GPU replay used
    model.eval()
    T = torch.from_numpy(prep_.transform(X)).to("cuda")
    yt = torch.from_numpy(y.astype(np.int64)).to("cuda")

    def score(values):
        with torch.inference_mode():
            p = torch.cat([model.predict_proba(values[s:s + P_BATCH]) for s in range(0, len(values), P_BATCH)]).float()
            acc = float((p.argmax(1) == yt).float().mean())
            ll = float(-torch.log(p.gather(1, yt[:, None]).clamp_min(1e-12)).mean())
        return acc, ll, p

    started = time.time()
    base_acc, base_ll, base_p = score(T)
    # The loaded model must reproduce the job's CPU OOF predictions on these rows.
    stored = np.load(job["work"] / "oof_proba.npy")[pick]
    agree = float(np.mean(base_p.cpu().numpy().argmax(1) == stored.argmax(1)))
    if agree < 0.999:
        raise RuntimeError(f"GPU reload agrees with the stored OOF argmax on only {agree:.4f} of rows")
    n = len(names)
    drops, rises = np.zeros((len(P_REPEATS), n)), np.zeros((len(P_REPEATS), n))
    for i, r in enumerate(P_REPEATS):
        rng = np.random.default_rng(r)
        for j in range(n):
            perm = torch.from_numpy(rng.permutation(len(rows))).to("cuda")
            Tp = T.clone()
            Tp[:, j], Tp[:, n + j] = T[perm, j], T[perm, n + j]
            acc, ll, _ = score(Tp)
            drops[i, j], rises[i, j] = base_acc - acc, ll - base_ll
        print(f"repeat {r}: done ({time.time() - started:.0f} s)", flush=True)
    np.save(out.with_name(out.stem + "_rows.npy"), rows)
    payload = {"names": names, "acc_drop": drops.mean(0).tolist(), "logloss_rise": rises.mean(0).tolist(),
               "acc_drop_per_repeat": drops.tolist(), "base_accuracy": base_acc, "base_logloss": base_ll,
               "stored_oof_argmax_agreement": agree, "rows": int(len(rows)), "classes": int(np.unique(y).size),
               "per_class": P_PER_CLASS, "sample_seed": 0, "repeat_seeds": list(P_REPEATS),
               "seconds": round(time.time() - started, 1), "device": torch.cuda.get_device_name(0)}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, out)


def permutation_ranking(names: list[str], folds: list[dict]) -> dict:
    out = STUDY / "ranking_P_scores.json"
    job = cascade_job(P_JOB, "tabm_ne", "fold0")
    if not np.array_equal(np.load(job["work"] / "oof_index.npy"), folds[0]["oof"]):
        raise RuntimeError(f"{P_JOB}: OOF rows differ from forward fold 0")
    if not out.exists():
        _log("ranking P: permutation importance on fold-0 OOF rows (GPU subprocess)")
        with open(STUDY / "ranking_P.log", "w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, "-m", "experiment.train_fs77_tabm", "_rank_p", str(out)],
                                  cwd=runner.ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=6 * 3600)
        if proc.returncode != 0:
            raise RuntimeError(f"ranking P failed (exit {proc.returncode}); see {STUDY / 'ranking_P.log'}")
    s = json.loads(out.read_text(encoding="utf-8"))
    rows = np.load(out.with_name(out.stem + "_rows.npy"))
    if s["names"] != names:
        raise RuntimeError("ranking P scores are not in v3 order")
    if s["per_class"] != P_PER_CLASS or s["repeat_seeds"] != list(P_REPEATS):
        raise RuntimeError(f"{out.name} was computed with other settings (smoke vs real?); refusing to reuse it")
    if not np.isin(rows, folds[0]["oof"]).all() or np.isin(rows, folds[1]["oof"]).any():
        raise RuntimeError("ranking P rows must be fold-0 OOF rows and never fold-1 OOF rows")
    drop, rise = dict(zip(names, s["acc_drop"])), dict(zip(names, s["logloss_rise"]))
    return {"ranking": arms.ranked(names, drop, rise), "acc_drop": drop, "logloss_rise": rise,
            "provenance": {"method": "permutation importance, mean accuracy drop (log-loss rise breaks ties)",
                           "job": P_JOB, "model_sha256": job["model_sha256"],
                           "result_json_sha256": job["result_json_sha256"], "scores_sha256": _sha256(out),
                           **{k: s[k] for k in s if k not in ("names", "acc_drop", "logloss_rise", "acc_drop_per_repeat")},
                           "rows": "fold-0 OOF", "sampled_rows": s["rows"]}}


def gain_ranking(names: list[str]) -> dict:
    import xgboost as xgb
    job = cascade_job(G_JOB, "xgboost", "fold0")
    if job["result"]["fit"].get("tune_source") != "training forward fold0 early-stop rows":
        raise RuntimeError(f"{G_JOB} was not early-stopped on fold-0 training rows")
    booster = xgb.Booster(model_file=str(job["model"]))
    if booster.num_features() != len(names):
        raise RuntimeError(f"{G_JOB}: {booster.num_features()} features, expected {len(names)}")
    gain = booster.get_score(importance_type="total_gain")
    keys = booster.feature_names or [f"f{j}" for j in range(len(names))]
    score = {n: float(gain.get(k, 0.0)) for k, n in zip(keys, names)}
    return {"ranking": arms.ranked(names, score), "total_gain": score,
            "provenance": {"method": "XGBoost total_gain; zero-gain features last in v3 order", "job": G_JOB,
                           "model_sha256": job["model_sha256"], "result_json_sha256": job["result_json_sha256"],
                           "rows": "fold-0 fit (early stop: fold-0 training early-stop rows)",
                           "zero_gain_features": int(sum(v == 0 for v in score.values()))}}


def fingerprint_ranking(R: Rows, names: list[str], folds: list[dict]) -> dict:
    fit = folds[0]["fit"]
    pick = arms.per_class_sample(fit, R.y_train[fit], FP_PER_CLASS, seed=0)
    if np.isin(pick, folds[1]["oof"]).any():
        raise RuntimeError("fingerprint rows overlap fold-1 OOF rows")
    X = np.asarray(np.load(runner.DATA / "X_train.npy", mmap_mode="r")[pick])
    score = arms.fingerprint_scores(X, R.y_train[pick], runner._arr("scenario_train")[pick], names)
    return {"ranking": arms.ranked(names, score), "score": score,
            "provenance": {"method": "I(f; scenario | class) / (I(f; class) + 1e-3), 32-bin plug-in MI, Miller-Madow; "
                                     "classes with >= 200 rows and >= 2 scenarios",
                           "rows": "fold-0 fit", "sampled_rows": int(len(pick)), "per_class": FP_PER_CLASS,
                           "sample_seed": 0}}


def rankings(R: Rows) -> dict:
    """P, G and FP over all v3 features, frozen together (rankings.json) before the first arm job."""
    path = STUDY / "rankings.json"

    def compute() -> dict:
        folds = runner.folds("77")
        p = permutation_ranking(R.names, folds)
        g = gain_ranking(R.names)
        _log("ranking G: done; ranking FP: scenario-fingerprint score on fold-0 fit rows")
        fp = fingerprint_ranking(R, R.names, folds)
        return {"P": p["ranking"], "G": g["ranking"], "FP": fp["ranking"],
                "scores": {"P_acc_drop": p["acc_drop"], "P_logloss_rise": p["logloss_rise"],
                           "G_total_gain": g["total_gain"], "FP_score": fp["score"]},
                "provenance": {"P": p["provenance"], "G": g["provenance"], "FP": fp["provenance"],
                               "plan": "ml_pipeline/notes_20261101/fs77_tabm_ne_plan.md", "smoke": runner.SMOKE}}

    out = arms.freeze(path, compute)
    if out["provenance"]["smoke"] != runner.SMOKE:
        raise RuntimeError(f"{path.name} was frozen in {'smoke' if out['provenance']['smoke'] else 'real'} mode")
    _log(f"rankings frozen ({path.name})")
    return out


# ----------------------------------------------------------------------------------------------- jobs
def fold_job(ctl, R: Rows, job_id: str, config: dict, seed: int) -> dict:
    r = ctl.run_job(job_id, "77", "tabm_ne", config, seed=seed, rows="fold1")
    if r is None or r.get("status") != "complete":
        raise RuntimeError(f"{job_id} failed")
    work = STUDY / r["work_dir"]
    idx = np.load(work / "oof_index.npy")
    out = {"job_id": job_id, "seed": seed, "work": str(work), "n_features": len(config["feature_list"]),
           **arms.role0_metrics(np.load(work / "val_proba.npy"), R.y_val, R.tune, R.tune_seen, R.tune_unseen,
                                R.n_classes),
           **arms.oof_metrics(np.load(work / "oof_proba.npy"), R.y_train[idx], R.n_classes)}
    out.update(best_epoch=r["fit"].get("best_epoch"), stop_reason=r["fit"].get("stop_reason"),
               wall_seconds=r.get("wall_seconds"))
    return out


def run_arm(ctl, R: Rows, base: dict, arm: str, names: list[str], seeds) -> list[dict]:
    return [fold_job(ctl, R, f"77_fs_{arm}_s{s}", make_config(base, names), s) for s in seeds]


def run_arm_adaptive(ctl, R: Rows, base: dict, arm: str, names: list[str], control: list[dict]) -> tuple[list, int | None]:
    """Seed by seed with the plan's early-drop rule (arms.early_drop)."""
    by = {r["seed"]: r for r in control}
    runs = []
    for s in SEEDS:
        if ctl.stopped():
            raise KeyboardInterrupt("STOP present")
        runs += run_arm(ctl, R, base, arm, names, (s,))
        if arms.early_drop([arms.J(r) - arms.J(by[r["seed"]]) for r in runs]):
            return runs, len(runs)
    return runs, None


def _hits(R: Rows, runs: list[dict]) -> np.ndarray:
    """Seed-averaged role-0 correctness per row."""
    return np.mean([np.load(Path(r["work"]) / "val_proba.npy")[R.tune].argmax(1) == R.y_val[R.tune] for r in runs], axis=0)


def summarize(R: Rows, control: list[dict], runs: list[dict], dropped: int | None, n_seeds: int) -> dict:
    by = {r["seed"]: r for r in control}
    s = {**arms.paired(control, runs), "n_features": runs[0]["n_features"], "seeds_run": len(runs),
         "dropped_after_seed": dropped, "runs": runs}
    s["admitted"] = dropped is None and arms.admitted(s, n_seeds)
    s["role0_diff_ci95"] = arms.scenario_diff_ci(_hits(R, runs), _hits(R, [by[r["seed"]] for r in runs]), R.scen[R.tune])
    return s


# ------------------------------------------------------------------------------------------------ main
def other_controllers() -> list[int]:
    """PIDs of another fs77 controller, ranking worker, or a runner job working in this study (one GPU job at a time)."""
    import psutil
    mine = {os.getpid()} | {p.pid for p in psutil.Process().parents()}
    busy = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = p.info["cmdline"] or []
        text = " ".join(cmd)
        if p.info["pid"] in mine:
            continue
        if ("experiment.train_fs77_tabm" in text and ("run" in cmd or "_rank_p" in cmd)) or \
                ("experiment.train_cascade_v3" in text and STUDY.name in text):
            busy.append(p.info["pid"])
    return busy


def quarantine_failed() -> None:
    """A crash marks the running job failed and the cache would re-raise it forever; move it aside so a restart retries."""
    for res in sorted((STUDY / "jobs").glob("*/result.json")):
        if json.loads(res.read_text(encoding="utf-8")).get("status") == "failed":
            dest = STUDY / "failed_jobs" / f"{res.parent.name}_{time.strftime('%Y%m%d_%H%M%S')}"
            dest.parent.mkdir(exist_ok=True)
            os.replace(res.parent, dest)
            _log(f"moved failed job {res.parent.name} to {dest.relative_to(STUDY)}; it will be retried")


def run() -> None:
    busy = other_controllers()
    if busy:
        raise RuntimeError(f"another fs77 controller or job is running (pids {busy}); one GPU job at a time")
    (STUDY / "controller_pid.json").write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    quarantine_failed()
    ctl = runner.Controller(log=_log)
    ctl.state(phase="running", smoke=runner.SMOKE)
    try:
        R = Rows()
        base = base_config()
        ctl.state(stage="rankings")
        rank = rankings(R)
        lists = arms.freeze(STUDY / "arms.json", arms.arm_lists(R.names, {k: rank[k] for k in ("P", "G", "FP")},
                                                                 SIZES, FP_DROP))
        ctl.state(stage="arms")
        control = run_arm(ctl, R, base, "C0", lists["C0"], SEEDS)
        results = {}
        for arm in ARM_ORDER:
            runs, dropped = run_arm_adaptive(ctl, R, base, arm, lists[arm], control)
            results[arm] = summarize(R, control, runs, dropped, len(SEEDS))
            _log(f"{arm}: J gain {100 * results[arm]['gain']:+.3f} pp, admitted {results[arm]['admitted']}"
                 + (f" (dropped after seed {dropped})" if dropped else ""))
        admitted = [a for a in ARM_ORDER if results[a]["admitted"]]
        chosen, combined = None, None
        if admitted:
            chosen = max(admitted, key=lambda a: results[a]["J"])
            topk = [a for a in admitted if a != FP_ARM]
            if FP_ARM in admitted and topk:
                # Plan: that top-k list minus the FP top 20; it replaces the single arm only with >= 0.1 pp more mean J.
                k_arm = max(topk, key=lambda a: results[a]["J"])
                combined = arms.freeze(STUDY / "combined_arm.json", {
                    "from": [k_arm, FP_ARM], "names": arms.drop_top(lists[k_arm], rank["FP"], FP_DROP)})
                ctl.state(stage="combined")
                runs = run_arm(ctl, R, base, "combined", combined["names"], SEEDS)
                results["combined"] = summarize(R, control, runs, None, len(SEEDS))
                if results["combined"]["J"] >= results[chosen]["J"] + arms.GAIN:
                    chosen = "combined"
        forced = False
        if chosen is None and runner.SMOKE:      # smoke only: exercise confirmation + final wiring
            chosen, forced = max(results, key=lambda a: results[a]["J"]), True
            _log(f"smoke: no arm admitted; forcing {chosen} through confirmation and final (wiring check only)")
        names_of = lambda a: combined["names"] if a == "combined" else lists[a]
        confirmation = None
        if chosen is not None:
            ctl.state(stage="confirmation", chosen=chosen)
            c0 = run_arm(ctl, R, base, "C0", lists["C0"], CONFIRM_SEEDS)
            runs = run_arm(ctl, R, base, chosen, names_of(chosen), CONFIRM_SEEDS)
            confirmation = summarize(R, c0, runs, None, len(CONFIRM_SEEDS))
            _log(f"confirmation {chosen}: J gain {100 * confirmation['gain']:+.3f} pp, passes {confirmation['admitted']}")
        confirmed = confirmation is not None and (confirmation["admitted"] or forced)
        decision = arms.freeze(STUDY / "decision.json", {
            "admitted": admitted, "chosen": chosen, "smoke_forced": forced, "confirmed": confirmed,
            "confirmation": None if confirmation is None else {k: v for k, v in confirmation.items() if k != "runs"},
            "names": names_of(chosen) if confirmed else None,
            "rule": "fold-1 decisions above stand; the all-row refit is calibrated on role 1 and reported on role 2 "
                    "(report-only); nothing is re-chosen on role 2"})
        fin = None
        if decision["confirmed"]:
            ctl.state(stage="final")
            fin = ctl.run_job(f"77_fs_final_{decision['chosen']}", "77", "tabm_ne", make_config(base, decision["names"]),
                              seed=FINAL_SEED, rows="full")
            if fin is None or fin.get("status") != "complete":
                raise RuntimeError("final refit failed")
        out = {"base": base, "rankings_provenance": rank["provenance"],
               "rankings_sha256": json.loads((STUDY / "rankings.json").read_text(encoding="utf-8"))["sha256"],
               "arms": {a: {"names_count": len(names_of(a))} | s for a, s in results.items()}, "control": control,
               "decision": decision, "final_job": None if fin is None else fin["job_id"]}
        (STUDY / "fs77.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        write_report(R, out, fin)
        ctl.state(phase="done", stage="report")
    except KeyboardInterrupt:
        ctl.state(phase="stopped")
        _log("stopped (STOP file); restart resumes from cached jobs")
    except Exception:
        ctl.state(phase="error")
        raise


# ---------------------------------------------------------------------------------------------- report
def _final_rows(R: Rows, fin: dict) -> list[str]:
    from framework.training.csv_v3 import report as reporting
    shipped = json.loads((CASCADE / "jobs" / BASE_JOB / "result.json").read_text(encoding="utf-8"))
    pct = lambda x: "-" if x is None else f"{100 * x:.2f}%"
    lines = ["| model | tuning acc | assessment acc (CI) | macro-F1 | seen / unseen | benign FAR | calibration | "
             "ECE raw -> cal (assess) |", "|---|---|---|---|---|---|---|---|"]
    for label, r in (("shipped TabM-NE (t008, 445 features)", shipped), (f"fs77 final ({fin['job_id']})", fin)):
        row = reporting._row(label, "", r)
        ci, ece = row["final_assessment_ci95"], row["assessment_ece"] or {}
        lines.append(f"| {label} | {pct(row['tuning_accuracy'])} | {pct(row['final_assessment_accuracy'])} "
                     f"({100 * ci[0]:.1f}-{100 * ci[1]:.1f}) | {100 * row['assessment_macro_f1_supported']:.2f} | "
                     f"{pct(row['assessment_seen_accuracy'])} / {pct(row['assessment_unseen_accuracy'])} | "
                     f"{pct(row['benign_false_alarm_rate'])} | {row['calibration']} | "
                     f"{ece.get('raw', float('nan')):.3f} -> {ece.get('calibrated', float('nan')):.3f} |")
    # Paired role-2 difference on the same rows (report-only; nothing is chosen on it).
    roles = runner._arr("validation_roles")
    a_rows = np.flatnonzero(roles == 2)
    proba = lambda job, r: np.load(job / "val_proba_cal.npy") if (job / "val_proba_cal.npy").exists() \
        else np.load(job.parent.parent / r["work_dir"] / "val_proba.npy")
    p_new, p_old = proba(runner.job_dir(fin["job_id"]), fin), proba(CASCADE / "jobs" / BASE_JOB, shipped)
    diff = (p_new[a_rows].argmax(1) == R.y_val[a_rows]).astype(float) - (p_old[a_rows].argmax(1) == R.y_val[a_rows])
    ci = arms.scenario_diff_ci(diff, np.zeros_like(diff), R.scen[a_rows])
    lines += ["", f"Paired assessment difference (fs77 final - shipped): {100 * diff.mean():+.2f} pp, scenario-bootstrap "
                  f"CI {100 * ci[0]:+.2f} to {100 * ci[1]:+.2f} pp (report-only)."]
    return lines


def write_report(R: Rows, o: dict, fin: dict | None) -> None:
    d = o["decision"]
    lines = ["# 77-class TabM-NE feature-selection test (csv_v3, 445 features)", "",
             "Plan: ml_pipeline/notes_20261101/fs77_tabm_ne_plan.md (pre-registered 2026-11-01). Validation only; "
             "held-out test untouched. Arms: fold-1 refits of the tuned config `77_tune_tabm_ne_t008` (fixed, not "
             f"re-tuned per subset), seeds {', '.join(map(str, SEEDS))}, paired with C0; J = 0.5 x role-0 acc + "
             "0.5 x fold-1 OOF acc. Seen/unseen = role-0 scenario present in training (prep.py).", ""]
    if runner.SMOKE:
        lines += ["**SMOKE RUN: wiring check only (300 fit rows per class, 1 epoch, one seed). No number here is a "
                  "result.**", ""]
    lines += ["## Arms (fold 1)", "",
              "| arm | features | seeds | mean J | J gain (pp) | SD (pp) | role-0 seen gain (pp) | role-0 unseen gain (pp) "
              "| role-0 diff CI (pp) | OOF macro-F1 change | admitted |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    c0 = o["control"]
    lines.append(f"| C0 (control) | {c0[0]['n_features']} | {len(c0)} | "
                 f"{100 * np.mean([arms.J(r) for r in c0]):.2f}% | | | | | | | |")
    for arm, s in o["arms"].items():
        ci = s["role0_diff_ci95"]
        note = f"no (dropped after seed {s['dropped_after_seed']})" if s["dropped_after_seed"] else \
            ("yes" if s["admitted"] else "no")
        lines.append(f"| {arm} | {s['n_features']} | {s['seeds_run']} | {100 * s['J']:.2f}% | {100 * s['gain']:+.3f} | "
                     f"{100 * s['gain_sd']:.3f} | {100 * s['seen_gain']:+.3f} | {100 * s['unseen_gain']:+.3f} | "
                     f"{100 * ci[0]:+.2f} to {100 * ci[1]:+.2f} | {s['oof_f1_change']:+.4f} | {note} |")
    lines += ["", "Admission (all must hold, paired over all seeds): mean J gain >= 0.1 pp and > 2 x SD; mean role-0 "
                  "unseen-scenario gain >= 0; mean fold-1 OOF macro-F1 change >= -0.005. Early drop: after seed 1 if "
                  "the J gain <= -0.1 pp, after seed 2 if the mean gain < 0. The role-0 CI is diagnostic only.", ""]
    if not d["admitted"]:
        lines.append("**Decision: no gain from feature selection; the shipped TabM-NE stays.**"
                     + (" (smoke: an arm was forced through the later stages for wiring only)" if d["smoke_forced"] else ""))
    else:
        lines.append(f"Admitted: {', '.join(d['admitted'])}. Chosen: {d['chosen']}.")
    conf = d.get("confirmation")
    if conf:
        lines += ["", f"Confirmation ({d['chosen']} vs C0, seeds {', '.join(map(str, CONFIRM_SEEDS))}): J gain "
                      f"{100 * conf['gain']:+.3f} pp (SD {100 * conf['gain_sd']:.3f}), unseen {100 * conf['unseen_gain']:+.3f} pp, "
                      f"seen {100 * conf['seen_gain']:+.3f} pp, OOF macro-F1 {conf['oof_f1_change']:+.4f}; passes: "
                      f"{conf['admitted']}." + ("" if d["confirmed"] or not d["admitted"] else " **Not confirmed.**")]
    lines += ["", "## Rankings (training forward fold 0 only; frozen before the first arm job)", "",
              f"`rankings.json` sha256 `{o['rankings_sha256']}`.", ""]
    rank = json.loads((STUDY / "rankings.json").read_text(encoding="utf-8"))["value"]
    for key in ("P", "G", "FP"):
        p = o["rankings_provenance"][key]
        detail = ", ".join(f"{k}={v}" for k, v in p.items() if k != "method")
        lines += [f"- **{key}**: {p['method']}. {detail}.", f"  Top 10: {', '.join(rank[key][:10])}."]
    lines += ["", "Caveat: the config is held fixed for every arm; a smaller input might prefer other settings.", ""]
    if fin is not None:
        lines += ["## Final model (all fit77 rows, early stop on role 0, calibration on role 1, assessment report-only)",
                  ""] + _final_rows(R, fin) + ["", "Adoption into the app is a separate owner decision.", ""]
    (STUDY / "REPORT_fs77.md").write_text("\n".join(lines), encoding="utf-8")
    _log("report written")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "start", "status", "stop", "_rank_p"])
    parser.add_argument("path", nargs="?")
    args = parser.parse_args(argv)
    STUDY.mkdir(parents=True, exist_ok=True)
    if args.command == "_rank_p":
        worker_rank_p(Path(args.path))
    elif args.command == "run":
        (STUDY / "STOP").unlink(missing_ok=True)
        run()
    elif args.command == "start":
        busy = other_controllers()
        if busy:
            print(f"not started: another fs77 controller or job is running (pids {busy})")
            return 1
        (STUDY / "STOP").unlink(missing_ok=True)
        log = open(STUDY / "controller.log", "a", encoding="utf-8")
        flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0   # DETACHED_PROCESS | NEW_PROCESS_GROUP
        proc = subprocess.Popen([sys.executable, "-m", "experiment.train_fs77_tabm", "run"], cwd=runner.ROOT,
                                stdout=log, stderr=subprocess.STDOUT, creationflags=flags, env=os.environ.copy())
        (STUDY / "controller_pid.json").write_text(json.dumps({"pid": proc.pid, "started": time.time()}))
        print(f"controller started, pid {proc.pid}; log: {STUDY / 'controller.log'}")
    elif args.command == "stop":
        (STUDY / "STOP").write_text("stop requested\n", encoding="utf-8")
        print("STOP written")
    else:
        state = STUDY / "state.json"
        print(state.read_text(encoding="utf-8") if state.exists() else "no state yet")
        print(f"rankings frozen: {(STUDY / 'rankings.json').exists()}; "
              f"{len(list((STUDY / 'jobs').glob('*/result.json')))} finished jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
