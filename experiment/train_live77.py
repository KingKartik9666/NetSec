"""Live (Npcap) 77-class XGBoost: feature-group ablation -> frozen selection -> size-aware tuning -> refit + calibration.

    python -m experiment.train_live77 start     # detached controller (log: <study>/controller.log)
    python -m experiment.train_live77 run       # foreground
    python -m experiment.train_live77 status
    python -m experiment.train_live77 stop      # STOP file; halts between jobs

Plan and rules: ml_pipeline/PIPELINE.md, "Live 77-class model" (Gate B approved 2026-10-31). Selection reads the
validation tuning role (0) and training forward-fold OOF rows only; calibration uses role 1; role 2 (assessment) is
read only after ``selection.json`` and ``size.json`` exist. The held-out test partition is never read.
One GPU job at a time (runner subprocesses).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("CASCADE_STUDY", "artifacts/training/live77_20261031_v1")

import numpy as np  # noqa: E402

from framework.training.csv_v3 import adapters_trees, evaluate, live77, runner  # noqa: E402

STUDY = runner.STUDY
SEEDS = (42, 43, 44)
GAIN = 0.001                       # 0.1 pp
SIZE_TOLERANCE = 0.001             # smallest model within 0.1 pp of the best tuning accuracy
F1_TOLERANCE = 0.005               # ... that also keeps tuning and OOF macro-F1 within 0.005 of the best point
TRUNCATION_STEP = 10               # rounds
TUNE_MAX, TUNE_MIN, TUNE_PATIENCE = 20, 8, 6
if runner.SMOKE:                   # CASCADE_SMOKE=1: wiring check only (tiny rows, 4 rounds, own folder)
    SEEDS, (TUNE_MAX, TUNE_MIN, TUNE_PATIENCE) = (42, 43), (3, 2, 1)
SEVEN = ["network_packets_all_count", "network_tcp-flags-ack_count", "network_tcp-flags-fin_count",
         "network_tcp-flags-psh_count", "network_tcp-flags-rst_count", "network_tcp-flags-syn_count",
         "network_tcp-flags-urg_count", "window_duration_seconds"]
EMULATED = ("port_reported", "unknown_formula")
REPORT_ONLY = ("time_delta", "protocol")


def _log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


# ------------------------------------------------------------------------------------------------ data
class Rows:
    def __init__(self):
        a = runner.task_arrays("77")
        self.y_val, self.y_train = a["y_val"], a["y_train"]
        self.roles = runner._arr("validation_roles")
        self.seen = runner._arr("validation_scenario_seen").astype(bool)
        self.scen = runner._arr("scenario_validation")
        self.tune = np.flatnonzero(self.roles == 0)
        self.tune_unseen = np.flatnonzero((self.roles == 0) & ~self.seen)
        self.names = runner.manifest()["feature_names"]
        self.n_classes = len(a["classes"])


def _f1(y, pred, n):
    f1, _ = evaluate._f1_per_class(np.asarray(y, int), np.asarray(pred, int), n)
    return float(f1[np.bincount(y, minlength=n) > 0].mean())


def tuning_metrics(R: Rows, proba: np.ndarray) -> dict:
    """Role-0 numbers only (never assessment)."""
    pred = proba.argmax(1)
    t, u = R.tune, R.tune_unseen
    return {"tune_acc": float(np.mean(pred[t] == R.y_val[t])), "tune_f1": _f1(R.y_val[t], pred[t], R.n_classes),
            "tune_unseen_acc": float(np.mean(pred[u] == R.y_val[u]))}


def oof_metrics(R: Rows, work: Path) -> dict:
    idx, p = np.load(work / "oof_index.npy"), np.load(work / "oof_proba.npy")
    y, pred = R.y_train[idx], p.argmax(1)
    return {"oof_acc": float(np.mean(pred == y)), "oof_f1": _f1(y, pred, R.n_classes)}


def emulate(X: np.ndarray, names: list[str], group: str) -> np.ndarray:
    """Replace a group's dataset values with what the live extractor will compute (see PIPELINE.md plan 2d)."""
    X = X.copy()
    col = {n: i for i, n in enumerate(names)}
    n = X[:, col["network_packets_all_count"]]
    if group == "port_reported":
        for d in ("all", "dst", "src"):
            c = col[f"network_ports_{d}_count"]
            X[:, c] = np.minimum(X[:, c], 65535)
        ports = X[:, col["network_ports_all_count"]]
        with np.errstate(divide="ignore", invalid="ignore"):
            X[:, col["network_packets_per_reported_port"]] = np.where(ports > 0, n / ports, np.nan)
    elif group == "unknown_formula":
        dur = X[:, col["window_duration_seconds"]]
        with np.errstate(divide="ignore", invalid="ignore"):
            X[:, col["network_interval-packets"]] = np.where(n > 1, 1000 * dur / n, 0)
            frag = X[:, col["network_fragmented-packets"]]
            X[:, col["network_fragmentation-score"]] = np.where(frag > 0, np.log(np.maximum(n, 1)) + 1.31, 0)
    else:
        raise ValueError(group)
    return X


# ----------------------------------------------------------------------------------------------- jobs
def base_config() -> dict:
    return adapters_trees.default_config("xgboost", "77")


def fold_job(ctl, R: Rows, job_id: str, config: dict, seed: int) -> dict:
    r = ctl.run_job(job_id, "77", "xgboost", config, seed=seed, rows="fold1")
    if r is None or r.get("status") != "complete":
        raise RuntimeError(f"{job_id} failed")
    work = STUDY / r["work_dir"]
    out = {"job_id": job_id, "seed": seed, "work": str(work),
           **tuning_metrics(R, np.load(work / "val_proba.npy")), **oof_metrics(R, work)}
    out["best_round"] = r["fit"].get("trainer", {}).get("best_iteration")
    out["stop_reason"] = r["fit"].get("stop_reason")
    return out


def arm_config(groups=None, feature_list=None) -> dict:
    c = base_config()
    if groups is not None:
        c["feature_groups"] = list(groups)
    else:
        c["feature_list"] = list(feature_list)
    return c


def emulated_tune_acc(R: Rows, job: dict, groups, group: str) -> float:
    names = live77.feature_names(groups)
    X = np.asarray(np.load(runner.DATA / "X_validation.npy", mmap_mode="r")[R.tune])[:, live77.columns(groups)]
    p = adapters_trees.predict_cpu("xgboost", Path(job["work"]) / "model", emulate(X, names, group))
    return float(np.mean(p.argmax(1) == R.y_val[R.tune]))


def ablation(ctl, R: Rows) -> dict:
    path = STUDY / "selection.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    exact = list(live77.EXACT)
    arms = {"exact": exact, **{f"+{g}": exact + [g] for g in live77.APPROXIMATE}}
    results = {}
    _log("ablation: baselines and arms (fold 1, tuning-role comparison)")
    results["seven_counts"] = [fold_job(ctl, R, "77_live_seven_counts_s42", arm_config(feature_list=SEVEN), 42)]
    for seed in SEEDS:
        for arm, groups in arms.items():
            results.setdefault(arm, []).append(fold_job(ctl, R, f"77_live_{arm.strip('+')}_s{seed}",
                                                        arm_config(groups), seed))
    for g in REPORT_ONLY:
        results[f"+{g}"] = [fold_job(ctl, R, f"77_live_{g}_s42", arm_config(exact + [g]), 42)]

    base = {s["seed"]: s for s in results["exact"]}
    sd_oof = float(np.std([s["oof_f1"] for s in results["exact"]], ddof=1))
    table, admitted = {}, []
    for g in live77.APPROXIMATE:
        runs = results[f"+{g}"]
        d = np.array([r["tune_acc"] - base[r["seed"]]["tune_acc"] for r in runs])
        du = np.array([r["tune_unseen_acc"] - base[r["seed"]]["tune_unseen_acc"] for r in runs])
        doof = np.array([r["oof_f1"] - base[r["seed"]]["oof_f1"] for r in runs])
        row = {"gain_mean": float(d.mean()), "gain_sd": float(d.std(ddof=1)), "unseen_gain_mean": float(du.mean()),
               "oof_f1_change_mean": float(doof.mean())}
        ok = d.mean() >= GAIN and d.mean() > 2 * d.std(ddof=1) and du.mean() >= 0 and doof.mean() >= -2 * sd_oof
        if ok and g in EMULATED:
            de = np.array([emulated_tune_acc(R, r, arms[f"+{g}"], g) - base[r["seed"]]["tune_acc"] for r in runs])
            row["emulated_gain_mean"] = float(de.mean())
            ok = de.mean() >= GAIN
        row["admitted"] = bool(ok)
        table[g] = row
        if ok:
            admitted.append(g)
    chosen = exact + admitted[:1]
    if len(admitted) >= 2:
        combo = exact + admitted
        results["combined"] = [fold_job(ctl, R, f"77_live_combined_s{s}", arm_config(combo), s) for s in SEEDS]
        dc = np.mean([r["tune_acc"] - base[r["seed"]]["tune_acc"] for r in results["combined"]])
        best_single = max(admitted, key=lambda g: table[g]["gain_mean"])
        chosen = combo if dc >= GAIN and dc >= table[best_single]["gain_mean"] else exact + [best_single]
        table["combined"] = {"gain_mean": float(dc), "kept": chosen == combo}
    elif admitted:
        chosen = exact + admitted
    names = live77.feature_names(chosen)
    selection = {"groups": chosen, "n_features": len(names), "feature_names": names,
                 "feature_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
                 "admission": table, "exact_oof_f1_sd": sd_oof, "runs": results,
                 "rule": "PIPELINE.md live plan step 2", "frozen": time.strftime("%Y-%m-%d %H:%M:%S")}
    path.write_text(json.dumps(selection, indent=1), encoding="utf-8")
    _log(f"selection frozen: {chosen} ({len(names)} features)")
    return selection


# ------------------------------------------------------------------------------------------- tuning
# Owner 2026-10-31: "train the best possible version of model , multiple iterations of training , optimization ,
# tuning , bayesian search optuna etc". Stage 1 broad multivariate TPE, stage 2 TPE in the region of the stage-1
# top 5, then seed repeats of the 5 best configs; size and config are chosen on seed-averaged curves.
BOUNDS = {  # name: (low, high, log)
    "max_depth": (3, 12, False), "lr": (0.01, 0.3, True), "min_child_weight": (0.5, 100.0, True),
    "subsample": (0.5, 1.0, False), "colsample_bytree": (0.3, 1.0, False), "colsample_bynode": (0.5, 1.0, False),
    "reg_lambda": (1e-3, 50.0, True), "reg_alpha": (1e-3, 10.0, True), "gamma": (1e-4, 2.0, True),
}
CATEGORIES = {"max_bin": [128, 256, 512], "weight_strategy": ["none", "sqrt_balanced"]}
# Owner 2026-10-31: "instead of extensive tuning , do adaptive and apply droputs if gains become marginal".
STAGES = (("broad", 40, 12, 8), ("local", 15, 6, 5))  # name, max trials, min trials, patience (stop: no +0.1 pp)
MAX_CONSECUTIVE_FAILURES = 3
TOP_K, REPEAT_SEEDS = 3, (43, 44)
# Trial pruning: compare a running trial's best fold-1 early-stop accuracy (training rows, never role 0) with finished
# trials at the same learning progress (rounds x learning rate); stop it after `strikes` checks `margin` below median.
PRUNE = {"min_refs": 6, "warmup": 10.0, "margin": 0.005, "strikes": 3, "poll_seconds": 30}
if runner.SMOKE:
    STAGES, TOP_K, REPEAT_SEEDS = (("broad", 3, 2, 1), ("local", 2, 1, 1)), 2, (43,)
PARAMS = tuple(BOUNDS) + tuple(CATEGORIES)


def config_from(params: dict, groups) -> dict:
    c = base_config()
    c.update(n_estimators=4000, runtime_seconds=5400, feature_groups=list(groups))
    c.update({k: params[k] for k in PARAMS if k in params})
    c["early_stopping_rounds"] = int(max(50, np.ceil(2 / c["lr"])))   # patience scales with the learning rate
    return c


def objective(job: dict) -> float:
    """Variance-reduced selection value: role-0 (shifted, 14.5k rows) and fold-1 OOF (all 77 classes) accuracy."""
    return 0.5 * job["tune_acc"] + 0.5 * job["oof_acc"] + 1e-4 * job["tune_f1"]


def _key(params: dict) -> str:
    return json.dumps({k: params[k] for k in sorted(params)}, sort_keys=True)


def suggest(trial, bounds: dict, categories: dict) -> dict:
    out = {}
    for k, (lo, hi, log) in bounds.items():
        out[k] = trial.suggest_int(k, int(lo), int(hi)) if k == "max_depth" else trial.suggest_float(k, lo, hi, log=log)
    for k, values in categories.items():
        out[k] = trial.suggest_categorical(k, values)
    return out


def local_space(top: list[dict]) -> tuple[dict, dict]:
    """Ranges spanned by the stage-1 top trials, widened by 25% (log scale where log), clipped to BOUNDS."""
    bounds = {}
    for k, (lo, hi, log) in BOUNDS.items():
        v = np.array([t["params"][k] for t in top], float)
        f = np.log if log else (lambda x: x)
        g = np.exp if log else (lambda x: x)
        a, b = f(v.min()), f(v.max())
        pad = 0.25 * max(b - a, (f(hi) - f(lo)) * 0.05)
        new_lo, new_hi = max(lo, float(g(a - pad))), min(hi, float(g(b + pad)))
        if k == "max_depth":
            new_lo, new_hi = int(np.floor(new_lo)), int(np.ceil(new_hi))
        bounds[k] = (new_lo, new_hi, log)
    cats = {k: sorted({t["params"][k] for t in top}, key=str) for k in CATEGORIES}
    return bounds, cats


def _best_curve(job: dict) -> np.ndarray | None:
    path = Path(job["work"]) / "model" / "history.json"
    if not path.exists():
        return None
    acc = [h["validation_accuracy"] for h in json.loads(path.read_text(encoding="utf-8"))]
    return np.maximum.accumulate(np.array(acc, float)) if acc else None


def _prune_watch(job_id: str, lr: float, refs: list, stop: threading.Event) -> None:
    """Write PRUNE into the running fit's model folder when it trails the finished trials (see PRUNE)."""
    if len(refs) < PRUNE["min_refs"]:
        return
    best, strikes = -1.0, 0
    while not stop.wait(PRUNE["poll_seconds"]):
        attempts = sorted(runner.job_dir(job_id).glob("attempt_*/model/progress.json"))
        if not attempts:
            continue
        try:
            prog = json.loads(attempts[-1].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        r, best = int(prog["round"]), max(best, float(prog["validation_accuracy"]))
        if r * lr < PRUNE["warmup"]:
            continue
        ref = float(np.median([c[min(max(int(round(r * lr / ref_lr)) - 1, 0), len(c) - 1)] for ref_lr, c in refs]))
        strikes = strikes + 1 if best < ref - PRUNE["margin"] else 0
        if strikes >= PRUNE["strikes"]:
            (attempts[-1].parent / "PRUNE").write_text(f"round {r}: best {best:.4f} < median {ref:.4f} - margin")
            _log(f"{job_id}: pruned at round {r} (best {best:.4f}, median {ref:.4f})")
            return


def run_stage(ctl, R: Rows, groups, name: str, bounds: dict, cats: dict, limits, enqueue=None,
              warm: list[dict] | None = None) -> list[dict]:
    import optuna
    from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    max_trials, min_trials, patience = limits
    storage = f"sqlite:///{(STUDY / 'optuna_live77.sqlite3').as_posix()}"
    probe = optuna.create_study(direction="maximize", study_name=f"live77_{name}", storage=storage, load_if_exists=True)
    # Seed depends on the trial count so a restart does not replay earlier startup suggestions.
    sampler = optuna.samplers.TPESampler(seed=42 + len(probe.trials), multivariate=True,
                                         n_startup_trials=10 if warm is None else 2)
    study = optuna.create_study(direction="maximize", study_name=f"live77_{name}", storage=storage,
                                load_if_exists=True, sampler=sampler)
    dists = {k: IntDistribution(int(lo), int(hi)) if k == "max_depth" else FloatDistribution(lo, hi, log=log)
             for k, (lo, hi, log) in bounds.items()} | {k: CategoricalDistribution(v) for k, v in cats.items()}
    if not study.trials:
        if warm:   # completed broad trials inside this box warm-start TPE (no retraining)
            for w in warm:
                inside = all(w["params"][k] in cats[k] if k in cats else bounds[k][0] <= w["params"][k] <= bounds[k][1]
                             for k in dists)
                if inside:
                    study.add_trial(optuna.trial.create_trial(params={k: w["params"][k] for k in dists},
                                                              distributions=dists, value=objective(w),
                                                              user_attrs={"job": w, "warm": True}))
        elif enqueue:
            study.enqueue_trial(enqueue)
    done = lambda: [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    own = lambda: [t for t in done() if not t.user_attrs.get("warm")]

    def evaluate(number: int, params: dict):
        job_id = f"77_live_{name}_t{number:03d}"
        refs = [(t.params["lr"], c) for t in done() if (c := _best_curve(t.user_attrs["job"])) is not None]
        stop = threading.Event()
        watcher = threading.Thread(target=_prune_watch, args=(job_id, params["lr"], refs, stop), daemon=True)
        watcher.start()
        try:
            return fold_job(ctl, R, job_id, config_from(params, groups), 42)
        except RuntimeError as exc:
            _log(f"{name} t{number:03d}: failed ({exc})")
            return None
        finally:
            stop.set()
            watcher.join(5)

    # Trials left RUNNING by a stop or crash are re-evaluated with their own params (a cache hit if they finished).
    for t in [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]:
        job = evaluate(t.number, t.params)
        if job is None:
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)
        else:
            optuna.trial.Trial(study, t._trial_id).set_user_attr("job", job)
            study.tell(t.number, objective(job))

    def flat() -> bool:
        vals = [t.value for t in sorted(own(), key=lambda t: t.number)]
        return len(vals) >= min_trials and max(vals[-patience:]) < max(vals[:-patience] or [-1]) + GAIN

    failures = 0
    while len(own()) < max_trials and not flat():
        if ctl.stopped():
            raise KeyboardInterrupt("STOP present")
        trial = study.ask()
        job = evaluate(trial.number, suggest(trial, bounds, cats))
        if job is None:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(f"{name}: {failures} consecutive failed trials")
            continue
        failures = 0
        trial.set_user_attr("job", job)
        study.tell(trial, objective(job))
        _log(f"{name} t{trial.number:03d}: done ({len(own())} complete)")
    return [t.user_attrs["job"] | {"params": t.params, "trial": f"{name}{t.number:03d}"} for t in own()]


def tune(ctl, R: Rows, groups) -> list[dict]:
    """Both search stages, then seed repeats of the TOP_K configs. Returns one entry per config with all seeds."""
    b = base_config()
    start = {k: b[k] for k in ("max_depth", "lr", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda",
                               "reg_alpha")} | {"colsample_bynode": 0.8, "gamma": 1e-4, "max_bin": 256,
                                                "weight_strategy": "none"}
    broad = run_stage(ctl, R, groups, STAGES[0][0], BOUNDS, CATEGORIES, STAGES[0][1:], enqueue=start)
    bounds, cats = local_space(sorted(broad, key=objective, reverse=True)[:TOP_K])
    local = run_stage(ctl, R, groups, STAGES[1][0], bounds, cats, STAGES[1][1:], warm=broad)
    unique = {}
    for t in sorted(broad + local, key=objective, reverse=True):
        unique.setdefault(_key(t["params"]), t)
    configs = []
    for t in list(unique.values())[:TOP_K]:
        runs = [t] + [fold_job(ctl, R, f"77_live_repeat_{t['trial']}_s{s}", config_from(t["params"], groups), s)
                      for s in REPEAT_SEEDS]
        configs.append({"trial": t["trial"], "params": t["params"], "runs": runs,
                        "mean_tune_acc": float(np.mean([r["tune_acc"] for r in runs])),
                        "mean_objective": float(np.mean([objective(r) for r in runs]))})
        _log(f"repeats {t['trial']}: done")
    (STUDY / "tuning.json").write_text(json.dumps({"broad": broad, "local": local, "local_space": {
        "bounds": bounds, "categories": cats}, "configs": configs}, indent=1, default=str), encoding="utf-8")
    return configs


def _curve(R: Rows, work: Path, cols, dm, y, X_train) -> list[dict]:
    """Role-0 accuracy/macro-F1 and fold-1 OOF macro-F1 at every 10-round truncation of one fitted model."""
    import xgboost as xgb
    booster = xgb.Booster(model_file=str(work / "model" / "model.ubj"))
    booster.set_param({"device": "cpu", "nthread": 4})
    rounds = booster.num_boosted_rounds()
    nodes_per_tree = [d.count("\n") for d in booster.get_dump()]
    cum_nodes = np.cumsum(np.add.reduceat(nodes_per_tree, np.arange(0, len(nodes_per_tree), R.n_classes)))
    oof_idx = np.load(work / "oof_index.npy") if (work / "oof_index.npy").exists() else None
    dm_oof = None if oof_idx is None else xgb.DMatrix(np.asarray(X_train[oof_idx])[:, cols], nthread=4)
    margin = np.zeros((len(y), R.n_classes))
    margin_oof = None if oof_idx is None else np.zeros((len(oof_idx), R.n_classes))
    points = []
    edges = list(range(0, rounds, TRUNCATION_STEP)) + [rounds]
    for a, b in zip(edges[:-1], edges[1:]):
        margin += booster.predict(dm, iteration_range=(a, b), output_margin=True)
        p = {"rounds": b, "nodes": int(cum_nodes[b - 1]), "tune_acc": float(np.mean(margin.argmax(1) == y)),
             "tune_f1": _f1(y, margin.argmax(1), R.n_classes)}
        if dm_oof is not None:
            margin_oof += booster.predict(dm_oof, iteration_range=(a, b), output_margin=True)
            p["oof_f1"] = _f1(R.y_train[oof_idx], margin_oof.argmax(1), R.n_classes)
        points.append(p)
    return points


def _latency(path: Path, rounds: int, X: np.ndarray) -> float:
    import xgboost as xgb
    booster = xgb.Booster(model_file=str(path))[0:rounds]
    booster.set_param({"device": "cpu", "nthread": 1})
    times = []
    for i in range(200):
        s = time.perf_counter()
        booster.inplace_predict(X[i % len(X)][None, :])
        times.append(time.perf_counter() - s)
    return float(np.median(times) * 1000)


def size_choice(R: Rows, configs: list[dict], groups) -> dict:
    """Seed-averaged curves of the TOP_K configs; smallest mean node count within 0.1 pp of the best mean accuracy,
    with mean tuning macro-F1 and mean OOF macro-F1 within F1_TOLERANCE of the best point."""
    import xgboost as xgb
    path = STUDY / "size.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    cols = live77.columns(groups)
    X = np.asarray(np.load(runner.DATA / "X_validation.npy", mmap_mode="r")[R.tune])[:, cols]
    y = R.y_val[R.tune]
    dm = xgb.DMatrix(X, nthread=4)
    X_train = np.load(runner.DATA / "X_train.npy", mmap_mode="r")
    curves = []
    for c in configs:
        per_seed = [_curve(R, Path(r["work"]), cols, dm, y, X_train) for r in c["runs"]]
        n = min(len(s) for s in per_seed)
        mean = [{k: (int(np.mean([s[i][k] for s in per_seed])) if k in ("rounds", "nodes")
                     else float(np.mean([s[i][k] for s in per_seed]))) for k in per_seed[0][i]} for i in range(n)]
        for i, p in enumerate(mean):
            p["rounds"] = per_seed[0][i]["rounds"]
        curves.append({"trial": c["trial"], "params": c["params"], "points": mean})
    pts = [(c, p) for c in curves for p in c["points"]]
    best = max(p["tune_acc"] for _, p in pts)
    top = max((p for _, p in pts), key=lambda p: (p["tune_acc"], -p["nodes"]))
    eligible = [(p["nodes"], -p["tune_acc"], c["trial"], p["rounds"]) for c, p in pts
                if p["tune_acc"] >= best - SIZE_TOLERANCE and p["tune_f1"] >= top["tune_f1"] - F1_TOLERANCE
                and p["oof_f1"] >= top["oof_f1"] - F1_TOLERANCE]
    nodes, neg_acc, trial, rounds = min(eligible)
    chosen = next(c for c in curves if c["trial"] == trial)
    big = max(pts, key=lambda cp: (cp[1]["tune_acc"], -cp[1]["nodes"]))
    model = lambda t: Path(next(c["runs"][0]["work"] for c in configs if c["trial"] == t)) / "model" / "model.ubj"
    # Seed-ensemble gain (information only: a 3x larger model, not adopted by the size rule).
    cfg = next(c for c in configs if c["trial"] == trial)
    ens = np.mean([np.load(Path(r["work"]) / "val_proba.npy")[R.tune] for r in cfg["runs"]], axis=0)
    out = {"best_mean_tune_acc": best, "tolerance": SIZE_TOLERANCE, "f1_tolerance": F1_TOLERANCE,
           "chosen": {"trial": trial, "rounds": rounds, "nodes": nodes, "tune_acc": -neg_acc, "params": chosen["params"],
                      "latency_ms_per_window_python": _latency(model(trial), rounds, X)},
           "largest_best": {"trial": big[0]["trial"], "rounds": big[1]["rounds"], "nodes": big[1]["nodes"],
                            "tune_acc": big[1]["tune_acc"],
                            "latency_ms_per_window_python": _latency(model(big[0]["trial"]), big[1]["rounds"], X)},
           "seed_ensemble_info": {"chosen_config_mean_single_acc": cfg["mean_tune_acc"],
                                  "three_seed_average_acc": float(np.mean(ens.argmax(1) == y))},
           "curves": curves, "frozen": time.strftime("%Y-%m-%d %H:%M:%S")}
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    _log(f"size chosen: {trial}, {rounds} rounds, {nodes} nodes, tune {-neg_acc:.4f} (best {best:.4f})")
    return out


# ------------------------------------------------------------------------------------- final + report
def _final_config(groups, params: dict, rounds: int) -> dict:
    c = config_from(params, groups)
    c.update(n_estimators=rounds, early_stopping_rounds=rounds + 1, runtime_seconds=10800, checkpoint_rule="last")
    return c


def final(ctl, R: Rows, groups, size: dict) -> dict:
    """All-rows fit. A probe at 1.5x the fold-1 rounds (2.25x if its best lies in the last 10%) finds the round count
    for the larger training set: argmax of the 3-point moving mean of role-0 accuracy (the size trade-off was already
    made once, in size_choice). The final refit uses exactly that count and keeps its last round."""
    import xgboost as xgb
    p, r0 = size["chosen"]["params"], size["chosen"]["rounds"]
    cols = live77.columns(groups)
    X = np.asarray(np.load(runner.DATA / "X_validation.npy", mmap_mode="r")[R.tune])[:, cols]
    dm = xgb.DMatrix(X, nthread=4)
    probes = []
    for i, factor in enumerate((1.5, 2.25)):
        probe_rounds = int(np.ceil(factor * r0 / TRUNCATION_STEP) * TRUNCATION_STEP)
        probe = ctl.run_job(f"77_live_final_probe{i}", "77", "xgboost", _final_config(groups, p, probe_rounds), 42,
                            rows="full")
        if probe is None or probe.get("status") != "complete":
            raise RuntimeError("final probe failed")
        pts = _curve(R, STUDY / probe["work_dir"], cols, dm, R.y_val[R.tune], None)
        acc = np.array([q["tune_acc"] for q in pts])
        smooth = np.convolve(np.pad(acc, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        k = int(np.argmax(smooth))
        probes.append({"probe_rounds": probe_rounds, "curve": pts, "best_index": k, "rounds": pts[k]["rounds"]})
        if pts[k]["rounds"] < 0.9 * probe_rounds:
            break
    r = probes[-1]["rounds"]
    (STUDY / "final_rounds.json").write_text(json.dumps({"fold1_rounds": r0, "probes": probes, "chosen_rounds": r},
                                                        indent=1), encoding="utf-8")
    _log(f"final rounds: {r} (fold-1 {r0}, probes {[q['probe_rounds'] for q in probes]})")
    fin = ctl.run_job("77_live_final", "77", "xgboost", _final_config(groups, p, r), seed=42, rows="full")
    if fin is None or fin.get("status") != "complete":
        raise RuntimeError("final refit failed")
    saved = fin["fit"].get("trainer", {}).get("saved_rounds")
    if saved != r and not runner.SMOKE:   # smoke overrides n_estimators
        raise RuntimeError(f"final refit saved {saved} rounds, expected {r} (runtime cap?)")
    return fin


def final_size(R: Rows, fin: dict, groups) -> dict:
    """Nodes and single-window CPU latency of the refitted model itself (not the fold-1 trial)."""
    import xgboost as xgb
    booster = xgb.Booster(model_file=str(STUDY / fin["work_dir"] / "model" / "model.ubj"))
    booster.set_param({"device": "cpu", "nthread": 1})
    X = np.asarray(np.load(runner.DATA / "X_validation.npy", mmap_mode="r")[R.tune[:200]])[:, live77.columns(groups)]
    times = []
    for i in range(200):
        s = time.perf_counter()
        booster.inplace_predict(X[i][None, :])
        times.append(time.perf_counter() - s)
    return {"rounds": booster.num_boosted_rounds(), "nodes": int(sum(d.count("\n") for d in booster.get_dump())),
            "latency_ms_per_window_python": float(np.median(times) * 1000)}


def window_choice(R: Rows, proba: np.ndarray) -> dict:
    """Pre-registered rule (PIPELINE.md live plan step 6), tuning role only."""
    dur = np.asarray(np.load(runner.DATA / "X_validation.npy", mmap_mode="r")[:, R.names.index("window_duration_seconds")])
    pred = proba.argmax(1)
    t = R.tune
    common = set(range(R.n_classes))
    for d in range(1, 11):
        common &= set(np.unique(R.y_val[t][dur[t] == d]).tolist())
    common = np.array(sorted(common))
    rng = np.random.default_rng(0)
    rows = {}
    for d in range(1, 11):
        m = t[(dur[t] == d) & np.isin(R.y_val[t], common)]
        y, pr, sc = R.y_val[m], pred[m], R.scen[m]
        point = _common_f1(y, pr, common, R.n_classes)
        codes, inv = np.unique(sc, return_inverse=True)
        boots = []
        for _ in range(500):
            pick = rng.integers(0, len(codes), len(codes))
            sel = np.concatenate([np.flatnonzero(inv == k) for k in pick])
            boots.append(_common_f1(y[sel], pr[sel], common, R.n_classes))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows[d] = {"rows": int(len(m)), "macro_f1_common": point, "ci95": [float(lo), float(hi)],
                   "accuracy": float(np.mean(y == pr))}
    best = max(rows, key=lambda d: rows[d]["macro_f1_common"])
    chosen = min(d for d in rows if rows[d]["ci95"][1] >= rows[best]["ci95"][0])
    return {"common_classes": int(len(common)), "per_length": rows, "best_length": best, "chosen_length": chosen}


def _common_f1(y, pred, common, n):
    # Average over the common classes present in y: a scenario resample often drops whole classes, and
    # scoring an absent class as F1 = 0 pushed every bootstrap value (and the CI) below the point estimate.
    y = np.asarray(y, int)
    f1, _ = evaluate._f1_per_class(y, np.asarray(pred, int), n)
    present = np.intersect1d(common, y)
    return float(f1[present].mean()) if len(present) else float("nan")


def write_report(R: Rows, selection: dict, trials: list[dict], size: dict, fin: dict) -> None:
    rep, cal = fin["report"], fin.get("calibration", {})
    crep = cal.get("report", rep)
    work = STUDY / fin["work_dir"]
    cal_path = runner.job_dir("77_live_final") / "val_proba_cal.npy"
    proba = np.load(cal_path) if cal_path.exists() else np.load(work / "val_proba.npy")
    win = window_choice(R, proba)
    (STUDY / "window.json").write_text(json.dumps(win, indent=1), encoding="utf-8")
    fsize = final_size(R, fin, selection["groups"])
    (STUDY / "final_size.json").write_text(json.dumps(fsize, indent=1), encoding="utf-8")
    runs = selection["runs"]
    mean = lambda arm, k: float(np.mean([r[k] for r in runs[arm]]))
    maj = float(np.mean(R.y_val[R.tune] == 0))
    a, t = crep["assessment"], crep["tuning"]
    q = cal.get("quality", {})
    lines = [
        "# Live 77-class XGBoost (Npcap-matched features)", "",
        f"Validation only; held-out test untouched. Selection: tuning role; calibration: role 1; assessment: report only.", "",
        "## Final model", "",
        "| metric | tuning | assessment |", "|---|---|---|",
        f"| accuracy | {100*t['accuracy']:.2f}% | {100*a['accuracy']:.2f}% (CI {100*a['accuracy_ci95_scenario_bootstrap'][0]:.1f}-{100*a['accuracy_ci95_scenario_bootstrap'][1]:.1f}) |",
        f"| macro-F1 (supported) | {t['macro_f1_supported']:.4f} | {a['macro_f1_supported']:.4f} |",
        f"| benign false-alarm rate | {100*t['benign_false_alarm_rate']:.2f}% | {100*a['benign_false_alarm_rate']:.2f}% |",
        f"| attack missed as benign | {100*t['attack_missed_as_benign_rate']:.2f}% | {100*a['attack_missed_as_benign_rate']:.2f}% |",
        f"| ECE raw -> calibrated | {q.get('tuning', {}).get('raw', {}).get('ece', float('nan')):.4f} -> {q.get('tuning', {}).get('calibrated', {}).get('ece', float('nan')):.4f} | {q.get('assessment', {}).get('raw', {}).get('ece', float('nan')):.4f} -> {q.get('assessment', {}).get('calibrated', {}).get('ece', float('nan')):.4f} |",
        f"| assessment seen / unseen scenarios | | {100*crep['assessment_seen_scenarios']['accuracy']:.2f}% / {100*crep['assessment_unseen_scenarios']['accuracy']:.2f}% |",
        "",
        f"Features: {selection['n_features']} ({', '.join(selection['groups'])}). Calibration: {cal.get('method')}.",
        f"Final model size: {fsize['rounds']} rounds x 77 trees, depth {size['chosen']['params']['max_depth']}, "
        f"{fsize['nodes']:,} nodes, {fsize['latency_ms_per_window_python']:.2f} ms/window (Python CPU, 1 thread). "
        f"Chosen fold-1 point: {size['chosen']['nodes']:,} nodes, tuning {100*size['chosen']['tune_acc']:.2f}%. "
        f"The largest best-accuracy candidate: {size['largest_best']['rounds']} rounds, {size['largest_best']['nodes']:,} nodes, "
        f"{size['largest_best']['latency_ms_per_window_python']:.2f} ms, tuning {100*size['largest_best']['tune_acc']:.2f}% "
        f"(fold-1 models).",
        f"Live window: {win['chosen_length']} s (best {win['best_length']} s; rule: shortest length whose CI reaches the best's).", "",
        "## Baselines and ablation (fold 1, tuning role, mean of seeds)", "",
        "| arm | tuning acc | unseen-scenario acc | OOF macro-F1 |", "|---|---|---|---|",
        f"| majority (benign) | {100*maj:.2f}% | | |",
    ]
    for arm in runs:
        lines.append(f"| {arm} ({len(runs[arm])} seeds) | {100*mean(arm,'tune_acc'):.2f}% | "
                     f"{100*mean(arm,'tune_unseen_acc'):.2f}% | {mean(arm,'oof_f1'):.4f} |")
    tj = json.loads((STUDY / "tuning.json").read_text(encoding="utf-8"))
    fr = json.loads((STUDY / "final_rounds.json").read_text(encoding="utf-8"))
    ens = size["seed_ensemble_info"]
    lines += ["", "Admission: " + json.dumps(selection["admission"]), "",
              f"Tuning: {len(tj['broad'])} broad + {len(tj['local'])} local Optuna TPE trials (fold 1); top {len(trials)} configs "
              f""
              f"repeated over {1 + len(REPEAT_SEEDS)} seeds. Best seed-averaged fold-1 tuning accuracy {100*size['best_mean_tune_acc']:.2f}%; "
              f"chosen {size['chosen']['trial']} at {size['chosen']['rounds']} rounds ({100*size['chosen']['tune_acc']:.2f}%). "
              f"All-rows round check: probes {[q['probe_rounds'] for q in fr['probes']]} -> {fr['chosen_rounds']} rounds.",
              f"Seed ensemble (information only, {1 + len(REPEAT_SEEDS)}x size): single {100*ens['chosen_config_mean_single_acc']:.2f}% vs "
              f"{1 + len(REPEAT_SEEDS)}-seed average {100*ens['three_seed_average_acc']:.2f}%.", ""]
    (STUDY / "REPORT_live77.md").write_text("\n".join(lines), encoding="utf-8")
    _log("report written")


def run() -> None:
    ctl = runner.Controller(log=_log)
    ctl.state(phase="running")
    try:
        R = Rows()
        selection = ablation(ctl, R)
        groups = selection["groups"]
        configs = tune(ctl, R, groups)
        size = size_choice(R, configs, groups)
        fin = final(ctl, R, groups, size)
        write_report(R, selection, configs, size, fin)
        ctl.state(phase="done")
    except KeyboardInterrupt:
        ctl.state(phase="stopped")
        _log("stopped (STOP file); restart resumes from cached jobs")
    except Exception:
        ctl.state(phase="error")
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "start", "status", "stop"])
    args = parser.parse_args(argv)
    STUDY.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        (STUDY / "STOP").unlink(missing_ok=True)
        run()
    elif args.command == "start":
        (STUDY / "STOP").unlink(missing_ok=True)
        log = open(STUDY / "controller.log", "a", encoding="utf-8")
        flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0
        proc = subprocess.Popen([sys.executable, "-m", "experiment.train_live77", "run"], cwd=runner.ROOT,
                                stdout=log, stderr=subprocess.STDOUT, creationflags=flags, env=os.environ.copy())
        (STUDY / "controller_pid.json").write_text(json.dumps({"pid": proc.pid, "started": time.time()}))
        print(f"controller started, pid {proc.pid}; log: {STUDY / 'controller.log'}")
    elif args.command == "stop":
        (STUDY / "STOP").write_text("stop requested\n", encoding="utf-8")
        print("STOP written")
    else:
        state = STUDY / "state.json"
        print(state.read_text(encoding="utf-8") if state.exists() else "no state yet")
        print(f"{len(list((STUDY / 'jobs').glob('*/result.json')))} finished jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
