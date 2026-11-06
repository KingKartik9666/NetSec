"""Sequential GPU training controller for the cascade v3 study (77 -> 8 -> 84).

For each task, in order and to completion before the next task starts:

1. ``baseline``  every family once with its frozen recipe (seed 42);
2. ``oof``       two forward folds for each hybrid branch, then the three hybrids (blend weights fitted on
                 out-of-fold training predictions, chosen on the validation tuning role);
3. ``tune``      Optuna TPE (Bayesian) search per family, stopped early once gains flatten; tuned hybrid
                 branches are refolded and every branch combination / weighting is searched;
4. ``repeat``    selected configurations refitted at seeds 43 and 44;
5. ``report``    metrics on the frozen validation roles, written to ``REPORT_<task>.md``.

Every fit runs in its own subprocess on CUDA, and every reported number comes from a CPU reload in a separate
process. One job runs at a time. Creating ``STOP`` in the study folder halts the controller between jobs;
completed jobs are never repeated on restart. The held-out test partition is never read.
"""
from __future__ import annotations

import importlib
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from framework.training.csv_v3 import evaluate, prep
from framework.training.csv_v3.contract import TaskData

ROOT = prep.ROOT
# CASCADE_DATA: an alternative data folder with the same arrays (e.g. v3 plus appended derived live features).
DATA = ROOT / os.environ["CASCADE_DATA"] if os.environ.get("CASCADE_DATA") else prep.DEFAULT_OUT
# Smoke mode exercises the whole controller with tiny row samples and one-epoch fits in its own folder.
SMOKE = os.environ.get("CASCADE_SMOKE") == "1"
DEFAULT_STUDY = "artifacts/training/cascade_v3_20261030_v1"
STUDY = ROOT / (os.environ.get("CASCADE_SMOKE_STUDY", "artifacts/training/cascade_v3_smoke") if SMOKE
                else os.environ.get("CASCADE_STUDY", DEFAULT_STUDY))
# Live feature subset for a whole study (JSON list of live77 groups). Job ids and Optuna studies do not encode the
# subset, so each subset needs its own study folder; run_job injects it into every config and checks cache hits.
FEATURE_GROUPS = json.loads(os.environ["CASCADE_FEATURE_GROUPS"]) if os.environ.get("CASCADE_FEATURE_GROUPS") else None
if FEATURE_GROUPS is not None and STUDY == ROOT / DEFAULT_STUDY:
    raise RuntimeError("CASCADE_FEATURE_GROUPS needs its own CASCADE_STUDY folder")
SMOKE_ROWS_PER_CLASS = 300
SMOKE_OVERRIDES = {
    "xgboost": {"n_estimators": 4, "early_stopping_rounds": 2},
    "kan": {"epochs": 1, "patience": 1}, "hem": {"epochs": 1, "patience": 1},
    "tabm_ne": {"epochs": 1, "patience": 1}, "tabm_hier": {"epochs": 1, "patience": 1},
    "tabm_hier_ne": {"epochs": 1, "patience": 1}, "tabpack": {"epochs": 1, "patience": 1},
}
PYTHON = sys.executable
CLI = "experiment.train_cascade_v3"

ADAPTERS = {
    "random_forest": "adapters_trees", "xgboost": "adapters_trees",
    "kan": "adapters_neural", "hem": "adapters_neural",
    "tabm_ne": "adapters_tabm", "tabm_hier": "adapters_tabm", "tabm_hier_ne": "adapters_tabm",
    "tabpack": "adapters_tabm",
}
FAMILIES = {"77": list(ADAPTERS), "84": list(ADAPTERS), "8": ["xgboost"]}
HYBRIDS = {"tabm_xgboost": ["tabm_ne", "xgboost"], "tabm_xgboost_kan": ["tabm_ne", "xgboost", "kan"],
           "tabpack_xgboost": ["tabpack", "xgboost"]}
OOF_BRANCHES = ["tabm_ne", "xgboost", "kan", "tabpack"]
# random_forest is the fixed conventional control; tabpack is a fixed 8-member auxiliary pack.
NOT_TUNED = {"random_forest", "tabpack"}
TUNE_LIMITS = {"xgboost": 24, "kan": 16, "hem": 8, "tabm_ne": 12, "tabm_hier": 12, "tabm_hier_ne": 12}
PLATEAU = {"min_trials": 6, "patience": 6, "min_gain": 0.001}   # stop: no +0.1 pp in 6 consecutive trials
REPEAT_SEEDS = (43, 44)
# Owner decision 2026-10-31: no direct 84-class retrain. "84xgb" is one 84-class XGBoost baseline control and
# "cascade" scores 77 + specialist as 84 classes (REPORT_cascade84.md). Task "84" remains runnable explicitly.
TASK_ORDER = ("77", "8", "84xgb", "cascade")
# Optuna objective after calibration. The 77 studies already ran on the raw objective, so 77 keeps it and
# re-selects every family across all its trials by the calibrated value instead.
CALIBRATED_OBJECTIVE_TASKS = {"8", "84"}


# ------------------------------------------------------------------------------------------------- data
def manifest() -> dict:
    return json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))


def _arr(name: str, mmap: bool = False) -> np.ndarray:
    return np.load(DATA / f"{name}.npy", mmap_mode="r" if mmap else None, allow_pickle=False)


def task_arrays(task: str) -> dict:
    """Labels, class names, fit mask and tuning rows for a task. Labels are contiguous 0..C-1."""
    m = manifest()
    roles = _arr("validation_roles")
    if task == "8":
        y_tr, y_va = _arr("y8_train"), _arr("y8_validation")
        return dict(classes=m["classes8"], benign=None, y_train=y_tr, y_val=y_va, fit=_arr("fit8_train"),
                    tune_rows=np.flatnonzero((roles == 0) & (y_va >= 0)), val_rows=np.flatnonzero(y_va >= 0))
    y_tr, y_va = _arr(f"y{task}_train"), _arr(f"y{task}_validation")
    return dict(classes=m[f"classes{task}"], benign=0, y_train=y_tr, y_val=y_va, fit=_arr(f"fit{task}_train"),
                tune_rows=np.flatnonzero(roles == 0), val_rows=np.arange(len(y_va)))


def _targeted_only(rows: np.ndarray) -> np.ndarray:
    """Drop whole-network sweep scenarios (specialist option: validation recon rows are all targeted)."""
    names = np.array(manifest()["scenario_names"], dtype=object)
    scen = _arr("scenario_train")[rows]
    return rows[~np.char.endswith(names[scen].astype(str), "whole-network")]


def folds(task: str) -> list[dict]:
    """Two forward folds over all training rows, intersected with the task's fit mask."""
    from framework.training.hybrid_data import forward_folds
    a = task_arrays(task)
    raw = forward_folds(a["y_train"], _arr("window_start_train"), _arr("window_end_train"),
                        _arr("row_digest_train"))
    fit = a["fit"]
    out = []
    for f in raw:
        fold = {k: v[fit[v]] for k, v in f.items()}
        if np.unique(a["y_train"][fold["fit"]]).size != len(a["classes"]):
            raise RuntimeError("fold fit rows lost class coverage after the fit mask")
        out.append(fold)
    return out


FEATURE_SCOPES = ("live", "csv")


def feature_columns(config: dict) -> tuple[np.ndarray | None, list[str]]:
    """Optional feature subset (``feature_groups`` or ``feature_list`` in a job config); None means all v3 columns.

    ``feature_scope`` "live" (default) refuses listed features that cannot exist live (log/artifact groups); "csv"
    is for dataset-only studies and allows any v3 feature. Listed columns always come back in v3 order.
    """
    groups, listed = config.get("feature_groups"), config.get("feature_list")
    scope = config.get("feature_scope", "live")
    if scope not in FEATURE_SCOPES:
        raise ValueError(f"unknown feature_scope {scope!r}")
    if groups is None and listed is None:
        return None, manifest()["feature_names"]
    from framework.training.csv_v3 import live77
    if groups is not None:
        return live77.columns(groups), live77.feature_names(groups)
    names = manifest()["feature_names"]
    if len(set(listed)) != len(listed) or not set(listed) <= set(names):
        raise ValueError("feature_list must hold distinct v3 feature names")
    if scope == "live" and any(live77.RATING[live77.group_of(n)] == "excluded" for n in listed):
        raise ValueError("feature_list contains features that cannot exist live")
    cols = np.array(sorted(names.index(n) for n in listed), dtype=np.int64)
    return cols, [names[i] for i in cols]


def scenario_weights(spec: dict | None, fit_rows: np.ndarray, y_train: np.ndarray) -> np.ndarray | None:
    """Per-row training-weight multipliers (scenario metadata used as weights only, never as an input).

    ``{"kind": "scenario", "rho": r}``: w ~ n(class, scenario)^-r, renormalised so each class keeps its total weight.
    """
    if spec is None:
        return None
    if spec.get("kind") != "scenario":
        raise ValueError(f"unknown row_weight {spec}")
    rho = float(spec["rho"])
    y, scen = y_train[fit_rows], _arr("scenario_train")[fit_rows]
    key = y.astype(np.int64) * 100_000 + scen.astype(np.int64)
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    w = counts[inverse].astype(np.float64) ** -rho
    class_total = np.bincount(y, minlength=y.max() + 1).astype(np.float64)
    class_w = np.bincount(y, weights=w, minlength=y.max() + 1)
    return (w * (class_total[y] / class_w[y])).astype(np.float32)


def build_task_data(request: dict) -> tuple[TaskData, dict]:
    """Assemble the arrays one job trains on, from the request's task, rows and config."""
    task, rows_kind = request["task"], request["rows"]
    a = task_arrays(task)
    x_tr, x_va = _arr("X_train", mmap=True), _arr("X_validation", mmap=True)
    config = dict(request["config"])
    rows_policy = config.pop("rows_policy", "all")
    row_weight = config.pop("row_weight", None)
    cols, names = feature_columns(config)
    config.pop("feature_groups", None)
    config.pop("feature_list", None)
    config.pop("feature_scope", None)
    if rows_kind == "full":
        fit_rows = np.flatnonzero(a["fit"])
        X_tune, y_tune = np.asarray(x_va[a["tune_rows"]]), a["y_val"][a["tune_rows"]]
        extra = {"tune_source": "validation tuning role"}
    else:
        fold = folds(task)[int(rows_kind[-1])]
        fit_rows = fold["fit"]
        X_tune, y_tune = np.asarray(x_tr[fold["early_stop"]]), a["y_train"][fold["early_stop"]]
        extra = {"tune_source": f"training forward {rows_kind} early-stop rows", "oof_rows": fold["oof"]}
    if rows_policy == "targeted_only":
        fit_rows = _targeted_only(fit_rows)
    elif rows_policy != "all":
        raise ValueError(f"unknown rows_policy {rows_policy}")
    if SMOKE:
        rng = np.random.default_rng(0)
        y = a["y_train"][fit_rows]
        fit_rows = np.sort(np.concatenate([rng.permutation(fit_rows[y == c])[:SMOKE_ROWS_PER_CLASS]
                                           for c in np.unique(y)]))
        config.update(SMOKE_OVERRIDES.get(request["family"], {}))
    X_fit = np.asarray(x_tr[fit_rows])
    if cols is not None:
        X_fit, X_tune = X_fit[:, cols], X_tune[:, cols]
    data = TaskData(task=task, X_fit=np.ascontiguousarray(X_fit, dtype=np.float32),
                    y_fit=a["y_train"][fit_rows].astype(np.int64),
                    X_tune=np.ascontiguousarray(X_tune, dtype=np.float32), y_tune=y_tune.astype(np.int64),
                    classes=list(a["classes"]), feature_names=names,
                    benign_index=a["benign"], extras={"rows_policy": rows_policy,
                                                      "row_weight": scenario_weights(row_weight, fit_rows, a["y_train"])})
    return data, {**extra, "config": config, "fit_rows": int(len(fit_rows))}


def adapter(family: str):
    return importlib.import_module(f"framework.training.csv_v3.{ADAPTERS[family]}")


# ------------------------------------------------------------------------------------------ job workers
def job_dir(job_id: str) -> Path:
    return STUDY / "jobs" / job_id


def worker_fit(request_path: Path) -> None:
    """Subprocess entry: train one model on CUDA and write ``result.json``."""
    request = json.loads(request_path.read_text(encoding="utf-8"))
    out = request_path.parent
    data, extra = build_task_data(request)
    data.validate()
    record = adapter(request["family"]).fit(request["family"], data, extra["config"], request["seed"], out / "model")
    record.update(fit_rows=extra["fit_rows"], tune_source=extra["tune_source"], rows=request["rows"])
    (out / "fit_result.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def worker_predict(request_path: Path) -> None:
    """Subprocess entry: CPU reload; validation (and OOF) probabilities; replay check against fit time."""
    request = json.loads(request_path.read_text(encoding="utf-8"))
    out = request_path.parent
    a = task_arrays(request["task"])
    family = request["family"]
    module = adapter(family)
    x_va = _arr("X_validation", mmap=True)
    rows = a["val_rows"]
    cols, _ = feature_columns(request["config"])
    select = (lambda x: x) if cols is None else (lambda x: x[:, cols])
    proba = np.asarray(module.predict_cpu(family, out / "model", select(np.asarray(x_va[rows]))), dtype=np.float32)
    full = np.zeros((len(a["y_val"]), proba.shape[1]), dtype=np.float32)
    full[rows] = proba
    np.save(out / "val_proba.npy", full)
    if request["rows"] != "full":
        fold = folds(request["task"])[int(request["rows"][-1])]
        x_tr = _arr("X_train", mmap=True)
        np.save(out / "oof_index.npy", fold["oof"])
        np.save(out / "oof_proba.npy", np.asarray(module.predict_cpu(family, out / "model", select(np.asarray(x_tr[fold["oof"]]))),
                                                   dtype=np.float32))
    (out / "predict_done.json").write_text(json.dumps({"rows": int(len(rows))}), encoding="utf-8")


# ------------------------------------------------------------------------------------------- evaluation
def score(task: str, proba: np.ndarray) -> dict:
    """Report on the frozen validation roles; task 8 is scored on recon8 rows only."""
    a = task_arrays(task)
    rows = a["val_rows"]
    focus = None
    if task == "77":
        focus = [a["classes"].index(prep.MERGED), a["classes"].index("recon_ping-sweep")]
    elif task == "84":
        focus = [a["classes"].index(c) for c in prep.RECON8]
    return evaluate.report(a["y_val"][rows], proba[rows], _arr("validation_roles")[rows],
                           _arr("validation_scenario_seen")[rows], _arr("scenario_validation")[rows],
                           len(a["classes"]), a["benign"], focus)


def tune_value(report: dict) -> float:
    """Selection value on the validation tuning role only: accuracy, macro-F1 (supported) as tie-break."""
    t = report["tuning"]
    return t["accuracy"] + 1e-4 * t["macro_f1_supported"]


def calibration_rows(task: str) -> dict:
    """Role rows of a task, plus tuning/assessment rows whose scenario never appears in the calibration role."""
    rows = task_arrays(task)["val_rows"]
    roles, scen = _arr("validation_roles")[rows], _arr("scenario_validation")[rows]
    out = {name: rows[roles == code] for code, name in evaluate.ROLE_NAMES.items()}
    cal_scenarios = np.unique(scen[roles == 1])
    for name, code in (("tuning_disjoint", 0), ("assessment_disjoint", 2)):
        out[name] = rows[(roles == code) & ~np.isin(scen, cal_scenarios)]
    return out


def _accuracy(proba: np.ndarray, y: np.ndarray, rows: np.ndarray) -> float | None:
    return float(np.mean(proba[rows].argmax(1) == y[rows])) if len(rows) else None


def calibrate_proba(task: str, proba: np.ndarray) -> tuple[dict, np.ndarray | None]:
    """Fit calibrators on the calibration role and pick one using the tuning role only.

    Temperature scaling is always available and never changes a prediction. A per-class-bias calibrator is
    adopted only if it raises the tuning value by ``PLATEAU['min_gain']`` AND raises accuracy by the same margin on
    tuning rows from scenarios absent from the calibration role. Assessment numbers are reported, never used.
    """
    from framework.training.csv_v3 import calibrate as cal
    a = task_arrays(task)
    y, rows = a["y_val"], a["val_rows"]
    if cal.is_one_hot(proba[rows]):
        return {"status": "not_calibratable", "reason": "one-hot output (hard vote or argmax), not probabilities"}, None
    r = calibration_rows(task)
    raw_value = tune_value(score(task, proba))
    raw_disjoint = _accuracy(proba, y, r["tuning_disjoint"])
    fitted = []
    for cfg in cal.candidates(proba[r["calibration"]], y[r["calibration"]], len(a["classes"])):
        p = np.zeros_like(proba)
        p[rows] = cal.apply(cfg, proba[rows])
        value = tune_value(score(task, p))
        fitted.append((cfg, p, value, _accuracy(p, y, r["tuning_disjoint"]), cal.nll(p[r["tuning"]], y[r["tuning"]])))
    # Confidence-only choice: keep the raw probabilities unless temperature lowers tuning-role ECE.
    tune_rows = r["tuning"]
    identity = ({"kind": "identity"}, proba, raw_value, raw_disjoint, cal.nll(proba[tune_rows], y[tune_rows]))
    temperature = next(f for f in fitted if f[0]["kind"] == "temperature")
    tuning_ece = lambda f: cal.ece(f[1][tune_rows], y[tune_rows])
    confidence = temperature if tuning_ece(temperature) < tuning_ece(identity) else identity
    # Decision change: must gain on all tuning rows AND on tuning scenarios the calibrator never saw; a tie on
    # the disjoint rows means the gain came from scenarios shared with the calibration role.
    decision = [f for f in fitted if f[0]["kind"] == "vector" and f[2] >= raw_value + PLATEAU["min_gain"]
                and raw_disjoint is not None and f[3] >= raw_disjoint + PLATEAU["min_gain"]]
    chosen = max(decision, key=lambda f: (f[2], -f[4])) if decision else confidence
    cfg, p = chosen[0], chosen[1]
    quality = {name: {"raw": cal.quality(proba[r[name]], y[r[name]]), "calibrated": cal.quality(p[r[name]], y[r[name]])}
               for name in ("tuning", "calibration", "assessment") if len(r[name])}
    info = {
        "status": "complete", "method": cfg["kind"], "config": cfg, "decisions_changed": bool(decision),
        "raw_tune_value": raw_value, "selection_value": chosen[2] if decision else raw_value,
        "tuning_disjoint_accuracy": {"raw": raw_disjoint, "calibrated": chosen[3], "rows": int(len(r["tuning_disjoint"]))},
        "assessment_disjoint_accuracy": {"raw": _accuracy(proba, y, r["assessment_disjoint"]),
                                         "calibrated": _accuracy(p, y, r["assessment_disjoint"]),
                                         "rows": int(len(r["assessment_disjoint"]))},
        "quality": quality, "report": score(task, p),
        "candidates": [{**{k: v for k, v in f[0].items() if k != "bias"}, "tune_value": f[2],
                        "tuning_disjoint_accuracy": f[3], "tuning_nll": f[4], "tuning_ece": tuning_ece(f)}
                       for f in [identity] + fitted],
    }
    return info, p


def selection_value(result: dict) -> float:
    """Tune value after calibration (equal to the raw value unless a decision-changing calibrator was adopted)."""
    return (result.get("calibration") or {}).get("selection_value", result["tune_value"])


# --------------------------------------------------------------------------------------------- controller
class Controller:
    def __init__(self, log=print):
        self.log = log
        STUDY.mkdir(parents=True, exist_ok=True)
        (STUDY / "jobs").mkdir(exist_ok=True)

    # -- bookkeeping
    def stopped(self) -> bool:
        return (STUDY / "STOP").exists()

    def state(self, **update) -> None:
        path = STUDY / "state.json"
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        state.update(update, updated=time.strftime("%Y-%m-%d %H:%M:%S"), pid=os.getpid())
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def ledger(self, **entry) -> None:
        with open(STUDY / "ledger.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}) + "\n")

    def result(self, job_id: str) -> dict | None:
        path = job_dir(job_id) / "result.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def calibrated(self, r: dict | None) -> dict | None:
        """Calibrate a complete full-row result once (also retrofits results written before calibration existed)."""
        if r is None or r.get("status") != "complete" or "report" not in r or "calibration" in r:
            return r
        info, p = calibrate_proba(r["task"], np.load(STUDY / r["work_dir"] / "val_proba.npy"))
        if p is not None:
            np.save(job_dir(r["job_id"]) / "val_proba_cal.npy", p)
        r["calibration"] = info
        self._write_result(r["job_id"], r)
        if info["status"] == "complete":
            self.ledger(job=r["job_id"], event="calibrated", method=info["method"],
                        decisions_changed=info["decisions_changed"], selection_value=info["selection_value"])
        return r

    # -- one fit job (train in a subprocess, CPU replay in another, then score)
    def run_job(self, job_id: str, task: str, family: str, config: dict, seed: int = 42, rows: str = "full",
                timeout: int = 6 * 3600) -> dict | None:
        if FEATURE_GROUPS is not None:
            config = {**config, "feature_groups": list(FEATURE_GROUPS)}
        done = self.result(job_id)
        if done is not None:
            cached = (done.get("config") or {}).get("feature_groups")
            if "config" in done and cached != config.get("feature_groups"):
                raise RuntimeError(f"{job_id}: cached feature_groups {cached} != requested {config.get('feature_groups')}")
            # A job id is reused only for the identical config and seed (else a reset study would silently get
            # another config's results).
            canon = lambda c: json.dumps(c, sort_keys=True, default=str)
            if "config" in done and (canon(done["config"]) != canon(config) or done.get("seed", seed) != seed):
                raise RuntimeError(f"{job_id}: cached result was trained with a different config or seed")
            return self.calibrated(done)
        if self.stopped():
            raise KeyboardInterrupt("STOP present")
        out = job_dir(job_id)
        start = time.time()
        for attempt in range(2):
            work = out / f"attempt_{attempt}"
            if (work / "predict_done.json").exists():
                break
            if work.exists():
                continue
            work.mkdir(parents=True)
            request = {"job_id": job_id, "task": task, "family": family, "config": config, "seed": seed,
                       "rows": rows, "attempt": attempt}
            (work / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
            self.state(current_job=job_id, task=task, family=family, attempt=attempt)
            self.log(f"[{task}] {job_id}: training {family} (rows={rows}, seed={seed}, attempt {attempt})")
            start = time.time()
            ok = True
            for stage in ("_fit", "_predict"):
                with open(work / f"{stage[1:]}.log", "w", encoding="utf-8") as log:
                    proc = subprocess.run([PYTHON, "-m", CLI, stage, str(work / "request.json")], cwd=ROOT,
                                          stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                if proc.returncode != 0:
                    ok = False
                    self.ledger(job=job_id, event="failed", stage=stage, attempt=attempt, code=proc.returncode)
                    self.log(f"[{task}] {job_id}: {stage} failed (exit {proc.returncode}); see {work}")
                    break
            if ok:
                break
        else:
            self._write_result(job_id, {"job_id": job_id, "status": "failed", "task": task, "family": family})
            return self.result(job_id)
        fit = json.loads((work / "fit_result.json").read_text(encoding="utf-8"))
        proba = np.load(work / "val_proba.npy")
        result = {"job_id": job_id, "status": "complete", "task": task, "family": family, "seed": seed,
                  "rows": rows, "config": config, "work_dir": str(work.relative_to(STUDY)),
                  "wall_seconds": round(time.time() - start, 1), "fit": fit}
        if rows == "full":
            report = score(task, proba)
            result["report"] = report
            result["tune_value"] = tune_value(report)
            fit_tune = fit.get("tune_accuracy")
            result["cpu_replay_tune_accuracy_matches_fit"] = (
                None if fit_tune is None else bool(abs(fit_tune - report["tuning"]["accuracy"]) < 2e-4))
        self._write_result(job_id, result)
        self.ledger(job=job_id, event="complete", tune_value=result.get("tune_value"))
        return self.calibrated(result)

    def _write_result(self, job_id: str, result: dict) -> None:
        job_dir(job_id).mkdir(parents=True, exist_ok=True)
        (job_dir(job_id) / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    # -- stage 1
    def baseline(self, task: str) -> dict:
        out = {}
        for family in FAMILIES[task]:
            config = adapter(family).default_config(family, task)
            out[family] = self.run_job(f"{task}_baseline_{family}", task, family, config)
        return out

    # -- hybrids
    def oof(self, task: str, family: str, config: dict, tag: str) -> list[dict]:
        return [self.run_job(f"{task}_oof_{tag}_{family}_fold{k}", task, family, config, rows=f"fold{k}")
                for k in (0, 1)]

    def _oof_matrix(self, task: str, fold_jobs: dict[str, list[dict]]) -> tuple[np.ndarray, np.ndarray]:
        y = task_arrays(task)["y_train"]
        probs, index = [], None
        for name, jobs in fold_jobs.items():
            idx = np.concatenate([np.load(STUDY / j["work_dir"] / "oof_index.npy") for j in jobs])
            p = np.concatenate([np.load(STUDY / j["work_dir"] / "oof_proba.npy") for j in jobs])
            if index is None:
                index = idx
            elif not np.array_equal(index, idx):
                raise RuntimeError("OOF rows differ between branches")
            probs.append(p)
        return np.stack(probs), y[index]

    def hybrid(self, task: str, name: str, branches: dict[str, dict], fold_jobs: dict[str, list[dict]],
               penalties=(None, 0.3, 1.0, 3.0, 10.0, 30.0)) -> dict:
        """Fit equal/global/classwise blends on OOF predictions; calibrate each; choose on the validation tuning role."""
        from framework.training import hybrid_blend
        job_id = f"{task}_hybrid_{name}"
        done = self.result(job_id)
        legacy = None
        if done is not None:
            if done.get("status") != "complete" or "calibration" in done:
                return done
            # Written before calibration: re-select the combiner by the calibrated value (no retraining needed).
            legacy = {"combiner": (done.get("combiner") or {}).get("label"), "tune_value": done.get("tune_value")}
            (job_dir(job_id) / "result_raw_selection.json").write_text(json.dumps(done, indent=2, default=str),
                                                                       encoding="utf-8")
        names = list(branches)
        oof_p, oof_y = self._oof_matrix(task, {n: fold_jobs[n] for n in names})
        oof_p = oof_p / oof_p.sum(2, keepdims=True)
        val_p = np.stack([np.load(STUDY / branches[n]["work_dir"] / "val_proba.npy") for n in names])
        rows = task_arrays(task)["val_rows"]
        candidates = [{"kind": "global", "names": names, "n_classes": oof_p.shape[2],
                       "weights": [1 / len(names)] * len(names), "label": "equal"}]
        for penalty in penalties:
            try:
                cfg = hybrid_blend.fit_blend(oof_p, oof_y, names, penalty=penalty)
                cfg["label"] = "global" if penalty is None else f"classwise_{penalty}"
                candidates.append(cfg)
            except RuntimeError as exc:
                self.log(f"[{task}] {name}: blend penalty {penalty} failed: {exc}")
        best, best_selection, scored = None, -1.0, []
        for cfg in candidates:
            p = np.zeros_like(val_p[0])
            p[rows] = hybrid_blend.blend(val_p[:, rows], cfg, names)
            report = score(task, p)
            value = tune_value(report)
            calibration, p_cal = calibrate_proba(task, p)
            selection = calibration.get("selection_value", value)
            scored.append({"label": cfg["label"], "tune_value": value, "selection_value": selection,
                           "calibration": calibration.get("method"),
                           "assessment_accuracy": report["assessment"]["accuracy"]})
            if selection > best_selection:
                best, best_selection = cfg, selection
                best_value, best_report, best_p, best_cal, best_p_cal = value, report, p, calibration, p_cal
        out = job_dir(job_id)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "val_proba.npy", best_p)
        if best_p_cal is not None:
            np.save(out / "val_proba_cal.npy", best_p_cal)
        result = {"job_id": job_id, "status": "complete", "task": task, "family": name, "branches": names,
                  "branch_jobs": {n: branches[n]["job_id"] for n in names}, "combiner": best,
                  "candidates": scored, "oof_rows": int(len(oof_y)), "report": best_report,
                  "tune_value": best_value, "calibration": best_cal, "work_dir": str(out.relative_to(STUDY))}
        if legacy:
            result["raw_selection_before_calibration"] = legacy
        self._write_result(job_id, result)
        self.ledger(job=job_id, event="complete", tune_value=best_value, selection_value=best_selection)
        return result

    def hybrids(self, task: str, branch_results: dict[str, dict], tag: str, search_combinations: bool) -> dict:
        fold_jobs = {}
        for family in OOF_BRANCHES:
            r = branch_results[family]
            if r is None or r.get("status") != "complete":
                self.log(f"[{task}] skipping OOF for {family}: no complete branch")
                continue
            fold_jobs[family] = self.oof(task, family, r["config"], tag)
        out = {}
        groups = dict(HYBRIDS)
        if search_combinations:
            usable = [f for f in OOF_BRANCHES if f in fold_jobs and all(j.get("status") == "complete" for j in fold_jobs[f])]
            seen = {frozenset(m) for m in groups.values()}
            for size in range(2, len(usable) + 1):
                for combo in itertools.combinations(usable, size):
                    if frozenset(combo) not in seen:
                        seen.add(frozenset(combo))
                        groups["combo_" + "+".join(combo)] = list(combo)
        for name, members in groups.items():
            if not all(m in fold_jobs and all(j.get("status") == "complete" for j in fold_jobs[m]) for m in members):
                self.log(f"[{task}] skipping hybrid {name}: missing branch")
                continue
            out[name] = self.hybrid(task, f"{tag}_{name}", {m: branch_results[m] for m in members}, fold_jobs)
        return out

    # -- stage 3
    def tune(self, task: str, family: str, base: dict) -> dict:
        """Optuna TPE on the tuning role; stops when the best value stops improving by PLATEAU['min_gain']."""
        import optuna
        from framework.training.csv_v3 import search

        if base is None or base.get("status") != "complete":
            return base
        limit = TUNE_LIMITS.get(family, 0) if task != "8" else 30
        if SMOKE:
            limit = 2
        storage = f"sqlite:///{(STUDY / f'optuna_{task}.sqlite3').as_posix()}"
        study = optuna.create_study(study_name=family, storage=storage, direction="maximize", load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, n_startup_trials=4))
        default = adapter(family).default_config(family, task)
        objective = selection_value if task in CALIBRATED_OBJECTIVE_TASKS else (lambda r: r["tune_value"])
        base = self.calibrated(base)
        state = {"best_value": objective(base), "history": [objective(base)], "since": 0, "n": 0, "results": [base]}

        def record(number: int, r: dict | None) -> None:
            r = self.calibrated(r)
            if r is None or r.get("status") != "complete":
                return
            value = objective(r)
            state["n"] += 1
            state["results"].append(r)
            state["history"].append(value)
            state["since"] = 0 if value > state["best_value"] + PLATEAU["min_gain"] else state["since"] + 1
            state["best_value"] = max(state["best_value"], value)
            self.log(f"[{task}] tune {family} t{number}: {value:.4f} "
                     f"(best {state['best_value']:.4f}, flat for {state['since']})")

        def flat() -> bool:
            return state["n"] >= PLATEAU["min_trials"] and state["since"] >= PLATEAU["patience"]

        # Replay finished trials and resume interrupted ones with their saved parameters (restart safety).
        for t in sorted(study.trials, key=lambda t: t.number):
            job_id = f"{task}_tune_{family}_t{t.number:03d}"
            if t.state == optuna.trial.TrialState.COMPLETE:
                record(t.number, self.result(job_id))
            elif t.state == optuna.trial.TrialState.RUNNING:
                config = search.suggest(family, task, optuna.trial.FixedTrial(t.params), default)
                r = self.run_job(job_id, task, family, config)
                if r and r.get("status") == "complete":
                    study.tell(t.number, objective(r))
                    record(t.number, r)
                else:
                    study.tell(t.number, state=optuna.trial.TrialState.FAIL)
        attempts = sum(t.state != optuna.trial.TrialState.RUNNING for t in study.trials)
        while attempts < limit and not flat():
            trial = study.ask()
            config = search.suggest(family, task, trial, default)
            r = self.run_job(f"{task}_tune_{family}_t{trial.number:03d}", task, family, config)
            attempts += 1
            if r is None or r.get("status") != "complete":
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                continue
            study.tell(trial, objective(r))
            record(trial.number, r)
        if flat():
            self.log(f"[{task}] tune {family}: gains flattened after {state['n']} trials; stopping")
        # Final pick across the baseline and every trial by the calibrated value (tuning role only).
        best = max(state["results"], key=selection_value)
        history = state["history"]
        summary = {"family": family, "trials": len(history) - 1, "objective": "calibrated" if task in CALIBRATED_OBJECTIVE_TASKS else "raw",
"baseline_value": objective(base), "best_value": state["best_value"],
                   "history": history, "best_job": best["job_id"], "best_job_objective_value": objective(best),
                   "best_selection_value": selection_value(best),
                   "best_calibration": (best.get("calibration") or {}).get("method"),
                   "best_decisions_changed": (best.get("calibration") or {}).get("decisions_changed")}
        (STUDY / f"tune_{task}_{family}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return best

    # -- stage 4
    def repeats(self, task: str, selected: dict[str, dict]) -> dict:
        out = {}
        for family, r in selected.items():
            if r is None or r.get("status") != "complete" or family not in ADAPTERS or family in NOT_TUNED:
                continue
            out[family] = [self.run_job(f"{task}_repeat_{family}_s{seed}", task, family, r["config"], seed=seed)
                           for seed in REPEAT_SEEDS]
        return out

    # -- a whole task, in order
    def run_task(self, task: str) -> None:
        self.state(task=task, stage="baseline")
        base = self.baseline(task)
        hybrids0 = {}
        if task != "8":
            self.state(stage="baseline_hybrids")
            hybrids0 = self.hybrids(task, base, "base", search_combinations=False)
        self.state(stage="tune")
        tuned = {}
        for family in FAMILIES[task]:
            tuned[family] = base[family] if family in NOT_TUNED else self.tune(task, family, base[family])
        hybrids1 = {}
        if task != "8":
            self.state(stage="tuned_hybrids")
            hybrids1 = self.hybrids(task, tuned, "tuned", search_combinations=True)
        self.state(stage="repeat")
        reps = self.repeats(task, tuned)
        self.state(stage="report")
        from framework.training.csv_v3 import report as reporting
        reporting.write(task, base, hybrids0, tuned, hybrids1, reps)
        self.state(stage="task_complete", completed_tasks=self._completed() + [task])
        self.log(f"[{task}] complete")

    def _completed(self) -> list[str]:
        path = STUDY / "state.json"
        return json.loads(path.read_text(encoding="utf-8")).get("completed_tasks", []) if path.exists() else []

    def run(self, tasks=TASK_ORDER) -> None:
        try:
            for task in tasks:
                if task in self._completed():
                    continue
                if task == "84xgb":
                    self.state(task=task, stage="control")
                    self.run_job("84_baseline_xgboost", "84", "xgboost", adapter("xgboost").default_config("xgboost", "84"))
                elif task == "cascade":
                    from framework.training.csv_v3 import cascade
                    self.state(task=task, stage="report")
                    info = cascade.write()
                    self.log(f"[cascade] stage1 {info['stage1_job']} + stage2 {info['stage2_job']}; REPORT_cascade84.md written")
                else:
                    self.run_task(task)
                    continue
                self.state(stage="task_complete", completed_tasks=self._completed() + [task])
                self.log(f"[{task}] complete")
            self.state(stage="all_complete")
        except KeyboardInterrupt:
            self.state(stage="stopped")
            self.log("STOP honoured; completed jobs are kept")
        except Exception:
            self.state(stage="error", error=traceback.format_exc()[-4000:])
            raise
