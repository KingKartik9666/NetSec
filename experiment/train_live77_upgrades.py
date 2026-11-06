"""Queued post-training upgrade tests for the live 77-class XGBoost (runs after experiment.train_live77 finishes).

    python -m experiment.train_live77_upgrades run     # the watchdog (experiment.live77_watch) starts this itself

Plan: ml_pipeline/notes_20261031/live77_upgrade_plan.md (research workflow wf_c07f6400-fc5). Owner 2026-10-31: "search
for upgrade techniques and apply them after training to see if they help"; "this live screening cant score less acc".
Every arm is a fold-1 refit of the frozen tuned config at its fixed round count, on fresh seeds 142/143/144 (never used
in tuning or size selection), paired against the control C0 on the same seeds. J = 0.5 x role-0 accuracy + 0.5 x
fold-1 OOF accuracy. Review 2026-10-31 (wf_d73c5310-4c9): only one pre-registered PRIMARY arm per technique can be
admitted (others are reported); admission = mean paired gain >= 0.1 pp and > 2 x SD and unseen-scenario gain >= 0, and
the chosen spec must pass the same rule again against C0 on new seeds 151/152/153 (confirmation). Size arms (mRMR) and
seed bagging are paired with those confirmation runs. Size arms are admitted only as non-inferior (J loss <= 0.1 pp,
unseen gain >= 0, OOF macro-F1 within 0.005, not worse than the gain-ranked control). The shift ranking and arm list are
frozen (sha256) before any arm is scored; the mRMR/gain lists before the size arms. The final rule is frozen before the
all-row refit: the fold-1 decision stands, role 2 is report-only. The held-out test partition is never read.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

# Set explicitly (not setdefault): the watchdog's environment carries the MAIN study's CASCADE_STUDY.
os.environ["CASCADE_STUDY"] = os.environ.get("LIVE77_UPGRADE_STUDY", "artifacts/training/live77_upgrades_20261031_v1")
os.environ["CASCADE_DATA"] = "dataset/experiments/csv_v3_live77_derived_20261031_v1"

import numpy as np  # noqa: E402

from experiment import train_live77 as L  # noqa: E402
from framework.training.csv_v3 import derived, runner  # noqa: E402

STUDY = runner.STUDY
MAIN = runner.ROOT / "artifacts/training/live77_20261031_v1"
SEEDS = (142, 143, 144)
CONFIRM_SEEDS = (151, 152, 153)
BAG_SEEDS = (154, 155, 156, 157, 158, 159)
PRIMARY = {"ports": "P2_drop_dynamic_and_3600_3820", "os": "T2_drop_os_dependent", "derived": "R4_all_derived",
           "weights": "WB_scenario_weight_0.5", "shift": "A1_drop_shift_top10"}
GAIN, F1_TOL = 0.001, 0.005
_log = L._log


# ------------------------------------------------------------------------------------------------ setup
def frozen(name: str, value):
    """Write once, then always return the stored value (pre-registration)."""
    path = STUDY / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["value"]
    text = json.dumps(value, sort_keys=True)
    path.write_text(json.dumps({"value": value, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                                "frozen": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    return value


def main_artifacts() -> dict:
    sel = json.loads((MAIN / "selection.json").read_text(encoding="utf-8"))
    size = json.loads((MAIN / "size.json").read_text(encoding="utf-8"))
    rounds = json.loads((MAIN / "final_rounds.json").read_text(encoding="utf-8"))
    final = json.loads((MAIN / "jobs" / "77_live_final" / "result.json").read_text(encoding="utf-8"))
    return {"names": sel["feature_names"], "params": size["chosen"]["params"], "fold_rounds": size["chosen"]["rounds"],
            "final_rounds": rounds["chosen_rounds"], "final": final}


def make_config(base: dict, names: list[str], rounds: int, row_weight=None, params=None) -> dict:
    c = L.base_config()
    c.update({k: v for k, v in (params or base["params"]).items() if k in L.PARAMS})
    c.update(n_estimators=int(rounds), early_stopping_rounds=int(rounds) + 1, runtime_seconds=10800,
             checkpoint_rule="last", feature_list=list(names))
    if row_weight is not None:
        c["row_weight"] = row_weight
    return c


def J(job: dict) -> float:
    return 0.5 * job["tune_acc"] + 0.5 * job["oof_acc"]


# --------------------------------------------------------------------------------------- feature lists
def _port(name: str) -> int | None:
    m = re.fullmatch(r"network_ports_(?:src|dst)__present_(\d+)", name)
    return int(m.group(1)) if m else None


def _drop(names, pred):
    return [n for n in names if not pred(n)]


OS_FAMILIES = ("network_ttl_", "network_window-size_", "network_mss_", "network_macs_")


def accuracy_arms(base: dict, shift: list[str]) -> dict:
    names = base["names"]
    dyn = lambda n: (_port(n) or 0) >= 49152
    block = lambda n: 3600 <= (_port(n) or 0) <= 3820
    os_drop = lambda n: n.startswith(OS_FAMILIES)
    ttl3 = {"network_ttl_avg", "network_ttl_max", "network_ttl_min"}
    hop = _drop(names, lambda n: n in ttl3) + derived.TTL_HOP
    p = base["params"]
    arms = {
        "P1_drop_dynamic_ports": {"family": "ports", "names": _drop(names, dyn)},
        "P2_drop_dynamic_and_3600_3820": {"family": "ports", "names": _drop(names, lambda n: dyn(n) or block(n))},
        "T1_ttl_reencode": {"family": "os", "names": hop},
        "T2_drop_os_dependent": {"family": "os", "names": _drop(names, os_drop)},
        "T3_ttl_reencode_drop_os": {"family": "os", "names": _drop(hop, lambda n: os_drop(n) and n.startswith(OS_FAMILIES[1:]))},
        "R1_ranges": {"family": "derived", "names": names + derived.RANGES},
        "R2_flag_ratios": {"family": "derived", "names": names + derived.RATIOS},
        "R3_ip_rates": {"family": "derived", "names": names + derived.IP_RATES},
        "R4_all_derived": {"family": "derived", "names": names + derived.RANGES + derived.RATIOS + derived.IP_RATES},
        "WB_scenario_weight_0.5": {"family": "weights", "names": names, "row_weight": {"kind": "scenario", "rho": 0.5}},
        "WC_scenario_weight_1.0": {"family": "weights", "names": names, "row_weight": {"kind": "scenario", "rho": 1.0}},
        "WD_scenario_weight_0.5_reg": {"family": "weights", "names": names, "row_weight": {"kind": "scenario", "rho": 0.5},
                                       "params": p | {"min_child_weight": 2 * p["min_child_weight"],
                                                      "reg_lambda": 2 * p["reg_lambda"]}},
    }
    for k in (5, 10, 20):
        drop = set(shift[:k])
        arms[f"A1_drop_shift_top{k}"] = {"family": "shift", "names": _drop(names, lambda n: n in drop)}
    return arms


# ------------------------------------------------------------------------------- MI / CMI / mRMR (CPU)
def _bins(col: np.ndarray, k: int = 32) -> np.ndarray:
    out = np.full(len(col), k, dtype=np.int64)          # NaN -> its own bin
    ok = ~np.isnan(col)
    v = col[ok]
    if v.size == 0:
        return out
    uniq = np.unique(v)
    if uniq.size <= k:
        out[ok] = np.searchsorted(uniq, v)
    else:
        edges = np.unique(np.quantile(v, np.linspace(0, 1, k + 1)[1:-1]))
        out[ok] = np.searchsorted(edges, v, side="right")
    return out


def _mi(a: np.ndarray, b: np.ndarray) -> float:
    """Plug-in MI (nats) with the Miller-Madow bias correction."""
    n = len(a)
    if n == 0:
        return 0.0
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    ka, kb = a.max() + 1, b.max() + 1
    joint = np.bincount(a * kb + b, minlength=ka * kb).astype(np.float64)
    pa, pb = np.bincount(a, minlength=ka) / n, np.bincount(b, minlength=kb) / n
    pj = joint[joint > 0] / n
    ia, ib = np.divmod(np.flatnonzero(joint), kb)
    mi = float(np.sum(pj * np.log(pj / (pa[ia] * pb[ib]))))
    return mi - (len(pj) - ka - kb + 1) / (2 * n)


def _fit_sample(R, names: list[str], per_class: int = 2500, seed: int = 0):
    fit = runner.folds("77")[1]["fit"]
    y = R.y_train[fit]
    rng = np.random.default_rng(seed)
    pick = np.sort(np.concatenate([rng.permutation(fit[y == c])[:per_class] for c in np.unique(y)]))
    all_names = runner.manifest()["feature_names"]
    cols = [all_names.index(n) for n in names]
    X = np.asarray(np.load(runner.DATA / "X_train.npy", mmap_mode="r")[pick])[:, cols]
    return X, R.y_train[pick], runner._arr("scenario_train")[pick]


def shift_ranking(R, names: list[str]) -> list[str]:
    """Rank features by I(f; scenario | class) / (I(f; class) + 1e-3) on fold-1 fit rows (fingerprint score)."""
    X, y, scen = _fit_sample(R, names)
    classes = [c for c in np.unique(y) if (y == c).sum() >= 200 and np.unique(scen[y == c]).size >= 2]
    w = np.array([(y == c).sum() for c in classes], float)
    w /= w.sum()
    scores = {}
    for j, n in enumerate(names):
        b = _bins(X[:, j])
        mi = max(_mi(b, y), 0.0)
        cmi = sum(wc * max(_mi(b[y == c], scen[y == c]), 0.0) for wc, c in zip(w, classes))
        scores[n] = cmi / (mi + 1e-3)
    return sorted(names, key=lambda n: -scores[n])


def mrmr_ranking(R, names: list[str], budget: int) -> list[str]:
    X, y, _ = _fit_sample(R, names, per_class=1500, seed=1)
    B = np.stack([_bins(X[:, j]) for j in range(len(names))], axis=1)
    rel = np.array([max(_mi(B[:, j], y), 0.0) for j in range(len(names))])
    chosen, red = [int(np.argmax(rel))], np.zeros(len(names))
    while len(chosen) < min(budget, len(names)):
        last = chosen[-1]
        for j in range(len(names)):
            if j not in chosen:
                red[j] += max(_mi(B[:, j], B[:, last]), 0.0)
        score = rel - red / len(chosen)
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return [names[j] for j in chosen]


def gain_ranking(job: dict, names: list[str]) -> list[str]:
    import xgboost as xgb
    booster = xgb.Booster(model_file=str(Path(job["work"]) / "model" / "model.ubj"))
    gain = booster.get_score(importance_type="total_gain")
    score = {n: gain.get(f"f{j}", 0.0) for j, n in enumerate(names)}
    return sorted(names, key=lambda n: -score[n])


# ------------------------------------------------------------------------------------------ arms + rules
def run_arm(ctl, R, base: dict, arm: str, spec: dict, rounds: int, seeds=SEEDS) -> list[dict]:
    return [L.fold_job(ctl, R, f"77_up_{arm}_s{s}",
                       make_config(base, spec["names"], rounds, spec.get("row_weight"), spec.get("params")), s)
            for s in seeds]


def run_arm_adaptive(ctl, R, base: dict, arm: str, spec: dict, rounds: int, control: list[dict]) -> tuple[list, int | None]:
    """Seed by seed; drop the arm when its gain is already marginal (owner 2026-10-31: "apply droputs if gains become
    marginal"): after seed 1 if J gain <= -0.1 pp, after seed 2 if the mean J gain < 0. A dropped arm cannot be admitted
    (admission needs all 3 seeds), so dropping only saves GPU time."""
    by = {r["seed"]: r for r in control}
    runs = []
    for i, s in enumerate(SEEDS):
        runs += run_arm(ctl, R, base, arm, spec, rounds, seeds=(s,))
        mean_gain = float(np.mean([J(r) - J(by[r["seed"]]) for r in runs]))
        if (i == 0 and mean_gain <= -GAIN) or (i == 1 and mean_gain < 0):
            return runs, i + 1
    return runs, None


def paired(control: list[dict], runs: list[dict]) -> dict:
    by = {r["seed"]: r for r in control}
    g = np.array([J(r) - J(by[r["seed"]]) for r in runs])
    gu = np.array([r["tune_unseen_acc"] - by[r["seed"]]["tune_unseen_acc"] for r in runs])
    gf = np.array([r["oof_f1"] - by[r["seed"]]["oof_f1"] for r in runs])
    return {"gain": float(g.mean()), "gain_sd": float(g.std(ddof=1)) if len(g) > 1 else 0.0,
            "unseen_gain": float(gu.mean()), "oof_f1_change": float(gf.mean()),
            "tune_acc": float(np.mean([r["tune_acc"] for r in runs])), "oof_acc": float(np.mean([r["oof_acc"] for r in runs]))}


def admitted(s: dict) -> bool:
    return s["gain"] >= GAIN and s["gain"] > 2 * s["gain_sd"] and s["unseen_gain"] >= 0


def role0_ci(R, control: list[dict], runs: list[dict]) -> list[float]:
    """Scenario-bootstrap 95% CI of the seed-averaged role-0 accuracy difference (arm - control), diagnostic only."""
    hit = lambda rs: np.mean([np.load(Path(r["work"]) / "val_proba.npy")[R.tune].argmax(1) == R.y_val[R.tune]
                              for r in rs], axis=0)
    diff = hit(runs) - hit(control)
    codes, inv = np.unique(R.scen[R.tune], return_inverse=True)
    sums, sizes = np.bincount(inv, weights=diff), np.bincount(inv).astype(float)
    draws = np.random.default_rng(0).integers(0, len(codes), size=(1000, len(codes)))
    est = sums[draws].sum(1) / sizes[draws].sum(1)
    return [float(x) for x in np.percentile(est, [2.5, 97.5])]


# ---------------------------------------------------------------------------------------- seed bagging
def _margins(job: dict, rows_X: np.ndarray, rounds: int) -> np.ndarray:
    import xgboost as xgb
    booster = xgb.Booster(model_file=str(Path(job["work"]) / "model" / "model.ubj"))
    booster.set_param({"device": "cpu", "nthread": 4})
    return booster.predict(xgb.DMatrix(rows_X, nthread=4), iteration_range=(0, rounds), output_margin=True)


def bagging(ctl, R, base: dict, spec: dict, rounds: int, control: list[dict]) -> dict:
    """E3-eq: three seeds truncated to rounds/3 each (same tree count as one model), margin-averaged."""
    extra = run_arm(ctl, R, base, "C_bag", spec, rounds, seeds=BAG_SEEDS)
    jobs = {r["seed"]: r for r in control + extra}
    names = runner.manifest()["feature_names"]
    cols = [names.index(n) for n in spec["names"]]
    Xt = np.asarray(np.load(runner.DATA / "X_validation.npy", mmap_mode="r")[R.tune])[:, cols]
    oof_idx = np.load(Path(control[0]["work"]) / "oof_index.npy")
    Xo = np.asarray(np.load(runner.DATA / "X_train.npy", mmap_mode="r")[oof_idx])[:, cols]
    third = max(1, rounds // 3)
    reps = []
    for r in range(3):
        trio = [jobs[CONFIRM_SEEDS[r]], jobs[BAG_SEEDS[r]], jobs[BAG_SEEDS[3 + r]]]
        mt = np.mean([_margins(j, Xt, third) for j in trio], axis=0)
        mo = np.mean([_margins(j, Xo, third) for j in trio], axis=0)
        acc_t = float(np.mean(mt.argmax(1) == R.y_val[R.tune]))
        acc_o = float(np.mean(mo.argmax(1) == R.y_train[oof_idx]))
        unseen = np.isin(R.tune, R.tune_unseen)
        acc_u = float(np.mean(mt.argmax(1)[unseen] == R.y_val[R.tune][unseen]))
        reps.append({"seed": CONFIRM_SEEDS[r], "tune_acc": acc_t, "oof_acc": acc_o, "tune_unseen_acc": acc_u,
                     "oof_f1": L._f1(R.y_train[oof_idx], mo.argmax(1), R.n_classes)})
    return {"stats": paired(control, reps), "replicates": reps, "rounds_each": third}


def merge_boosters(paths: list[Path], out: Path) -> None:
    """One gbtree whose margin is the mean of the inputs' margins (leaves scaled by 1/k; equal base_score)."""
    import xgboost as xgb
    models = []
    for p in paths:
        b = xgb.Booster(model_file=str(p))
        models.append(json.loads(b.save_raw("json")))
    k = len(models)
    merged = models[0]
    gb = merged["learner"]["gradient_booster"]["model"]
    trees, info = [], []
    for m in models:
        mm = m["learner"]["gradient_booster"]["model"]
        for t in mm["trees"]:
            leaf = np.array(t["left_children"]) == -1
            for key in ("split_conditions", "base_weights"):
                v = np.array(t[key], dtype=np.float64)
                v[leaf] /= k
                t[key] = v.tolist()
            t["id"] = len(trees)
            trees.append(t)
        info += mm["tree_info"]
    per_round = int(merged["learner"]["learner_model_param"]["num_class"])
    gb["trees"], gb["tree_info"] = trees, info
    gb["gbtree_model_param"]["num_trees"] = str(len(trees))
    gb["iteration_indptr"] = list(range(0, len(trees) + 1, per_round))
    booster = xgb.Booster()
    booster.load_model(bytearray(json.dumps(merged).encode()))
    booster.save_model(str(out))


# --------------------------------------------------------------------------------------- post-hoc (CPU)
def posthoc(R, proba: np.ndarray, raw: np.ndarray, cal_rows: np.ndarray) -> dict:
    """Report-only: logit (prior) adjustment, confidence abstain curve, APS conformal sets on role 0."""
    y, t = R.y_val, R.tune
    prior = np.bincount(R.y_train[np.flatnonzero(runner.task_arrays("77")["fit"])], minlength=R.n_classes) / \
        runner.task_arrays("77")["fit"].sum()
    out = {"logit_adjustment": {}}
    unseen = np.isin(t, R.tune_unseen)
    for tau in (0.0, 0.25, 0.5, 0.75, 1.0):
        lp = np.log(np.clip(proba[t], 1e-12, 1)) - tau * np.log(np.clip(prior, 1e-12, 1))
        pred = lp.argmax(1)
        out["logit_adjustment"][tau] = {"tune_acc": float(np.mean(pred == y[t])), "tune_f1": L._f1(y[t], pred, R.n_classes),
                                        "unseen_acc": float(np.mean(pred[unseen] == y[t][unseen]))}
    conf = proba[t].max(1)
    correct = proba[t].argmax(1) == y[t]
    order = np.argsort(-conf)
    out["abstain_msp"] = {f"{c:.2f}": float(correct[order[:max(1, int(c * len(order)))]].mean())
                          for c in (1.0, 0.98, 0.95, 0.9, 0.8)}
    # APS on the RAW probabilities (the calibrator was fitted on role 1, so role 1 stays clean for the threshold);
    # threshold from role 1, coverage and set size on role 0 (dataset coverage, not live coverage).
    proba = raw
    def aps_scores(p, labels):
        srt = np.argsort(-p, axis=1)
        cum = np.cumsum(np.take_along_axis(p, srt, 1), axis=1)
        rank = np.argmax(srt == labels[:, None], axis=1)
        return cum[np.arange(len(p)), rank]
    s_cal = aps_scores(proba[cal_rows], y[cal_rows])
    out["conformal_aps"] = {}
    for alpha in (0.05, 0.10):
        q = float(np.quantile(s_cal, min(1.0, np.ceil((len(s_cal) + 1) * (1 - alpha)) / len(s_cal)), method="higher"))
        srt = np.argsort(-proba[t], axis=1)
        cum = np.cumsum(np.take_along_axis(proba[t], srt, 1), axis=1)
        size = np.minimum((cum < q).sum(1) + 1, R.n_classes)
        covered = np.array([y[t][i] in srt[i, :size[i]] for i in range(len(t))])
        out["conformal_aps"][alpha] = {"q": q, "coverage": float(covered.mean()),
                                       "unseen_coverage": float(covered[unseen].mean()), "mean_set_size": float(size.mean())}
    return out


# ------------------------------------------------------------------------------------------------ main
def run() -> None:
    ctl = runner.Controller(log=_log)
    R = L.Rows()
    base = main_artifacts()
    N = base["fold_rounds"]
    C0 = {"names": base["names"]}
    control = run_arm(ctl, R, base, "C0", C0, N)
    shift = frozen("shift_ranking.json", shift_ranking(R, base["names"]))
    arms = frozen("arms.json", accuracy_arms(base, shift))
    frozen("primary_arms.json", PRIMARY)
    results = {}
    for arm, spec in arms.items():
        if ctl.stopped():
            raise KeyboardInterrupt("STOP present")
        try:
            runs, dropped = run_arm_adaptive(ctl, R, base, arm, spec, N, control)
        except RuntimeError as exc:
            _log(f"{arm}: failed ({exc})")
            continue
        s = {"family": spec["family"], "primary": PRIMARY.get(spec["family"]) == arm, **paired(control, runs),
             "seeds_run": len(runs), "dropped_after_seed": dropped}
        s["passes_rule"] = dropped is None and admitted(s)
        s["admitted"] = s["primary"] and s["passes_rule"]
        s["role0_diff_ci95"] = role0_ci(R, control, runs)
        s["provisional"] = s["admitted"] and s["role0_diff_ci95"][0] <= 0
        results[arm] = s
        _log(f"{arm}: done" + (f" (dropped after seed {dropped})" if dropped else ""))
    best = {s["family"]: arm for arm, s in results.items() if s["admitted"]}
    chosen_arm, chosen = "C0", dict(C0)
    if best:
        top = max(best.values(), key=lambda a: results[a]["gain"])
        chosen_arm, chosen = top, dict(arms[top])
        if len(best) >= 2:
            combo = {"names": list(base["names"]), "family": "combined"}
            for fam, arm in best.items():
                spec = arms[arm]
                removed = set(base["names"]) - set(spec["names"])
                added = [n for n in spec["names"] if n not in base["names"]]
                combo["names"] = [n for n in combo["names"] if n not in removed] + \
                                 [n for n in added if n not in combo["names"]]
                for key in ("row_weight", "params"):
                    if key in spec:
                        combo[key] = spec[key]
            combo = frozen("combined_arm.json", combo)
            runs = run_arm(ctl, R, base, "combined", combo, N)
            s = {"family": "combined", "primary": True, **paired(control, runs)}
            s["admitted"] = s["passes_rule"] = admitted(s)
            s["role0_diff_ci95"] = role0_ci(R, control, runs)
            results["combined"] = s
            if s["admitted"] and s["gain"] >= results[top]["gain"] - 0.0005:
                chosen_arm, chosen = "combined", combo
    # Confirmation on new seeds: the chosen spec must pass the rule again against C0 (winner's-curse guard).
    confirm_c0 = run_arm(ctl, R, base, "C0", C0, N, seeds=CONFIRM_SEEDS)
    confirmation = None
    if chosen_arm != "C0":
        confirm_runs = run_arm(ctl, R, base, chosen_arm, chosen, N, seeds=CONFIRM_SEEDS)
        confirmation = paired(confirm_c0, confirm_runs)
        confirmation["passes_rule"] = admitted(confirmation)
        if not confirmation["passes_rule"]:
            _log(f"{chosen_arm} failed confirmation; keeping C0")
            chosen_arm, chosen, chosen_control = "C0", dict(C0), confirm_c0
        else:
            chosen_control = confirm_runs
    else:
        chosen_control = confirm_c0
    # Size arms (non-inferiority) on the chosen feature list, paired with the fresh confirmation runs.
    size_results = {}
    names = chosen["names"]
    mr = frozen(f"mrmr_ranking_{chosen_arm}.json", mrmr_ranking(R, names, 160))
    gr = frozen(f"gain_ranking_{chosen_arm}.json", gain_ranking(chosen_control[0], names))
    for K in (96, 160):
        if K >= len(names):
            continue
        for tag, rank in (("M", mr), ("G", gr)):
            spec = dict(chosen, names=[n for n in names if n in set(rank[:K])])
            runs = run_arm(ctl, R, base, f"{tag}{K}_on_{chosen_arm}", spec, N, seeds=CONFIRM_SEEDS)
            size_results[f"{tag}{K}"] = paired(chosen_control, runs)
    size_pick = None
    for K in (96, 160):
        m, g = size_results.get(f"M{K}"), size_results.get(f"G{K}")
        if m and g and m["gain"] >= -GAIN and m["unseen_gain"] >= 0 and m["oof_f1_change"] >= -F1_TOL and m["gain"] >= g["gain"]:
            size_pick = K
            break
    if size_pick:
        chosen = dict(chosen, names=[n for n in names if n in set(mr[:size_pick])])
        chosen_control = run_arm(ctl, R, base, f"M{size_pick}_on_{chosen_arm}", chosen, N, seeds=CONFIRM_SEEDS)
    # Seed bagging at equal tree count, paired with the confirmation-seed runs of the chosen spec.
    bag = bagging(ctl, R, base, chosen, N, chosen_control)
    plan = frozen("final_choice.json", {
        "arm": chosen_arm, "size_pick": size_pick, "bagging": admitted(bag["stats"]), "spec": chosen,
        "confirmation": confirmation,
        "rule": "fold-1 decisions above stand; the all-row refit is reported against the main final on role 2 "
                "(report-only, paired scenario-bootstrap CI); nothing is re-chosen on role 2"})
    # Final all-row refit(s) + calibration.
    Rf = base["final_rounds"]
    spec = plan["spec"]
    all_names = runner.manifest()["feature_names"]
    Xv = np.asarray(np.load(runner.DATA / "X_validation.npy", mmap_mode="r"))[:, [all_names.index(n) for n in spec["names"]]]
    if not plan["bagging"]:
        fin = ctl.run_job("77_up_final", "77", "xgboost",
                          make_config(base, spec["names"], Rf, spec.get("row_weight"), spec.get("params")), 42, rows="full")
        raw = np.load(STUDY / fin["work_dir"] / "val_proba.npy")
        cal_path = runner.job_dir("77_up_final") / "val_proba_cal.npy"
        proba = np.load(cal_path) if cal_path.exists() else raw
        report = fin.get("calibration", {}).get("report", fin["report"])
    else:
        import xgboost as xgb
        each = max(1, Rf // 3)
        fins = [ctl.run_job(f"77_up_final_s{s}", "77", "xgboost",
                            make_config(base, spec["names"], each, spec.get("row_weight"), spec.get("params")), s, rows="full")
                for s in (42, 43, 44)]
        merged = STUDY / "final_merged_model.ubj"
        merge_boosters([STUDY / f["work_dir"] / "model" / "model.ubj" for f in fins], merged)
        mb = xgb.Booster(model_file=str(merged))
        mb.set_param({"device": "cpu", "nthread": 4})
        margin = mb.predict(xgb.DMatrix(Xv, nthread=4), output_margin=True)
        mean_margin = np.mean([_margins({"work": str(STUDY / f["work_dir"])}, Xv, each) for f in fins], axis=0)
        if not np.allclose(margin, mean_margin, atol=1e-4):
            raise RuntimeError("merged booster does not reproduce the mean margin")
        raw = np.exp(margin - margin.max(1, keepdims=True))
        raw = (raw / raw.sum(1, keepdims=True)).astype(np.float32)
        np.save(STUDY / "final_merged_val_proba.npy", raw)
        info, proba = runner.calibrate_proba("77", raw)
        proba = raw if proba is None else proba
        report = info.get("report") or runner.score("77", raw)
        (STUDY / "final_merged_calibration.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    roles = runner._arr("validation_roles")
    extra = posthoc(R, proba, raw, np.flatnonzero(roles == 1))
    main = base["final"].get("calibration", {}).get("report", base["final"]["report"])
    # Paired role-2 comparison (report-only): upgraded vs main final on the same assessment rows.
    mjob = runner.ROOT / "artifacts/training/live77_20261031_v1/jobs/77_live_final"
    mpath = mjob / "val_proba_cal.npy" if (mjob / "val_proba_cal.npy").exists() else MAIN / base["final"]["work_dir"] / "val_proba.npy"
    a_rows = np.flatnonzero(roles == 2)
    diff = (proba[a_rows].argmax(1) == R.y_val[a_rows]).astype(float) - (np.load(mpath)[a_rows].argmax(1) == R.y_val[a_rows])
    codes, inv = np.unique(R.scen[a_rows], return_inverse=True)
    sums, sizes = np.bincount(inv, weights=diff), np.bincount(inv).astype(float)
    draws = np.random.default_rng(0).integers(0, len(codes), size=(1000, len(codes)))
    paired_ci = [float(x) for x in np.percentile(sums[draws].sum(1) / sizes[draws].sum(1), [2.5, 97.5])]
    out = {"arms": results, "size_arms": size_results, "bagging": bag, "plan": plan, "posthoc": extra,
           "final_report": report, "main_final_report": main,
           "assessment_paired_diff": {"mean": float(diff.mean()), "ci95": paired_ci}}
    (STUDY / "upgrades.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    write_report(out)
    (STUDY / "upgrades_done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    ctl.state(phase="done")


def write_report(o: dict) -> None:
    pct = lambda x: f"{100 * x:.2f}%"
    a, m = o["final_report"], o["main_final_report"]
    lines = ["# Live 77-class XGBoost: post-training upgrade tests", "",
             "Fold-1 arms vs control C0 (same tuned config, fixed rounds), fresh seeds 142/143/144; "
             "J = 0.5 x role-0 acc + 0.5 x OOF acc. Only the primary arm of each technique can be admitted; the chosen "
             "spec must pass again on seeds 151-153. Selection: role 0 + training OOF only.", "",
             "| arm | family | primary | J gain (pp) | SD | unseen gain (pp) | role-0 diff CI (pp) | OOF macro-F1 change | admitted |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm, s in o["arms"].items():
        ci = s.get("role0_diff_ci95", [float("nan")] * 2)
        lines.append(f"| {arm} | {s['family']} | {'yes' if s.get('primary') else ''} | {100*s['gain']:+.2f} | "
                     f"{100*s['gain_sd']:.2f} | {100*s['unseen_gain']:+.2f} | {100*ci[0]:+.2f} to {100*ci[1]:+.2f} | "
                     f"{s['oof_f1_change']:+.4f} | {'yes' + (' (provisional)' if s.get('provisional') else '') if s['admitted'] else 'no'} |")
    conf = o["plan"].get("confirmation")
    d = o["assessment_paired_diff"]
    lines += ["", "TTL arms are a re-encoding of window aggregates (55% of windows mix initial TTLs), not full OS invariance.",
              "", "Confirmation on new seeds: " + (json.dumps(conf) if conf else "none (C0 chosen)"),
              "", "Size arms (non-inferiority, mRMR vs gain-ranked control): " + json.dumps(o["size_arms"]),
              "", f"Seed bagging at equal tree count: J gain {100*o['bagging']['stats']['gain']:+.2f} pp, "
              f"adopted: {o['plan']['bagging']}.",
              f"Final choice: {o['plan']['arm']}, size pick {o['plan']['size_pick']}, {len(o['plan']['spec']['names'])} features "
              f"(C0 = a reproduction of the main final).", "",
              "| final model | assessment acc (CI) | assessment macro-F1 | benign FPR (assess) |", "|---|---|---|---|",
              f"| main (no upgrades) | {pct(m['assessment']['accuracy'])} ({100*m['assessment']['accuracy_ci95_scenario_bootstrap'][0]:.1f}-"
              f"{100*m['assessment']['accuracy_ci95_scenario_bootstrap'][1]:.1f}) | {m['assessment']['macro_f1_supported']:.4f} | "
              f"{pct(m['assessment']['benign_false_alarm_rate'])} |",
              f"| upgraded | {pct(a['assessment']['accuracy'])} ({100*a['assessment']['accuracy_ci95_scenario_bootstrap'][0]:.1f}-"
              f"{100*a['assessment']['accuracy_ci95_scenario_bootstrap'][1]:.1f}) | {a['assessment']['macro_f1_supported']:.4f} | "
              f"{pct(a['assessment']['benign_false_alarm_rate'])} |", "",
              f"Paired assessment difference (upgraded - main): {100*d['mean']:+.2f} pp, scenario-bootstrap CI "
              f"{100*d['ci95'][0]:+.2f} to {100*d['ci95'][1]:+.2f} pp (report-only).", "",
              "Report-only output layers (role 0; conformal on raw probabilities with a role-1 threshold): " + json.dumps(o["posthoc"]), ""]
    (STUDY / "REPORT_upgrades.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    import sys
    command = (argv or sys.argv[1:] or ["run"])[0]
    STUDY.mkdir(parents=True, exist_ok=True)
    if command == "run":
        try:
            run()
        except KeyboardInterrupt:
            _log("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
