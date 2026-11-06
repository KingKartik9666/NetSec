"""Paired arm comparison for fold-1 feature tests (shared helpers; import-safe: no environment or study changes).

An arm is a fold-1 refit of one fixed config on a feature list, paired with the control C0 on the same seed.
J = 0.5 x role-0 accuracy + 0.5 x fold-1 OOF accuracy (live77 upgrades rule). Role 0 is split into seen and unseen
scenarios with the ``prep.py`` definition (a validation scenario is "seen" when it occurs anywhere in training).
Logic copied (not imported) from ``experiment/train_live77.py`` and ``experiment/train_live77_upgrades.py``: those
drivers set CASCADE_STUDY / CASCADE_DATA at import. Only roles 0 and training OOF rows are scored here.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from framework.training.csv_v3 import evaluate

GAIN = 0.001                       # 0.1 pp
F1_TOL = 0.005                     # OOF macro-F1 may fall by at most this much


# ---------------------------------------------------------------------------------------- rows + metrics
def role_split(roles: np.ndarray, scen_val: np.ndarray, scen_train: np.ndarray, role: int = 0):
    """Rows of one validation role, and its seen/unseen-scenario parts (``prep.py``: np.isin(scen_va, scen_tr))."""
    rows = np.flatnonzero(roles == role)
    seen = np.isin(scen_val[rows], scen_train)
    return rows, rows[seen], rows[~seen]


def macro_f1(y, pred, n: int) -> float:
    """Macro-F1 over the classes present in ``y``."""
    y = np.asarray(y, int)
    f1, _ = evaluate._f1_per_class(y, np.asarray(pred, int), n)
    return float(f1[np.bincount(y, minlength=n) > 0].mean())


def role0_metrics(proba: np.ndarray, y_val: np.ndarray, tune, seen, unseen, n: int) -> dict:
    """Role-0 numbers only (never calibration or assessment rows). ``proba`` covers all validation rows."""
    pred = proba.argmax(1)
    acc = lambda rows: float(np.mean(pred[rows] == y_val[rows])) if len(rows) else float("nan")
    return {"tune_acc": acc(tune), "tune_f1": macro_f1(y_val[tune], pred[tune], n),
            "tune_seen_acc": acc(seen), "tune_unseen_acc": acc(unseen)}


def oof_metrics(proba: np.ndarray, y: np.ndarray, n: int) -> dict:
    pred = proba.argmax(1)
    return {"oof_acc": float(np.mean(pred == y)), "oof_f1": macro_f1(y, pred, n)}


def J(job: dict) -> float:
    return 0.5 * job["tune_acc"] + 0.5 * job["oof_acc"]


# ------------------------------------------------------------------------------------------ paired rules
def paired(control: list[dict], runs: list[dict]) -> dict:
    """Mean paired differences (arm - control on the same seed) and the SD of the paired J gains."""
    by = {r["seed"]: r for r in control}
    diff = lambda key: np.array([r[key] - by[r["seed"]][key] for r in runs], float)
    g = np.array([J(r) - J(by[r["seed"]]) for r in runs])
    return {"gain": float(g.mean()), "gain_sd": float(g.std(ddof=1)) if len(g) > 1 else 0.0,
            "gains": [float(x) for x in g], "seeds": [r["seed"] for r in runs],
            "seen_gain": float(diff("tune_seen_acc").mean()), "unseen_gain": float(diff("tune_unseen_acc").mean()),
            "tune_f1_change": float(diff("tune_f1").mean()), "oof_f1_change": float(diff("oof_f1").mean()),
            "J": float(np.mean([J(r) for r in runs])), "J_control": float(np.mean([J(by[r["seed"]]) for r in runs])),
            "tune_acc": float(np.mean([r["tune_acc"] for r in runs])), "oof_acc": float(np.mean([r["oof_acc"] for r in runs]))}


def admitted(s: dict, n_seeds: int = 3) -> bool:
    """Plan rule: all seeds run; mean J gain >= 0.1 pp and > 2 x SD; unseen gain >= 0; OOF macro-F1 change >= -0.005."""
    return (len(s["gains"]) >= n_seeds and s["gain"] >= GAIN and s["gain"] > 2 * s["gain_sd"]
            and s["unseen_gain"] >= 0 and s["oof_f1_change"] >= -F1_TOL)


def early_drop(gains: list[float]) -> bool:
    """Drop an arm after seed 1 if its J gain <= -0.1 pp, after seed 2 if the mean J gain < 0 (saves GPU time only:
    a dropped arm cannot be admitted)."""
    mean = float(np.mean(gains))
    return (len(gains) == 1 and mean <= -GAIN) or (len(gains) == 2 and mean < 0)


def scenario_diff_ci(hits_arm: np.ndarray, hits_control: np.ndarray, scen: np.ndarray, reps: int = 1000,
                     seed: int = 0) -> list[float]:
    """Scenario-bootstrap 95% CI of an accuracy difference (arm - control) on one role's rows, resampling whole
    scenarios; per-row hits may be seed-averaged. Arms use it on role 0 (diagnostic only)."""
    diff = np.asarray(hits_arm, float) - np.asarray(hits_control, float)
    codes, inv = np.unique(scen, return_inverse=True)
    sums, sizes = np.bincount(inv, weights=diff), np.bincount(inv).astype(float)
    draws = np.random.default_rng(seed).integers(0, len(codes), size=(reps, len(codes)))
    est = sums[draws].sum(1) / sizes[draws].sum(1)
    return [float(x) for x in np.percentile(est, [2.5, 97.5])]


# --------------------------------------------------------------------------------------- feature lists
def ranked(names: list[str], *scores: dict) -> list[str]:
    """Names sorted by descending scores (later dicts break ties); remaining ties keep the given (v3) order."""
    pos = {n: i for i, n in enumerate(names)}
    return sorted(names, key=lambda n: tuple(-float(s.get(n, 0.0)) for s in scores) + (pos[n],))


def top_k(names: list[str], ranking: list[str], k: int) -> list[str]:
    """The first ``k`` of a ranking, returned in v3 order."""
    keep = set(ranking[:k])
    return [n for n in names if n in keep]


def drop_top(names: list[str], ranking: list[str], k: int) -> list[str]:
    drop = set(ranking[:k])
    return [n for n in names if n not in drop]


def arm_lists(names: list[str], rankings: dict[str, list[str]], sizes=(300, 200), fp_drop: int = 20) -> dict:
    """C0, P<k>, G<k> and FP<k> feature lists (v3 order). Every ranking must be a permutation of ``names``."""
    for key, rank in rankings.items():
        if sorted(rank) != sorted(names) or len(set(rank)) != len(rank):
            raise ValueError(f"ranking {key} is not a permutation of the v3 features")
    out = {"C0": list(names)}
    for tag in ("P", "G"):
        for k in sizes:
            out[f"{tag}{k}"] = top_k(names, rankings[tag], k)
    out[f"FP{fp_drop}"] = drop_top(names, rankings["FP"], fp_drop)
    return out


# ------------------------------------------------------------------------------------- pre-registration
def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def freeze(path: Path, value):
    """Write once with a sha256 of the value; afterwards return the stored value.

    ``value`` may be a zero-argument callable (called only when the file is missing, for expensive values).
    A stored file whose sha256 does not match its value, or a plain ``value`` that differs from the stored one,
    raises: a frozen file is never changed.
    """
    path = Path(path)
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if _sha(stored["value"]) != stored["sha256"]:
            raise RuntimeError(f"{path.name}: stored value does not match its sha256")
        if not callable(value) and _sha(json.loads(json.dumps(value))) != stored["sha256"]:
            raise RuntimeError(f"{path.name} is frozen and differs from the requested value")
        return stored["value"]
    value = json.loads(json.dumps(value() if callable(value) else value))
    # Atomic: a write cut short must not leave a half file that blocks every restart.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"value": value, "sha256": _sha(value), "frozen": time.strftime("%Y-%m-%d %H:%M:%S")},
                              indent=1), encoding="utf-8")
    for attempt in range(20):           # Windows: a reader (antivirus, indexer) can hold the target briefly
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)
    return value


# -------------------------------------------------------------------- fingerprint score (MI / CMI, CPU)
def bins(col: np.ndarray, k: int = 32) -> np.ndarray:
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


def mi(a: np.ndarray, b: np.ndarray) -> float:
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
    out = float(np.sum(pj * np.log(pj / (pa[ia] * pb[ib]))))
    return out - (len(pj) - ka - kb + 1) / (2 * n)


def fingerprint_scores(X: np.ndarray, y: np.ndarray, scen: np.ndarray, names: list[str],
                       min_rows: int = 200) -> dict[str, float]:
    """I(f; scenario | class) / (I(f; class) + 1e-3), averaged over classes with >= min_rows rows and >= 2 scenarios
    (the live77 upgrades ``shift_ranking`` estimator)."""
    classes = [c for c in np.unique(y) if (y == c).sum() >= min_rows and np.unique(scen[y == c]).size >= 2]
    w = np.array([(y == c).sum() for c in classes], float)
    w /= w.sum()
    scores = {}
    for j, n in enumerate(names):
        b = bins(X[:, j])
        m = max(mi(b, y), 0.0)
        cmi = sum(wc * max(mi(b[y == c], scen[y == c]), 0.0) for wc, c in zip(w, classes))
        scores[n] = float(cmi / (m + 1e-3))
    return scores


def per_class_sample(rows: np.ndarray, y: np.ndarray, per_class: int, seed: int = 0) -> np.ndarray:
    """Up to ``per_class`` rows of each class (``y`` aligned with ``rows``), sorted; one generator over the classes."""
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.permutation(rows[y == c])[:per_class] for c in np.unique(y)]))
