"""Metrics for cascade v3 on the original 30,000 validation rows and their frozen positional roles.

Roles come from the frozen ten-model study (``validation_roles.npy``: -1 excluded, 0 tuning,
1 calibration, 2 assessment) so new and frozen models are scored on identical rows. Selection uses the
tuning role only; the assessment role is the headline. Intervals resample whole scenarios
(``label_full``), because windows inside one scenario overlap and are not independent.
"""
from __future__ import annotations

import numpy as np

ROLE_NAMES = {0: "tuning", 1: "calibration", 2: "assessment"}


def _f1_per_class(y: np.ndarray, pred: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    tp = np.bincount(y[y == pred], minlength=n).astype(float)
    support = np.bincount(y, minlength=n).astype(float)
    predicted = np.bincount(pred, minlength=n).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros(n), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(n), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n), where=(precision + recall) > 0)
    return f1, recall


def metrics(y: np.ndarray, pred: np.ndarray, n_classes: int, benign_index: int | None = 0) -> dict:
    """Accuracy, declared/supported macro-F1, benign false-alarm and attack-miss rates."""
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    f1, recall = _f1_per_class(y, pred, n_classes)
    supported = np.bincount(y, minlength=n_classes) > 0
    out = {
        "rows": int(len(y)),
        "accuracy": float(np.mean(y == pred)) if len(y) else float("nan"),
        "macro_f1_declared": float(f1.mean()),
        "macro_f1_supported": float(f1[supported].mean()) if supported.any() else float("nan"),
        "supported_classes": int(supported.sum()),
    }
    if benign_index is not None:
        benign = y == benign_index
        attack = ~benign
        out["benign_false_alarm_rate"] = float(np.mean(pred[benign] != benign_index)) if benign.any() else None
        out["attack_missed_as_benign_rate"] = float(np.mean(pred[attack] == benign_index)) if attack.any() else None
    out["per_class_recall"] = {int(c): float(recall[c]) for c in np.flatnonzero(supported)}
    return out


def scenario_bootstrap_ci(y: np.ndarray, pred: np.ndarray, scenarios: np.ndarray, reps: int = 1000,
                          seed: int = 0) -> tuple[float, float]:
    """95% percentile interval for accuracy, resampling whole scenarios with replacement."""
    correct = (np.asarray(y) == np.asarray(pred)).astype(float)
    codes, inverse = np.unique(scenarios, return_inverse=True)
    hits = np.bincount(inverse, weights=correct, minlength=len(codes))
    sizes = np.bincount(inverse, minlength=len(codes)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(codes), size=(reps, len(codes)))
    acc = hits[draws].sum(1) / sizes[draws].sum(1)
    lo, hi = np.percentile(acc, [2.5, 97.5])
    return float(lo), float(hi)


def report(y: np.ndarray, proba: np.ndarray, roles: np.ndarray, seen: np.ndarray, scenarios: np.ndarray,
           n_classes: int, benign_index: int | None, focus_classes: list[int] | None = None) -> dict:
    """Full report: per role, seen/unseen split of the assessment role, scenario-bootstrap CI."""
    pred = np.asarray(proba).argmax(1)
    out = {"full_validation": metrics(y, pred, n_classes, benign_index)}
    for code, name in ROLE_NAMES.items():
        mask = roles == code
        out[name] = metrics(y[mask], pred[mask], n_classes, benign_index)
    assess = roles == 2
    for label, mask in (("assessment_seen_scenarios", assess & seen), ("assessment_unseen_scenarios", assess & ~seen)):
        m = metrics(y[mask], pred[mask], n_classes, benign_index)
        out[label] = {k: v for k, v in m.items() if k != "per_class_recall"}
    out["assessment"]["accuracy_ci95_scenario_bootstrap"] = scenario_bootstrap_ci(y[assess], pred[assess], scenarios[assess])
    if focus_classes:
        recall = out["full_validation"]["per_class_recall"]
        out["focus_class_recall"] = {int(c): recall.get(int(c)) for c in focus_classes}
    return out
