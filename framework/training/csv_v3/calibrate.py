"""Post-hoc probability calibration for cascade v3 predictors, fitted on the validation calibration role only.

Both calibrator kinds operate on smoothed log-probabilities ``z = log((1 - alpha) p + alpha / K)``. The
smoothing weight ``alpha`` is fitted with the temperature (not a hard-coded floor), because some families
(random forest) emit exact zeros and a fixed clip value would otherwise decide the fitted temperature.

* ``temperature``  ``softmax(z / T)`` -- never changes the predicted class, only confidence.
* ``vector``       ``softmax(a * z + b)`` -- a scale and a per-class bias with an L2 penalty ``lam``. The bias
  can change predictions (a class-prior correction). Classes with fewer than ``MIN_SUPPORT`` calibration rows
  keep ``b = 0`` and the free biases are constrained to mean zero, so unsupported classes keep the average
  offset instead of being pushed down by a shared shift.

The runner decides whether a decision-changing calibrator is adopted (validation tuning role only, with a
scenario-disjoint check); this module only fits, applies and measures. One-hot outputs (HEM's hard vote, frozen
argmax predictions) carry no probabilities and are reported as not calibratable.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

EPS = 1e-7          # reporting floor for NLL only
MIN_SUPPORT = 10
LAMBDAS = (1e-3, 1e-2, 1e-1)
ECE_BINS = 15
# alpha grid: lower end keeps alpha/K above EPS (EPS stays a reporting floor); the 0.5 cap is a deliberate
# regulariser (tuning-role NLL is better at <=0.5 than at the larger alpha calibration-role NLL would pick).
ALPHA_BOUNDS = (1e-5, 0.5)


def logits(p: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if alpha > 0:
        p = (1.0 - alpha) * p + alpha / p.shape[1]
    return np.log(np.clip(p, EPS, 1.0))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def is_one_hot(p: np.ndarray) -> bool:
    p = np.asarray(p)
    return bool(np.all((p == 0) | (p == 1)))


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    q = np.array(p, dtype=np.float64)
    q[np.arange(len(y)), y] -= 1.0
    return float(np.mean((q ** 2).sum(1)))


def ece(p: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float:
    """Top-label expected calibration error with equal-width confidence bins."""
    conf, pred = p.max(1), p.argmax(1)
    idx = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    hit = (pred == y).astype(float)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def quality(p: np.ndarray, y: np.ndarray) -> dict:
    return {"rows": int(len(y)), "accuracy": float(np.mean(p.argmax(1) == y)) if len(y) else float("nan"),
            "nll": nll(p, y), "ece": ece(p, y), "brier": brier(p, y),
            "zero_true_prob_rate": float(np.mean(p[np.arange(len(y)), y] < EPS)) if len(y) else float("nan")}


def _temperature_nll(z: np.ndarray, y: np.ndarray, log_t: float) -> float:
    zz = z / np.exp(log_t)
    zz = zz - zz.max(1, keepdims=True)
    return float(np.mean(np.log(np.exp(zz).sum(1)) - zz[np.arange(len(y)), y]))


def fit_temperature(p: np.ndarray, y: np.ndarray) -> dict:
    """Jointly fit the temperature and the smoothing weight alpha (log-alpha grid, bounded 1-D search for T).

    Temperature is applied to smoothed probabilities, so it never changes the predicted class.
    """
    best = None
    for alpha in np.geomspace(*ALPHA_BOUNDS, 14):
        z = logits(p, alpha)
        res = minimize_scalar(lambda t: _temperature_nll(z, y, t), bounds=(-3.0, 3.0), method="bounded")
        if best is None or res.fun < best[0] - 1e-9:
            best = (float(res.fun), float(alpha), float(np.exp(res.x)))
    return {"kind": "temperature", "temperature": best[2], "alpha": best[1], "train_nll": best[0],
            "alpha_at_cap": bool(np.isclose(best[1], ALPHA_BOUNDS[1]))}


def fit_vector(p: np.ndarray, y: np.ndarray, n_classes: int, lam: float, alpha: float = 0.0,
               temperature: float = 1.0) -> dict:
    z = logits(p, alpha)
    support = np.bincount(y, minlength=n_classes)
    free = np.flatnonzero(support >= MIN_SUPPORT)
    rows = np.arange(len(y))
    onehot = np.zeros((len(y), n_classes))
    onehot[rows, y] = 1.0

    def unpack(theta):
        b = np.zeros(n_classes)
        if len(free):
            b[free] = theta[1:] - theta[1:].mean()      # mean-zero: no shared shift against unsupported classes
        return np.exp(theta[0]), b

    def loss_grad(theta):
        a, b = unpack(theta)
        s = a * z + b
        s = s - s.max(1, keepdims=True)
        lse = np.log(np.exp(s).sum(1))
        loss = np.mean(lse - s[rows, y]) + 0.5 * lam * float(b[free] @ b[free])
        g = (np.exp(s - lse[:, None]) - onehot) / len(y)
        grad = np.empty_like(theta)
        grad[0] = a * float((g * z).sum())
        if len(free):
            gb = g.sum(0)[free]
            grad[1:] = (gb - gb.mean()) + lam * b[free]
        return loss, grad

    theta0 = np.zeros(1 + len(free))
    theta0[0] = -np.log(temperature)
    res = minimize(loss_grad, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    a, b = unpack(res.x)
    return {"kind": "vector", "lam": lam, "alpha": alpha, "scale": float(a), "bias": b.tolist(),
            "free_classes": int(len(free)), "converged": bool(res.success), "train_objective": float(res.fun)}


def candidates(p_cal: np.ndarray, y_cal: np.ndarray, n_classes: int) -> list[dict]:
    temp = fit_temperature(p_cal, y_cal)
    out = [temp]
    out += [fit_vector(p_cal, y_cal, n_classes, lam, temp["alpha"], temp["temperature"]) for lam in LAMBDAS]
    return out


def apply(cfg: dict, p: np.ndarray) -> np.ndarray:
    z = logits(p, cfg.get("alpha", 0.0))
    if cfg["kind"] == "temperature":
        return softmax(z / cfg["temperature"]).astype(np.float32)
    if cfg["kind"] == "vector":
        return softmax(cfg["scale"] * z + np.asarray(cfg["bias"])).astype(np.float32)
    raise ValueError(f"unknown calibrator {cfg['kind']}")
