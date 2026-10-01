"""Auditable multiclass label-smoothed cross entropy for partial-input research.

The true softmax Hessian has off-diagonal terms. XGBoost receives its standard
positive diagonal upper bound, 2*p*(1-p), rather than the true diagonal alone.
No inference dependency on Torch or CUDA is introduced by this module.
"""
from __future__ import annotations

import numpy as np


def _checked(margins, labels, weights, smoothing):
    z = np.asarray(margins, dtype=np.float64)
    y = np.asarray(labels)
    if z.ndim != 2 or z.shape[1] < 2 or not len(z) or not np.isfinite(z).all():
        raise ValueError("Finite nonempty multiclass margins are required")
    if y.shape != (len(z),) or not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= z.shape[1]).any():
        raise ValueError("Integer labels must match the rows and class range")
    if not 0 <= smoothing < 1:
        raise ValueError("Label smoothing must be in [0, 1)")
    w = np.ones(len(z)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(z),) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Finite nonnegative row weights are required")
    return z, y, w


def smoothed_loss(margins, labels, weights=None, smoothing=0.05):
    """Weighted sum CE; target=(1-epsilon)*one_hot + epsilon/K."""
    z, y, w = _checked(margins, labels, weights, smoothing)
    shifted = z - z.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(np.sum(w * (-(1-smoothing)*logp[np.arange(len(z)), y] - smoothing*logp.mean(axis=1))))


def smoothed_derivatives(margins, labels, weights=None, smoothing=0.05):
    """Return gradient, true Hessian diagonal and diagonal solver upper bound."""
    z, y, w = _checked(margins, labels, weights, smoothing)
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    gradient = p.copy() - smoothing/z.shape[1]
    gradient[np.arange(len(z)), y] -= 1-smoothing
    gradient *= w[:, None]
    diagonal = p*(1-p)*w[:, None]
    return gradient, diagonal, np.maximum(2*diagonal, 1e-6)


class CudaSmoothedObjective:
    """Compute custom gradients on existing Torch CUDA; return XGBoost NumPy ABI.

    XGBoost 3.1 accepts NumPy/CuPy gradient arrays, not Torch tensors. Host
    transfers are explicit; softmax, gradient and Hessian arithmetic run on CUDA.
    """
    def __init__(self, labels, weights, smoothing=0.05):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the custom objective; no CPU fallback")
        if not 0 <= smoothing < 1:
            raise ValueError("Label smoothing must be in [0, 1)")
        self.torch = torch
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long, device="cuda:0")
        self.weights = torch.as_tensor(np.asarray(weights), dtype=torch.float32, device="cuda:0")[:, None]
        self.smoothing = smoothing
        self.calls = 0

    def __call__(self, margins, data):
        t = self.torch
        z = t.as_tensor(margins, dtype=t.float32, device="cuda:0")
        if z.ndim != 2 or z.shape[0] != self.labels.numel():
            raise ValueError("Custom objective margins differ from training rows")
        p = t.softmax(z, dim=1)
        hessian = (2*p*(1-p)*self.weights).clamp_min_(1e-6)
        gradient = p - self.smoothing/z.shape[1]
        gradient[t.arange(len(z), device=z.device), self.labels] -= 1-self.smoothing
        gradient *= self.weights
        self.calls += 1
        return gradient.cpu().numpy(), hessian.cpu().numpy()
