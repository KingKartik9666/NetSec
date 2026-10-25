"""Differentiable neural loss ablations with the prior native reductions intact.

Multiclass weights use PyTorch CE's sum-of-target-weights normalization. Binary
weights use the original HEM adaptation's weighted arithmetic mean. Focal loss
modulates each example by (1 - p_true)**gamma; it does not detach p_true.
"""
from __future__ import annotations

import math


def _validate(loss: str, smoothing: float, gamma: float) -> None:
    if loss not in {"native_ce", "label_smoothed_ce", "focal"}:
        raise ValueError(f"Unsupported neural loss: {loss}")
    if not math.isfinite(smoothing) or not 0 <= smoothing <= 1:
        raise ValueError("smoothing must be finite and in [0,1]")
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("focal gamma must be finite and nonnegative")


def multiclass_loss(logits, targets, weights=None, *, loss="native_ce", smoothing=.05, gamma=2.):
    """Weighted softmax CE, label smoothing, or genuine multiclass focal loss."""
    import torch
    from torch.nn import functional as F

    _validate(loss, smoothing, gamma)
    if loss == "native_ce" or (loss == "focal" and gamma == 0):
        return F.cross_entropy(logits, targets, weight=weights)
    if loss == "label_smoothed_ce":
        return F.cross_entropy(logits, targets, weight=weights, label_smoothing=smoothing)
    ce = F.cross_entropy(logits, targets, reduction="none")
    focal = (-torch.expm1(-ce)).pow(gamma) * ce
    if weights is None:
        return focal.mean()
    target_weights = weights[targets]
    return (focal * target_weights).sum() / target_weights.sum()


def binary_loss(logits, targets, weights=None, *, loss="native_ce", smoothing=.05, gamma=2.):
    """Binary BCE variants, weighted by the original unsmoothed class label."""
    import torch
    from torch.nn import functional as F

    _validate(loss, smoothing, gamma)
    binary_targets = targets.to(dtype=logits.dtype)
    if loss == "label_smoothed_ce":
        binary_targets = binary_targets * (1 - smoothing) + .5 * smoothing
    values = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction="none")
    if loss == "focal" and gamma:
        values = (-torch.expm1(-values)).pow(gamma) * values
    if weights is not None:
        values = values * weights[targets.long()]
    return values.mean()
