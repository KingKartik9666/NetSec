"""Shared contract between the cascade v3 runner and the per-family adapters.

Every adapter module exposes::

    FAMILIES: tuple[str, ...]
    def default_config(family: str, task: str) -> dict
    def fit(family: str, data: TaskData, config: dict, seed: int, out_dir: Path) -> dict
    def predict_cpu(family: str, out_dir: Path, X: np.ndarray) -> np.ndarray   # (n, n_classes) float32

``fit`` trains on CUDA (the fixed RF control is CPU by design), writes its checkpoint inside ``out_dir``
and returns a JSON-serialisable record. It must never read files outside the arrays it is given, never
touch the held-out test partition, and never fall back silently to another device or model. The runner
calls ``fit`` in a fresh subprocess per job, then calls ``predict_cpu`` in another fresh process to
produce the CPU replay used for every reported metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TASKS = ("77", "8", "84")


@dataclass(frozen=True)
class TaskData:
    """Arrays for one fit. Labels are contiguous ``0..n_classes-1``."""

    task: str                     # "77", "8" or "84"
    X_fit: np.ndarray             # float32 (n_fit, F); NaN allowed, no inf
    y_fit: np.ndarray             # int64 (n_fit,)
    X_tune: np.ndarray            # float32 (n_tune, F): early stopping / checkpoint selection only
    y_tune: np.ndarray            # int64 (n_tune,)
    classes: list[str]            # class names in index order
    feature_names: list[str]
    benign_index: int | None      # 0 for "77"/"84"; None for the recon specialist "8"
    extras: dict = field(default_factory=dict)   # optional, adapter-specific, documented by the runner

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def validate(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task}")
        for name, x in (("X_fit", self.X_fit), ("X_tune", self.X_tune)):
            if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != len(self.feature_names):
                raise ValueError(f"{name} must be float32 (n, {len(self.feature_names)})")
            if np.isinf(x).any():
                raise ValueError(f"{name} contains inf")
        for name, y in (("y_fit", self.y_fit), ("y_tune", self.y_tune)):
            if y.dtype != np.int64 or y.min() < 0 or y.max() >= self.n_classes:
                raise ValueError(f"{name} must be int64 in [0, {self.n_classes})")
        if np.unique(self.y_fit).size != self.n_classes:
            raise ValueError("every class must occur in y_fit")
        if self.benign_index is not None and self.classes[self.benign_index] != "benign":
            raise ValueError("benign_index does not point at 'benign'")
