"""Cheap full-validation telemetry; unsupported declared classes stay explicit."""
from __future__ import annotations

import numpy as np


def snapshot(truth: np.ndarray, prediction: np.ndarray, n_classes: int, benign_index: int) -> dict:
    matrix = np.bincount(n_classes * truth + prediction, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    actual, proposed, correct = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    recall = np.divide(correct, actual, out=np.zeros(n_classes), where=actual != 0)
    precision = np.divide(correct, proposed, out=np.zeros(n_classes), where=proposed != 0)
    f1 = np.divide(2 * correct, actual + proposed, out=np.zeros(n_classes), where=(actual + proposed) != 0)
    benign, attack = truth == benign_index, truth != benign_index
    return dict(accuracy=float(np.mean(truth == prediction)), macro_precision=float(precision.mean()),
                macro_recall=float(recall.mean()), macro_f1=float(f1.mean()),
                false_positive_rate=float(np.mean(prediction[benign] != benign_index)) if benign.any() else None,
                attack_missed_as_benign_rate=float(np.mean(prediction[attack] == benign_index)) if attack.any() else None)
