"""Declared-label, supported-label and confidence diagnostics, without calibration."""
from __future__ import annotations

import numpy as np

from framework.training.tracking import classification_metrics


def measure(truth, probabilities, classes, benign_index):
    truth = np.asarray(truth, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(truth), len(classes)) or not np.isfinite(p).all():
        raise ValueError('Invalid probability matrix')
    if np.any(p < 0) or not np.allclose(p.sum(1), 1., atol=2e-5):
        raise ValueError('Probabilities must be nonnegative and sum to one')
    predictions = p.argmax(1)
    result = classification_metrics(truth, predictions, classes, benign_index)
    supported = [item['f1-score'] for item in result['per_class'].values() if item['support']]
    result['macro_f1_declared'] = result.pop('macro_f1')
    result['macro_f1_supported'] = float(np.mean(supported))
    result['supported_classes'] = len(supported)
    result['nll'] = float(-np.log(np.maximum(p[np.arange(len(truth)), truth], 1e-15)).mean())
    result['multiclass_brier'] = float(np.mean(np.sum(p*p, axis=1)
                                       - 2*p[np.arange(len(truth)), truth] + 1))
    confidence = p.max(1)
    bins = np.minimum((confidence*15).astype(int), 14)
    result['uncalibrated_ece15'] = float(sum(
        np.mean(bins == b)*abs(np.mean(predictions[bins == b] == truth[bins == b])
                              - np.mean(confidence[bins == b]))
        for b in range(15) if np.any(bins == b)))
    recon = np.array([name.startswith('recon_') for name in classes])[truth]
    result['recon_exact_accuracy'] = float(np.mean(predictions[recon] == truth[recon])) if recon.any() else None
    result['nonrecon_exact_accuracy'] = float(np.mean(predictions[~recon] == truth[~recon])) if (~recon).any() else None
    return result
