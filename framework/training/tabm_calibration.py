"""Post-selection calibration; alert policy and class argmax remain distinct."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from experiment.calibrate_datasense_models import temperature_fit, assessment, confidence_metrics
from experiment.datasense_quality import temperature_probabilities


def choose_threshold(truth, attack_probability, false_alarm_cap=.02):
    attack = np.asarray(truth) != 0
    if len(np.unique(attack)) != 2:
        raise ValueError('Threshold tuning requires benign and attack support')
    fpr, tpr, thresholds = roc_curve(attack, attack_probability, drop_intermediate=False)
    eligible = np.flatnonzero((fpr <= false_alarm_cap) & np.isfinite(thresholds))
    if not len(eligible):
        # All observations tied: an explicit no-alert threshold satisfies the cap.
        return float(np.nextafter(1., 2.))
    index = max(eligible, key=lambda i: (tpr[i], -fpr[i], thresholds[i]))
    return float(thresholds[index])


def binary_metrics(truth, probability, threshold):
    attack = np.asarray(truth) != 0; pred = probability >= threshold
    return dict(accuracy=float(np.mean(attack == pred)),
        attack_recall=float(pred[attack].mean()) if attack.any() else None,
        false_positive_rate=float(pred[~attack].mean()) if (~attack).any() else None,
        brier=float(np.mean((probability-attack)**2)))


def fit_calibration(probabilities, truth, roles, manifest):
    p = np.asarray(probabilities, dtype=float); y = np.asarray(truth)
    tune, fit, assess_ids = [np.flatnonzero(np.asarray(roles) == i) for i in (0, 1, 2)]
    if any(len(a) == 0 for a in (tune, fit, assess_ids)):
        raise ValueError('Distinct tuning, calibration and assessment roles required')
    temperature = temperature_fit(p[fit], y[fit])
    calibrated = temperature_probabilities(p, temperature)
    np.testing.assert_array_equal(calibrated.argmax(1), p.argmax(1))
    raw_attack = 1-p[:, 0]
    threshold = choose_threshold(y[tune], raw_attack[tune])
    features = logit(np.clip(raw_attack, 1e-6, 1-1e-6))[:, None]
    estimator = LogisticRegression(C=1., random_state=42, max_iter=1000).fit(features[fit], y[fit] != 0)
    slope, intercept = float(estimator.coef_[0, 0]), float(estimator.intercept_[0])
    # Monotonic confidence correction preserves the independently tuned alert policy.
    monotonic_fallback = slope <= 0
    if monotonic_fallback:
        slope, intercept = 1., 0.
    attack_calibrated = expit(slope*features[:, 0]+intercept)
    attack_true = y[assess_ids] != 0
    subtype = p[:, 1:].argmax(1)+1
    gate = np.where(raw_attack >= threshold, subtype, 0)
    result = dict(temperature=temperature, binary_platt=dict(slope=slope, intercept=intercept,
        monotonic_identity_fallback=monotonic_fallback), raw_attack_threshold=threshold,
        threshold_false_alarm_cap=.02, threshold_fit_role='tuning', calibration_fit_role='calibration',
        calibration_rows=len(fit), assessment_rows=len(assess_ids), predictions_unchanged=True,
        before=assessment(p[assess_ids], y[assess_ids], manifest),
        after=assessment(calibrated[assess_ids], y[assess_ids], manifest),
        binary_before=binary_metrics(y[assess_ids], raw_attack[assess_ids], threshold),
        binary_probability_after=dict(brier=float(np.mean((attack_calibrated[assess_ids]-attack_true)**2)),
            reliability=confidence_metrics(attack_true, attack_calibrated[assess_ids])),
        subtype_accuracy_on_true_attacks=float(np.mean(subtype[assess_ids][attack_true] == y[assess_ids][attack_true])) if attack_true.any() else None,
        gated_exact_accuracy=float(np.mean(gate[assess_ids] == y[assess_ids])),
        gated_exact_attack_recall=float(np.mean(gate[assess_ids][attack_true] == y[assess_ids][attack_true])) if attack_true.any() else None,
        assessment_used_for_fitting_or_selection=False, test_loaded=False,
        warning='Previously reused within-capture validation; exploratory, not an independent test.')
    return result, calibrated, attack_calibrated


def apply_calibration(probabilities, calibration):
    p = np.asarray(probabilities, dtype=float)
    cfg = calibration['binary_platt']
    return (temperature_probabilities(p, calibration['temperature']),
            expit(cfg['slope']*logit(np.clip(1-p[:, 0], 1e-6, 1-1e-6))+cfg['intercept']))
