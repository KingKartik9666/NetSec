"""Separate calibration fits, frozen alert thresholds and honest HEM confidence."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from experiment.calibrate_datasense_models import temperature_fit, confidence_metrics
from experiment.datasense_quality import temperature_probabilities
from framework.training.tabm_calibration import choose_threshold, binary_metrics
from framework.training.tracking import classification_metrics


def validate_output(output, rows, classes):
    pred = np.asarray(output['pred'])
    if pred.shape != (rows,) or np.any((pred < 0) | (pred >= classes)):
        raise ValueError('Prediction schema differs from the sealed class vocabulary')
    p = output.get('probabilities', output.get('multiclass_probabilities'))
    if p is None or p.shape != (rows, classes) or not np.isfinite(p).all():
        raise ValueError('Invalid probability dimensions or nonfinite values')
    if np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=5e-6):
        raise ValueError('Probabilities must be nonnegative and sum to one')


def attack_probability(output):
    if 'probabilities' in output: return 1-output['probabilities'][:, 0]
    # This is explicitly the HEM binary component, not the hard ensemble vote.
    return output['binary_probabilities']


def correctness_features(output):
    pred = output['pred']; p = output['multiclass_probabilities']
    attack = output['binary_probabilities']
    return np.column_stack((np.log(np.clip(p[np.arange(len(pred)), pred], 1e-8, 1)),
        np.log(np.clip(np.where(pred != 0, attack, 1-attack), 1e-8, 1)),
        output['knn_predictions'] == pred))


def logistic_fit(x, target, monotonic=False):
    x = np.asarray(x); target = np.asarray(target, int)
    if len(np.unique(target)) < 2:
        prior = (target.sum()+1)/(len(target)+2)
        return dict(coefficients=[0.]*x.shape[1], intercept=float(logit(prior)),
                    fallback='smoothed constant; one target class in calibration')
    model = LogisticRegression(C=1., random_state=42, max_iter=2000).fit(x, target)
    coefficients, intercept = model.coef_[0], float(model.intercept_[0])
    if monotonic and coefficients[0] <= 0:
        return dict(coefficients=[1.], intercept=0., fallback='nonpositive slope; identity')
    return dict(coefficients=coefficients.tolist(), intercept=intercept, fallback=None)


def logistic_apply(x, config):
    return expit(np.asarray(x) @ np.asarray(config['coefficients']) + config['intercept'])


def fit(cal_output, cal_y, tune_output, tune_y):
    raw = attack_probability(cal_output)
    result = dict(method='temperature_and_monotonic_platt' if 'probabilities' in cal_output else 'HEM_hard_vote_correctness_and_binary_component_platt',
        fit_role='calibration', threshold_fit_role='tune', calibration_rows=len(cal_y),
        raw_attack_threshold=choose_threshold(tune_y, attack_probability(tune_output)),
        threshold_false_alarm_cap=.02,
        binary_platt=logistic_fit(logit(np.clip(raw, 1e-6, 1-1e-6))[:, None], cal_y != 0, True),
        classification_rule='unchanged argmax or HEM hard vote; alert policy reported separately',
        assessment_used=False, test_loaded=False)
    if 'probabilities' in cal_output:
        result['temperature'] = temperature_fit(cal_output['probabilities'], cal_y)
    else:
        result['correctness'] = logistic_fit(correctness_features(cal_output), cal_output['pred'] == cal_y)
    return result


def apply(output, config):
    raw = attack_probability(output)
    attack = logistic_apply(logit(np.clip(raw, 1e-6, 1-1e-6))[:, None], config['binary_platt'])
    if 'probabilities' in output:
        probabilities = temperature_probabilities(output['probabilities'], config['temperature'])
        np.testing.assert_array_equal(probabilities.argmax(1), output['pred'])
        return probabilities.max(1), attack, probabilities
    return logistic_apply(correctness_features(output), config['correctness']), attack, None


def wilson(successes, total):
    if not total: return None
    z = 1.959963984540054; p = successes/total; den = 1+z*z/total
    center = (p+z*z/(2*total))/den
    radius = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [float(center-radius), float(center+radius)]


def measure(output, y, manifest, config):
    validate_output(output, len(y), len(manifest['classes']))
    pred = output['pred']; raw = attack_probability(output)
    confidence, attack, p = apply(output, config)
    score = classification_metrics(y, pred, manifest['classes'], 0)
    alert = raw >= config['raw_attack_threshold']; benign = y == 0
    score['screening'] = dict(binary_metrics(y, raw, config['raw_attack_threshold']),
        benign_rows=int(benign.sum()), attack_rows=int((~benign).sum()),
        false_alarms=int(alert[benign].sum()), detected_attacks=int(alert[~benign].sum()),
        false_positive_rate_95ci=wilson(alert[benign].sum(), benign.sum()),
        attack_recall_95ci=wilson(alert[~benign].sum(), (~benign).sum()),
        component='model probability mass' if p is not None else 'HEM binary component (separate from ensemble vote)',
        calibrated_binary_brier=float(np.mean((attack-(y != 0))**2)))
    score['calibrated_correctness'] = confidence_metrics(pred == y, confidence)
    if p is not None:
        def metrics(q):
            return dict(nll=float(-np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)).mean()),
                multiclass_brier=float(np.mean(np.sum(q*q, 1)-2*q[np.arange(len(y)), y]+1)))
        score['raw_probability'] = metrics(output['probabilities'])
        score['calibrated_probability'] = metrics(p)
        score['raw_correctness'] = confidence_metrics(pred == y, output['probabilities'].max(1))
    return score
