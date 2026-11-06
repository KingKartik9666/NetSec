"""Independent study selection and restart safety contracts."""
import copy
import importlib
from pathlib import Path

import numpy as np
import pytest


def runner():
    return importlib.import_module('framework.training.csv_redesign.runner')


def score(accuracy=.8, macro=.7, fpr=.01):
    return {'accuracy': accuracy, 'macro_f1': macro, 'false_positive_rate': fpr}


def test_representation_selection_rejects_accuracy_gain_with_macro_or_false_alarm_regression():
    records = {arm: [dict(seed=s, selection_metrics=score()) for s in (42, 43, 44)]
               for arm in ('base_unfiltered', 'base_clean', 'enriched_clean')}
    records['base_unfiltered'] = [dict(seed=s, selection_metrics=score(.79, .69, .02)) for s in (42, 43, 44)]
    records['enriched_clean'] = [dict(seed=s, selection_metrics=score(.85, .60, .03)) for s in (42, 43, 44)]
    assert runner().select_representation(records)['arm'] == 'base_clean'
    records['enriched_clean'] = [dict(seed=s, selection_metrics=score(.85, .71, .009)) for s in (42, 43, 44)]
    assert runner().select_representation(records)['arm'] == 'enriched_clean'


def test_matched_comparison_requires_complete_same_seed_set():
    records = {'base_clean': [dict(seed=42, selection_metrics=score())],
               'enriched_clean': [dict(seed=43, selection_metrics=score(.9))]}
    with pytest.raises(ValueError, match='seed'):
        runner().select_representation(records)


def test_resumed_request_fingerprint_cannot_be_edited():
    from framework.training.live_study.protocol import fingerprint
    request = dict(family='xgboost', seed=42, arm='base_clean')
    request['request_fingerprint'] = fingerprint(request)
    runner().validate_request(request)
    request['seed'] = 43
    with pytest.raises(ValueError, match='request'):
        runner().validate_request(request)


def test_family_configuration_preserves_true_seed_and_larger_csv_budget():
    import optuna
    cfg = runner().configuration(optuna.trial.FixedTrial({'max_depth': 5, 'learning_rate': .05,
        'min_child_weight': 1., 'subsample': .8, 'colsample_bytree': .9,
        'reg_lambda': 2., 'reg_alpha': .01, 'weight_strategy': 'none'}), 'xgboost', False)
    assert cfg['n_estimators'] == 1800
    assert cfg['runtime_seconds'] >= 7200
    assert cfg['early_stopping_rounds'] >= 100


def test_worker_views_keep_challenge_outside_fit_tune_and_restore_only_original_fitting_rows():
    worker = importlib.import_module('framework.training.csv_redesign.worker')
    roles = np.array([0, -1, 1, 2, 3, 4, -1], dtype=np.int8)
    unfiltered = np.array([True, True, False, False, False, False, False])
    manifest = {'feature_names': ['a', 'b', 'c'], 'base_feature_indices': [0, 1],
                'selected_feature_indices': [0, 1, 2], 'feature_groups': {'port_geometry': [2]}}
    indices, changed = worker.arm_view('base_unfiltered', manifest, roles, unfiltered)
    assert indices == [0, 1]
    np.testing.assert_array_equal(changed, [0, 0, 1, 2, 3, 4, -1])
    indices, unchanged = worker.arm_view('enrich_no_port_geometry', manifest, roles, unfiltered)
    assert indices == [0, 1]
    np.testing.assert_array_equal(unchanged, roles)
    np.testing.assert_array_equal(roles, [0, -1, 1, 2, 3, 4, -1])


def test_challenge_result_is_not_a_selection_metric():
    candidate = {'selection_metrics': score(.8), 'challenge_metrics': score(.99)}
    assert runner().selection_value(candidate) == .8


def test_path_boundary_rejects_outside_study(tmp_path):
    with pytest.raises(ValueError, match='study'):
        runner().study_path(tmp_path)
