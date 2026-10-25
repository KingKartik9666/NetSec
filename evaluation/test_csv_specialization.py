"""Contracts preventing live-profile leakage into the dedicated CSV study."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_csv_ranking_ignores_live_metrics_and_prefers_accuracy():
    from experiment.csv_study_policy import ranking
    high_accuracy = {'model_id': 'kan', 'metrics': {'accuracy': .91, 'macro_f1': .5}}
    high_f1 = {'model_id': 'kan', 'metrics': {'accuracy': .85, 'macro_f1': .8}}
    assert ranking(high_accuracy) > ranking(high_f1)
    xgb = {'model_id': 'xgboost', 'metrics': {'accuracy': .91, 'macro_f1': .5},
           'profile_metrics': {'packet_counts_7': {'accuracy': 1., 'macro_f1': 1.}}}
    assert ranking(xgb) == (.91, .5)


def test_csv_loading_keeps_original_rows_missing_values_and_locks_test(tmp_path):
    from experiment.train_csv_models import load_csv_data
    manifest = {'feature_names': ['a', 'b'], 'classes': ['benign', 'attack'], 'benign_index': 0}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    for split, values, labels, ids in [('train', [[0., np.nan], [2., 3.]], [0, 1], [10, 20]),
                                      ('validation', [[1., 4.]], [1], [30])]:
        for kind, data in [('X', values), ('y', labels), ('row_ids', ids)]:
            np.save(tmp_path / f'{kind}_{split}.npy', np.array(data))
    (tmp_path / 'X_test.npy').write_bytes(b'locked, deliberately invalid')
    _, train, y, val, v, provenance = load_csv_data(tmp_path)
    assert train.shape == (2, 2) and val.shape == (1, 2)
    assert train[0, 0] == 0 and np.isnan(train[0, 1])
    assert y.tolist() == [0, 1] and v.tolist() == [1]
    assert provenance['masked_training_rows'] == 0
    assert provenance['effective_training_rows'] == 2
    assert provenance['test_loaded'] is False


def test_csv_policy_has_only_three_families_and_independent_ablations():
    from experiment.csv_study_policy import controls, independent_trials
    base = controls()
    assert set(base) == {'xgboost', 'kan', 'hem'}
    for trial in independent_trials():
        original = base[trial['model_id']]
        changed = [key for key in trial['config'] if trial['config'][key] != original.get(key)]
        assert len(changed) == 1
        assert trial['input_profile'] == 'full_csv'


def test_quantile_preprocessing_is_fitted_on_training_only():
    from framework.training.neural import _fit_preprocessing
    train = np.array([[0.], [1.], [2.], [1000.]], dtype=np.float32)
    val = np.array([[1.], [1e9]], dtype=np.float32)
    pipeline, X, V = _fit_preprocessing('kan', train, val, {'scaling': 'quantile'}, 42)
    assert X[1, 0] == pytest.approx(-1 / 3, abs=1e-5)
    np.testing.assert_allclose(V[:, 0], [-1 / 3, 1.], atol=1e-5)
    assert pipeline.n_features_in_ == 1


@pytest.mark.parametrize('family', ['xgboost', 'kan', 'hem'])
def test_csv_cuda_fit_and_cpu_reload(family, tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA research verification')
    from experiment.train_csv_models import train_candidate, reload_predictions
    rng = np.random.default_rng(73)
    X = rng.normal(size=(90, 5)).astype(np.float32)
    y = np.tile(np.arange(3), 30)
    X[:, 0] += y * 4
    config = {'num_boost_round': 3, 'early_stopping_rounds': 2, 'max_depth': 2,
              'weight_strategy': 'none', 'epochs': 2, 'batch_size': 32,
              'hidden': [8], 'binary_hidden': [8], 'pca_components': 3,
              'knn_k': 1, 'knn_max_rows': 60, 'selection_metric': 'accuracy',
              'scaling': 'quantile', 'scheduler': 'cosine'}
    result = train_candidate(family, X[:60], y[:60], X[60:], y[60:], 3, 0, config, 42, tmp_path)
    reloaded = reload_predictions(family, Path(result['checkpoint_path']), X[60:])
    np.testing.assert_array_equal(reloaded, result['predictions'])
    assert result['device'].startswith('cuda')
    assert 'accuracy' in result['selection_criterion']
    if family != 'xgboost':
        best = max(result['history'], key=lambda r: (r['validation_accuracy'], r['validation_macro_f1']))
        assert result['best_epoch'] == best['epoch']


def test_csv_transition_preserves_deadline_and_keeps_rf_out_of_search():
    from experiment.optimize_csv_models import transition_state
    old = {'started': 100., 'deadline': 86500., 'search_deadline': 64900., 'status': 'stopped'}
    state = transition_state(old, now=1000.)
    assert state['deadline'] == 86500. and state['search_deadline'] == 64900.
    assert state['target_accuracy'] == .96
    assert state['optimized_families'] == ['xgboost', 'kan', 'hem']
    assert state['live_stage']['status'] == 'blocked'
    with pytest.raises(ValueError, match='expired'):
        transition_state(old, now=86501.)


def test_csv_request_rejects_partial_profile_and_pins_representation(tmp_path):
    from experiment.optimize_csv_models import prepare_request
    (tmp_path / 'manifest.json').write_text('{}')
    trial = {'model_id': 'xgboost', 'input_profile': 'packet_counts_7', 'config': {}}
    with pytest.raises(ValueError, match='full_csv'):
        prepare_request(trial, tmp_path, 'partition')


def test_csv_fit_timeout_uses_its_own_stop_and_state_files(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time
    from experiment import optimize_models as lifecycle
    real = subprocess.Popen
    def sleeper(command, **kwargs):
        if 'experiment.train_csv_models' in command:
            return real([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        return real(command, **kwargs)
    monkeypatch.setattr(lifecycle.subprocess, 'Popen', sleeper)
    monkeypatch.setattr(lifecycle, 'ROOT', tmp_path)
    state = {'started': time.time(), 'deadline': time.time() + .5}
    folder = tmp_path / 'csv'
    result = lifecycle.run_fit({'model_id': 'kan'}, folder / 'runs' / 'one', state,
                                study=folder, worker_module='experiment.train_csv_models',
                                on_update=lambda _: None)
    assert result is None
    saved = json.loads((folder / 'state.json').read_text())
    assert saved['active_trial'] is None
    failure = next((folder / 'runs').rglob('failure.json'))
    assert 'deadline' in json.loads(failure.read_text())['reason']


def test_combinations_require_independent_measured_improvements():
    from experiment.csv_study_policy import supported_combination
    control = {'config': {'lr': .01, 'scaling': 'minmax'}, 'metrics': {'accuracy': .8, 'macro_f1': .7}}
    good = {'variant': 'ablation_0', 'config': {'lr': .02, 'scaling': 'minmax'},
            'metrics': {'accuracy': .81, 'macro_f1': .7}}
    bad = {'variant': 'ablation_1', 'config': {'lr': .01, 'scaling': 'quantile'},
           'metrics': {'accuracy': .79, 'macro_f1': .9}}
    assert supported_combination('kan', control, [good, bad]) is None
    bad['metrics']['accuracy'] = .82
    assert supported_combination('kan', control, [good, bad])['config'] == {'lr': .02, 'scaling': 'quantile'}


def test_resume_freezes_an_existing_combination_request(tmp_path):
    from experiment.optimize_csv_models import frozen_trial
    folder = tmp_path / 'trial' / 'attempt_000'
    folder.mkdir(parents=True)
    original = {'model_id': 'kan', 'variant': 'supported_combination', 'seed': 42,
                'config': {'lr': .001}, 'input_profile': 'full_csv', 'research_policy': 'csv_v1'}
    (folder / 'request.json').write_text(json.dumps(original))
    changed = {**original, 'config': {'lr': .002}}
    assert frozen_trial(folder.parent, changed)['config'] == {'lr': .001}


def test_failed_repeats_leave_study_resumable():
    from experiment.optimize_csv_models import terminal_status
    assert terminal_status({'models': {'kan': {'repeats_complete': False}}}, stopped=False, expired=False) == 'incomplete'
    assert terminal_status({'models': {'kan': {'repeats_complete': True}}}, stopped=False, expired=False) == 'completed'


def test_progress_snapshot_reports_false_alarms_and_recall():
    from framework.training.csv_monitoring import snapshot
    scores = snapshot(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 2, 0)
    assert scores['accuracy'] == .75
    assert scores['macro_recall'] == .75
    assert scores['false_positive_rate'] == .5
    assert scores['attack_missed_as_benign_rate'] == 0.


def test_plateau_search_expands_capacity_and_logs_its_reason():
    from experiment.csv_adaptive_policy import adaptive_decision
    base = {'model_id': 'kan', 'variant': 'control', 'config': {'hidden': [96, 48], 'grid_size': 5},
            'metrics': {'accuracy': .8, 'macro_f1': .6, 'macro_recall': .5, 'false_positive_rate': .01}}
    rows = [{**base, 'completed_at': i, 'variant': f'adaptive_{i:03d}'} for i in range(8)]
    trial = adaptive_decision('kan', rows, 9, set())
    assert trial['decision']['plateau'] is True
    assert trial['decision']['recent_accuracy_gain'] == 0.
    assert trial['config'] != base['config']
    assert trial['decision']['monitored_metrics']['macro_recall'] == .5
