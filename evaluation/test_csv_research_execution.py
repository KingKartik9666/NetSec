"""Lifecycle contracts for open-ended research without changing deployment."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_research_state_has_no_absolute_deadline():
    from experiment.optimize_csv_research import initial_state
    state = initial_state()
    assert state['deadline'] is None
    assert state['target_accuracy'] == .95
    assert state['test_loaded'] is False
    assert state['live_training_started'] is False


def test_cuda_evidence_is_required_not_assumed_from_model_name():
    from experiment.optimize_csv_research import cuda_verified
    assert not cuda_verified({'model_id': 'xgboost', 'device': 'cpu'})
    assert cuda_verified({'device': 'cuda:0', 'device_evidence': {'parameter_devices': ['cuda:0']}})
    assert cuda_verified({'device': 'cuda:0', 'gpu_training_verified': True})
    assert not cuda_verified({'device': 'cuda:0'})


def test_watchdog_escalation_is_independent_of_overall_deadline():
    from experiment.optimize_csv_research import trial_budget
    assert trial_budget({}) == 1200
    assert trial_budget({'watchdog_seconds': 3600}) == 3600
    assert trial_budget({'watchdog_seconds': 14400}) == 14400
    with pytest.raises(ValueError):
        trial_budget({'watchdog_seconds': 0})


def test_failed_memory_fit_reduces_batch_without_dropping_rows():
    from experiment.optimize_csv_research import oom_retry
    original = {'model_id': 'hem', 'config': {'batch_size': 512, 'knn_batch_size': 64}, 'pipeline': {}, 'seed': 42}
    new = oom_retry(original)
    assert new['config']['batch_size'] == 256
    assert new['config']['knn_batch_size'] == 32
    assert original['config']['batch_size'] == 512
    assert 'training_row_limit' not in new['config']


def test_pipeline_reload_preserves_raw_schema(tmp_path):
    from experiment.train_csv_research import reload_research
    import joblib
    import xgboost as xgb
    from framework.training.research_features import ResearchTransform
    X = np.array([[0., 1.], [1., 0.], [0., 2.], [2., 0.]], dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    transform = ResearchTransform(['a', 'b'], {'feature_selection': 'none', 'feature_engineering': False}).fit(X, y)
    joblib.dump(transform, tmp_path / 'research_transform.joblib')
    model = xgb.train({'objective': 'multi:softprob', 'num_class': 2, 'max_depth': 2,
                      'min_child_weight': 0, 'nthread': 1}, xgb.DMatrix(X, label=y), 3)
    model.save_model(tmp_path / 'model.ubj')
    result = {'model_id': 'xgboost', 'checkpoint_path': str(tmp_path / 'model.ubj')}
    predicted, probability = reload_research(result, X)
    np.testing.assert_array_equal(predicted, model.predict(xgb.DMatrix(X)).argmax(1))
    assert probability.shape == (4, 2)
    with pytest.raises(ValueError):
        reload_research(result, np.ones((1, 3), dtype=np.float32))


@pytest.mark.parametrize('family', ['xgboost', 'kan', 'hem'])
def test_research_pipeline_cuda_fit_and_raw_cpu_reload(family, tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA research verification')
    from experiment.train_csv_research import fit_base
    rng = np.random.default_rng(42)
    y = np.tile(np.arange(3), 30)
    X = rng.normal(size=(90, 5)).astype(np.float32)
    X[:, 0] += 4*y
    manifest = {'classes': ['benign', 'a', 'b'], 'benign_index': 0,
                'feature_names': [f'f{i}' for i in range(5)]}
    request = {'model_id': family, 'seed': 42,
               'pipeline': {'feature_selection': 'none', 'feature_engineering': False,
                            'resampling': 'none', 'order': 'selection_then_resampling'},
               'config': {'num_boost_round': 3, 'max_depth': 2, 'epochs': 2,
                          'hidden': [8], 'binary_hidden': [8], 'pca_components': 3,
                          'batch_size': 32, 'knn_k': 1, 'knn_max_rows': 60,
                          'scheduler': 'plateau', 'loss': 'label_smoothed_ce'}}
    result = fit_base(request, manifest, X[:60], y[:60], np.arange(60), X[60:], y[60:], tmp_path)
    assert result['checkpoint_reload_verified'] is True
    assert result['original_training_rows'] == 60
    assert result['synthetic_training_rows'] == 0


def test_xgboost_honors_prune_and_records_real_training_monitor(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA research verification')
    from framework.training.csv_xgboost import train_csv_xgboost
    X = np.random.default_rng(42).normal(size=(60, 3)).astype(np.float32)
    y = np.tile(np.arange(3), 20)
    (tmp_path / 'PRUNE').touch()
    result = train_csv_xgboost(X[:45], y[:45], X[45:], y[45:], 3,
                              {'num_boost_round': 8, 'max_depth': 2,
                               'original_training_rows': 40}, 42, tmp_path)
    assert len(result['history']) == 1
    assert result['pruned'] is True
    assert result['history'][0]['training_monitor_rows'] == 40
    assert 0 <= result['history'][0]['training_accuracy'] <= 1


def test_stop_preserves_scheduled_request_for_restart(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from experiment import optimize_csv_research as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'STUDY', tmp_path)
    monkeypatch.setattr(module, 'all_locks', nullcontext)
    monkeypatch.setattr(module.lifecycle, 'protect_descendants', lambda: None)
    monkeypatch.setattr(module, 'pin_inputs', lambda: {})
    monkeypatch.setattr(module, 'success_report', lambda *a: {'success': False})
    definition = {'source_input_hashes': {}, 'controls': {}, 'queue': []}
    rows = []
    for family in module.FAMILIES:
        row = {'model_id': family, 'seed': 42, 'fingerprint': family,
               'metrics': {'accuracy': .8, 'macro_f1': .7}, 'config': {}}
        rows.append(row)
        definition['controls'][family] = row
    monkeypatch.setattr(module, 'completed_results', lambda **kw: rows)
    scheduled = {'model_id': 'kan', 'seed': 42, 'config': {'lr': .002},
                 'variant': 'tpe_0001', 'optuna_trial_number': 1}
    state = {**module.initial_state(), 'scheduled_request': scheduled}
    module.lifecycle.write(tmp_path / 'state.json', state)
    module.lifecycle.write(tmp_path / 'adopted_results.json', [])
    definition['adopted_sha256'] = module.lifecycle.digest(tmp_path / 'adopted_results.json')
    module.lifecycle.write(tmp_path / 'study.json', definition)
    monkeypatch.setattr(module, 'handoff', lambda s: module.lifecycle.write(tmp_path / 'state.json', s))
    monkeypatch.setattr(module, 'prepare', lambda r, d: r)
    calls = []
    def interrupted(request, *args):
        calls.append(request)
        (tmp_path / 'STOP').touch()
        return None
    monkeypatch.setattr(module, 'run_trial', interrupted)
    module.controller()
    assert calls == [scheduled]
    saved = json.loads((tmp_path / 'state.json').read_text())
    assert saved['status'] == 'stopped'
    assert saved['scheduled_request'] == scheduled


def test_research_integrity_covers_preprocessor_not_only_checkpoint(tmp_path, monkeypatch):
    from experiment import optimize_csv_research as module
    monkeypatch.setattr(module.lifecycle, 'ROOT', tmp_path)
    (tmp_path / 'model.ubj').write_bytes(b'model')
    (tmp_path / 'research_transform.joblib').write_bytes(b'preprocessor')
    result = {'status': 'completed', 'checkpoint_path': 'model.ubj',
              'checkpoint_sha256': module.lifecycle.digest(tmp_path / 'model.ubj')}
    module.lifecycle.write(tmp_path / 'result.json', result)
    module.lifecycle.write(tmp_path / 'bundle_integrity.json', {
        'research_transform.joblib': module.lifecycle.digest(tmp_path / 'research_transform.joblib')})
    assert module.verify_result(tmp_path / 'result.json') == result
    (tmp_path / 'research_transform.joblib').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='bundle corruption'):
        module.verify_result(tmp_path / 'result.json')


def test_watchdog_kills_owned_worker_and_failure_is_not_repeated(tmp_path, monkeypatch):
    import subprocess
    import sys
    from experiment import optimize_csv_research as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'STUDY', tmp_path)
    request = {'model_id': 'xgboost', 'config': {}, 'seed': 42, 'fingerprint': 'timeout-fixture'}
    monkeypatch.setattr(module, 'prepare', lambda r, d: r)
    monkeypatch.setattr(module, 'fingerprint', lambda r: r['fingerprint'])
    monkeypatch.setattr(module, 'trial_budget', lambda r: .05)
    monkeypatch.setattr(module, 'completed_results', lambda **kw: [])
    monkeypatch.setattr(module, 'handoff', lambda s: None)
    popen = subprocess.Popen
    children = []
    def sleeper(*args, **kwargs):
        if args and args[0][0] == 'taskkill':
            return popen(*args, **kwargs)
        child = popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(module.subprocess, 'Popen', sleeper)
    state = module.initial_state()
    assert module.run_trial(request, state, {}) is None
    assert children[0].poll() is not None
    failure = next((tmp_path / 'runs').glob('*/attempt_*/failure.json'))
    assert json.loads(failure.read_text())['kind'] == 'timeout'
    assert module.run_trial(request, state, {}) is None
    assert len(children) == 1


def test_ensemble_rejects_unverified_cpu_member(tmp_path, monkeypatch):
    from experiment import train_csv_research as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    def bad_member(*args):
        checkpoint = args[-1] / 'model.ubj'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'not CUDA')
        return {'checkpoint_path': str(checkpoint), 'predictions': np.zeros(6),
                'probabilities': np.ones((6, 2))*.5, 'device': 'cpu',
                'device_evidence': {}, 'effective_training_rows': 6, 'synthetic_training_rows': 0}
    monkeypatch.setattr(module, 'fit_base', bad_member)
    with pytest.raises(RuntimeError, match='CUDA'):
        module.fit_ensemble({'model_id': 'xgboost', 'seed': 42, 'pipeline': {}, 'config': {}},
                            {'classes': ['a', 'b']}, np.ones((6, 2)), np.zeros(6),
                            np.arange(6), np.ones((6, 2)), np.zeros(6), tmp_path)


def test_oof_calibration_separates_fit_stopping_and_prediction_ids(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA research verification')
    from experiment.train_csv_research import oof_temperature
    rng = np.random.default_rng(42)
    y = np.tile(np.arange(3), 30)
    X = rng.normal(size=(90, 4)).astype(np.float32)
    X[:, 0] += 4*y
    result = oof_temperature({'model_id': 'xgboost', 'seed': 42, 'pipeline': {},
                              'config': {'max_depth': 2, 'num_boost_round': 2}},
                             {'feature_names': ['a', 'b', 'c', 'd'], 'classes': ['a', 'b', 'c'],
                              'benign_index': 0}, X, y, np.arange(90), tmp_path)
    assert result['outer_validation_used'] is False
    assert .1 <= result['temperature'] <= 10
    folds = json.loads((tmp_path / 'oof_provenance.json').read_text())['folds']
    predicted = []
    for fold in folds:
        fit, stop, held = (set(fold[k]) for k in ('fit_row_ids', 'stopping_row_ids', 'prediction_row_ids'))
        assert not fit & stop and not fit & held and not stop & held
        assert fit | stop | held == set(range(90))
        predicted.extend(held)
    assert sorted(predicted) == list(range(90))


def test_pending_memory_recovery_does_not_skip_queued_experiment(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from experiment import optimize_csv_research as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'STUDY', tmp_path)
    monkeypatch.setattr(module, 'all_locks', nullcontext)
    monkeypatch.setattr(module.lifecycle, 'protect_descendants', lambda: None)
    monkeypatch.setattr(module, 'pin_inputs', lambda: {})
    monkeypatch.setattr(module, 'success_report', lambda *a: {'success': False})
    definition = {'source_input_hashes': {}, 'controls': {}, 'queue': [{'variant': 'next_solo'}]}
    rows = []
    for family in module.FAMILIES:
        row = {'model_id': family, 'seed': 42, 'fingerprint': family,
               'metrics': {'accuracy': .8, 'macro_f1': .7}, 'config': {}}
        rows.append(row)
        definition['controls'][family] = row
    monkeypatch.setattr(module, 'completed_results', lambda **kw: rows)
    state = {**module.initial_state(), 'pending_requests': [{'variant': 'memory_recovery'}]}
    module.lifecycle.write(tmp_path / 'state.json', state)
    module.lifecycle.write(tmp_path / 'adopted_results.json', [])
    definition['adopted_sha256'] = module.lifecycle.digest(tmp_path / 'adopted_results.json')
    module.lifecycle.write(tmp_path / 'study.json', definition)
    monkeypatch.setattr(module, 'handoff', lambda s: module.lifecycle.write(tmp_path / 'state.json', s))
    monkeypatch.setattr(module, 'prepare', lambda r, d: r)
    calls = []
    def run(request, *args):
        calls.append(request['variant'])
        if len(calls) == 2:
            (tmp_path / 'STOP').touch()
        return None
    monkeypatch.setattr(module, 'run_trial', run)
    module.controller()
    assert calls == ['memory_recovery', 'next_solo']
    saved = json.loads((tmp_path / 'state.json').read_text())
    assert saved['queue_index'] == 0  # interrupted next_solo remains scheduled


def test_adoption_rejects_incompatible_split_before_relabelling():
    from experiment.optimize_csv_research import adoption_request
    definition = {'prepared_manifest_sha256': 'manifest', 'protocol_id': 'protocol',
                  'train_row_ids_sha256': 'train', 'validation_row_ids_sha256': 'val'}
    row = {'model_id': 'kan', 'seed': 42, 'config': {}, 'input_profile': 'full_csv',
           **definition, 'train_row_ids_sha256': 'wrong-training-partition',
           'checkpoint_reload_verified': True, 'device': 'cuda:0',
           'device_evidence': {'parameter_devices': ['cuda:0']}}
    with pytest.raises(ValueError, match='identity'):
        adoption_request(row, definition)


def test_adoption_rejects_changed_real_labels_even_when_row_ids_match(tmp_path, monkeypatch):
    from experiment import optimize_csv_research as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'PREPARED', tmp_path / 'prepared')
    module.PREPARED.mkdir()
    names = ['manifest.json'] + [f'{kind}_{split}.npy' for kind in ('X', 'y', 'row_ids')
                                for split in ('train', 'validation')]
    hashes = {}
    for name in names:
        path = module.PREPARED / name
        path.write_bytes(name.encode())
        hashes[str(path.relative_to(tmp_path))] = module.lifecycle.digest(path)
    (module.PREPARED / 'y_train.npy').write_bytes(b'changed real category labels')
    with pytest.raises(ValueError, match='Prior.*identity'):
        module.verify_prior_inputs({'source_input_hashes': hashes})


def test_plateau_capacity_is_available_before_pair_queue_finishes():
    from experiment.optimize_csv_research import initial_state, next_capacity
    rows = [{'model_id': 'xgboost', 'seed': 42, 'variant': f'solo_{i}',
             'completed_at': i, 'config': {'learning_rate': .08, 'reg_lambda': 3},
             'metrics': {'accuracy': .8, 'macro_f1': .7},
             'history': [{'training_accuracy': .9, 'training_monitor_rows': 4096,
                          'validation_accuracy': .8}]} for i in range(7)]
    state = initial_state()
    trial = next_capacity(rows, state)
    assert trial['config']['max_depth'] == 8
    assert trial['decision']['training_validation_gap'] == pytest.approx(.1)
    assert trial['decision']['training_monitor_rows'] == 4096
    assert trial['decision']['diagnosis'] == 'possible_overfitting'
    assert state['queue_index'] == 0
    assert next_capacity(rows, state) is None


def test_logon_recovery_honors_explicit_stop(tmp_path, monkeypatch):
    import sys
    from experiment import optimize_csv_research as module
    monkeypatch.setattr(module, 'STUDY', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['optimize_csv_research', 'recover'])
    calls = []
    monkeypatch.setattr(module, 'launch', lambda: calls.append('launched'))
    (tmp_path / 'study.json').write_text('{}')
    (tmp_path / 'STOP').touch()
    module.main()
    assert calls == []
    (tmp_path / 'STOP').unlink()
    module.main()
    assert calls == ['launched']
