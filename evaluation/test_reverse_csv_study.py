"""Reversed-partition diagnostics must remain isolated from forward selection."""
import importlib
import importlib.util
import json

import numpy as np
import pytest


def module(name):
    path = 'experiment.' + name
    assert importlib.util.find_spec(path), f'{path} is not implemented'
    return importlib.import_module(path)


@pytest.fixture
def prepared(tmp_path):
    manifest = {'feature_names': ['a', 'b'], 'classes': ['benign', 'absent_in_reverse_fit', 'attack'],
                'benign_index': 0}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    rows = {'test': ([[0., np.nan], [.1, 1.], [9., 2.], [10., 3.]], [0, 0, 2, 2], [30, 31, 32, 33]),
            'validation': ([[.05, 1.], [8., 2.], [5., 1.]], [0, 2, 1], [20, 21, 22]),
            'train': ([[0., 1.], [9., 2.], [5., 3.]], [0, 2, 1], [10, 11, 12])}
    for split, (values, labels, ids) in rows.items():
        for kind, data in (('X', values), ('y', labels), ('row_ids', ids)):
            np.save(tmp_path / f'{kind}_{split}.npy', np.asarray(data))
    return tmp_path


@pytest.mark.parametrize('status,report,active', [('running', True, None), ('stopped', True, None),
                                                 ('success', False, None), ('success', True, {'pid': 1})])
def test_reverse_gate_does_not_start_early(status, report, active):
    assert not module('reverse_csv_study').forward_ready({'status': status, 'active_trial': active}, {'success': report})


def test_reverse_gate_accepts_only_completed_repeat_validated_study():
    assert module('reverse_csv_study').forward_ready({'status': 'success', 'active_trial': None}, {'success': True})


def test_loader_does_not_open_locked_test_before_gate(prepared):
    (prepared / 'X_test.npy').write_bytes(b'locked deliberately invalid sentinel')
    with pytest.raises(PermissionError, match='forward'):
        module('train_reverse_csv').load_partitions(prepared, ready=False, ledger=prepared / 'access.json')
    assert not (prepared / 'access.json').exists()


def test_reverse_roles_and_retirement_ledger(prepared):
    result = module('train_reverse_csv').load_partitions(prepared, ready=True, ledger=prepared / 'access.json')
    assert result['fit']['ids'].tolist() == [30, 31, 32, 33]
    assert result['stopping']['ids'].tolist() == [20, 21, 22]
    assert result['evaluation']['ids'].tolist() == [10, 11, 12]
    assert result['fit']['X'][0, 0] == 0 and np.isnan(result['fit']['X'][0, 1])
    assert json.loads((prepared / 'access.json').read_text())['held_out_test_retired'] is True


def test_reverse_rejects_overlapping_rows(prepared):
    np.save(prepared / 'row_ids_test.npy', np.array([10, 31, 32, 33]))
    with pytest.raises(ValueError, match='overlap'):
        module('train_reverse_csv').load_partitions(prepared, ready=True, ledger=prepared / 'access.json')


def test_frozen_reverse_recipes_have_no_learned_forward_checkpoint():
    rows = [{'model_id': f, 'seed': 42, 'variant': 'control', 'config': {'epochs': 2},
             'pipeline': {}, 'checkpoint_path': 'forward_weights.pt', 'checkpoint_reload_verified': True,
             'cuda_verified': True, 'metrics': {'accuracy': .8, 'macro_f1': .7}}
            for f in ('xgboost', 'kan', 'hem')]
    requests = module('reverse_csv_study').freeze_recipes(rows)
    assert [r['model_id'] for r in requests] == ['random_forest', 'xgboost', 'kan', 'hem']
    assert all('checkpoint_path' not in r for r in requests)
    assert all(r['fresh_initialization'] is True and r['seed'] == 42 for r in requests)
    from framework.training.fixed_random_forest import FIXED_PARAMETERS
    assert requests[0]['config'] == FIXED_PARAMETERS


def test_fresh_rf_keeps_fixed_settings_and_aligns_missing_classes(prepared):
    worker = module('train_reverse_csv')
    partitions = worker.load_partitions(prepared, ready=True, ledger=prepared / 'access.json')
    trained = worker.fit_rf(partitions, prepared / 'fresh_rf')
    from framework.training.fixed_random_forest import FIXED_PARAMETERS
    assert trained['parameters'] == FIXED_PARAMETERS
    predicted, probability = worker.predict_rf(trained, partitions['evaluation']['X'])
    assert probability.shape == (3, 3)
    assert (probability[:, 1] == 0).all()
    assert set(predicted).issubset({0, 2})
    import joblib
    fitted = joblib.load(trained['checkpoint_path'])
    assert fitted.named_steps['imputer'].statistics_[0] == pytest.approx(4.55)
    assert trained['global_class_ids'] == [0, 2]


def test_queued_state_does_not_touch_test_arrays(tmp_path, monkeypatch):
    controller = module('reverse_csv_study')
    monkeypatch.setattr(controller, 'STUDY', tmp_path)
    controller.queue()
    state = json.loads((tmp_path / 'state.json').read_text())
    assert state['status'] == 'waiting_for_forward_success'
    assert state['held_out_test_retired'] is False
    assert state['active_trial'] is None
    assert not (tmp_path / 'test_access.json').exists()


def test_rf_reversed_fit_executes_complete_diagnostic(prepared, monkeypatch):
    controller = module('reverse_csv_study')
    worker = module('train_reverse_csv')
    from experiment.optimize_models import digest, write
    from framework.training.fixed_random_forest import FIXED_PARAMETERS
    monkeypatch.setattr(worker, 'ROOT', prepared)
    forward = prepared / 'forward'
    monkeypatch.setattr(controller, 'FORWARD', forward)
    write(forward / 'state.json', {'status': 'success', 'active_trial': None})
    write(forward / 'report.json', {'success': True})
    directory = prepared / 'reverse/runs/random_forest/attempt_000'
    request = {'model_id': 'random_forest', 'config': dict(FIXED_PARAMETERS), 'pipeline': {},
               'seed': 42, 'fresh_initialization': True, 'prepared': '.',
               'roles': {'fit': 'test', 'stopping': 'validation', 'evaluation': 'train'},
               'dataset_hashes': {p.name: digest(p) for p in prepared.glob('*.npy')}}
    write(directory / 'request.json', request)
    worker.fit(directory / 'request.json')
    result = json.loads((directory / 'result.json').read_text())
    assert result['training_rows'] == 4 and result['evaluation_rows'] == 3
    assert result['held_out_test_retired'] is True
    assert result['evaluation_used_for_early_stopping'] is False
    assert result['classes_without_reverse_training_support'] == ['absent_in_reverse_fit']
    assert (prepared / 'reverse/test_access.json').exists()
    assert not (prepared / 'reverse/runs/test_access.json').exists()
    assert (directory / 'original_train_predictions.npy').is_file()


def test_reverse_watchdog_stops_owned_worker_and_does_not_repeat_failure(tmp_path, monkeypatch):
    import subprocess
    import sys
    controller = module('reverse_csv_study')
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    monkeypatch.setattr(controller, 'STUDY', tmp_path)
    monkeypatch.setattr(controller, 'FORWARD', tmp_path / 'forward')
    popen = subprocess.Popen
    workers = []
    def launch_worker(*args, **kwargs):
        if args and args[0][0] == 'taskkill':
            return popen(*args, **kwargs)
        child = popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        workers.append(child)
        return child
    monkeypatch.setattr(controller.subprocess, 'Popen', launch_worker)
    state = {'active_trial': None}
    request = {'model_id': 'xgboost', 'watchdog_seconds': .05}
    assert controller.run_request(request, state) is False
    assert workers[0].poll() is not None
    failure = json.loads(next((tmp_path / 'runs').glob('*/attempt_*/failure.json')).read_text())
    assert failure['kind'] == 'timeout'
    assert controller.run_request(request, state) is False
    assert len(workers) == 1


def test_reverse_recovery_respects_pause(tmp_path, monkeypatch):
    import sys
    controller = module('reverse_csv_study')
    monkeypatch.setattr(controller, 'STUDY', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['reverse_csv_study', 'recover'])
    (tmp_path / 'spec.json').write_text('{}')
    (tmp_path / 'STOP').touch()
    def unexpected_launch():
        raise AssertionError('Paused reversed diagnostic must not recover automatically')
    monkeypatch.setattr(controller, 'launch', unexpected_launch)
    controller.main()


def test_reverse_checkpoint_reuse_rejects_changed_request(tmp_path, monkeypatch):
    controller = module('reverse_csv_study')
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    monkeypatch.setattr(controller, 'STUDY', tmp_path)
    from experiment.optimize_models import digest, write
    directory = tmp_path / 'runs/xgboost/attempt_000'
    original = {'model_id': 'xgboost', 'watchdog_seconds': 14400, 'config': {'max_depth': 6}}
    write(directory / 'request.json', original)
    checkpoint = directory / 'model.ubj'
    checkpoint.write_bytes(b'owned checkpoint integrity fixture')
    write(directory / 'result.json', {**original, 'status': 'completed', 'checkpoint_path': str(checkpoint),
                                     'checkpoint_sha256': digest(checkpoint)})
    with pytest.raises(ValueError, match='immutable'):
        controller.run_request({**original, 'config': {'max_depth': 8}}, {})


def test_reverse_worker_rejects_rf_tuning_before_touching_data(tmp_path):
    worker = module('train_reverse_csv')
    from experiment.optimize_models import write
    request = {'model_id': 'random_forest', 'seed': 42, 'fresh_initialization': True,
               'config': {'n_estimators': 999}, 'roles': {'fit': 'test', 'stopping': 'validation', 'evaluation': 'train'}}
    write(tmp_path / 'request.json', request)
    with pytest.raises(ValueError, match='fixed RF'):
        worker.fit(tmp_path / 'request.json')
