"""Lifecycle tests without research training or GPU dependencies."""
import json
import time
from pathlib import Path

import pytest

from experiment.optimize_models import StudyLock, verified, ranking, search_configs, remaining


def test_duplicate_controller_and_release(tmp_path):
    with StudyLock(tmp_path / 'lock'):
        with pytest.raises(RuntimeError):
            with StudyLock(tmp_path / 'lock'):
                pass
    with StudyLock(tmp_path / 'lock'):
        pass


def test_checkpoint_integrity(tmp_path):
    import hashlib
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'original')
    result = {'status': 'completed', 'checkpoint_path': str(checkpoint),
              'checkpoint_sha256': hashlib.sha256(b'original').hexdigest()}
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(result))
    assert verified(path)
    checkpoint.write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        verified(path)


def test_deadline_does_not_reset():
    assert remaining({'deadline': time.time() - 1}) == 0
    assert 0 < remaining({'deadline': time.time() + 30}) <= 30


def test_ranking_uses_live_profile_for_xgboost():
    result = {'model_id': 'xgboost', 'metrics': {'accuracy': .99, 'macro_f1': .9},
              'profile_metrics': {'packet_counts_7': {'accuracy': .7, 'macro_f1': .4}}}
    assert ranking(result) == (.7, .4)


def test_search_preserves_architecture_and_excludes_rf():
    configs = {'xgboost': {'max_depth': 6, 'learning_rate': .08, 'reg_lambda': 3,
                         'weight_strategy': 'sqrt_balanced', 'subsample': .8},
               'kan': {'hidden': [96, 48], 'grid_size': 5, 'spline_order': 3,
                       'lr': .002, 'weight_decay': .001, 'weight_strategy': 'sqrt_balanced'},
               'hem': {'hidden': [512, 256, 32, 16], 'binary_hidden': [1024, 256, 64, 16],
                       'lr': .001, 'weight_decay': .0001, 'weight_strategy': 'none'}}
    trials = search_configs(configs, allow_architecture=False)
    assert {t['model_id'] for t in trials} == set(configs)
    for t in trials:
        before = configs[t['model_id']]
        for key in ('hidden', 'binary_hidden', 'grid_size', 'spline_order', 'max_depth'):
            if key in before:
                assert t['config'][key] == before[key]


def test_architecture_search_changes_three_families_only():
    controls = {'xgboost': {'max_depth': 6, 'learning_rate': .08},
                'kan': {'hidden': [96, 48], 'grid_size': 5},
                'hem': {'hidden': [512, 256, 32, 16], 'binary_hidden': [1024, 256, 64, 16],
                        'pca_components': 16, 'knn_k': 1}}
    architecture = [t for t in search_configs(controls) if t.get('architecture_modified')]
    assert {t['model_id'] for t in architecture} == set(controls)
    assert any(t['config']['max_depth'] != 6 for t in architecture if t['model_id'] == 'xgboost')
    assert any(t['config']['hidden'] != [96, 48] for t in architecture if t['model_id'] == 'kan')
    assert any(t['config']['hidden'] != [512, 256, 32, 16] for t in architecture if t['model_id'] == 'hem')


def test_xgboost_capacity_options_are_validated_and_not_booster_params():
    from experiment.fit_optimization import partial_options
    rounds, patience, params = partial_options({'max_depth': 8, 'num_boost_round': 800, 'early_stopping_rounds': 60})
    assert rounds == 800 and patience == 60 and params == {'max_depth': 8}
    with pytest.raises(ValueError):
        partial_options({'num_boost_round': 0})


def test_capacity_fit_preserves_request_options_for_resume(tmp_path, monkeypatch):
    from experiment import fit_optimization as wrapper
    from experiment import train_four_models as original
    from experiment import train_partial_xgboost as partial
    request = {'model_id': 'xgboost', 'config': {'max_depth': 8, 'num_boost_round': 800,
                                              'early_stopping_rounds': 60}}
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(request))
    def fake_trial(sanitized):
        cleaned = json.loads(sanitized.read_text())
        assert 'num_boost_round' not in cleaned['config']
        assert partial.MAX_ROUNDS == 800 and partial.PATIENCE == 60
        (tmp_path / 'result.json').write_text(json.dumps({'status': 'completed',
                                                        'config': cleaned['config']}))
        return 0
    monkeypatch.setattr(partial, 'trial', fake_trial)
    monkeypatch.setattr(original, 'fit', lambda p: partial.trial(p))
    wrapper.fit(path)
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['config'] == request['config']
    assert result['effective_trainer_limits']['num_boost_round'] == 800


def test_refresh_waits_for_wrapper_exit_before_stop(tmp_path, monkeypatch):
    import subprocess
    import sys
    import threading
    import experiment.optimize_models as controller
    monkeypatch.setattr(controller, 'STUDY', tmp_path / 'study')
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    request_file = tmp_path / 'trial/request.json'
    request_file.parent.mkdir()
    request_file.write_text('{}')
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', str(request_file)])
    controller.write(controller.STUDY / 'state.json',
                     {'deadline': time.time() + 20, 'status': 'running',
                      'active_trial': {'directory': 'trial', 'pid': child.pid}})
    controller.write(controller.STUDY / 'refresh_request.json',
                     {'active_directory': 'trial', 'active_pid': child.pid})
    monkeypatch.setattr(controller, 'verified', lambda path: {'status': 'completed'})
    launched = threading.Event()
    monkeypatch.setattr(controller, 'launch', launched.set)
    thread = threading.Thread(target=controller.refresh_after_fit, daemon=True)
    try:
        thread.start()
        time.sleep(.3)
        assert not (controller.STUDY / 'STOP').exists()
        assert not launched.is_set()
        child.kill()
        child.wait(timeout=5)
        thread.join(timeout=5)
        assert launched.is_set() and not thread.is_alive()
        assert json.loads((controller.STUDY / 'state.json').read_text())['deadline'] > time.time()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_inner_xgboost_result_is_not_a_completed_wrapped_fit(tmp_path):
    path = tmp_path / 'result.json'
    path.write_text(json.dumps({'status': 'completed', 'checkpoint': 'model.ubj'}))
    assert verified(path) is None


def test_stop_terminates_owned_process_tree():
    import subprocess
    import sys
    from experiment.optimize_models import terminate_tree
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                             start_new_session=__import__('os').name != 'nt')
    try:
        terminate_tree(child)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()


def test_resume_preserves_deadline_and_detaches(tmp_path, monkeypatch):
    import experiment.optimize_models as controller
    controller.write(tmp_path / 'state.json', {'started': time.time(), 'deadline': time.time() + 60})
    before = (tmp_path / 'state.json').read_bytes()
    monkeypatch.setattr(controller, 'STUDY', tmp_path)
    calls = []
    class Child:
        pid = 123
    monkeypatch.setattr(controller.subprocess, 'Popen', lambda *a, **kw: (calls.append(kw) or Child()))
    controller.launch()
    assert before == (tmp_path / 'state.json').read_bytes()
    assert calls[0]['close_fds'] and calls[0]['stdin'] == controller.subprocess.DEVNULL
    assert calls[0]['creationflags'] if __import__('os').name == 'nt' else calls[0]['start_new_session']


def test_expired_resume_never_launches(tmp_path, monkeypatch):
    import experiment.optimize_models as controller
    controller.write(tmp_path / 'state.json', {'deadline': time.time() - 1})
    monkeypatch.setattr(controller, 'STUDY', tmp_path)
    monkeypatch.setattr(controller.subprocess, 'Popen', lambda *a, **kw: pytest.fail('expired study launched'))
    controller.launch()


@pytest.mark.skipif(__import__('os').name != 'nt', reason='Windows Job Object semantics')
def test_controller_death_kills_trial_descendant(tmp_path):
    import subprocess
    import sys
    import psutil
    pid_file = tmp_path / 'descendant.pid'
    script = ('from experiment.optimize_models import protect_descendants; '
              'import subprocess,sys,time; from pathlib import Path; '
              'job=protect_descendants(); '
              'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
              f'Path({str(pid_file)!r}).write_text(str(child.pid)); time.sleep(60)')
    parent = subprocess.Popen([sys.executable, '-B', '-c', script])
    descendant = None
    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            if parent.poll() is not None:
                pytest.fail('Job Object setup failed')
            time.sleep(.05)
        assert pid_file.exists()
        descendant = psutil.Process(int(pid_file.read_text()))
        parent.kill()
        parent.wait(timeout=5)
        descendant.wait(timeout=5)
        assert not descendant.is_running()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if descendant and descendant.is_running():
            descendant.kill()


@pytest.mark.parametrize('stop_requested', [False, True])
def test_real_fit_deadline_and_stop_clear_active_trial(tmp_path, monkeypatch, stop_requested):
    import subprocess
    import sys
    import experiment.optimize_models as controller
    monkeypatch.setattr(controller, 'STUDY', tmp_path)
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    real_popen = subprocess.Popen
    def sleeper(*args, **kwargs):
        if 'experiment.fit_optimization' in args[0]:
            return real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        return real_popen(*args, **kwargs)
    monkeypatch.setattr(controller.subprocess, 'Popen', sleeper)
    monkeypatch.setattr(controller, 'handoff', lambda state: None)
    state = {'started': time.time(), 'deadline': time.time() + 1.5}
    if stop_requested:
        import threading
        timer = threading.Timer(.2, lambda: (tmp_path / 'STOP').touch())
        timer.start()
    result = controller.run_fit({'model_id': 'kan'}, tmp_path / 'trial', state)
    assert result is None and state['active_trial'] is None
    failure = json.loads(next((tmp_path / 'trial').glob('attempt_*/failure.json')).read_text())
    assert 'deadline' in failure['reason'] or 'stop' in failure['reason']


def test_missing_fixed_rf_blocks_search(tmp_path, monkeypatch):
    import experiment.optimize_models as controller
    import experiment.train_four_models as original
    monkeypatch.setattr(controller, 'STUDY', tmp_path / 'study')
    monkeypatch.setattr(controller, 'BASE', tmp_path / 'base')
    controller.write(controller.STUDY / 'state.json',
                     {'started': time.time(), 'deadline': time.time() + 60})
    controller.write(controller.BASE / 'study.json',
                     {'identity': {}, 'protocol_id': 'fixture', 'configurations':
                      [{'model_id': 'random_forest', 'variant': 'fixed_baseline', 'config': {}}]})
    monkeypatch.setattr(controller, 'protect_descendants', lambda: None)
    monkeypatch.setattr(controller, 'pin_inputs', lambda: {})
    monkeypatch.setattr(original, 'verify_common_partitions', lambda *a: {})
    monkeypatch.setattr(controller, 'request_for', lambda trial, definition: trial)
    monkeypatch.setattr(controller, 'run_fit', lambda *a: None)
    monkeypatch.setattr(controller, 'handoff', lambda *a: None)
    with pytest.raises(RuntimeError, match='random_forest'):
        controller.controller()
    state = json.loads((controller.STUDY / 'state.json').read_text())
    assert state['status'] == 'failed' and state['phase'] == 'prerequisites_incomplete'


def test_auxiliary_artifact_corruption_rejects_resume(tmp_path):
    import experiment.optimize_models as controller
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'weights')
    aux = tmp_path / 'preprocessing.joblib'
    aux.write_bytes(b'original preprocessing')
    controller.write(tmp_path / 'result.json', {'status': 'completed',
                     'checkpoint_path': str(checkpoint), 'checkpoint_sha256': controller.digest(checkpoint)})
    controller.seal_assets(tmp_path)
    assert controller.verified(tmp_path / 'result.json')
    aux.write_bytes(b'wrong preprocessing')
    with pytest.raises(ValueError, match='asset integrity'):
        controller.verified(tmp_path / 'result.json')
