"""Queue a separately versioned reversed-split diagnostic after forward success.

The waiting controller uses no ML imports or GPU. Four fresh fits execute only
after the original repeat-validated completion gate and exclusive study locks.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiment import optimize_models as lifecycle

ROOT = lifecycle.ROOT
STUDY = ROOT / 'artifacts/training/reversed_csv_v1'
FORWARD = ROOT / 'artifacts/training/csv_research_v3'
PREPARED = ROOT / 'dataset/prepared/datasense_full_protocols'
MODULE = 'experiment.reverse_csv_study'
WORKER = 'experiment.train_reverse_csv'
ROLES = {'fit': 'test', 'stopping': 'validation', 'evaluation': 'train'}


def forward_ready(state: dict, report: dict) -> bool:
    return state.get('status') == 'success' and not state.get('active_trial') and report.get('success') is True


def freeze_recipes(rows: list[dict]) -> list[dict]:
    """Copy parameters, never learned weights or fitted forward preprocessing."""
    from framework.training.fixed_random_forest import FIXED_PARAMETERS
    requests = [{'model_id': 'random_forest', 'config': dict(FIXED_PARAMETERS), 'pipeline': {},
                 'seed': 42, 'fresh_initialization': True}]
    for family in ('xgboost', 'kan', 'hem'):
        eligible = [r for r in rows if r['model_id'] == family and r['seed'] == 42
                    and r.get('checkpoint_reload_verified') is True]
        leader = max(eligible, key=lambda r: (r['metrics']['accuracy'], r['metrics']['macro_f1']))
        requests.append({'model_id': family, 'config': deepcopy(leader['config']),
            'pipeline': deepcopy(leader.get('pipeline', {})), 'seed': 42, 'fresh_initialization': True,
            'forward_recipe_reference': {'variant': leader['variant'], 'fingerprint': leader.get('fingerprint'),
                                         'metrics': leader['metrics']}})
    return requests


def queue() -> None:
    """Record deferred authorization without reading any test arrays."""
    with lifecycle.StudyLock(STUDY / 'controller.lock'):
        if (STUDY / 'spec.json').exists():
            return
        sources = ['experiment/reverse_csv_study.py', 'experiment/train_reverse_csv.py']
        lifecycle.write(STUDY / 'spec.json', {'protocol': 'reversed_csv_v1', 'roles': ROLES,
            'families': ['random_forest', 'xgboost', 'kan', 'hem'], 'after': 'forward_repeat_validated_success',
            'source_hashes': {p: lifecycle.digest(ROOT / p) for p in sources}, 'watchdog_seconds': 14400,
            'forward_definition_sha256': lifecycle.digest(FORWARD / 'study.json'),
            'authorization': 'User explicitly requested train on testing and test on training after current training'})
        lifecycle.write(STUDY / 'state.json', {'status': 'waiting_for_forward_success', 'active_trial': None,
            'held_out_test_retired': False, 'queued_at': time.time(), 'roles': ROLES})


def run_request(request: dict, state: dict) -> bool:
    from experiment.optimize_csv_research import verify_result
    directory = STUDY / 'runs' / request['model_id']
    for path in directory.glob('attempt_*/request.json'):
        if json.loads(path.read_text()) != request:
            raise ValueError('Reversed immutable request changed; create a new study version')
    for path in directory.glob('attempt_*/result.json'):
        if verify_result(path):
            return True
    attempts = sorted(directory.glob('attempt_*'))
    if attempts and (attempts[-1] / 'failure.json').exists():
        kind = json.loads((attempts[-1] / 'failure.json').read_text())['kind']
        if kind not in ('interrupted', 'controller_restart'):
            return False
    attempt = directory / f'attempt_{len(attempts):03d}'
    lifecycle.write(attempt / 'request.json', request)
    deadline = time.time() + request['watchdog_seconds']
    with (attempt / 'fit.log').open('a', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-B', '-u', '-m', WORKER, '--fit', str(attempt / 'request.json')],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, 'IMDAI_FIT_DEADLINE': str(deadline)},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0, start_new_session=os.name != 'nt')
        state['active_trial'] = {'model_id': request['model_id'], 'pid': child.pid,
                                'directory': attempt.relative_to(ROOT).as_posix()}
        lifecycle.write(STUDY / 'state.json', state)
        reason = None
        while child.poll() is None:
            if (STUDY / 'STOP').exists() or (FORWARD / 'STOP').exists():
                reason = 'interrupted'
            elif time.time() >= deadline:
                reason = 'timeout'
            if reason:
                lifecycle.terminate_tree(child)
                break
            time.sleep(1)
        result = verify_result(attempt / 'result.json')
        if not result:
            lifecycle.write(attempt / 'failure.json', {'kind': reason or 'failed', 'exit_code': child.returncode})
    state['active_trial'] = None
    lifecycle.write(STUDY / 'state.json', state)
    return result is not None


def execute(state: dict) -> None:
    """Caller owns all forward CUDA locks before this function reads test bytes."""
    from experiment import optimize_csv_research as forward
    from experiment.csv_research_policy import success_report
    from experiment.train_reverse_csv import retire_test
    definition = json.loads((FORWARD / 'study.json').read_text())
    if forward.pin_inputs() != definition['source_input_hashes']:
        raise ValueError('Forward source/inputs changed; cannot freeze reverse protocol')
    rows = forward.completed_results(verify=True)
    controls = {f: next(r for r in rows if r['fingerprint'] == definition['controls'][f]['fingerprint'])
                for f in forward.FAMILIES}
    if not success_report(rows, controls)['success']:
        raise ValueError('Forward repeat-validated completion evidence does not verify')
    retire_test(STUDY / 'test_access.json')
    state.update(status='running', held_out_test_retired=True)
    requests_file = STUDY / 'requests.json'
    if not requests_file.exists():
        hashes = {p.name: lifecycle.digest(p) for p in [PREPARED / 'manifest.json'] +
                  [PREPARED / f'{kind}_{split}.npy' for kind in ('X', 'y', 'row_ids')
                   for split in ('test', 'validation', 'train')]}
        requests = freeze_recipes(rows)
        for request in requests:
            request.update(roles=ROLES, prepared=PREPARED.relative_to(ROOT).as_posix(),
                           dataset_hashes=hashes, watchdog_seconds=14400)
        lifecycle.write(requests_file, requests)
        state['requests_sha256'] = lifecycle.digest(requests_file)
        lifecycle.write(STUDY / 'state.json', state)
    if lifecycle.digest(requests_file) != state.get('requests_sha256'):
        raise ValueError('Frozen reversed requests changed; preserve evidence')
    if state.get('active_trial'):
        prior = ROOT / state['active_trial']['directory']
        if not (prior / 'result.json').exists():
            lifecycle.write(prior / 'failure.json', {'kind': 'controller_restart'})
        state['active_trial'] = None
    outcomes = {}
    for request in json.loads(requests_file.read_text()):
        if (STUDY / 'STOP').exists() or (FORWARD / 'STOP').exists():
            state.update(status='stopped'); break
        outcomes[request['model_id']] = run_request(request, state)
    else:
        state['status'] = 'completed' if all(outcomes.values()) else 'completed_with_failures'
    state['outcomes'] = outcomes
    state['updated_at'] = time.time()
    lifecycle.write(STUDY / 'state.json', state)


def watch() -> None:
    with lifecycle.StudyLock(STUDY / 'controller.lock'):
        job = lifecycle.protect_descendants()
        state = json.loads((STUDY / 'state.json').read_text())
        state['pid'] = os.getpid()
        try:
            spec = json.loads((STUDY / 'spec.json').read_text())
            if any(lifecycle.digest(ROOT / p) != expected for p, expected in spec['source_hashes'].items()):
                raise ValueError('Reverse controller source changed; version before resuming')
            if lifecycle.digest(FORWARD / 'study.json') != spec['forward_definition_sha256']:
                raise ValueError('Forward study definition changed')
            while not (STUDY / 'STOP').exists():
                source, report = [json.loads((FORWARD / p).read_text()) for p in ('state.json', 'report.json')]
                if forward_ready(source, report) and not (FORWARD / 'STOP').exists():
                    from experiment.optimize_csv_research import all_locks
                    try:
                        locks = all_locks()
                    except RuntimeError:
                        time.sleep(5); continue
                    with locks:
                        if os.name == 'nt':
                            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
                        execute(state)
                    return
                state.update(status='waiting_for_forward_success', forward_status=source['status'], updated_at=time.time())
                lifecycle.write(STUDY / 'state.json', state)
                time.sleep(5)
            state.update(status='stopped', active_trial=None)
        except Exception as exc:
            state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if os.name == 'nt':
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lifecycle.write(STUDY / 'state.json', state)


def launch() -> None:
    """Detached waiting only; current training's recovery entry is preserved."""
    with lifecycle.StudyLock(STUDY / 'controller.lock'):
        state = json.loads((STUDY / 'state.json').read_text())
        if state['status'].startswith('completed'):
            print('Reversed diagnostic already finished'); return
        marker = STUDY / 'launch.json'
        if marker.exists():
            import psutil
            try:
                process = psutil.Process(json.loads(marker.read_text())['pid'])
                if MODULE in ' '.join(process.cmdline()):
                    raise RuntimeError('Reverse watcher already launching')
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        (STUDY / 'STOP').unlink(missing_ok=True)
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        with (STUDY / 'controller.log').open('a', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-B', '-u', '-m', MODULE, '_watch'], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                close_fds=True, creationflags=flags, start_new_session=os.name != 'nt')
        lifecycle.write(marker, {'pid': child.pid})
        if os.name == 'nt':
            startup = Path(os.environ['APPDATA']) / 'Microsoft/Windows/Start Menu/Programs/Startup/IMDAI-Reversed-Resume.vbs'
            command = subprocess.list2cmdline([str(Path(sys.executable).with_name('pythonw.exe')), '-B', '-m', MODULE, 'recover'])
            startup.write_text('Set shell = CreateObject("WScript.Shell")\n'
                + 'shell.CurrentDirectory = "' + str(ROOT).replace('"', '""') + '"\n'
                + 'shell.Run "' + command.replace('"', '""') + '", 0, False\n', encoding='utf-8')
        print(json.dumps({'queued_watcher_pid': child.pid, 'training_started': False}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('queue', 'start', 'status', 'stop', 'resume', 'recover', '_watch'))
    command = parser.parse_args().command
    if command in ('queue', 'start'):
        queue(); launch()
    elif command == 'status':
        print((STUDY / 'state.json').read_text() if (STUDY / 'state.json').exists() else 'Not queued')
    elif command == 'stop':
        STUDY.mkdir(parents=True, exist_ok=True); (STUDY / 'STOP').touch()
    elif command == 'resume':
        launch()
    elif command == 'recover':
        if (STUDY / 'spec.json').exists() and not (STUDY / 'STOP').exists():
            launch()
    else:
        watch()


if __name__ == '__main__':
    main()
