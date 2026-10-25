"""Three requested hybrids: training-only cross-fitting, calibration and CPU export."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np

from experiment.train_verified_clean import read, write
from experiment.datasense_quality import digest
from experiment.optimize_models import StudyLock, protect_descendants
from framework.training.hybrid_protocol import ROOT, DATA, STUDY, initialize, verify_protocol, relative

MODULE = 'experiment.train_hybrids'


def fit(request_path):
    from framework.training.hybrid_bundle import branch_predict
    from framework.training.hybrid_data import IndexedRows
    protocol = read(STUDY/'study.json'); verify_protocol(protocol)
    request_path = Path(request_path); request = read(request_path); output = request_path.parent
    expected = next(j for j in protocol['jobs'] if j['job'] == request['job'])
    if request != expected:
        raise ValueError('Worker recipe differs from the frozen protocol')
    family, fold = request['family'], request['fold']
    manifest = read(DATA/'model/manifest.json')
    X = np.load(DATA/'model/X_train.npy', mmap_mode='r'); y = np.load(DATA/'model/y_train.npy')
    started = time.monotonic()
    if fold is None:
        fit_ids = np.arange(len(y)); roles = np.load(STUDY/'validation_roles.npy')
        V = np.load(DATA/'model/X_validation.npy', mmap_mode='r')
        v = np.load(DATA/'model/y_validation.npy')
        early_X, early_y = V[roles==0], v[roles==0]
        oof_ids = None
    else:
        folder = STUDY/'folds'/str(fold)
        fit_ids = np.load(folder/('fit_strict.npy' if request['arm']=='strict' else 'fit.npy'))
        early_ids = np.load(folder/'early_stop.npy'); oof_ids = np.load(folder/'oof.npy')
        early_X, early_y = X[early_ids], y[early_ids]
    print(f'START {request["job"]} rows={len(fit_ids)} early_stop={len(early_y)} CUDA', flush=True)
    if family in ('tabm_ne','tabpack'):
        from framework.training.hybrid_neural import train
        details = train(family, IndexedRows(X,fit_ids), y[fit_ids], early_X, early_y, manifest,
            request['config'], request['seed'], output, 'cuda')
    else:
        from experiment.train_csv_models import train_candidate
        details = train_candidate(family, X[fit_ids], y[fit_ids], early_X, early_y,
            len(manifest['classes']), 0, request['config'], request['seed'], output)
        details = {k:v for k,v in details.items() if k not in ('predictions','probabilities','checkpoint_path')}
    checkpoint = output/('model.ubj' if family=='xgboost' else 'model.pt')
    schema = DATA/'model/manifest.json'
    assets = [checkpoint, schema]+([output/'preprocessing.joblib'] if family=='kan' else [])
    branch = dict(family=family, checkpoint=relative(checkpoint), schema_manifest=relative(schema), assets={relative(p):digest(p) for p in assets})
    del early_X
    if oof_ids is None:
        raw = V; prefix = 'validation'
        ids = np.load(DATA/'model/row_ids_validation.npy')
    else:
        raw = IndexedRows(X,oof_ids) if family in ('tabm_ne','tabpack') else X[oof_ids]
        prefix = 'oof'
        ids = np.load(DATA/'model/row_ids_train.npy')[oof_ids]
    p = branch_predict(branch, raw)
    if p.shape != (len(ids), len(manifest['classes'])):
        raise ValueError('Output class/row dimensions differ')
    replay = branch_predict(branch, raw[:min(128,len(raw))])
    np.testing.assert_allclose(p[:len(replay)], replay, atol=2e-6, rtol=1e-5)
    np.testing.assert_array_equal(p[:len(replay)].argmax(1), replay.argmax(1))
    np.save(output/f'{prefix}_probabilities.npy', p)
    np.save(output/f'{prefix}_row_ids.npy', ids)
    result = dict(request, details=details, status='completed', training_rows=len(fit_ids),
        early_stopping_rows=len(early_y), output_rows=len(ids), run_directory=relative(output),
        cpu_checkpoint_reload_verified=True, elapsed_seconds=time.monotonic()-started,
        fitting_partition='train', early_stopping_partition='train' if fold is not None else 'validation_tuning',
        oof_used_for_checkpoint_selection=False, test_loaded=False,
        artifact_hashes={relative(p):digest(p) for p in output.iterdir() if p.suffix in ('.npy','.pt','.ubj','.joblib')})
    write(output/'result.json', result)
    print(f'COMPLETE {request["job"]}', flush=True)


def attempts(job):
    return sorted((STUDY/'runs'/job).glob('attempt_*'))


def needs_attempt(job):
    history = attempts(job)
    if any((p/'result.json').exists() for p in history):
        return False
    # Failed attempts remain skipped for diagnosis; resume restarts interrupted
    # attempts in a new directory without overwriting their evidence.
    return not history or not (history[-1]/'failure.json').exists()


def resources_available(request=None):
    import psutil
    from experiment.run_xgboost_followup import memory_available
    for proc in psutil.process_iter(['cmdline']):
        try:
            args = proc.info['cmdline'] or []
            if '--fit' in args and any(a.startswith('experiment.train_') for a in args):
                return False
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    memory = memory_available()
    physical_minimum = 2.5 if request and request['family'] in ('tabm_ne','tabpack') else 4.
    return memory['physical_gib'] >= physical_minimum and memory['commit_gib'] >= 6.


def terminate_owned(child):
    import psutil
    try:
        for proc in reversed(psutil.Process(child.pid).children(recursive=True)):
            try: proc.kill()
            except psutil.NoSuchProcess: pass
    except psutil.NoSuchProcess:
        pass
    if child.poll() is None:
        child.kill()
    child.wait()


def launch_fit(request, state, protocol):
    parent = STUDY/'runs'/request['job']; parent.mkdir(parents=True, exist_ok=True)
    output = parent/f'attempt_{len(attempts(request["job"])):03d}'; output.mkdir()
    write(output/'request.json', request)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (output/'fit.log').open('a', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable,'-B','-u','-m',MODULE,'--fit',str(output/'request.json')],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        state.update(status='running', active_job=dict(job=request['job'], pid=child.pid, run_directory=relative(output)))
        started = time.monotonic()
        while child.poll() is None:
            stop = (STUDY/'STOP').exists()
            if stop or time.monotonic()-started > protocol['per_fit_seconds']:
                terminate_owned(child)
                write(output/('interrupted.json' if stop else 'failure.json'),
                    dict(reason='User stop' if stop else 'Per-fit watchdog expired', at=time.time()))
                break
            state['heartbeat_at'] = time.time(); write(STUDY/'state.json', state); time.sleep(5)
    if not (output/'result.json').exists() and not (output/'interrupted.json').exists():
        if not (output/'failure.json').exists():
            write(output/'failure.json', dict(exit_code=child.returncode, reason='No complete result; inspect fit.log'))
    state['active_job'] = None


def controller(launch_id=None):
    from framework.training.hybrid_evaluation import completed_jobs, finalize_available
    with StudyLock(STUDY/'controller.lock'):
        job_handle = protect_descendants()  # Retained until controller termination.
        protocol = read(STUDY/'study.json'); verify_protocol(protocol)
        state = dict(status='starting', pid=os.getpid(), started_at=time.time(),
            planned_fits=len(protocol['jobs']), active_job=None, test_loaded=False, launch_id=launch_id)
        write(STUDY/'state.json', state)
        while not (STUDY/'STOP').exists():
            done = completed_jobs()
            state['completed_jobs'] = list(done)
            state['failed_jobs'] = [j['job'] for j in protocol['jobs'] if j['job'] not in done and not needs_attempt(j['job'])]
            state['status'] = 'fitting_combiners'; write(STUDY/'state.json', state)
            finalize_available(protocol)
            state['completed_hybrids'] = [p.parent.name for p in (STUDY/'hybrids').glob('*/result.json')]
            queue = [j for j in protocol['jobs'] if needs_attempt(j['job'])]
            state['pending_fits'] = len(queue)
            if not queue:
                state['status'] = 'completed' if len(state['completed_hybrids'])==3 else 'completed_with_failures'
                write(STUDY/'state.json', state); return
            eligible = next((request for request in queue if resources_available(request)), None)
            if eligible is None:
                state.update(status='waiting_for_training_or_memory', heartbeat_at=time.time())
                write(STUDY/'state.json', state); time.sleep(10); continue
            try:
                with ExitStack() as locks:
                    for name in ('optimization_24h_v1','csv_specialization_v2','csv_research_v3'):
                        locks.enter_context(StudyLock(ROOT/'artifacts/training'/name/'controller.lock'))
                    launch_fit(eligible, state, protocol)
            except RuntimeError as exc:
                if 'holds the lock' not in str(exc):
                    raise
                state.update(status='waiting_for_existing_gpu_controller', heartbeat_at=time.time())
                write(STUDY/'state.json', state); time.sleep(10)
        state.update(status='stopped', active_job=None); write(STUDY/'state.json', state)


def wait_for_controller(child, launch_id, timeout=30.):
    started = time.monotonic()
    while time.monotonic()-started < timeout:
        exit_code = child.poll()
        if exit_code is not None:
            raise RuntimeError(f'Hybrid controller startup failed with exit {exit_code}; see controller.log')
        path = STUDY/'state.json'
        if path.exists():
            state = read(path)
            if state.get('launch_id') == launch_id:
                if state['status'] == 'controller_failed':
                    raise RuntimeError(f'Hybrid controller startup failed: {state.get("error")}')
                return state
        time.sleep(.1)
    terminate_owned(child)
    raise RuntimeError('Hybrid controller startup timed out; see controller.log')


def start():
    import psutil
    # Serialize launchers separately. The child can acquire controller.lock
    # immediately, even when it is scheduled before this parent finishes.
    with StudyLock(STUDY/'launch.lock'):
        with StudyLock(STUDY/'controller.lock'):
            pass
        if (STUDY/'launch.json').exists():
            launch = read(STUDY/'launch.json')
            try:
                proc = psutil.Process(launch['pid'])
                if MODULE in proc.cmdline() and '_run' in proc.cmdline():
                    raise RuntimeError('Hybrid controller already launched')
            except psutil.NoSuchProcess:
                pass
        (STUDY/'STOP').unlink(missing_ok=True)
        launch_id = uuid.uuid4().hex
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        with (STUDY/'controller.log').open('a', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable,'-B','-u','-m',MODULE,'_run','--launch-id',launch_id], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=flags, close_fds=True, start_new_session=os.name!='nt')
        write(STUDY/'launch.json', dict(pid=child.pid, at=time.time(), launch_id=launch_id))
        wait_for_controller(child, launch_id)
    print(f'Launched hybrid controller PID {child.pid}; {STUDY}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=('initialize','start','resume','status','stop','_run'))
    parser.add_argument('--device', choices=('cuda',))
    parser.add_argument('--fit', type=Path)
    parser.add_argument('--launch-id', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.fit:
        try:
            fit(args.fit)
        except Exception as exc:
            write(args.fit.parent/'failure.json', dict(error=str(exc), type=type(exc).__name__, traceback=traceback.format_exc()))
            raise
    elif args.command in ('initialize','start','resume'):
        if args.command=='start' and args.device!='cuda':
            parser.error('Research training requires explicit --device cuda; app defaults are unchanged')
        if args.command=='resume' and not (STUDY/'study.json').exists():
            parser.error('No pinned study to resume; use start --device cuda')
        initialize()
        if args.command!='initialize': start()
        else: print(STUDY)
    elif args.command=='_run':
        try: controller(args.launch_id)
        except Exception as exc:
            write(STUDY/'controller_failure.json', dict(error=str(exc), traceback=traceback.format_exc(), at=time.time()))
            if (STUDY/'state.json').exists():
                state = read(STUDY/'state.json'); state.update(status='controller_failed', error=str(exc))
                write(STUDY/'state.json', state)
            raise
    elif args.command=='status':
        if not (STUDY/'state.json').exists(): print('Prepared; not started' if (STUDY/'study.json').exists() else 'Not initialized')
        else:
            import json, psutil
            state = read(STUDY/'state.json')
            try: state['controller_alive'] = MODULE in psutil.Process(state['pid']).cmdline()
            except psutil.NoSuchProcess: state['controller_alive'] = False
            print(json.dumps(state, indent=2))
    elif args.command=='stop':
        STUDY.mkdir(parents=True, exist_ok=True); (STUDY/'STOP').touch()
    else:
        parser.error('Choose a command')


if __name__ == '__main__':
    main()
