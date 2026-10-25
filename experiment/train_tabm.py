"""Isolated TabM tuning/calibration on verified DataSense v2; test stays locked."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from experiment.datasense_quality import digest
from experiment.optimize_models import StudyLock
from experiment.train_verified_clean import read, write, training_paths, split_validation
from framework.training.tracking import classification_metrics, validate_manifest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'dataset/experiments/datasense_verified_clean_20261014_v2'
STUDY = ROOT/'artifacts/training/tabm_verified_v2_20261016_v1'
MODULE = 'experiment.train_tabm'
FAMILIES = ('tabm_ne', 'tabm_hier', 'tabm_hier_ne')
SOURCES = ('experiment/train_tabm.py', 'framework/training/tabm_data.py',
    'framework/training/tabm_models.py', 'framework/training/tabm_fit.py',
    'framework/training/tabm_calibration.py', 'experiment/train_verified_clean.py',
    'experiment/datasense_quality.py', 'experiment/calibrate_datasense_models.py',
    'framework/training/tracking.py', 'requirements-tabm.txt')


def configurations():
    base = dict(k=16, width=256, depth=3, dropout=.1, d_embedding=12, n_bins=16,
        scaling='log_robust', lr=.002, weight_decay=.0003, weight_power=0., smoothing=0.,
        subtype_weight=1., batch_size=512, epochs=60, patience=8, runtime_seconds=3600)
    options = [({}, 'full'), ({'scaling':'quantile'}, 'full'),
        ({'k':32, 'width':384, 'd_embedding':16, 'n_bins':32, 'lr':.001}, 'full'),
        ({'weight_power':.5, 'smoothing':.02, 'lr':.001}, 'full'),
        ({'scaling':'quantile', 'weight_power':.25, 'dropout':.15}, 'full'),
        ({}, 'strict'),
        ({'width':512, 'depth':4, 'lr':.0007, 'weight_decay':.001, 'dropout':.2}, 'full'),
        ({'n_bins':48, 'd_embedding':16, 'lr':.001, 'subtype_weight':1.5}, 'full')]
    rows = []
    for i, (changes, arm) in enumerate(options):
        for family in FAMILIES:
            cfg = dict(base, **changes)
            if family == 'tabm_ne':
                cfg['subtype_weight'] = 1.
            rows.append(dict(family=family, arm=arm, config=cfg, seed=42,
                             recipe=f'{family}_{arm}_c{i}', phase='search'))
    return rows


def choose_best(rows, family):
    eligible = [r for r in rows if r['family']==family and r['phase']=='search']
    return max(eligible, key=lambda r: (r['selection_metrics']['accuracy'], r['selection_metrics']['macro_f1']))


def initialize(device='cpu'):
    STUDY.mkdir(parents=True, exist_ok=True)
    manifest = read(DATA/'model/manifest.json'); validate_manifest(manifest)
    if len(manifest['feature_names']) != 655 or len(manifest['classes']) != 84 or manifest['benign_index'] != 0:
        raise ValueError('Pinned DataSense schema differs')
    if not read(DATA/'verification/integrity.json')['passed'] or not read(DATA/'verification/independent_feature_replay.json')['passed']:
        raise ValueError('Dataset integrity/replay gate failed')
    expected = read(DATA/'verification/output_hashes.json')
    names = {'model/manifest.json', 'verification/integrity.json', 'verification/independent_feature_replay.json',
        'model/window_start_validation.npy', 'model/window_end_validation.npy', 'audit/validation/model_input_sha256.npy'}
    for arm in ('full', 'strict'):
        names.update(p.relative_to(DATA).as_posix() for p in training_paths(DATA, arm).values())
        paths = training_paths(DATA, arm)
        train_ids = np.load(paths['row_ids_train'], mmap_mode='r')
        validation_ids = np.load(paths['row_ids_validation'], mmap_mode='r')
        if len(np.unique(train_ids)) != len(train_ids) or np.intersect1d(train_ids,validation_ids).size:
            raise ValueError('Duplicate training IDs or training/validation overlap')
    hashes = {}
    for name in sorted(names):
        value = digest(DATA/name)
        if name in expected and expected[name] != value:
            raise ValueError(f'Dataset changed: {name}')
        hashes[name] = value
    roles = split_validation(np.load(DATA/'model/y_validation.npy'), np.load(DATA/'model/window_start_validation.npy'),
        np.load(DATA/'model/window_end_validation.npy'), np.load(DATA/'audit/validation/model_input_sha256.npy'))
    existing = ROOT/'artifacts/training/verified_clean_four_models_20261016_v1/validation_roles.npy'
    if existing.exists():
        np.testing.assert_array_equal(roles, np.load(existing))
    role_path = STUDY/'validation_roles.npy'
    if role_path.exists():
        np.testing.assert_array_equal(roles, np.load(role_path))
    else:
        np.save(role_path, roles)
    protocol = dict(device=device, configurations=configurations(), repeat_seeds=[43, 44], max_fits=30,
        input_hashes=hashes, source_hashes={name:digest(ROOT/name) for name in SOURCES},
        roles_sha256=digest(role_path), validation_roles={str(i):int(np.sum(roles==i)) for i in (-1,0,1,2)},
        packages={p:importlib.metadata.version(p) for p in ('tabm','rtdl-num-embeddings','torch','numpy','scikit-learn')},
        selection='Tuning accuracy then macro F1 only', calibration='Fit role1 after selection; assess role2',
        per_fit_seconds=7200, test_loaded=False, auto_promote=False,
        warning='Reused within-capture validation; exploratory; no guaranteed accuracy target.')
    path = STUDY/'study.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Pinned protocol changed; preserve this study and version a new one')
    write(path, protocol)
    for name in SOURCES:
        target = STUDY/'source_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes((ROOT/name).read_bytes())
    return protocol


def verify_inputs(protocol, paths):
    for path in paths:
        if digest(path) != protocol['input_hashes'][path.relative_to(DATA).as_posix()]:
            raise ValueError(f'Changed pinned input: {path.name}')
    if digest(STUDY/'validation_roles.npy') != protocol['roles_sha256']:
        raise ValueError('Changed validation roles')
    for name, expected in protocol['source_hashes'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Changed pinned source: {name}')
    for name, expected in protocol['packages'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f'Changed pinned package: {name}')


def fit(path):
    from framework.training.tabm_fit import train, predict
    from framework.training.tabm_data import TabMPreprocessor
    request = read(path); protocol = read(STUDY/'study.json'); output = path.parent
    paths = training_paths(DATA, request['arm'])
    verify_inputs(protocol, [*paths.values(), DATA/'model/manifest.json'])
    X, y, V, v = [np.load(paths[key], mmap_mode='r') for key in ('X_train','y_train','X_validation','y_validation')]
    manifest = read(DATA/'model/manifest.json'); roles = np.load(STUDY/'validation_roles.npy')
    tuning = np.flatnonzero(roles==0)
    if X.shape != (len(y),655) or V.shape != (len(v),655) or len(np.unique(y)) != 84:
        raise ValueError('Training/validation dimensions or class coverage differs')
    started = time.monotonic()
    prep_path = STUDY/'preprocessing'/f'{request["arm"]}_{request["config"]["scaling"]}_bins{request["config"]["n_bins"]}.json'
    if prep_path.exists():
        saved = read(prep_path)
        if saved['training_sha256'] != protocol['input_hashes'][paths['X_train'].relative_to(DATA).as_posix()]:
            raise ValueError('Preprocessor belongs to different training observations')
        prep = TabMPreprocessor.from_dict(saved['preprocessor'])
    else:
        prep = TabMPreprocessor(request['config']['scaling'], request['config']['n_bins']).fit(X, manifest['feature_names'])
        write(prep_path, dict(training_sha256=protocol['input_hashes'][paths['X_train'].relative_to(DATA).as_posix()],
            training_rows=len(y), preprocessor=prep.to_dict(), fit_partition='train', test_loaded=False))
    print(f'START {request["recipe"]} seed={request["seed"]} rows={len(y)} device={protocol["device"]}', flush=True)
    model, prep, details = train(X,y,V[tuning],v[tuning],manifest,request,output,protocol['device'],prep)
    inference_start = time.monotonic(); p = predict(model,prep,V)
    details['cpu_validation_seconds'] = time.monotonic()-inference_start
    np.save(output/'validation_probabilities.npy',p)
    pred = p.argmax(1); np.save(output/'validation_predictions.npy',pred)
    result = dict(request, **details, status='completed', training_rows=len(y), validation_rows=len(v),
        metrics=classification_metrics(v,pred,manifest['classes'],0),
        elapsed_seconds=time.monotonic()-started, test_loaded=False, unknown_attack_evaluated=False,
        live_validated=False, artifact_hashes={name:digest(output/name) for name in ('model.pt','validation_probabilities.npy','validation_predictions.npy')})
    # Select on the deployed CPU computation for the frozen tuning observations.
    result['selection_metrics'] = classification_metrics(v[tuning],pred[tuning],manifest['classes'],0)
    write(output/'result.json',result)
    print(f'COMPLETE {request["recipe"]}: full={result["metrics"]["accuracy"]:.6f} tuning={result["selection_metrics"]["accuracy"]:.6f}',flush=True)


def completed():
    rows=[]
    for path in sorted((STUDY/'runs').glob('*/attempt_*/result.json')):
        row=read(path)
        if row['status']=='completed':
            for name, expected in row['artifact_hashes'].items():
                if digest(path.parent/name)!=expected:
                    raise ValueError(f'Changed artifact: {path.parent/name}')
            row['run_directory']=path.parent.relative_to(ROOT).as_posix(); rows.append(row)
    return rows


def calibrate(result):
    from framework.training.tabm_calibration import fit_calibration, apply_calibration
    run=ROOT/result['run_directory']; destination=STUDY/'calibration'/f'{result["family"]}_s{result["seed"]}'
    result_hash=digest(run/'result.json')
    if (destination/'result.json').exists():
        saved=read(destination/'result.json')
        if saved['model_result_sha256']!=result_hash:
            raise ValueError('Calibration belongs to another checkpoint')
        for name,expected in saved['artifact_hashes'].items():
            if digest(destination/name)!=expected:
                raise ValueError(f'Changed calibration artifact: {name}')
        return saved
    p=np.load(run/'validation_probabilities.npy'); y=np.load(DATA/'model/y_validation.npy')
    roles=np.load(STUDY/'validation_roles.npy'); manifest=read(DATA/'model/manifest.json')
    result_cal,p_cal,binary=fit_calibration(p,y,roles,manifest)
    result_cal.update(family=result['family'],seed=result['seed'],model_result_sha256=result_hash,
                      full_validation_accuracy=result['metrics']['accuracy'])
    destination.mkdir(parents=True,exist_ok=True)
    write(destination/'parameters.json',result_cal)
    replay_p,replay_binary=apply_calibration(p,read(destination/'parameters.json'))
    np.testing.assert_allclose(replay_p,p_cal,atol=0,rtol=0)
    np.testing.assert_allclose(replay_binary,binary,atol=0,rtol=0)
    np.save(destination/'class_probabilities.npy',p_cal);np.save(destination/'attack_probabilities.npy',binary)
    ids=np.load(DATA/'model/row_ids_validation.npy')
    for role,name in ((1,'calibration'),(2,'assessment')):
        np.save(destination/f'{name}_row_ids.npy',ids[roles==role])
    result_cal['calibrator_reload_verified']=True
    result_cal['artifact_hashes']={p.name:digest(p) for p in destination.iterdir() if p.name!='result.json'}
    write(destination/'result.json',result_cal);return result_cal


def attempt_needed(request):
    parent=STUDY/'runs'/f'{request["recipe"]}_s{request["seed"]}'
    attempts=sorted(parent.glob('attempt_*'))
    if any((p/'result.json').exists() for p in attempts):return False
    return not attempts or (len(attempts)<3 and not (attempts[-1]/'failure.json').exists())


def resources_available():
    import psutil
    from experiment.run_xgboost_followup import memory_available
    for proc in psutil.process_iter(['pid','cmdline']):
        if proc.pid==os.getpid():continue
        command=' '.join(proc.info['cmdline'] or [])
        if 'experiment.train_' in command and '--fit' in command:return False
    memory=memory_available()
    return memory['physical_gib']>=2.5 and memory['commit_gib']>=5.


def launch_trial(request,state,protocol):
    parent=STUDY/'runs'/f'{request["recipe"]}_s{request["seed"]}';parent.mkdir(parents=True,exist_ok=True)
    out=parent/f'attempt_{len(list(parent.glob("attempt_*"))):03d}';out.mkdir()
    write(out/'request.json',request)
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (out/'fit.log').open('a',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-B','-u','-m',MODULE,'--fit',str(out/'request.json')],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        state.update(status='running',active_trial=dict(request,pid=child.pid,directory=out.relative_to(ROOT).as_posix()))
        started=time.monotonic()
        while child.poll() is None:
            stop=(STUDY/'STOP').exists();expired=time.monotonic()-started>protocol['per_fit_seconds']
            if stop or expired:
                import psutil
                try:
                    descendants=psutil.Process(child.pid).children(recursive=True)
                    for descendant in reversed(descendants):
                        try:descendant.kill()
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:pass
                child.kill();child.wait()
                write(out/('interrupted.json' if stop else 'failure.json'),dict(reason='User stop' if stop else 'Watchdog expired'))
                break
            state['heartbeat_at']=time.time();write(STUDY/'state.json',state);time.sleep(5)
    if child.returncode and not any((out/name).exists() for name in ('failure.json','interrupted.json')):
        write(out/'failure.json',dict(exit_code=child.returncode,reason='See fit.log; no substituted result'))
    state['active_trial']=None


def controller():
    with StudyLock(STUDY/'controller.lock'):
        protocol=read(STUDY/'study.json')
        state=dict(status='starting',pid=os.getpid(),started_at=time.time(),active_trial=None,test_loaded=False,planned_fits=30)
        while not (STUDY/'STOP').exists():
            rows=completed();queue=deepcopy(protocol['configurations'])
            state['completed_fits']={f:sum(r['family']==f for r in rows) for f in FAMILIES}
            state['completed_fits_total']=len(rows)
            state['best']={}
            for family in FAMILIES:
                if any(r['family']==family and r['phase']=='search' for r in rows):
                    best=choose_best(rows,family)
                    state['best'][family]=dict(recipe=best['recipe'],seed=best['seed'],
                        tuning_accuracy=best['selection_metrics']['accuracy'],
                        accuracy=best['metrics']['accuracy'],macro_recall=best['metrics']['macro_recall'],
                        macro_f1=best['metrics']['macro_f1'])
            for family in FAMILIES:
                requests=[q for q in queue if q['family']==family]
                if all(not attempt_needed(q) for q in requests) and any(r['family']==family for r in rows):
                    best=choose_best(rows,family);write(STUDY/'selected'/f'{family}.json',best)
                    calibrate(best)
                    for seed in protocol['repeat_seeds']:
                        queue.append({key:deepcopy(best[key]) for key in ('family','arm','config','recipe')}|dict(seed=seed,phase='repeat'))
                    for row in rows:
                        if row['family']==family and row['phase']=='repeat':calibrate(row)
            pending=[q for q in queue if attempt_needed(q)]
            state.update(pending_fits=len(pending),heartbeat_at=time.time(),calibrations=len(list((STUDY/'calibration').glob('*/result.json'))))
            if not pending:
                state.update(status='completed' if len(rows)==30 else 'completed_with_failures',active_trial=None)
                write(STUDY/'state.json',state);write(STUDY/'results.json',dict(state=state,runs=rows));return
            if not resources_available():
                state['status']='waiting_for_existing_fit_or_memory';write(STUDY/'state.json',state);time.sleep(10);continue
            try:
                with ExitStack() as locks:
                    if protocol['device']=='cuda':
                        for name in ('optimization_24h_v1','csv_specialization_v2','csv_research_v3'):
                            locks.enter_context(StudyLock(ROOT/'artifacts/training'/name/'controller.lock'))
                    launch_trial(pending[0],state,protocol)
            except RuntimeError as exc:
                if 'holds the lock' not in str(exc):raise
                state['status']='waiting_for_existing_gpu_controller';write(STUDY/'state.json',state);time.sleep(10)
        state.update(status='stopped',active_trial=None);write(STUDY/'state.json',state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',choices=('start','resume','status','stop','_run','initialize'))
    parser.add_argument('--device',choices=('cpu','cuda'),default=None)
    parser.add_argument('--fit',type=Path)
    args=parser.parse_args()
    if args.fit:
        fit(args.fit);return
    if args.command in ('start','resume','initialize'):
        pinned=read(STUDY/'study.json')['device'] if (STUDY/'study.json').exists() else 'cpu'
        initialize(args.device or pinned)
        if args.command=='initialize':print(STUDY);return
        with StudyLock(STUDY/'controller.lock'):
            import psutil
            if (STUDY/'launch.json').exists():
                previous=read(STUDY/'launch.json')['pid']
                if psutil.pid_exists(previous):
                    command=' '.join(psutil.Process(previous).cmdline())
                    if MODULE in command and '_run' in command:
                        raise RuntimeError('TabM controller already launched')
            (STUDY/'STOP').unlink(missing_ok=True)
            flags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            with (STUDY/'controller.log').open('a',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,'-B','-u','-m',MODULE,'_run'],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=flags,
                    close_fds=True,start_new_session=os.name!='nt')
            write(STUDY/'launch.json',dict(pid=child.pid,started_at=time.time()))
        print(json.dumps(dict(launched_pid=child.pid,study=STUDY.as_posix())))
    elif args.command=='_run':
        try:controller()
        except Exception as exc:
            write(STUDY/'controller_failure.json',dict(type=type(exc).__name__,error=str(exc),time=time.time()))
            raise
    elif args.command=='stop':
        STUDY.mkdir(parents=True,exist_ok=True);(STUDY/'STOP').touch()
    elif args.command=='status':
        print((STUDY/'state.json').read_text() if (STUDY/'state.json').exists() else 'Not started')
    else:parser.error('Choose start, resume, initialize, status or stop')


if __name__=='__main__':main()
