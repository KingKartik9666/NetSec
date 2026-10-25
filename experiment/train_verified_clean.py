"""Isolated, resumable four-family training on verified DataSense v2.

Never loads held-out test arrays. The earlier studies and fixed RF remain intact.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiment.datasense_quality import digest, temperature_probabilities
from experiment.optimize_models import StudyLock
from framework.training.tracking import classification_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset/experiments/datasense_verified_clean_20261014_v2'
STUDY = ROOT / 'artifacts/training/verified_clean_four_models_20261016_v1'
FAMILIES = ('random_forest', 'xgboost', 'kan', 'hem')
MODULE = 'experiment.train_verified_clean'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def training_paths(data, arm):
    if arm not in ('full', 'strict'):
        raise ValueError('Only full or strict training is allowed; no test input')
    base = Path(data) / 'model'
    train = base if arm == 'full' else base / 'strict_train'
    return {**{f'{k}_train': train / f'{k}_train.npy' for k in ('X', 'y', 'row_ids')},
            **{f'{k}_validation': base / f'{k}_validation.npy' for k in ('X', 'y', 'row_ids')}}


def configurations():
    """Frozen 24-fit search; eight selected-recipe seed repeats follow."""
    rf = dict(n_estimators=100, max_depth=None, min_samples_leaf=1,
              max_features='sqrt', class_weight=None, n_jobs=2)
    xgb = dict(n_estimators=800, max_depth=6, max_bin=128, lr=.08,
               weight_strategy='none', loss='native_ce', early_stopping_rounds=60,
               runtime_seconds=3300, selection_metric='accuracy')
    neural = dict(epochs=40, batch_size=512, patience=8, lr=.001,
                  weight_decay=.0001, weight_strategy='none', scaling='log_minmax',
                  selection_metric='accuracy', scheduler='plateau', runtime_seconds=3300)
    kan = dict(neural, hidden=[64, 32], grid_size=5, spline_order=3)
    hem = dict(neural, hidden=[512, 256, 64, 32], binary_hidden=[512, 128, 32, 16],
               pca_components=32, knn_k=1, knn_max_rows=480405, knn_batch_size=32)
    options = {
        'random_forest': [rf, dict(rf, n_estimators=200, max_depth=24, min_samples_leaf=2),
                          dict(rf, n_estimators=200, class_weight='balanced_subsample')],
        'xgboost': [xgb, dict(xgb, max_depth=8, lr=.05, reg_lambda=4.),
                    dict(xgb, max_depth=6, weight_strategy='sqrt_balanced',
                         loss='label_smoothed_ce', smoothing=.05)],
        'kan': [kan, dict(kan, hidden=[128, 64], lr=.0005),
                dict(kan, weight_strategy='sqrt_balanced', scaling='quantile')],
        'hem': [hem, dict(hem, pca_components=64, knn_k=3, knn_batch_size=16),
                dict(hem, weight_strategy='sqrt_balanced', scaling='quantile')],
    }
    return [dict(family=f, arm=a, config=deepcopy(options[f][i]), seed=42,
                 recipe=f'{f}_{a}_c{i}', phase='search')
            for i in range(3) for a in ('strict', 'full') for f in FAMILIES]


def split_validation(labels, starts, ends, hashes):
    """Early half tunes; later half splits into calibration/assessment, purged."""
    from experiment.datasense_quality import calibration_partition
    from framework.data.research_data import prevent_duplicate_crossing
    labels, starts, ends, hashes = map(np.asarray, (labels, starts, ends, hashes))
    first = calibration_partition(labels, starts, ends, hashes)
    codes = np.full(len(labels), -1, dtype=np.int8)
    codes[first == 0] = 0
    later = np.flatnonzero(first == 1)
    second = calibration_partition(labels[later], starts[later], ends[later], hashes[later])
    codes[later[second == 0]] = 1
    codes[later[second == 1]] = 2
    return prevent_duplicate_crossing(codes, hashes)[0]


def choose_best(rows, family):
    return max((r for r in rows if r['family'] == family and r.get('phase', 'search') == 'search'),
               key=lambda r: (r['selection_metrics']['accuracy'], r['selection_metrics']['macro_f1']))


def initialize():
    STUDY.mkdir(parents=True, exist_ok=True)
    expected = read(DATA / 'verification/output_hashes.json')
    names = {'model/manifest.json', 'summary.json', 'verification/integrity.json',
             'audit/validation/model_input_sha256.npy', 'model/window_start_validation.npy',
             'model/window_end_validation.npy'}
    for arm in ('full', 'strict'):
        names.update(p.relative_to(DATA).as_posix() for p in training_paths(DATA, arm).values())
    hashes = {}
    for name in sorted(names):
        value = digest(DATA / name)
        if name in expected and value != expected[name]:
            raise ValueError(f'Verified cleaned input changed: {name}')
        hashes[name] = value
    integrity = read(DATA / 'verification/integrity.json')
    if not integrity['passed'] or not integrity['source_and_input_hashes_unchanged']:
        raise ValueError('Dataset integrity gate failed')
    manifest = read(DATA / 'model/manifest.json')
    for arm in ('full', 'strict'):
        paths = training_paths(DATA, arm)
        arrays = {k: np.load(p, mmap_mode='r') for k, p in paths.items()}
        for split in ('train', 'validation'):
            X, y, ids = [arrays[f'{k}_{split}'] for k in ('X', 'y', 'row_ids')]
            if X.shape != (len(y), 655) or len(ids) != len(y) or len(np.unique(ids)) != len(ids):
                raise ValueError('Dataset dimensions or unique IDs failed')
        if np.intersect1d(arrays['row_ids_train'], arrays['row_ids_validation']).size:
            raise ValueError('Training/validation ID overlap')
        if len(np.unique(arrays['y_train'])) != len(manifest['classes']):
            raise ValueError('Training arm lost a class')
    codes = split_validation(np.load(DATA / 'model/y_validation.npy'),
                             np.load(DATA / 'model/window_start_validation.npy'),
                             np.load(DATA / 'model/window_end_validation.npy'),
                             np.load(DATA / 'audit/validation/model_input_sha256.npy'))
    if any(np.sum(codes == role) < 100 for role in (0, 1, 2)):
        raise ValueError('Insufficient validation role coverage')
    partition = STUDY / 'validation_roles.npy'
    if partition.exists():
        np.testing.assert_array_equal(np.load(partition), codes)
    else:
        np.save(partition, codes)
    hashes_code = {name: digest(ROOT / name) for name in (
        'experiment/train_verified_clean.py', 'experiment/train_csv_models.py',
        'framework/training/neural.py', 'framework/training/csv_xgboost.py',
        'framework/training/trees.py', 'experiment/calibrate_datasense_models.py',
        'experiment/datasense_quality.py')}
    protocol = dict(dataset=DATA.relative_to(ROOT).as_posix(), target_accuracy=.90,
                    target_scope='Each family: full validation accuracy; separately report untouched-role assessment',
                    configurations=configurations(), repeat_seeds=[43, 44], max_fits=32,
                    input_hashes=hashes, source_hashes=hashes_code, roles_sha256=digest(partition),
                    validation_roles={str(i): int(np.sum(codes == i)) for i in (-1, 0, 1, 2)},
                    selection='Tuning-role accuracy; macro F1 tie-break. No calibration/assessment-based selection.',
                    calibration='Earlier later-half rows fit calibrators; purged final quarter assesses them.',
                    per_fit_seconds=7200, no_overall_deadline=True, rf_retuning_authorized=True,
                    test_loaded=False, app_promotion=False,
                    warning='Previously reused within-capture validation; not an independent final test. No guarantee of 90%.')
    if (STUDY / 'study.json').exists() and read(STUDY / 'study.json') != protocol:
        raise ValueError('Pinned study differs; preserve it and version a new study')
    write(STUDY / 'study.json', protocol)
    for name in hashes_code:
        dest = STUDY / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes((ROOT / name).read_bytes())
    return protocol


def neural_full_outputs(family, checkpoint, V):
    """CPU neural scores plus explicit CUDA exact KNN for HEM's hard vote."""
    import joblib
    import torch
    from scipy.special import expit, softmax
    from framework.training.neural import _models, _clean, cuda_knn_predict, hem_closed_set_vote
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    transformed = joblib.load(checkpoint.parent / 'preprocessing.joblib').transform(_clean(V)).astype(np.float32)
    models = _models(family, saved['input_size'], saved['n_classes'], saved['config'], torch, device='cpu')
    logits = {}
    for name, model in models.items():
        model.load_state_dict(saved['state_dicts'][name]); model.eval()
        with torch.no_grad():
            logits[name] = np.concatenate([model(torch.from_numpy(transformed[i:i+256])).numpy()
                                            for i in range(0, len(V), 256)])
    if family == 'kan':
        p = softmax(logits['kan'], axis=1)
        return p.argmax(1), p, None
    multi = softmax(logits['multiclass'], axis=1)
    binary = expit(logits['binary'][:, 0])
    with np.load(checkpoint.parent / 'knn_references.npz') as refs:
        knn = cuda_knn_predict(transformed, refs['X'], refs['y'], saved['n_classes'],
                               saved['config']['knn_k'], saved['config']['knn_batch_size'])
    pred = hem_closed_set_vote(multi.argmax(1), binary >= .5, knn, saved['benign_index'])
    return pred, None, dict(multiclass_probabilities=multi, binary_probabilities=binary, knn_predictions=knn)


def fit(path):
    import joblib
    from experiment.train_csv_models import train_candidate, reload_outputs
    request = read(path); out = path.parent
    protocol = read(STUDY / 'study.json')
    family, arm = request['family'], request['arm']
    paths = training_paths(DATA, arm)
    for p in [*paths.values(), DATA / 'model/manifest.json']:
        if digest(p) != protocol['input_hashes'][p.relative_to(DATA).as_posix()]:
            raise ValueError('Training input hash changed')
    if digest(STUDY / 'validation_roles.npy') != protocol['roles_sha256']:
        raise ValueError('Validation roles changed')
    arrays = {k: np.load(p, mmap_mode='r') for k, p in paths.items()}
    X, y, V, v = [arrays[k] for k in ('X_train', 'y_train', 'X_validation', 'y_validation')]
    manifest = read(DATA / 'model/manifest.json'); n = len(manifest['classes'])
    codes = np.load(STUDY / 'validation_roles.npy'); tuning = np.flatnonzero(codes == 0)
    started = time.time()
    print(f'START {family} {arm} seed={request["seed"]} rows={len(y)} features={X.shape[1]} tuning={len(tuning)}', flush=True)
    if family == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        model = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                          ('classifier', RandomForestClassifier(**request['config'], random_state=request['seed']))])
        model.fit(X, y)
        p = model.predict_proba(V); pred = p.argmax(1)
        checkpoint = out / 'model.joblib'; joblib.dump(model, checkpoint, compress=3)
        reload = joblib.load(checkpoint).predict_proba(V)
        np.testing.assert_allclose(reload, p, atol=1e-12, rtol=0)
        details = dict(device='cpu', backend='sklearn RandomForestClassifier; separately authorized tuning',
                       checkpoint_path=str(checkpoint))
        components = None; count = len(v)
    else:
        details = train_candidate(family, X, y, V[tuning], v[tuning], n, manifest['benign_index'],
                                  request['config'], request['seed'], out)
        tune_pred = details.pop('predictions'); details.pop('probabilities', None)
        checkpoint = Path(details['checkpoint_path'])
        if family == 'xgboost':
            pred, p = reload_outputs(family, checkpoint, V); components = None
        else:
            pred, p, components = neural_full_outputs(family, checkpoint, V)
        np.testing.assert_array_equal(pred[tuning], tune_pred)
        count = min(128, len(v)) if family == 'hem' else len(v)
        replay, replay_p = reload_outputs(family, checkpoint, V[:count])
        np.testing.assert_array_equal(replay, pred[:count])
        if p is not None:
            np.testing.assert_allclose(replay_p, p[:count], atol=2e-6, rtol=1e-5)
    np.save(out / 'validation_predictions.npy', pred)
    if p is not None:
        np.save(out / 'validation_probabilities.npy', p)
    if components is not None:
        np.savez(out / 'validation_components.npz', **components)
    metrics = classification_metrics(v, pred, manifest['classes'], manifest['benign_index'])
    selection = classification_metrics(v[tuning], pred[tuning], manifest['classes'], manifest['benign_index'])
    result = dict(request, **{k: val for k, val in details.items() if k != 'checkpoint_path'},
                  status='completed', checkpoint_path=checkpoint.relative_to(ROOT).as_posix(),
                  checkpoint_reload_verified=True, reload_device='cpu', reload_rows=count,
                  training_rows=len(y), validation_rows=len(v), feature_count=X.shape[1],
                  metrics=metrics, selection_metrics=selection, elapsed_seconds=time.time()-started,
                  test_loaded=False, live_validated=False, unknown_attack_evaluated=False,
                  research_scope='HEM is upstream-derived closed-set adaptation; previously reused within-capture validation.')
    result['artifact_hashes'] = {p.name: digest(p) for p in out.iterdir()
                                  if p.suffix in ('.joblib', '.pt', '.ubj', '.npy', '.npz')}
    write(out / 'result.json', result)
    print(f'COMPLETE {family} full-validation={metrics["accuracy"]:.6f} tuning={selection["accuracy"]:.6f}', flush=True)


def calibrate(result):
    import joblib
    from scipy.special import logit
    from sklearn.linear_model import LogisticRegression
    from experiment.calibrate_datasense_models import temperature_fit, assessment, confidence_metrics
    family = result['family']; run = ROOT / result['run_directory']
    out = STUDY / 'calibration' / family
    selected_hash = digest(run / 'result.json')
    if (out / 'result.json').exists():
        old = read(out / 'result.json')
        if old['model_result_sha256'] != selected_hash:
            raise ValueError('Calibrator belongs to another selected model')
        return old
    out.mkdir(parents=True, exist_ok=True)
    codes = np.load(STUDY / 'validation_roles.npy')
    fit_ids, assess_ids = np.flatnonzero(codes == 1), np.flatnonzero(codes == 2)
    y = np.load(DATA / 'model/y_validation.npy'); manifest = read(DATA / 'model/manifest.json')
    pred = np.load(run / 'validation_predictions.npy'); correct = pred == y
    np.save(out / 'calibration_row_ids.npy', np.load(DATA / 'model/row_ids_validation.npy')[fit_ids])
    np.save(out / 'assessment_row_ids.npy', np.load(DATA / 'model/row_ids_validation.npy')[assess_ids])
    if family != 'hem':
        p = np.load(run / 'validation_probabilities.npy')
        t = temperature_fit(p[fit_ids], y[fit_ids])
        calibrated = temperature_probabilities(p, t)
        np.testing.assert_array_equal(calibrated.argmax(1), pred)
        write(out / 'temperature.json', dict(temperature=t))
        np.testing.assert_allclose(temperature_probabilities(p, read(out / 'temperature.json')['temperature']), calibrated)
        np.save(out / 'calibrated_probabilities.npy', calibrated)
        before = assessment(p[assess_ids], y[assess_ids], manifest)
        after = assessment(calibrated[assess_ids], y[assess_ids], manifest)
        raw = p.max(1); features = logit(np.clip(raw, 1e-6, 1-1e-6))[:, None]
        temperature_result = dict(temperature=t, before=before, after=after)
    else:
        with np.load(run / 'validation_components.npz') as parts:
            mc, bc = parts['multiclass_probabilities'], parts['binary_probabilities']
            voted = mc[np.arange(len(y)), pred]
            matching = np.where(pred == manifest['benign_index'], 1-bc, bc)
            raw = .5*(voted+matching)
            features = np.column_stack([np.log(np.clip(voted, 1e-12, 1)),
                np.log(np.clip(matching, 1e-12, 1)), (parts['knn_predictions'] == mc.argmax(1)).astype(float)])
        temperature_result = None
    if len(np.unique(correct[fit_ids])) != 2:
        raise ValueError('Correctness calibration needs both correct and incorrect examples')
    estimator = LogisticRegression(C=1., random_state=42, max_iter=1000).fit(features[fit_ids], correct[fit_ids])
    confidence = estimator.predict_proba(features)[:, list(estimator.classes_).index(True)]
    joblib.dump(estimator, out / 'correctness_calibrator.joblib')
    replay = joblib.load(out / 'correctness_calibrator.joblib').predict_proba(features)[:, 1]
    np.testing.assert_allclose(replay, confidence, atol=0, rtol=0)
    np.save(out / 'correctness_confidence.npy', confidence)
    entry = dict(family=family, model_result_sha256=selected_hash, calibration_rows=len(fit_ids),
                 assessment_rows=len(assess_ids), temperature=temperature_result,
                 correctness_before=confidence_metrics(correct[assess_ids], raw[assess_ids]),
                 correctness_after=confidence_metrics(correct[assess_ids], confidence[assess_ids]),
                 assessment_accuracy=float(correct[assess_ids].mean()),
                 full_validation_accuracy=result['metrics']['accuracy'], predictions_unchanged=True,
                 calibrator_reload_verified=True, test_loaded=False,
                 meaning='Estimated correctness of unchanged decisions; HEM confidence is not ensemble class probabilities.',
                 assessment_used_for_fitting_or_selection=False,
                 limitation='Previously reused source validation; exploratory, not independent final test.')
    write(out / 'result.json', entry)
    return entry


def completed():
    rows = []
    for p in sorted((STUDY / 'runs').glob('*/attempt_*/result.json')):
        row = read(p)
        if row['status'] == 'completed':
            for name, expected in row['artifact_hashes'].items():
                if digest(p.parent / name) != expected:
                    raise ValueError(f'Changed completed artifact: {p.parent / name}')
            row['run_directory'] = p.parent.relative_to(ROOT).as_posix(); rows.append(row)
    return rows


def available_resources(gpu=False):
    import psutil
    # Fits are serialized here; CPU RF may coexist with the old CUDA fit only
    # when both physical and Windows commit headroom are available.
    for proc in psutil.process_iter(['pid', 'cmdline']):
        if proc.pid == os.getpid():
            continue
        command = ' '.join(proc.info['cmdline'] or [])
        if gpu and 'experiment.train_' in command and ('--fit' in command or 'train_verified_clean' in command):
            return False
    from experiment.run_xgboost_followup import memory_available
    memory = memory_available()
    return memory['physical_gib'] >= (3.5 if gpu else 4.) and memory['commit_gib'] >= 6.


def needs_attempt(request):
    parent = STUDY/'runs'/f'{request["recipe"]}_s{request["seed"]}'
    attempts = sorted(parent.glob('attempt_*'))
    if any((p/'result.json').exists() for p in attempts):
        return False
    # A crashed controller/worker can resume in a new directory; failed fits
    # remain recorded and do not loop indefinitely.
    return not attempts or (len(attempts) < 3 and not (attempts[-1]/'failure.json').exists())


def launch_trial(request, state, protocol):
    recipe = request['recipe'] + '_s' + str(request['seed'])
    parent = STUDY / 'runs' / recipe; parent.mkdir(parents=True, exist_ok=True)
    attempts = list(parent.glob('attempt_*'))
    if not needs_attempt(request):
        return False  # Do not silently loop on failures; resume creates no overwrites.
    out = parent / f'attempt_{len(attempts):03d}'; out.mkdir()
    write(out / 'request.json', request)
    environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', IMDAI_FIT_DEADLINE=str(time.time()+protocol['per_fit_seconds']-30))
    state.update(status='running', active_trial=dict(request, directory=out.relative_to(ROOT).as_posix()))
    with (out / 'fit.log').open('a', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-B', '-u', '-m', MODULE, '--fit', str(out/'request.json')],
                                  cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        state['active_trial']['pid'] = child.pid; write(STUDY/'state.json', state)
        started = time.time()
        while child.poll() is None:
            if (STUDY/'STOP').exists() or time.time()-started > protocol['per_fit_seconds']:
                import psutil
                parent_process = psutil.Process(child.pid)
                descendants = parent_process.children(recursive=True)
                for proc in reversed(descendants):
                    try: proc.kill()
                    except psutil.NoSuchProcess: pass
                child.kill(); child.wait()
                write(out/'failure.json', dict(reason='Explicit stop or per-fit watchdog', test_loaded=False))
                break
            state['heartbeat_at'] = time.time(); write(STUDY/'state.json', state); time.sleep(5)
    if child.returncode and not (out/'failure.json').exists():
        write(out/'failure.json', dict(exit_code=child.returncode, reason='See fit.log; no substitute result'))
    state['active_trial'] = None
    return True


def controller():
    with StudyLock(STUDY/'controller.lock'):
        protocol = initialize()
        state = dict(status='starting', pid=os.getpid(), started_at=time.time(), active_trial=None,
                     target_accuracy=.90, test_loaded=False)
        # An existing orphan worker must finish before any next worker starts.
        while not (STUDY/'STOP').exists():
            rows = completed()
            state['completed_fits'] = {f: sum(r['family']==f for r in rows) for f in FAMILIES}
            state['best'] = {f: dict(accuracy=choose_best(rows, f)['metrics']['accuracy'],
                                    tuning_accuracy=choose_best(rows, f)['selection_metrics']['accuracy'])
                             for f in FAMILIES if any(r['family']==f for r in rows)}
            queue = list(protocol['configurations'])
            # Repeats start only once every family has had its full search.
            search_done = all(any(r['recipe']==q['recipe'] and r['seed']==42 for r in rows)
                              or list((STUDY/'runs'/f'{q["recipe"]}_s42').glob('attempt_*/failure.json'))
                              for q in queue)
            if search_done:
                for family in FAMILIES:
                    if not any(r['family']==family for r in rows):
                        continue
                    best = choose_best(rows, family)
                    write(STUDY/'selected'/f'{family}.json', best)
                    calibrate(best)
                    for seed in protocol['repeat_seeds']:
                        queue.append({k: deepcopy(best[k]) for k in ('family','arm','config','recipe')} |
                                     dict(seed=seed, phase='repeat'))
            pending = [q for q in queue if needs_attempt(q)]
            state['pending_fits'] = len(pending)
            if not pending:
                failures = list((STUDY/'runs').glob('*/attempt_*/failure.json'))
                state.update(status='completed' if len(rows)==32 else 'completed_with_failures',
                             failed_attempts=len(failures),
                             target_met={f: all(any(r['family']==f and r['seed']==s and r['recipe']==choose_best(rows,f)['recipe']
                                                      and r['metrics']['accuracy']>=.90 for r in rows) for s in (42,43,44))
                                         for f in FAMILIES if f in state['best']})
                write(STUDY/'state.json', state); return
            request = pending[0]
            if not available_resources(gpu=request['family'] != 'random_forest'):
                cpu = next((q for q in pending if q['family']=='random_forest'), None)
                if cpu is not None and available_resources(gpu=False):
                    launch_trial(cpu, state, protocol); continue
                state.update(status='waiting_for_existing_fit_or_memory', heartbeat_at=time.time())
                write(STUDY/'state.json', state); time.sleep(10); continue
            if request['family'] == 'random_forest':
                launch_trial(request, state, protocol)
            else:
                try:
                    # Same locks as the old research controller; never compete for its CUDA device.
                    with ExitStack() as locks:
                        for name in ('optimization_24h_v1','csv_specialization_v2','csv_research_v3'):
                            locks.enter_context(StudyLock(ROOT/'artifacts/training'/name/'controller.lock'))
                        launch_trial(request, state, protocol)
                except RuntimeError as exc:
                    if 'holds the lock' not in str(exc): raise
                    # Still allow pending RF fits when GPU ownership is unavailable.
                    cpu = next((q for q in pending if q['family']=='random_forest'), None)
                    if cpu is not None: launch_trial(cpu, state, protocol)
                    else:
                        state.update(status='waiting_for_existing_gpu_controller', heartbeat_at=time.time())
                        write(STUDY/'state.json', state); time.sleep(10)
        state.update(status='stopped', active_trial=None); write(STUDY/'state.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=('start','resume','status','stop','_run'))
    parser.add_argument('--fit', type=Path)
    args = parser.parse_args()
    if args.fit:
        fit(args.fit); return
    if args.command in ('start', 'resume'):
        initialize()
        with StudyLock(STUDY/'controller.lock'):
            import psutil
            if (STUDY/'launch.json').exists() and psutil.pid_exists(read(STUDY/'launch.json')['pid']):
                raise RuntimeError('Study controller already launched')
            (STUDY/'STOP').unlink(missing_ok=True)
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            with (STUDY/'controller.log').open('a', encoding='utf-8') as log:
                child = subprocess.Popen([sys.executable,'-B','-u','-m',MODULE,'_run'], cwd=ROOT,
                                          stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                          creationflags=flags, close_fds=True, start_new_session=os.name!='nt')
            write(STUDY/'launch.json', dict(pid=child.pid, started_at=time.time()))
        print(json.dumps(dict(launched_pid=child.pid, study=STUDY.as_posix())))
    elif args.command == '_run':
        controller()
    elif args.command == 'stop':
        STUDY.mkdir(parents=True, exist_ok=True); (STUDY/'STOP').touch()
    elif args.command == 'status':
        print((STUDY/'state.json').read_text() if (STUDY/'state.json').exists() else 'Not started')
    else:
        parser.error('Choose start, resume, status, stop, or --fit')


if __name__ == '__main__':
    main()
