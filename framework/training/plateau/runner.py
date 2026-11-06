"""Bounded, resumable CPU ablations; existing studies and validation stay sealed."""
from __future__ import annotations

import gc
import importlib.metadata
from pathlib import Path
import shutil
import time

import numpy as np

from experiment.optimize_models import StudyLock
from framework.data.research_data import stratified_cap
from framework.training.tracking import validate_manifest
from .data import (TRAIN_FILES, check_roles, common_evaluation_mask, feature_arms,
                   load_training_array, read_json, row_hashes, sha256, verify_pins,
                   write_json)
from .metrics import measure

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/'dataset/experiments/datasense_verified_clean_20261014_v2/model'
FOLDS = ROOT/'artifacts/training/all_models_wrapup_20261023_v1/folds/0'
STUDY = ROOT/'artifacts/research/plateau_20261029_v2'
SOURCES = [f'framework/training/plateau/{name}.py' for name in
           ('__init__', 'data', 'metrics', 'runner')] + [
    'experiment/investigate_plateau.py', 'experiment/optimize_models.py',
    'framework/data/research_data.py', 'framework/training/tracking.py']
PACKAGES = ('numpy', 'xgboost', 'scipy', 'scikit-learn', 'psutil')


def initialize(study=STUDY):
    """Pin one pilot specification before any model is fitted."""
    study = Path(study)
    with StudyLock(study/'controller.lock'):
        if (study/'protocol.json').exists():
            protocol = read_json(study/'protocol.json')
            verify(protocol, study)
            return protocol
        manifest = read_json(DATA/'manifest.json')
        validate_manifest(manifest)
        names, classes = manifest['feature_names'], manifest['classes']
        if len(names) != 655 or len(classes) != 84:
            raise ValueError('Expected frozen 655-feature, 84-class schema')
        paths = [DATA/'manifest.json', *[DATA/name for name in sorted(TRAIN_FILES)]]
        paths += [FOLDS/f'{role}{suffix}.npy' for role in ('fit', 'early_stop')
                  for suffix in ('', '_row_ids')]
        pins = {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}
        expected = read_json(DATA.parent/'verification/output_hashes.json')
        for p in paths:
            if p.is_relative_to(DATA.parent):
                key = p.relative_to(DATA.parent).as_posix()
                if key not in expected or sha256(p) != expected[key]:
                    raise ValueError(f'Canonical dataset pin mismatch: {key}')
        historical = read_json(ROOT/'artifacts/training/all_models_wrapup_20261023_v1/study.json')
        # The sealed continuation keeps role hashes under protocol/fold hashes.
        fold_pins = historical.get('role_hashes', historical.get('fold_hashes', {}))
        for p in paths:
            if p.is_relative_to(FOLDS):
                relative = p.relative_to(ROOT).as_posix()
                if relative not in fold_pins or sha256(p) != fold_pins[relative]:
                    raise ValueError(f'Frozen fold pin missing or changed: {relative}')
        arrays = {name: load_training_array(DATA, f'{name}_train.npy')
                  for name in ('X', 'y', 'row_ids', 'window_start', 'window_end')}
        X, y, ids = (arrays[name] for name in ('X', 'y', 'row_ids'))
        if X.shape != (len(y), len(names)) or len(ids) != len(y):
            raise ValueError('Training array/schema alignment mismatch')
        roles = {}
        for role, cap in (('fit', 24000), ('early_stop', 8000)):
            index = np.load(FOLDS/f'{role}.npy', allow_pickle=False)
            np.testing.assert_array_equal(ids[index], np.load(FOLDS/f'{role}_row_ids.npy', allow_pickle=False))
            roles[role] = stratified_cap(index, y, cap, 20261029)
        fit, candidate = roles['fit'], roles['early_stop']
        check_roles(y, arrays['window_start'], arrays['window_end'], fit, candidate)
        if np.unique(y[fit]).size != len(classes):
            raise ValueError('Pilot fitting sample lost class coverage')
        A, B = np.array(X[fit]), np.array(X[candidate])
        arms, groups = feature_arms(A, names)
        keep, collisions = common_evaluation_mask(A, B, arms)
        evaluate = candidate[keep]
        if not len(evaluate):
            raise ValueError('Projected duplicate purging emptied evaluation')
        prepared = study/'prepared'
        prepared.mkdir(exist_ok=True)
        values = {'X_fit': A, 'y_fit': np.array(y[fit]), 'X_evaluate': B[keep],
                  'y_evaluate': np.array(y[evaluate]), 'fit_indices': fit,
                  'evaluation_indices': evaluate, 'fit_row_ids': np.array(ids[fit]),
                  'evaluation_row_ids': np.array(ids[evaluate]),
                  'purged_row_ids': np.array(ids[candidate[~keep]])}
        for name, value in values.items():
            np.save(prepared/f'{name}.npy', value, allow_pickle=False)
        source_pins = {name: sha256(ROOT/name) for name in SOURCES}
        for name in SOURCES:
            target = study/'source_snapshot'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, target)
        jobs = [dict(arm=arm, seed=seed) for seed in (42, 43, 44) for arm in arms]
        protocol = dict(version=1, study='existing-model representation pilot',
            fit_rows=len(fit), evaluation_rows=len(evaluate), feature_names=names,
            classes=classes, benign_index=manifest['benign_index'], arms=arms,
            duplicate_groups_fitting_only=groups, projected_collision_purge=collisions,
            fit_support=np.bincount(y[fit], minlength=len(classes)).tolist(),
            evaluation_support_before_purge=np.bincount(y[candidate], minlength=len(classes)).tolist(),
            evaluation_support=np.bincount(y[evaluate], minlength=len(classes)).tolist(),
            evaluation_distinct_vectors={arm: int(len(np.unique(row_hashes(B[keep][:, columns]))))
                                         for arm, columns in arms.items()},
            jobs=jobs, rounds=96, checkpoints=[32, 96], threads=1,
            wall_seconds=6*3600, per_fit_seconds=1800, input_hashes=pins,
            source_hashes=source_pins,
            prepared_hashes={p.relative_to(study).as_posix(): sha256(p) for p in prepared.glob('*.npy')},
            packages={name: importlib.metadata.version(name) for name in PACKAGES},
            source_validation_loaded=False, held_out_test_loaded=False,
            auto_promote=False, device='cpu', selection='No model selection or calibration in this pilot',
            caveats=[
                'Existing training-derived vocabulary includes feature selection using later internal folds.',
                'Historical early_stop role is reused development data; not an independent holdout.',
                'Uses original full fitting fold, not the historically label-filtered strict subset.',
                'All arms share rows novel relative to fitting under every tested projection.',
                'Small fixed-budget XGBoost fits do not reproduce frozen-model or live accuracy.',
                'This does not test extension of the original 215-round depth-9 checkpoint.',
                'No new scenario-heldout evidence, source-validation assessment, or label changes.',
            ])
        verify_pins(pins, ROOT)
        write_json(study/'protocol.json', protocol)
        (study/'protocol.sha256').write_text(sha256(study/'protocol.json')+'\n', encoding='utf-8')
        write_json(study/'state.json', dict(status='prepared', completed=0, planned=len(jobs)))
        return protocol


def verify(protocol, study):
    study = Path(study)
    if sha256(study/'protocol.json') != (study/'protocol.sha256').read_text().strip():
        raise ValueError('Pinned protocol changed')
    verify_pins(protocol['source_hashes'], ROOT)
    verify_pins(protocol['input_hashes'], ROOT)
    verify_pins(protocol['prepared_hashes'], study)
    for name, version in protocol['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Pinned package changed: {name}')


def train_one(out, X, y, V, truth, names, classes, benign_index, *, seed, rounds,
              checkpoints, threads, stop_path, deadline, heartbeat=None):
    import xgboost as xgb
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if Path(stop_path).exists() or time.time() >= deadline:
        raise InterruptedError('STOP requested or budget expired')
    if rounds < 1 or any(not 0 < n <= rounds for n in checkpoints):
        raise ValueError('Invalid round budget')
    started = time.time()

    def check_budget():
        if Path(stop_path).exists() or time.time() >= deadline:
            raise InterruptedError('STOP requested or budget expired; incomplete fit')

    class Progress(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            check_budget()
            if heartbeat is not None and (epoch == 0 or (epoch+1) % 4 == 0):
                heartbeat(epoch+1, time.time()-started)
            return False

    parameters = dict(objective='multi:softprob', num_class=len(classes), tree_method='hist',
                      device='cpu', max_depth=5, max_bin=64, eta=.1, min_child_weight=5.,
                      reg_lambda=1., subsample=.9, colsample_bytree=1., colsample_bynode=1.,
                      nthread=threads, seed=seed, verbosity=0)
    train = xgb.DMatrix(X, label=y, feature_names=names, nthread=threads)
    evaluate = xgb.DMatrix(V, feature_names=names, nthread=threads)
    model = xgb.train(parameters, train, num_boost_round=rounds, callbacks=[Progress()])
    if model.num_boosted_rounds() != rounds:
        raise ValueError('Unequal round budget; refusing comparable result')
    metrics, fit_metrics = {}, {}
    for count in checkpoints:
        check_budget()
        metrics[str(count)] = measure(truth, model.predict(evaluate, iteration_range=(0, count)), classes, benign_index)
        check_budget()
        fit_metrics[str(count)] = measure(y, model.predict(train, iteration_range=(0, count)), classes, benign_index)
    check_budget()
    probabilities = model.predict(evaluate)
    check_budget()
    model.save_model(out/'model.ubj')
    check_budget()
    replay = xgb.Booster(params={'device': 'cpu', 'nthread': threads})
    replay.load_model(out/'model.ubj')
    np.testing.assert_allclose(replay.predict(evaluate), probabilities, atol=1e-7, rtol=0)
    check_budget()
    np.save(out/'probabilities.npy', probabilities, allow_pickle=False)
    np.save(out/'predictions.npy', probabilities.argmax(1), allow_pickle=False)
    result = dict(status='completed', seed=seed, rounds=rounds, parameters=parameters,
                  feature_names=names, fit_rows=len(y), evaluation_rows=len(truth),
                  cpu_reload_verified=True, metrics_by_round=metrics,
                  fit_metrics_by_round=fit_metrics, elapsed_seconds=time.time()-started,
                  artifact_hashes={p.name: sha256(p) for p in out.iterdir()
                                   if p.suffix in ('.ubj', '.npy')})
    check_budget()
    write_json(out/'result.json', result)
    return result


def completed_results(study, protocol):
    found = {}
    for job in protocol['jobs']:
        key = f'{job["arm"]}_s{job["seed"]}'
        for path in sorted((study/'runs'/key).glob('attempt_*/result.json')):
            result = read_json(path)
            if result.get('status') != 'completed':
                continue
            request_path = path.parent/'request.json'
            receipt_path = path.parent/'receipt.json'
            if not request_path.is_file():
                raise ValueError('Completed result lacks protocol binding request')
            request = read_json(request_path)
            if request != dict(job, protocol_sha256=sha256(study/'protocol.json')):
                raise ValueError('Completed result binding changed')
            if not receipt_path.is_file():
                continue  # Process died before sealing; retry in a new attempt.
            receipt = read_json(receipt_path)
            if receipt != dict(result_sha256=sha256(path), request_sha256=sha256(request_path)):
                raise ValueError('Completed result binding changed')
            if set(result.get('artifact_hashes', {})) != {'model.ubj', 'predictions.npy', 'probabilities.npy'}:
                raise ValueError('Missing required completed result artifacts')
            verify_pins(result['artifact_hashes'], path.parent)
            if result['rounds'] != protocol['rounds'] or result['seed'] != job['seed']:
                raise ValueError('Saved result does not match pinned job')
            expected = [protocol['feature_names'][i] for i in protocol['arms'][job['arm']]]
            if result['feature_names'] != expected:
                raise ValueError('Saved result schema mismatch')
            found[key] = dict(job, result=path.relative_to(study).as_posix(),
                             metrics=result['metrics_by_round'][str(protocol['rounds'])])
    return found


def write_report(study, protocol, results):
    lines = ['# Existing-model representation pilot', '',
             f'{len(results)} of {len(protocol["jobs"])} fixed-budget fits completed.', '',
             'Exploratory internal training-fold results; not original validation or live accuracy.', '',
             '| Arm | Seed | Accuracy | Declared macro F1 | Benign false alarms |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in results.values():
        m = row['metrics']
        fpr = 'unsupported' if m['false_positive_rate'] is None else f'{100*m["false_positive_rate"]:.3f}%'
        lines.append(f'| {row["arm"]} | {row["seed"]} | {100*m["accuracy"]:.3f}% | '
                     f'{100*m["macro_f1_declared"]:.3f}% | {fpr} |')
    lines += ['', '## Interpretation limits', '', *['- '+text for text in protocol['caveats']], '']
    (study/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(study/'results.json', list(results.values()))


def run(study=STUDY):
    import psutil
    study = Path(study)
    with StudyLock(study/'controller.lock'):
        protocol = read_json(study/'protocol.json')
        verify(protocol, study)
        old = read_json(study/'state.json')
        started = old.get('started_at', time.time())
        deadline = started + protocol['wall_seconds']
        process = psutil.Process()
        if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS'):
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        state = dict(started_at=started, deadline=deadline, pid=process.pid,
                     process_created=process.create_time(), planned=len(protocol['jobs']))

        def update(status, **extra):
            state.update(status=status, heartbeat=time.time(), **extra)
            write_json(study/'state.json', state)

        try:
            results = completed_results(study, protocol)
            for job in protocol['jobs']:
                key = f'{job["arm"]}_s{job["seed"]}'
                if key in results:
                    continue
                while psutil.virtual_memory().available < .9*1024**3:
                    if (study/'STOP').exists() or time.time() >= deadline:
                        raise InterruptedError('Stopped while waiting for memory')
                    update('waiting_for_memory', completed=len(results), current=job)
                    time.sleep(10)
                if (study/'STOP').exists() or time.time() >= deadline:
                    raise InterruptedError('STOP requested or study budget expired')
                verify_pins(protocol['source_hashes'], ROOT)
                verify_pins(protocol['prepared_hashes'], study)
                folder = study/'runs'/key
                folder.mkdir(parents=True, exist_ok=True)
                attempt = folder/f'attempt_{len(list(folder.glob("attempt_*"))):04d}'
                attempt.mkdir()
                write_json(attempt/'request.json', dict(job, protocol_sha256=sha256(study/'protocol.json')))
                update('running', completed=len(results), current=job, attempt=attempt.relative_to(study).as_posix())
                columns = protocol['arms'][job['arm']]
                A = np.load(study/'prepared/X_fit.npy', mmap_mode='r', allow_pickle=False)[:, columns]
                B = np.load(study/'prepared/X_evaluate.npy', mmap_mode='r', allow_pickle=False)[:, columns]
                train_one(attempt, A, np.load(study/'prepared/y_fit.npy', allow_pickle=False),
                          B, np.load(study/'prepared/y_evaluate.npy', allow_pickle=False),
                          [protocol['feature_names'][i] for i in columns], protocol['classes'],
                          protocol['benign_index'], seed=job['seed'], rounds=protocol['rounds'],
                          checkpoints=protocol['checkpoints'], threads=protocol['threads'],
                          stop_path=study/'STOP', deadline=min(deadline, time.time()+protocol['per_fit_seconds']),
                          heartbeat=lambda rounds, elapsed: update('running', rounds=rounds, fit_seconds=elapsed))
                write_json(attempt/'receipt.json', dict(result_sha256=sha256(attempt/'result.json'),
                                                        request_sha256=sha256(attempt/'request.json')))
                del A, B
                gc.collect()
                results = completed_results(study, protocol)
                write_report(study, protocol, results)
            verify(protocol, study)
            update('completed', completed=len(results), current=None)
            write_json(study/'completion.json', dict(state, source_validation_loaded=False,
                                                    held_out_test_loaded=False, auto_promote=False))
        except InterruptedError as exc:
            update('stopped', reason=str(exc))
        except Exception as exc:
            update('failed', error=f'{type(exc).__name__}: {exc}')
            raise

