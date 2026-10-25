"""Resumable bounded follow-up, independent of the existing CUDA controller."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil

from experiment.xgboost_followup import (ARMS, BASE, DEFAULT, ROOT, SEEDS, SOURCE,
    TUNING, arrays, digest, eligible, feature_indexes, matrix, read, recipe)
from framework.training.tracking import classification_metrics, write_json


def memory_available():
    if os.name == 'nt':
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ('total_physical', 'physical',
                'total_commit', 'commit', 'total_virtual', 'virtual', 'extended')]
        status = Status()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('Cannot read Windows memory availability')
        return dict(physical_gib=status.physical / 2**30, commit_gib=status.commit / 2**30)
    return dict(physical_gib=psutil.virtual_memory().available / 2**30,
                commit_gib=psutil.virtual_memory().available / 2**30)


@contextmanager
def single_runner(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.runner.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == 'nt':
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def verify_result(path):
    result = read(path / 'result.json')
    if result['status'] != 'completed' or not result['checkpoint_reload_verified']:
        raise ValueError(f'Incomplete result {path}')
    for name, expected in result['hashes'].items():
        if digest(path / name) != expected:
            raise ValueError(f'Changed result artifact: {path / name}')
    return result


def initialize(directory, source):
    prepared = source / 'prepared'
    manifest = read(prepared / 'manifest.json')
    # This whitelist cannot include locked-test arrays.
    inputs = [prepared / name for name in ('manifest.json', 'feature_transform.json', 'training_keep.npy')]
    inputs += [prepared / f'{key}_{split}.npy' for split in ('train', 'validation')
               for key in ('X', 'y', 'extra', 'row_ids', 'window_start', 'window_end', 'feature_hash')]
    hashes = {p.relative_to(source).as_posix(): digest(p) for p in inputs}
    for name, expected in manifest['input_hashes'].items():
        if name.endswith(('_train.npy', '_validation.npy')) and hashes[f'prepared/{name}'] != expected:
            raise ValueError(f'Original prepared input changed: {name}')
    code = [ROOT / name for name in ('experiment/xgboost_followup.py',
        'experiment/run_xgboost_followup.py', 'experiment/report_xgboost_followup.py',
        'experiment/audit_xgboost_data.py', 'framework/training/tracking.py',
        'docs/XGBOOST_FOLLOWUP_20261013.md')]
    protocol = dict(source=source.relative_to(ROOT).as_posix(), input_hashes=hashes,
                    source_hashes={p.relative_to(ROOT).as_posix(): digest(p) for p in code},
                    seeds=list(SEEDS), arms=list(ARMS), tuning=TUNING,
                    representation_cells=15, reusable_cells=3,
                    new_fits_minimum=18, new_fits_maximum=20,
                    test_loaded=False, device='cpu', threads=4)
    path = directory / 'protocol.json'
    if path.exists():
        if read(path) != protocol:
            raise ValueError('Pinned protocol, source code or input bytes changed; use a new study')
    else:
        write_json(path, protocol)
        snapshots = directory / 'source_snapshot'
        snapshots.mkdir()
        for p in code:
            (snapshots / p.name).write_bytes(p.read_bytes())
    return manifest


def adopt(source, output, arm):
    """Verify the old seed-42 result and replay its whole validation checkpoint."""
    import xgboost as xgb
    old = source / 'runs' / arm
    request, result = read(old / 'request.json'), read(old / 'result.json')
    manifest = read(source / 'prepared/manifest.json')
    indexes, names = feature_indexes(arm, manifest, read(source / 'prepared/feature_transform.json'))
    expected = dict(BASE, objective='multi:softprob', num_class=len(manifest['classes']),
                    seed=42, base_score=.5, device='cpu', tree_method='hist', nthread=4,
                    disable_default_eval_metric=True)
    if request['parameters'] != expected or request['rounds'] != 79 or request['smoothing'] != .05:
        raise ValueError('Historical arm is not a matched control')
    if request['input_manifest_sha256'] != digest(source / 'prepared/manifest.json'):
        raise ValueError('Historical input identity changed')
    if request['transform_sha256'] != digest(source / 'prepared/feature_transform.json'):
        raise ValueError('Historical transform changed')
    if result['feature_names'] != names or result['checkpoint_sha256'] != digest(old / 'model.ubj'):
        raise ValueError('Historical model identity changed')
    data = arrays(source)['validation']
    truth = np.asarray(data['y'])
    probabilities = np.load(old / 'validation_probabilities.npy')
    predictions = np.load(old / 'validation_predictions.npy')
    np.testing.assert_array_equal(probabilities.argmax(1), predictions)
    measured = classification_metrics(truth, predictions, manifest['classes'], manifest['benign_index'])
    for key in ('accuracy', 'macro_recall', 'macro_f1', 'false_positive_rate'):
        if abs(measured[key] - result['metrics'][key]) > 1e-12:
            raise ValueError('Historical saved metrics differ from predictions')
    model = xgb.Booster(model_file=str(old / 'model.ubj'))
    model.set_param({'device': 'cpu', 'nthread': 4})
    validation = xgb.DMatrix(matrix(data, np.arange(len(truth)), indexes), nthread=4)
    replay = model.predict(validation)
    np.testing.assert_allclose(replay, probabilities, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(replay.argmax(1), predictions)
    output.mkdir(parents=True, exist_ok=True)
    paths = {key: (old / name).relative_to(ROOT).as_posix() for key, name in
             [('model', 'model.ubj'), ('probabilities', 'validation_probabilities.npy'),
              ('predictions', 'validation_predictions.npy'), ('result', 'result.json'),
              ('request', 'request.json')]}
    return dict(recipe=recipe(arm, 42), metrics=measured, feature_names=names,
                feature_indexes=indexes.tolist(), paths=paths, adopted=True,
                checkpoint_reload_verified=True, source_result_sha256=digest(old / 'result.json'),
                artifact_hashes={key: digest(ROOT / value) for key, value in paths.items()},
                old_elapsed_seconds=result['elapsed_seconds'])


def record_result(output, request):
    result = verify_result(output)
    if result['recipe'] != request:
        raise ValueError('Completed fit has a different request')
    paths = {key: (output / name).relative_to(ROOT).as_posix() for key, name in
             [('model', 'model.ubj'), ('probabilities', 'probabilities.npy'),
              ('predictions', 'predictions.npy'), ('truth', 'truth.npy'),
              ('scoring_ids', 'scoring_row_ids.npy'), ('fit_ids', 'fit_row_ids.npy'),
              ('result', 'result.json'), ('request', 'request.json')]}
    return dict(recipe=request, metrics=result['metrics'], paths=paths, adopted=False,
                feature_names=result['feature_names'], feature_indexes=result['feature_indexes'],
                checkpoint_reload_verified=True, elapsed_seconds=result['elapsed_seconds'],
                source_result_sha256=digest(output / 'result.json'),
                artifact_hashes={key: digest(ROOT / value) for key, value in paths.items()})


class Study:
    def __init__(self, directory, source):
        self.directory, self.source = directory, source
        self.state = dict(status='starting', pid=os.getpid(), started_at=time.time(),
                          completed=[], active=None, test_loaded=False)
        self.records = {}

    def update(self, **values):
        self.state.update(values, updated_at=time.time())
        self.state['completed_count'] = len(self.records)
        self.state['new_fit_count'] = sum(not r['adopted'] for r in self.records.values())
        write_json(self.directory / 'state.json', self.state)

    def record(self, name, value):
        self.records[name] = value
        write_json(self.directory / 'records' / f'{name}.json', value)
        self.state['completed'] = list(self.records)
        self.update()
        return value

    def existing(self, name):
        path = self.directory / 'records' / f'{name}.json'
        if not path.exists():
            return None
        value = read(path)
        for key, expected in value['artifact_hashes'].items():
            if digest(ROOT / value['paths'][key]) != expected:
                raise ValueError(f'Recorded artifact changed: {name}/{key}')
        self.records[name] = value
        self.state['completed'] = list(self.records)
        self.update()
        return value

    def resources(self):
        while True:
            available = memory_available()
            if available['commit_gib'] >= 5. and available['physical_gib'] >= .5:
                return
            self.update(status='waiting_for_memory', memory=available)
            if (self.directory / 'STOP').exists():
                raise InterruptedError('Follow-up stop requested')
            time.sleep(20)

    def trial(self, name, request):
        previous = self.existing(name)
        if previous:
            if previous['recipe'] != request:
                raise ValueError('Recorded request differs')
            return previous
        parent = self.directory / 'runs' / name
        for complete in sorted(parent.glob('attempt_*/result.json')):
            return self.record(name, record_result(complete.parent, request))
        # Do not duplicate a surviving worker after a controller interruption.
        for process in psutil.process_iter(['pid', 'cmdline']):
            command = ' '.join(process.info['cmdline'] or [])
            if 'experiment.xgboost_followup' in command and str(parent) in command:
                self.update(status='waiting_for_existing_worker', active=name, worker_pid=process.pid)
                while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    time.sleep(20)
                return self.trial(name, request)
        self.resources()
        parent.mkdir(parents=True, exist_ok=True)
        attempt = parent / f'attempt_{len(list(parent.glob("attempt_*"))):03d}'
        pending = parent / f'{attempt.name}_request.json'
        write_json(pending, request)
        log_path = parent / f'{attempt.name}.log'
        began = time.monotonic()
        with log_path.open('w', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'experiment.xgboost_followup',
                '--source', str(self.source), '--output', str(attempt), '--request', str(pending)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            reason = None
            while child.poll() is None:
                available = memory_available()
                progress_path = attempt / 'progress.json'
                self.update(status='running', active=name, worker_pid=child.pid,
                            active_output=attempt.relative_to(ROOT).as_posix(), memory=available,
                            progress=read(progress_path) if progress_path.exists() else None)
                if (self.directory / 'STOP').exists():
                    reason = 'Explicit follow-up stop'
                elif available['commit_gib'] < .75:
                    reason = 'Available Windows commit below 0.75 GiB'
                elif time.monotonic()-began > 2700:
                    reason = '45-minute per-fit watchdog'
                if reason:
                    process = psutil.Process(child.pid)
                    children = process.children(recursive=True)
                    for descendant in children:
                        try: descendant.terminate()
                        except psutil.NoSuchProcess: pass
                    if child.poll() is None:
                        child.terminate()
                    child.wait(timeout=30)
                    break
                time.sleep(5)
        if reason or child.returncode:
            write_json(parent / f'{attempt.name}_failure.json',
                       dict(reason=reason, returncode=child.returncode, log=log_path.relative_to(ROOT).as_posix()))
            raise RuntimeError(f'{name} failed: {reason or child.returncode}; see {log_path}')
        return self.record(name, record_result(attempt, request))

    def run(self):
        initialize(self.directory, self.source)
        self.update(stage='replication_and_ablation', representation_total=15,
                    total_new_fits_range=[18, 20])
        for arm in ('original_control', 'cleaned', 'cleaned_enriched'):
            name = f'{arm}_s42_base'
            if not self.existing(name):
                self.resources()
                self.record(name, adopt(self.source, self.directory / 'runs' / name / 'adopted', arm))
        order = [(arm, seed) for seed in (43, 44) for arm in ('original_control', 'cleaned_enriched')]
        order += [(arm, seed) for arm in ('cleaned', 'ports', 'rates') for seed in SEEDS
                  if not (arm == 'cleaned' and seed == 42)]
        for arm, seed in order:
            self.trial(f'{arm}_s{seed}_base', recipe(arm, seed))
        from experiment.report_xgboost_followup import summarize_arms, finish
        summary, arm = summarize_arms(self.records)
        write_json(self.directory / 'representation_comparison.json', summary)
        self.update(stage='tuning', selected_arm=arm)
        base = [self.records[f'{arm}_s{s}_base'] for s in SEEDS]
        candidates = [('base', base[0])]
        for variant in TUNING:
            result = self.trial(f'{arm}_s42_{variant}', recipe(arm, 42, variant))
            if eligible(result['metrics'], base[0]['metrics']):
                candidates.append((variant, result))
        chosen, best = max(candidates, key=lambda pair: (pair[1]['metrics']['accuracy'],
                                                        pair[1]['metrics']['macro_f1']))
        selected = base
        if chosen != 'base' and best['metrics']['accuracy'] > base[0]['metrics']['accuracy']:
            self.update(stage='tuned_recipe_replication')
            repeated = [best] + [self.trial(f'{arm}_s{s}_{chosen}', recipe(arm, s, chosen)) for s in (43, 44)]
            if (all(eligible(a['metrics'], b['metrics']) for a, b in zip(repeated, base)) and
                np.mean([x['metrics']['accuracy'] for x in repeated]) > np.mean([x['metrics']['accuracy'] for x in base])):
                selected = repeated
        final_recipe = selected[0]['recipe']
        write_json(self.directory / 'selection.json', dict(recipe=final_recipe,
                   selected=selected, representation_comparison=summary,
                   rule='Mean accuracy with per-seed recall/FPR guards; no best-seed selection'))
        self.update(stage='training_only_calibration', selected_recipe=final_recipe)
        folds = [self.trial(f'calibration_fold_{fold}', dict(final_recipe, fold=fold)) for fold in range(3)]
        self.resources()
        finish(self.directory, self.source, self.records, selected, folds)
        # Re-check every frozen input and code file after all computation.
        initialize(self.directory, self.source)
        self.update(status='completed', stage='completed', active=None, worker_pid=None,
                    selected_recipe=final_recipe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--source', type=Path, default=SOURCE)
    args = parser.parse_args()
    directory, source = args.directory.resolve(), args.source.resolve()
    with single_runner(directory):
        study = Study(directory, source)
        try:
            study.run()
        except BaseException as exc:
            study.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()
