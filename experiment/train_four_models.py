"""Fresh, bounded four-family fits on identical CIC-IIoT-2025 observations.

Original artifacts are read-only. The only selectable forest in this extension
is a fixed conventional RF. Neural/boosted fits require CUDA, with no fallback.
The final test is neither loaded nor scored by this controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/training/four_models_v2'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_common_partitions(numeric: Path, protocol: Path) -> dict:
    """Check train/validation identities without touching the held-out test."""
    result = {'test_loaded': False, 'partitions': {}}
    for split in ['train', 'validation']:
        values = {}
        for kind in ['row_ids', 'y']:
            left = np.load(numeric / f'{kind}_{split}.npy', mmap_mode='r')
            right = np.load(protocol / f'{kind}_{split}.npy', mmap_mode='r')
            if not np.array_equal(left, right):
                raise ValueError(f'{split} {kind} row/label identity differs across representations')
            values[kind + '_sha256'] = sha256(numeric / f'{kind}_{split}.npy')
            values['rows'] = len(left)
        result['partitions'][split] = values
    train_ids = np.load(numeric / 'row_ids_train.npy', mmap_mode='r')
    valid_ids = np.load(numeric / 'row_ids_validation.npy', mmap_mode='r')
    if len(np.unique(train_ids)) != len(train_ids) or len(np.unique(valid_ids)) != len(valid_ids):
        raise ValueError('Duplicate row IDs within a partition')
    if np.intersect1d(train_ids, valid_ids).size:
        raise ValueError('Training/validation row overlap')
    return result


def study_configurations(originals: dict) -> list[dict]:
    """Independent one-parameter ablations; RF never has an enhanced trial."""
    trials = []
    for family in ['xgboost', 'kan', 'hem']:
        control = dict(originals[family])
        if family == 'xgboost':
            if 'warm_start' not in control:
                control['n_estimators'] = min(int(control.get('n_estimators', 500)), 500)
                control['verbose_eval'] = 50
            control.setdefault('reg_lambda', 1.0)
            change = {'reg_lambda': float(control['reg_lambda']) * 2}
        elif family == 'kan':
            control.setdefault('spline_regularization', 0.0)
            change = {'spline_regularization': 0.0001}
        else:
            control.setdefault('weight_decay', 0.0001)
            change = {'weight_decay': 0.001}
        trials += [{'model_id': family, 'variant': 'control', 'config': control},
                   {'model_id': family, 'variant': 'enhanced', 'config': {**control, **change}}]
    trials.append({'model_id': 'random_forest', 'variant': 'fixed_baseline',
                   'config': dict(originals['random_forest'])})
    return trials


def fit(request_path: Path) -> None:
    from framework.training.tracking import classification_metrics, write_json
    request = json.loads(request_path.read_text(encoding='utf-8'))
    family = request['model_id']
    prepared = ROOT / request['prepared']
    output = request_path.parent
    manifest = json.loads((prepared / 'manifest.json').read_text())
    if sha256(prepared / 'manifest.json') != request['prepared_manifest_sha256']:
        raise ValueError('Prepared manifest changed after study definition')
    arrays = [np.load(prepared / f'{kind}_{split}.npy', mmap_mode='r')
              for split in ['train', 'validation'] for kind in ['X', 'y']]
    X_train, y_train, X_val, y_val = arrays
    started = time.perf_counter()
    if family == 'random_forest':
        from framework.training.fixed_random_forest import train_fixed_random_forest
        trained = train_fixed_random_forest(X_train, y_train, X_val, y_val,
                                           len(manifest['classes']), request['config'], request['seed'], output)
    elif family == 'xgboost':
        # Reuse current depth-six masked-input architecture and its numerically
        # verified CUDA objective, keeping the original study untouched.
        from experiment.train_partial_xgboost import load_data, trial
        _, _, _, source_profiles = load_data()
        partial_request = {**request, 'source_profiles': source_profiles}
        partial_request_path = output / 'partial_request.json'
        write_json(partial_request_path, partial_request)
        if trial(partial_request_path):
            raise RuntimeError('Current-architecture XGBoost fit failed; see preserved result and log')
        trained = json.loads((output / 'result.json').read_text())
        trained['checkpoint_path'] = str(ROOT / trained['checkpoint'])
        probabilities = np.load(output / 'full_validation_probabilities.npy')
        trained['probabilities'] = probabilities
        trained['predictions'] = probabilities.argmax(1)
    else:
        from framework.training.neural import train_neural
        trained = train_neural(family, X_train, y_train, X_val, y_val,
                               len(manifest['classes']), manifest['benign_index'],
                               request['config'], request['seed'], output)
    predicted = trained.pop('predictions')
    probabilities = trained.pop('probabilities', None)
    np.save(output / 'validation_predictions.npy', predicted)
    metrics = classification_metrics(y_val, predicted, manifest['classes'], manifest['benign_index'])
    if probabilities is not None:
        np.save(output / 'validation_probabilities.npy', probabilities)
        metrics['top_three_accuracy'] = float(np.mean(np.any(np.argsort(-probabilities, axis=1)[:, :3] == y_val[:, None], axis=1)))
    else:
        metrics['top_three_accuracy'] = None
    result = {**request, **trained, 'status': 'completed', 'metrics': metrics,
              'elapsed_seconds': time.perf_counter() - started, 'training_rows': len(y_train),
              'validation_rows': len(y_val), 'feature_names': manifest['feature_names'],
              'classes': manifest['classes'], 'benign_index': manifest['benign_index'],
              'test_loaded': False, 'checkpoint_sha256': sha256(Path(trained['checkpoint_path']))}
    result['checkpoint_path'] = str(Path(result['checkpoint_path']).relative_to(ROOT)).replace('\\', '/')
    result['artifact_bytes'] = sum(p.stat().st_size for p in output.iterdir() if p.suffix in ['.pt', '.ubj', '.joblib', '.npz'])
    write_json(output / 'result.json', result)
    print(json.dumps({'model_id': family, 'variant': request['variant'], 'accuracy': metrics['accuracy'],
                      'macro_f1': metrics['macro_f1'], 'seconds': result['elapsed_seconds']}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path)
    parser.add_argument('--models', nargs='+', choices=['random_forest', 'xgboost', 'kan', 'hem'], default=['xgboost', 'kan', 'hem', 'random_forest'])
    parser.add_argument('--per-fit-seconds', type=int, default=1200)
    args = parser.parse_args()
    if args.fit:
        fit(args.fit)
        return
    from framework.training.tracking import validate_manifest, write_json
    numeric = ROOT / 'dataset/prepared/datasense_full_train'
    protocol = ROOT / 'dataset/prepared/datasense_full_protocols'
    identity = verify_common_partitions(numeric, protocol)
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in [numeric, protocol]]
    for manifest in manifests:
        validate_manifest(manifest)
    if manifests[0]['classes'] != manifests[1]['classes'] or manifests[0]['source_sha256'] != manifests[1]['source_sha256']:
        raise ValueError('Source or class order differs across prepared representations')
    if manifests[1]['feature_names'][:len(manifests[0]['feature_names'])] != manifests[0]['feature_names']:
        raise ValueError('Numeric feature order differs')
    originals = {}
    for family in ['kan', 'hem']:
        record = json.loads((ROOT / f'artifacts/training/datasense_all_models_full_v1/selected/{family}.json').read_text())['seed_42_record']
        originals[family] = record['config']
    from experiment.train_partial_xgboost import configurations
    originals['xgboost'] = next(x for x in configurations() if x['id'] == 'fresh_d6_smooth')
    originals['random_forest'] = {'n_estimators': 100, 'criterion': 'gini', 'bootstrap': True,
                                  'max_features': 'sqrt', 'max_depth': None, 'n_jobs': 2}
    configs = study_configurations(originals)
    protocol_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    definition = {'dataset': 'local DataSense CSV; official dataset family CIC-IIoT-2025',
                  'source_sha256': manifests[0]['source_sha256'], 'protocol_id': protocol_id,
                  'identity': identity, 'configurations': configs, 'seed': 42,
                  'per_fit_seconds': args.per_fit_seconds, 'max_fits': 7,
                  'test_loaded': False, 'selection': 'validation macro F1; control wins ties; RF not tuned'}
    STUDY.mkdir(parents=True, exist_ok=True)
    definition_path = STUDY / 'study.json'
    if definition_path.exists() and json.loads(definition_path.read_text()) != definition:
        raise ValueError('Study definition differs; create a separately versioned study')
    write_json(definition_path, definition)
    for trial in configs:
        if trial['model_id'] not in args.models:
            continue
        prepared = protocol if trial['model_id'] == 'xgboost' else numeric
        directory = STUDY / 'runs' / (trial['model_id'] + '_' + trial['variant'] + '_s42')
        directory.mkdir(parents=True, exist_ok=True)
        result_file = directory / 'result.json'
        if result_file.exists():
            result = json.loads(result_file.read_text())
            if result['status'] == 'completed' and sha256(ROOT / result['checkpoint_path']) == result['checkpoint_sha256']:
                print(f"Verified existing completed trial: {directory.name}", flush=True)
                continue
            raise ValueError('Existing result/checkpoint failed integrity; preserve and investigate')
        request = {**trial, 'seed': 42, 'prepared': prepared.relative_to(ROOT).as_posix(),
                   'prepared_manifest_sha256': sha256(prepared / 'manifest.json'), 'protocol_id': protocol_id}
        request_file = directory / 'request.json'
        if request_file.exists() and json.loads(request_file.read_text()) != request:
            raise ValueError('Existing request differs; no overwrite permitted')
        write_json(request_file, request)
        print(f"Training {directory.name} on ALL {identity['partitions']['train']['rows']} eligible rows", flush=True)
        with (directory / 'fit.log').open('a', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'experiment.train_four_models', '--fit', str(request_file)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                exit_code = child.wait(timeout=args.per_fit_seconds)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                write_json(directory / 'failure.json', {'status': 'failed', 'reason': 'documented per-fit runtime budget exceeded'})
                continue
        if exit_code:
            write_json(directory / 'failure.json', {'status': 'failed', 'exit_code': exit_code, 'reason': 'Inspect fit.log; no substitute result'})
        else:
            result = json.loads(result_file.read_text())
            print(f"Completed {directory.name}: accuracy={result['metrics']['accuracy']:.6f}, macroF1={result['metrics']['macro_f1']:.6f}", flush=True)
    print('Study controller finished; inspect all result/failure files before selecting.', flush=True)


if __name__ == '__main__':
    main()
