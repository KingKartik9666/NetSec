"""Fresh reversed-split fits; original training rows are evaluation only.

This explicitly retires the original held-out test when the deferred gate opens.
It is a diagnostic protocol, never part of forward hyperparameter selection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from experiment.optimize_models import ROOT, digest, write


def retire_test(ledger: Path) -> None:
    """Record retirement before accessing the former held-out test."""
    if not ledger.exists():
        write(ledger, {'held_out_test_retired': True, 'retired_at': time.time(),
            'authorization': 'User requested original-test training after forward completion',
            'purpose': 'reversed-split diagnostic; no untouched-test claim'})


def load_partitions(prepared: Path, *, ready: bool, ledger: Path) -> dict:
    """Fit=test, stop=validation, evaluation=train; never exchange array files."""
    if not ready:
        raise PermissionError('Current forward study must finish before test access')
    retire_test(ledger)
    from framework.training.tracking import validate_manifest
    manifest = json.loads((prepared / 'manifest.json').read_text())
    validate_manifest(manifest)
    result = {'manifest': manifest}
    for role, split in (('fit', 'test'), ('stopping', 'validation'), ('evaluation', 'train')):
        X, y, ids = [np.load(prepared / f'{kind}_{split}.npy', mmap_mode='r') for kind in ('X', 'y', 'row_ids')]
        if X.shape != (len(y), len(manifest['feature_names'])) or y.ndim != 1 or not len(y):
            raise ValueError(f'Invalid {role} dimensions')
        if ids.ndim != 1 or len(ids) != len(y) or len(np.unique(ids)) != len(ids):
            raise ValueError(f'Invalid {role} row identities')
        if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= len(manifest['classes']):
            raise ValueError(f'Invalid {role} real labels')
        result[role] = {'X': X, 'y': y, 'ids': ids, 'original_partition': split}
    for a, b in (('fit', 'stopping'), ('fit', 'evaluation'), ('stopping', 'evaluation')):
        if np.intersect1d(result[a]['ids'], result[b]['ids']).size:
            raise ValueError('Reversed partitions overlap')
    return result


def fit_rf(partitions: dict, directory: Path) -> dict:
    """Keep conventional RF settings fixed; map absent training classes honestly."""
    from framework.training.fixed_random_forest import FIXED_PARAMETERS, train_fixed_random_forest
    training, stopping = partitions['fit'], partitions['stopping']
    observed = np.unique(training['y'])
    eligible = np.isin(stopping['y'], observed)
    if len(observed) < 2 or not eligible.any():
        raise ValueError('Fixed RF reverse fit lacks two training classes or usable stopping rows')
    # RF has no early stopping. This slice only permits its unchanged strict
    # compatibility check; final metrics retain every evaluation class/row.
    trained = train_fixed_random_forest(training['X'], np.searchsorted(observed, training['y']),
        stopping['X'][eligible], np.searchsorted(observed, stopping['y'][eligible]),
        int(len(observed)), dict(FIXED_PARAMETERS), 42, directory)
    trained.pop('predictions'); trained.pop('probabilities')
    trained.update(global_class_ids=observed.tolist(), n_declared_classes=len(partitions['manifest']['classes']))
    write(directory / 'class_alignment.json', {'global_class_ids': observed.tolist(),
           'n_declared_classes': trained['n_declared_classes'], 'unseen_class_probability': 0})
    return trained


def predict_rf(trained: dict, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import joblib
    checkpoint = Path(trained['checkpoint_path'])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    local = joblib.load(checkpoint).predict_proba(raw)
    probability = np.zeros((len(raw), trained['n_declared_classes']), dtype=local.dtype)
    probability[:, trained['global_class_ids']] = local
    return probability.argmax(1), probability


def score_partition(trained: dict, family: str, partition: dict, manifest: dict) -> tuple[dict, np.ndarray]:
    """Bounded CPU batches; evaluation labels cannot select a checkpoint."""
    from experiment.train_csv_research import reload_research
    from framework.training.tracking import classification_metrics
    chunks, top_three, has_probability = [], 0, True
    batch = 512 if family in ('kan', 'hem') else 8192
    for start in range(0, len(partition['y']), batch):
        if time.time() >= float(os.environ.get('IMDAI_FIT_DEADLINE', 'inf')) - 15:
            raise TimeoutError('Reverse evaluation exhausted fit watchdog; no partial accuracy reported')
        raw = partition['X'][start:start+batch]
        predicted, probability = (predict_rf(trained, raw) if family == 'random_forest' else
            reload_research({'model_id': family, 'checkpoint_path': trained['checkpoint_path']}, raw))
        chunks.append(predicted)
        if probability is None:
            has_probability = False
        else:
            ranks = np.argsort(-probability, axis=1)[:, :3]
            top_three += int((ranks == partition['y'][start:start+batch, None]).any(1).sum())
    predicted = np.concatenate(chunks)
    metrics = classification_metrics(partition['y'], predicted, manifest['classes'], manifest['benign_index'])
    metrics['top_three_accuracy'] = top_three / len(predicted) if has_probability else None
    return metrics, predicted


def fit(path: Path) -> None:
    from experiment.reverse_csv_study import forward_ready, FORWARD
    from experiment.train_csv_research import fit_base, fit_ensemble, oof_temperature
    request = json.loads(path.read_text())
    if not request.get('fresh_initialization') or request['model_id'] not in ('random_forest', 'xgboost', 'kan', 'hem'):
        raise ValueError('Reverse fit requires a fresh recognized family')
    if request['roles'] != {'fit': 'test', 'stopping': 'validation', 'evaluation': 'train'}:
        raise ValueError('Reversed dataset roles changed')
    if request['model_id'] == 'random_forest':
        from framework.training.fixed_random_forest import FIXED_PARAMETERS
        if request['config'] != FIXED_PARAMETERS or request['seed'] != 42:
            raise ValueError('Reversed fixed RF cannot tune its settings')
    state, report = [json.loads((FORWARD / name).read_text()) for name in ('state.json', 'report.json')]
    ready = forward_ready(state, report) and not (FORWARD / 'STOP').exists()
    if not ready:
        raise PermissionError('Forward completion gate closed')
    prepared = ROOT / request['prepared']
    directory = path.parent
    ledger = directory.parents[2] / 'test_access.json'
    retire_test(ledger)
    for name, expected in request['dataset_hashes'].items():
        if digest(prepared / name) != expected:
            raise ValueError('Frozen reversed inputs changed')
    partitions = load_partitions(prepared, ready=ready, ledger=ledger)
    manifest, training, stopping = partitions['manifest'], partitions['fit'], partitions['stopping']
    family = request['model_id']
    print(f'REVERSED DIAGNOSTIC fresh {family}: train original-test={len(training["y"])}; stop original-validation; evaluate original-train', flush=True)
    calibration = None
    if family == 'random_forest':
        trained = fit_rf(partitions, directory)
    else:
        pipeline = request.get('pipeline', {})
        if pipeline.get('calibration', 'none') != 'none':
            calibration = oof_temperature(request, manifest, training['X'], training['y'], training['ids'], directory)
        trainer = fit_ensemble if pipeline.get('ensemble', 'none') != 'none' else fit_base
        trained = trainer(request, manifest, training['X'], training['y'], training['ids'],
                          stopping['X'], stopping['y'], directory)
        trained.pop('predictions'); trained.pop('probabilities', None)
        if calibration:
            write(directory / 'calibration.json', calibration)
    stopping_metrics, _ = score_partition(trained, family, stopping, manifest)
    metrics, predicted = score_partition(trained, family, partitions['evaluation'], manifest)
    np.save(directory / 'original_train_predictions.npy', predicted)
    checkpoint = Path(trained['checkpoint_path'])
    result = {**request, **trained, 'status': 'completed', 'completed_at': time.time(),
        'checkpoint_path': checkpoint.relative_to(ROOT).as_posix(), 'checkpoint_sha256': digest(checkpoint),
        'evaluation_metrics': metrics, 'stopping_metrics': stopping_metrics, 'calibration': calibration,
        'evaluation_rows': len(predicted), 'training_rows': len(training['y']),
        'classes_without_reverse_training_support': [manifest['classes'][i] for i in range(len(manifest['classes']))
                                                    if i not in np.unique(training['y'])],
        'held_out_test_retired': True, 'original_test_used_for_training': True,
        'evaluation_used_for_early_stopping': False, 'previous_weights_reused': False,
        'scope': 'Reversed-split diagnostic; fixed schema/hyperparameters informed by prior work; not untouched-test performance'}
    assets = {str(p.relative_to(directory)): digest(p) for p in directory.rglob('*')
              if p.is_file() and p.name not in ('result.json', 'bundle_integrity.json', 'progress.json', 'fit.log')}
    write(directory / 'bundle_integrity.json', assets)
    write(directory / 'result.json', result)
    print(json.dumps({'model': family, 'reverse_evaluation_accuracy': metrics['accuracy']}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    fit(parser.parse_args().fit)


if __name__ == '__main__':
    main()
