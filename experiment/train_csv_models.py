"""One isolated, full-CSV CUDA trial with CPU checkpoint verification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from experiment.optimize_models import ROOT, digest, seal_assets, write


def load_csv_data(prepared: Path):
    """Load all original train/validation observations, never masked views or test."""
    manifest = json.loads((prepared / 'manifest.json').read_text())
    arrays = [np.load(prepared / f'{kind}_{split}.npy', mmap_mode='r')
              for split in ('train', 'validation') for kind in ('X', 'y', 'row_ids')]
    X, y, ids, V, v, vids = arrays
    if X.shape != (len(y), len(manifest['feature_names'])) or V.shape != (len(v), X.shape[1]):
        raise ValueError('Prepared CSV dimensions differ from ordered schema')
    if len(ids) != len(y) or len(vids) != len(v) or np.intersect1d(ids, vids).size:
        raise ValueError('CSV train/validation row identities overlap or differ in length')
    if len(np.unique(ids)) != len(ids) or len(np.unique(vids)) != len(vids):
        raise ValueError('Duplicate CSV row IDs')
    return manifest, X, y, V, v, dict(original_training_rows=len(y), effective_training_rows=len(y),
               masked_training_rows=0, test_loaded=False,
               train_row_ids_sha256=digest(prepared / 'row_ids_train.npy'),
               validation_row_ids_sha256=digest(prepared / 'row_ids_validation.npy'))


def train_candidate(family, X, y, V, v, n_classes, benign_index, config, seed, directory):
    if family == 'xgboost':
        from framework.training.csv_xgboost import train_csv_xgboost
        return train_csv_xgboost(X, y, V, v, n_classes, config, seed, directory, benign_index=benign_index)
    if family not in {'kan', 'hem'}:
        raise ValueError('RF is fixed and excluded from CSV upgrades')
    from framework.training.neural import train_neural
    if config.get('selection_metric') != 'accuracy':
        raise ValueError('CSV neural checkpoints must select full validation accuracy')
    return train_neural(family, X, y, V, v, n_classes, benign_index, config, seed, directory)


def reload_outputs(family: str, checkpoint: Path, X: np.ndarray):
    """Explicit CPU-only inference; no CUDA fallback is used for reload checks."""
    if family == 'xgboost':
        import xgboost as xgb
        model = xgb.Booster(model_file=str(checkpoint))
        model.set_param({'device': 'cpu', 'nthread': 2})
        probabilities = model.predict(xgb.DMatrix(X, nthread=2))
        return probabilities.argmax(1), probabilities
    import joblib
    import torch
    from framework.training.neural import _models, _clean, cuda_knn_predict, hem_closed_set_vote
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['candidate'] != family:
        raise ValueError('Saved family differs from request')
    transformed = joblib.load(checkpoint.parent / 'preprocessing.joblib').transform(_clean(X)).astype(np.float32)
    models = _models(family, saved['input_size'], saved['n_classes'], saved['config'], torch, device='cpu')
    for name, model in models.items():
        model.load_state_dict(saved['state_dicts'][name])
        model.eval()
    with torch.no_grad():
        if family == 'kan':
            probabilities = np.concatenate([torch.softmax(models['kan'](torch.from_numpy(transformed[i:i + 512])), 1).numpy()
                                            for i in range(0, len(transformed), 512)])
            return probabilities.argmax(1), probabilities
        batch = torch.from_numpy(transformed)
        multi = models['multiclass'](batch).argmax(1).numpy()
        binary = (models['binary'](batch)[:, 0] >= 0).numpy()
    with np.load(checkpoint.parent / 'knn_references.npz') as refs:
        knn = cuda_knn_predict(transformed, refs['X'], refs['y'], saved['n_classes'],
                               saved['config']['knn_k'], device='cpu')
    return hem_closed_set_vote(multi, binary, knn, saved['benign_index']), None


def reload_predictions(family: str, checkpoint: Path, X: np.ndarray) -> np.ndarray:
    return reload_outputs(family, checkpoint, X)[0]


def fit(path: Path) -> None:
    from framework.training.tracking import classification_metrics
    request = json.loads(path.read_text())
    if request.get('input_profile') != 'full_csv':
        raise ValueError('This worker only accepts the full_csv study profile')
    prepared = ROOT / request['prepared']
    if digest(prepared / 'manifest.json') != request['prepared_manifest_sha256']:
        raise ValueError('Prepared manifest changed')
    manifest, X, y, V, v, provenance = load_csv_data(prepared)
    started = time.perf_counter()
    print(f"CSV FIT {request['model_id']} {request['variant']} rows={len(y)} columns={X.shape[1]} masked_rows=0", flush=True)
    trained = train_candidate(request['model_id'], X, y, V, v, len(manifest['classes']),
                              manifest['benign_index'], request['config'], request['seed'], path.parent)
    predictions = trained.pop('predictions')
    probabilities = trained.pop('probabilities', None)
    # KAN CPU inference is inexpensive enough to verify every validation row;
    # HEM exact nearest-neighbor CPU reload is deliberately bounded to 128 rows.
    count = min(128, len(v)) if request['model_id'] == 'hem' else len(v)
    reloaded, cpu_probabilities = reload_outputs(request['model_id'], Path(trained['checkpoint_path']), V[:count])
    if not np.array_equal(reloaded, predictions[:count]):
        raise RuntimeError('Saved CPU predictions differ from validation-selected predictions')
    if probabilities is None and request['model_id'] == 'kan':
        probabilities = cpu_probabilities
    metrics = classification_metrics(v, predictions, manifest['classes'], manifest['benign_index'])
    metrics['top_three_accuracy'] = (float(np.mean(np.any(np.argsort(-probabilities, axis=1)[:, :3] == v[:, None], axis=1)))
                                     if probabilities is not None else None)
    np.save(path.parent / 'validation_predictions.npy', predictions)
    if probabilities is not None:
        np.save(path.parent / 'validation_probabilities.npy', probabilities)
    result = {**request, **trained, **provenance, 'status': 'completed', 'metrics': metrics,
              'training_rows': len(y), 'validation_rows': len(v), 'feature_names': manifest['feature_names'],
              'classes': manifest['classes'], 'benign_index': manifest['benign_index'],
              'checkpoint_sha256': digest(Path(trained['checkpoint_path'])),
              'checkpoint_path': Path(trained['checkpoint_path']).relative_to(ROOT).as_posix(),
              'checkpoint_reload_verified': True, 'reload_device': 'cpu', 'reload_prediction_rows': count,
              'elapsed_seconds': time.perf_counter() - started, 'completed_at': time.time(), 'live_validated': False,
              'scope': 'Full CSV only; not validated for partial CSV or live capture; HEM is a closed-set upstream-derived adaptation'}
    seal_assets(path.parent)
    write(path.parent / 'result.json', result)
    print(json.dumps({'model': request['model_id'], 'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    fit(parser.parse_args().fit)
