"""Isolated research fit: raw CSV -> train-only recipe -> CUDA -> CPU reload.

Only training/validation partitions are loaded. Synthetic observations never
become authentic HEM neighbors. This worker never updates the deployed app.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np

from experiment.optimize_models import ROOT, digest, write
from experiment.train_csv_models import load_csv_data, train_candidate, reload_outputs
from framework.training.tracking import classification_metrics


def resolved_checkpoint(result: dict) -> Path:
    path = Path(result['checkpoint_path'])
    return path if path.is_absolute() else ROOT / path


def reload_research(result: dict, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Load an entire saved pipeline explicitly on CPU using raw input columns."""
    checkpoint = resolved_checkpoint(result)
    if checkpoint.suffix == '.json':
        manifest = json.loads(checkpoint.read_text())
        if manifest.get('kind') != 'same_family_seed_ensemble':
            raise ValueError('Unrecognized research bundle')
        outputs = [reload_research(member, raw) for member in manifest['members']]
        if result['model_id'] == 'hem':
            votes = np.stack([p for p, _ in outputs])
            counts = np.stack([(votes == i).sum(0) for i in range(manifest['n_classes'])], axis=1)
            return counts.argmax(1), None
        probability = np.mean([p for _, p in outputs], axis=0)
    else:
        transform_file = checkpoint.parent / 'research_transform.joblib'
        transform = joblib.load(transform_file)
        predicted, probability = reload_outputs(result['model_id'], checkpoint, transform.transform(raw))
        if probability is None:
            return predicted, None
    calibration = checkpoint.parent / 'calibration.json'
    if calibration.exists():
        temperature = json.loads(calibration.read_text())['temperature']
        z = np.log(np.clip(probability, 1e-12, 1.)) / temperature
        z -= z.max(axis=1, keepdims=True)
        probability = np.exp(z)
        probability /= probability.sum(axis=1, keepdims=True)
    return probability.argmax(1), probability


def fit_base(request: dict, manifest: dict, X, y, ids, V, v, directory: Path) -> dict:
    """Every final fit retains all supplied authentic originals before augmentation."""
    from framework.training.research_features import ResearchTransform
    from framework.training.research_augmentation import augment_training
    pipeline = request.get('pipeline', {})
    transform_config = {k: pipeline[k] for k in ('feature_selection', 'feature_engineering') if k in pipeline}
    transform = ResearchTransform(manifest['feature_names'], transform_config).fit(X, y, seed=request['seed'])
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(transform, directory / 'research_transform.joblib')
    distance_columns = (getattr(transform, 'raw_distance_columns', None)
                        if pipeline.get('order') == 'selection_then_resampling' else None)
    augmentation_config = {**pipeline.get('resampling_params', {}), 'method': pipeline.get('resampling', 'none')}
    augmented, labels, augmentation = augment_training(X, y, ids, manifest['feature_names'],
             augmentation_config, request['seed'], directory / 'augmentation', distance_columns=distance_columns)
    if len(labels) < len(y) or not np.array_equal(labels[:len(y)], y):
        raise ValueError('Augmentation dropped or relabelled original training observations')
    if not np.array_equal(augmented[:len(X)], X, equal_nan=True):
        raise ValueError('Augmentation changed original training observations')
    if pipeline.get('resampling', 'none') != 'none' and len(labels) == len(y):
        raise ValueError('Requested augmentation generated no eligible synthetic rows; unsupported recipe')
    config = {**request['config'], 'original_training_rows': len(y)}
    config['selection_metric'] = 'accuracy'
    trained = train_candidate(request['model_id'], transform.transform(augmented), labels,
               transform.transform(V), v, len(manifest['classes']), manifest['benign_index'],
               config, request['seed'], directory)
    if trained.get('pruned'):
        raise InterruptedError('Optuna pruned this research fit')
    trained.update(original_training_rows=len(y), effective_training_rows=len(labels),
                   synthetic_training_rows=len(labels)-len(y), augmentation=augmentation,
                   feature_names=transform.output_names, raw_feature_names=manifest['feature_names'])
    # Reload the complete raw-schema pipeline, not only the classifier.
    count = min(128, len(v)) if request['model_id'] == 'hem' else len(v)
    info = {'model_id': request['model_id'], 'checkpoint_path': trained['checkpoint_path']}
    pred, prob = reload_research(info, V[:count])
    if not np.array_equal(pred, trained['predictions'][:count]):
        raise RuntimeError('Research raw-input CPU reload differs from CUDA-selected predictions')
    trained.update(checkpoint_reload_verified=True, reload_device='cpu', reload_prediction_rows=count)
    if request['model_id'] == 'kan' and trained.get('probabilities') is None:
        trained['probabilities'] = prob
    return trained


def oof_temperature(request: dict, manifest: dict, X, y, ids, directory: Path) -> dict:
    """Calibrate using genuine held-fold predictions; outer validation is absent."""
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from scipy.optimize import minimize_scalar
    if request['model_id'] == 'hem':
        raise ValueError('HEM hard voting has no calibrated probability API')
    folds = StratifiedKFold(3, shuffle=True, random_state=request['seed'])
    probabilities = np.zeros((len(y), len(manifest['classes'])), dtype=np.float32)
    provenance = []
    for fold, (available, hold) in enumerate(folds.split(X, y)):
        fit_ids, stop_ids = train_test_split(available, test_size=.1, stratify=y[available], random_state=42+fold)
        inner = {**request, 'pipeline': {**request.get('pipeline', {}), 'calibration': 'none', 'ensemble': 'none'}}
        config = dict(request['config'])
        remaining = float(os.environ.get('IMDAI_FIT_DEADLINE', 'inf')) - time.time() - 180
        config['runtime_seconds'] = min(config.get('runtime_seconds', 1050), remaining/(4-fold))
        inner['config'] = config
        trained = fit_base(inner, manifest, X[fit_ids], y[fit_ids], ids[fit_ids], X[stop_ids], y[stop_ids], directory / f'oof_{fold}')
        _, probability = reload_research({'model_id': request['model_id'], 'checkpoint_path': trained['checkpoint_path']}, X[hold])
        probabilities[hold] = probability
        provenance.append({'fold': fold, 'fit_row_ids': ids[fit_ids].tolist(),
                           'stopping_row_ids': ids[stop_ids].tolist(), 'prediction_row_ids': ids[hold].tolist()})
    if not np.isfinite(probabilities).all() or np.any(probabilities.sum(1) <= 0):
        raise ValueError('Incomplete out-of-fold probability coverage')
    logp = np.log(np.clip(probabilities.astype(np.float64), 1e-12, 1.))
    def loss(log_temperature):
        z = logp / np.exp(log_temperature)
        z -= z.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y]))
    optimum = minimize_scalar(loss, bounds=(np.log(.1), np.log(10.)), method='bounded')
    np.save(directory / 'oof_probabilities.npy', probabilities)
    write(directory / 'oof_provenance.json', {'folds': provenance, 'outer_validation_used': False,
                    'note': 'Training-only stratified folds; overlapping source windows inside training are not independent captures'})
    return {'temperature': float(np.exp(optimum.x)), 'training_oof_log_loss': float(optimum.fun),
            'method': 'scalar temperature scaling', 'outer_validation_used': False,
            'fit_rows': len(y), 'argmax_unchanged': True}


def fit_ensemble(request: dict, manifest: dict, X, y, ids, V, v, directory: Path) -> dict:
    """Uniform same-family seed average/hard vote; no fitted validation weights."""
    members, predictions, probabilities = [], [], []
    for offset in range(3):
        member = {**request, 'seed': request['seed']+offset,
                  'pipeline': {**request.get('pipeline', {}), 'ensemble': 'none', 'calibration': 'none'}}
        config = dict(request['config'])
        remaining = float(os.environ.get('IMDAI_FIT_DEADLINE', 'inf')) - time.time() - 180
        config['runtime_seconds'] = min(config.get('runtime_seconds', 1050), remaining/(3-offset))
        member['config'] = config
        trained = fit_base(member, manifest, X, y, ids, V, v, directory / f'member_{offset}')
        from experiment.optimize_csv_research import cuda_verified
        if not cuda_verified(trained):
            raise RuntimeError('Ensemble member lacks verified CUDA execution')
        predictions.append(trained.pop('predictions'))
        probability = trained.pop('probabilities', None)
        if probability is not None:
            probabilities.append(probability)
        members.append({'model_id': request['model_id'],
                        'checkpoint_path': Path(trained['checkpoint_path']).relative_to(ROOT).as_posix(),
                        'seed': member['seed'], 'details': trained})
    checkpoint = directory / 'ensemble.json'
    write(checkpoint, {'kind': 'same_family_seed_ensemble', 'n_classes': len(manifest['classes']),
                      'members': members, 'combination': 'uniform hard vote' if request['model_id']=='hem' else 'uniform probability average'})
    if request['model_id'] == 'hem':
        votes = np.stack(predictions)
        predicted = np.stack([(votes==i).sum(0) for i in range(len(manifest['classes']))], axis=1).argmax(1)
        probability = None
    else:
        probability = np.mean(probabilities, axis=0)
        predicted = probability.argmax(1)
    count = min(128,len(v)) if request['model_id']=='hem' else len(v)
    cpu, _ = reload_research({'model_id':request['model_id'],'checkpoint_path':str(checkpoint)},V[:count])
    if not np.array_equal(cpu,predicted[:count]):
        raise RuntimeError('Ensemble CPU reload mismatch')
    return {'checkpoint_path': str(checkpoint), 'predictions': predicted, 'probabilities': probability,
            'device': 'cuda:0', 'gpu_training_verified': True,
            'device_evidence': {'members': [m['details']['device_evidence'] for m in members]},
            'checkpoint_reload_verified': True, 'reload_device': 'cpu', 'reload_prediction_rows': count,
            'backend': 'explicit same-family seed ensemble', 'members': members,
            'original_training_rows':len(y), 'effective_training_rows':members[0]['details']['effective_training_rows'],
            'synthetic_training_rows':members[0]['details']['synthetic_training_rows']}


def fit(path: Path) -> None:
    request = json.loads(path.read_text())
    if request.get('input_profile') != 'full_csv' or request['model_id'] not in ('xgboost','kan','hem'):
        raise ValueError('Only the three upgraded full-CSV families are eligible')
    prepared = ROOT / request['prepared']
    if digest(prepared / 'manifest.json') != request['prepared_manifest_sha256']:
        raise ValueError('Training manifest changed')
    manifest,X,y,V,v,provenance = load_csv_data(prepared)
    ids = np.load(prepared / 'row_ids_train.npy', mmap_mode='r')
    directory=path.parent
    print(f"RESEARCH CUDA FIT {request['model_id']} real_rows={len(y)} columns={X.shape[1]} recipe={request.get('pipeline')}",flush=True)
    started=time.perf_counter()
    pipeline=request.get('pipeline',{})
    calibration=None
    if pipeline.get('calibration','none')!='none':
        if pipeline.get('ensemble','none')!='none':
            raise ValueError('Ensemble calibration requires ensemble OOF predictions; not supported')
        calibration=oof_temperature(request,manifest,X,y,ids,directory)
    trained=(fit_ensemble if pipeline.get('ensemble','none')!='none' else fit_base)(request,manifest,X,y,ids,V,v,directory)
    predicted=trained.pop('predictions')
    probability=trained.pop('probabilities',None)
    if calibration:
        write(directory/'calibration.json',calibration)
        predicted,probability=reload_research({'model_id':request['model_id'],'checkpoint_path':trained['checkpoint_path']},V)
    metrics=classification_metrics(v,predicted,manifest['classes'],manifest['benign_index'])
    metrics['top_three_accuracy']=(float(np.mean((np.argsort(-probability,axis=1)[:,:3]==v[:,None]).any(1))) if probability is not None else None)
    np.save(directory/'validation_predictions.npy',predicted)
    if probability is not None: np.save(directory/'validation_probabilities.npy',probability)
    checkpoint=Path(trained['checkpoint_path'])
    from experiment.optimize_csv_research import cuda_verified
    verified_cuda = cuda_verified(trained)
    if not verified_cuda:
        raise RuntimeError('Research fit lacks verified CUDA execution evidence')
    result={**request,**provenance,**trained,'status':'completed','metrics':metrics,
            'checkpoint_path':checkpoint.relative_to(ROOT).as_posix(),'checkpoint_sha256':digest(checkpoint),
            'classes':manifest['classes'],'raw_feature_names':manifest['feature_names'],
            'benign_index':manifest['benign_index'],'validation_rows':len(v),'calibration':calibration,
            'training_rows':len(y),'masked_training_rows':0,'test_loaded':False,'live_validated':False,
            'cuda_verified':verified_cuda,'completed_at':time.time(),'elapsed_seconds':time.perf_counter()-started,
            'scope':'Real full-CSV source validation; synthetic additions labelled; HEM-derived closed-set experiment'}
    assets={str(p.relative_to(directory)):digest(p) for p in directory.rglob('*') if p.is_file() and
            p.name not in ('result.json','bundle_integrity.json','progress.json','fit.log','PRUNE','execution.json')}
    write(directory/'bundle_integrity.json',assets)
    write(directory/'result.json',result)
    print(json.dumps({'model':request['model_id'],'accuracy':metrics['accuracy'],'recall':metrics['macro_recall'],'f1':metrics['macro_f1']}),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',type=Path,required=True)
    path=parser.parse_args().fit
    try:
        fit(path)
    except Exception as exc:
        reason=str(exc)
        kind=('pruned' if isinstance(exc,InterruptedError) else 'cuda_oom' if
              ('out of memory' in reason.lower() or 'bad_alloc' in reason.lower()) else 'failed')
        write(path.parent/'worker_failure.json',{'kind':kind,'error_type':type(exc).__name__,'message':reason,'test_loaded':False})
        raise


if __name__=='__main__': main()
