"""CPU replication of the isolated XGBoost audit; train/validation only."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiment.audit_xgboost_data import DEFAULT as SOURCE, DURATION_NAMES, cpu_smoothed_objective
from framework.training.tracking import classification_metrics, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'dataset/experiments/xgboost_followup_20261013_v1'
ARMS = ('original_control', 'cleaned', 'ports', 'rates', 'cleaned_enriched')
SEEDS = (42, 43, 44)
BASE = dict(max_depth=6, learning_rate=.08, max_bin=128, subsample=.8,
            colsample_bytree=1., colsample_bynode=.9, min_child_weight=3.,
            reg_lambda=3., reg_alpha=.05, gamma=0.)
TUNING = {'depth8': {'max_depth': 8}, 'rounds160': {'rounds': 160},
          'unsmoothed': {'smoothing': 0.}}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def feature_indexes(arm, manifest, transform):
    """Select by names so duration/rates cannot accidentally enter port arms."""
    if arm not in ARMS:
        raise ValueError(f'Unknown arm: {arm}')
    base = list(manifest['feature_names'])
    extra = list(transform['extra_feature_names'])
    names = base + extra
    indexes = list(range(len(base)))
    if arm in ('ports', 'rates', 'cleaned_enriched'):
        for offset, name in enumerate(extra, start=len(base)):
            is_rate = name in DURATION_NAMES
            if arm == 'cleaned_enriched' or (arm == 'rates' and is_rate) or (arm == 'ports' and not is_rate):
                indexes.append(offset)
    return np.asarray(indexes, dtype=np.int64), [names[i] for i in indexes]


def arrays(source, splits=('train', 'validation')):
    prepared = Path(source) / 'prepared'
    return {s: {key: np.load(prepared / f'{key}_{s}.npy', mmap_mode='r')
                for key in ('X', 'y', 'row_ids', 'window_start', 'window_end', 'feature_hash', 'extra')}
            for s in splits}


def matrix(array, rows, indexes):
    """Materialize only the requested rows and columns in bounded chunks."""
    width = array['X'].shape[1]
    out = np.empty((len(rows), len(indexes)), dtype=np.float32)
    base_cols = indexes[indexes < width]
    extra_cols = indexes[indexes >= width] - width
    for offset in range(0, len(rows), 8192):
        selected = rows[offset:offset+8192]
        out[offset:offset+len(selected), :len(base_cols)] = array['X'][np.ix_(selected, base_cols)]
        if len(extra_cols):
            out[offset:offset+len(selected), len(base_cols):] = array['extra'][np.ix_(selected, extra_cols)]
    return out


def temporal_folds(labels, starts, ends, hashes, folds=3):
    """Held class-conditional time blocks, with strict window/copy purging."""
    labels, starts, ends, hashes = map(np.asarray, (labels, starts, ends, hashes))
    if not (labels.shape == starts.shape == ends.shape == hashes.shape) or np.any(ends < starts):
        raise ValueError('Invalid fold provenance')
    codes = np.full(len(labels), -1, dtype=np.int8)
    classes = np.unique(labels)
    for label in classes:
        ids = np.flatnonzero(labels == label)
        times = np.unique(starts[ids])
        if len(times) < folds * 2:
            raise ValueError(f'Insufficient time blocks for class {label}')
        cuts = [times[(len(times) * i)//folds] for i in range(1, folds)]
        codes[ids] = np.searchsorted(cuts, starts[ids], side='right')
    result = []
    for fold in range(folds):
        held = np.flatnonzero(codes == fold)
        fit = codes != fold
        for label in classes:
            block = held[labels[held] == label]
            low, high = starts[block].min(), ends[block].max()
            fit &= ~((labels == label) & (starts <= high) & (ends >= low))
        fit &= ~np.isin(hashes, np.unique(hashes[held]))
        fitting = np.flatnonzero(fit)
        if not np.array_equal(np.unique(labels[fitting]), classes):
            raise ValueError('Purging removed a training class')
        result.append((fitting, held))
    if not np.array_equal(np.sort(np.concatenate([hold for _, hold in result])), np.arange(len(labels))):
        raise ValueError('Incomplete held-fold coverage')
    return result


def recipe(arm, seed, variant='base'):
    if variant not in ('base', *TUNING):
        raise ValueError('Unknown recipe')
    params = dict(BASE)
    extra = dict(TUNING.get(variant, {}))
    rounds, smoothing = extra.pop('rounds', 79), extra.pop('smoothing', .05)
    params.update(extra)
    return dict(arm=arm, seed=int(seed), variant=variant, rounds=rounds,
                smoothing=smoothing, parameters=params, threads=4)


def fit(source, output, request):
    """Fixed-round CPU fit, with complete checkpoint replay and pinned inputs."""
    import xgboost as xgb
    output, source = Path(output), Path(source)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    manifest = read(source / 'prepared/manifest.json')
    transform = read(source / 'prepared/feature_transform.json')
    data = arrays(source, ('train',) if 'fold' in request else ('train', 'validation'))
    indexes, names = feature_indexes(request['arm'], manifest, transform)
    all_rows = np.arange(len(data['train']['y']))
    if request['arm'] != 'original_control':
        all_rows = all_rows[np.load(source / 'prepared/training_keep.npy')]
    fit_rows = all_rows
    scoring_split = 'validation'
    score_rows = np.arange(len(data['validation']['y'])) if 'fold' not in request else None
    if 'fold' in request:
        blocks = temporal_folds(*(data['train'][key][all_rows] for key in
                                  ('y', 'window_start', 'window_end', 'feature_hash')))
        fitting, held = blocks[request['fold']]
        fit_rows, score_rows = all_rows[fitting], all_rows[held]
        scoring_split = 'train'
    labels = np.asarray(data['train']['y'][fit_rows], dtype=np.int64)
    truth = np.asarray(data[scoring_split]['y'][score_rows], dtype=np.int64)
    parameters = dict(request['parameters'], objective='multi:softprob',
                      num_class=len(manifest['classes']), seed=request['seed'],
                      base_score=.5, device='cpu', tree_method='hist',
                      nthread=request['threads'], disable_default_eval_metric=True)
    provenance = dict(request, training_rows=len(fit_rows), scoring_rows=len(score_rows),
                      scoring_split=scoring_split, test_loaded=False, parameters=parameters,
                      feature_names=names, xgboost_version=xgb.__version__,
                      input_manifest_sha256=digest(source / 'prepared/manifest.json'),
                      transform_sha256=digest(source / 'prepared/feature_transform.json'))
    write_json(output / 'request.json', provenance)
    np.save(output / 'fit_row_ids.npy', data['train']['row_ids'][fit_rows])
    np.save(output / 'scoring_row_ids.npy', data[scoring_split]['row_ids'][score_rows])
    raw = matrix(data['train'], fit_rows, indexes)
    training = xgb.QuantileDMatrix(raw, label=labels, max_bin=parameters['max_bin'], nthread=request['threads'])
    del raw
    raw = matrix(data[scoring_split], score_rows, indexes)
    scoring = xgb.DMatrix(raw, nthread=request['threads'])
    del raw
    gc.collect()
    history = []
    class Progress(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if epoch % 10 == 0 or epoch+1 == request['rounds']:
                entry = dict(round=epoch+1, rounds=request['rounds'],
                             elapsed_seconds=time.monotonic()-started)
                # Held-fold labels never choose rounds, recipes or stopping.
                if scoring_split == 'validation':
                    entry['accuracy'] = float(np.mean(model.predict(scoring).argmax(1) == truth))
                history.append(entry)
                write_json(output / 'progress.json', entry)
                print(json.dumps(entry), flush=True)
            return False
    model = xgb.train(parameters, training, num_boost_round=request['rounds'],
                      obj=cpu_smoothed_objective(labels, request['smoothing']), callbacks=[Progress()])
    checkpoint = output / 'model.ubj'
    model.save_model(checkpoint)
    probabilities = model.predict(scoring)
    loaded = xgb.Booster(model_file=str(checkpoint))
    loaded.set_param({'device': 'cpu', 'nthread': request['threads']})
    replay = loaded.predict(scoring)
    np.testing.assert_allclose(replay, probabilities, atol=1e-7, rtol=0)
    predictions = probabilities.argmax(1)
    np.testing.assert_array_equal(replay.argmax(1), predictions)
    np.save(output / 'probabilities.npy', probabilities)
    np.save(output / 'predictions.npy', predictions)
    np.save(output / 'truth.npy', truth)
    metrics = classification_metrics(truth, predictions, manifest['classes'], manifest['benign_index'])
    result = dict(status='completed', recipe=request, metrics=metrics,
                  feature_names=names, feature_indexes=indexes.tolist(),
                  checkpoint_reload_verified=True, device='cpu', history=history,
                  elapsed_seconds=time.monotonic()-started, test_loaded=False,
                  hashes={name: digest(output / name) for name in
                          ('model.ubj', 'request.json', 'probabilities.npy', 'predictions.npy',
                           'truth.npy', 'fit_row_ids.npy', 'scoring_row_ids.npy')})
    write_json(output / 'result.json', result)
    print(json.dumps({'completed': str(output), 'accuracy': metrics['accuracy'],
                      'macro_f1': metrics['macro_f1']}), flush=True)
    return result


def eligible(candidate, control):
    return (candidate['macro_recall'] >= control['macro_recall'] - .01 and
            candidate['false_positive_rate'] <= control['false_positive_rate'] + .005)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    fit(args.source, args.output, read(args.request))
