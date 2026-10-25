"""Isolated DataSense audit and CPU XGBoost data ablations; never load test arrays.

The original source, preparation and active GPU study are immutable inputs.
Only train/validation source row IDs enter analysis or feature extraction.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from framework.training.tracking import classification_metrics, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'dataset/experiments/xgboost_data_audit_20261012_v1'
PARENT = ROOT / 'dataset/prepared/datasense_full_protocols'
BASELINE = ROOT / 'artifacts/training/csv_specialization_v2/runs/xgboost_ablation_01_weight_strategy_s42/attempt_000'
PORT_FIELDS = ('network_ports_src', 'network_ports_dst')
COUNT_FIELDS = ('network_packets_all_count', 'network_packets_src_count',
                'network_packets_dst_count', 'log_messages_count')
DURATION_NAMES = ['window_duration_seconds'] + [f'{name}__per_second' for name in COUNT_FIELDS]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _parse_ports(text):
    if not isinstance(text, str) or len(text) > 2_000_000:
        raise ValueError('Port field must be bounded list text')
    try:
        values = json.loads(text.replace("'", '"'))
    except (ValueError, RecursionError) as exc:
        raise ValueError('Malformed port list') from exc
    if not isinstance(values, list):
        raise ValueError('Expected port list')
    ports = set()
    for value in values:
        if isinstance(value, str) and value.isascii() and value.isdigit():
            value = int(value)
        if type(value) is not int or not 0 <= value <= 65535:
            raise ValueError('Invalid port number')
        ports.add(value)
    return tuple(sorted(ports))


_small_ports = lru_cache(maxsize=4096)(_parse_ports)


def parse_ports(text):
    return _small_ports(text) if isinstance(text, str) and len(text) <= 2048 else _parse_ports(text)


def choose_vocabulary(counts, maximum=32):
    # Only counts collected from training observations are accepted here.
    return sorted(np.flatnonzero(counts >= 5).tolist(), key=lambda p: (-int(counts[p]), p))[:maximum]


def port_vector(ports, vocabulary):
    observed = set(ports)
    return np.asarray([float(p in observed) for p in vocabulary] +
                      [float(bool(observed.difference(vocabulary)))], dtype=np.float32)


def duration_features(counts, starts, ends):
    seconds = (ends - starts).astype(np.float64) / 1e9
    valid = np.isfinite(seconds) & (seconds > 0)
    result = np.full((len(seconds), 5), np.nan, dtype=np.float32)
    result[valid, 0] = seconds[valid]
    result[valid, 1:] = counts[valid] / seconds[valid, None]
    return result


def load_local(directory):
    prepared = directory / 'prepared'
    manifest = json.loads((prepared / 'manifest.json').read_text())
    arrays = {s: {k: np.load(prepared / f'{k}_{s}.npy', mmap_mode='r')
                  for k in ('X', 'y', 'row_ids', 'window_start', 'window_end', 'feature_hash')}
              for s in ('train', 'validation')}
    return manifest, arrays


def audit_vectors(array, classes):
    X, y, hashes = array['X'], array['y'], array['feature_hash']
    frame = pd.DataFrame({'hash': hashes, 'label': y})
    group = frame.groupby('hash').agg(rows=('label', 'size'), labels=('label', 'nunique'))
    conflicts = group[group.labels > 1].sort_values('rows', ascending=False)
    details = []
    for key, item in conflicts.iterrows():
        indexes = np.flatnonzero(hashes == key)
        reference = X[indexes[0]]
        # Verify actual equality, so a hash collision cannot be a label conflict.
        for block in np.array_split(indexes, max(1, len(indexes) // 4096)):
            if not np.array_equal(X[block], np.broadcast_to(reference, (len(block), X.shape[1])), equal_nan=True):
                raise ValueError('Hash collision; cannot characterize as identical features')
        values, counts = np.unique(y[indexes], return_counts=True)
        details.append({'rows': len(indexes), 'all_zero': bool(np.all(reference == 0)),
                        'class_counts': {classes[int(k)]: int(n) for k, n in zip(values, counts)},
                        'example_source_row_ids': array['row_ids'][indexes[:5]].tolist()})
    pairs = frame.groupby(['hash', 'label']).size()
    bound = int(pairs.groupby(level=0).max().sum())
    zero = np.all(X == 0, axis=1)
    return dict(rows=len(y), distinct_vectors=len(group), duplicate_excess=int((group.rows-1).sum()),
                conflicting_groups=len(details), conflicting_rows=int(conflicts.rows.sum()),
                feature_identical_conflicts=details, zero_rows=int(zero.sum()),
                empirical_max_correct_from_identical_vectors=bound,
                empirical_max_accuracy_from_identical_vectors=bound/len(y),
                note='Feature conflicts do not prove incorrect source labels. Empirical maximum is not an achievable generalization score.')


def prepare(directory):
    prepared = directory / 'prepared'
    if prepared.exists():
        raise FileExistsError('Preparation already exists; preserve the completed experiment')
    copy = json.loads((directory / 'copy_manifest.json').read_text())
    source = directory / 'source/combined_dataset.csv'
    if digest(source) != copy['sha256']:
        raise ValueError('Source copy integrity failed')
    original = json.loads((PARENT / 'manifest.json').read_text())
    prepared.mkdir()
    input_hashes = {}
    for split in ('train', 'validation'):
        for prefix in ('X', 'y', 'row_ids', 'window_start', 'window_end', 'feature_hash'):
            name = f'{prefix}_{split}.npy'
            shutil.copy2(PARENT / name, prepared / name)
            input_hashes[name] = digest(prepared / name)
    manifest = dict(original)
    manifest.update(source=source.relative_to(ROOT).as_posix(), test_loaded=False,
                    rows={s: original['rows'][s] for s in ('train', 'validation')},
                    original_manifest_sha256=digest(PARENT / 'manifest.json'), input_hashes=input_hashes)
    write_json(prepared / 'manifest.json', manifest)
    _, arrays = load_local(directory)
    train, val = arrays['train'], arrays['validation']
    if np.intersect1d(train['row_ids'], val['row_ids']).size:
        raise ValueError('Split row overlap')
    report = {'source_copy': copy, 'test_loaded': False, 'partitions': {}, 'numeric_columns': [],
              'policy': 'All audit/fit rules use training only. Validation rows and labels are never filtered or changed.'}
    for split, a in arrays.items():
        report['partitions'][split] = audit_vectors(a, original['classes'])
    report['cross_partition_identical_feature_hashes'] = int(np.intersect1d(train['feature_hash'], val['feature_hash']).size)
    for i, name in enumerate(original['feature_names']):
        z = train['X'][:, i]
        good = np.isfinite(z)
        unique = np.unique(z[good])
        report['numeric_columns'].append({'feature': name, 'missing': int(np.isnan(z).sum()),
            'infinite': int(np.isinf(z).sum()), 'negative': int((z < 0).sum()),
            'distinct_train_values': len(unique), 'min': float(unique.min()) if len(unique) else None,
            'max': float(unique.max()) if len(unique) else None})
    keep = ~np.all(train['X'] == 0, axis=1)
    if set(np.unique(train['y'][keep])) != set(np.unique(train['y'])):
        raise ValueError('No-signal removal would remove an entire class')
    np.save(prepared / 'training_keep.npy', keep)
    class_table = []
    for i, name in enumerate(original['classes']):
        class_table.append({'class': name, 'training': int(np.sum(train['y'] == i)),
            'cleaned_training': int(np.sum(train['y'][keep] == i)),
            'validation': int(np.sum(val['y'] == i))})
    pd.DataFrame(class_table).to_csv(directory / 'class_counts.csv', index=False)
    pd.DataFrame(report['numeric_columns']).to_csv(directory / 'numeric_audit.csv', index=False)
    baseline = json.loads((BASELINE / 'result.json').read_text())
    cm = np.array(baseline['metrics']['confusion_matrix']); np.fill_diagonal(cm, 0)
    confusions = []
    for flat in np.argsort(-cm.ravel())[:30]:
        i, j = np.unravel_index(flat, cm.shape)
        confusions.append({'actual': original['classes'][i], 'predicted': original['classes'][j], 'rows': int(cm[i, j])})
    report['baseline_top_confusions'] = confusions
    pd.DataFrame(confusions).to_csv(directory / 'baseline_confusions.csv', index=False)
    write_json(directory / 'audit.json', report)
    print(f'Audit saved: train conflicts={report["partitions"]["train"]["conflicting_rows"]}; removed zero rows={(~keep).sum()}', flush=True)

    # Learn identity-free port vocabularies using training observations only.
    lookup = np.full(original['source_rows'], -1, dtype=np.int32)
    split_lookup = np.zeros(original['source_rows'], dtype=np.int8)
    for code, (split, a) in enumerate(arrays.items(), 1):
        lookup[a['row_ids']] = np.arange(len(a['y']))
        split_lookup[a['row_ids']] = code
    frequencies = {f: np.zeros(65536, dtype=np.int64) for f in PORT_FIELDS}
    invalid = Counter(); invalid_examples = []
    csv.field_size_limit(8 * 1024 * 1024)
    verified_rows = Counter()
    for pass_id in (1, 2):
        print(f'Source pass {pass_id}/2: only original train/validation IDs are inspected', flush=True)
        with source.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream); headers = next(reader)
            positions = {name: headers.index(name) for name in (*PORT_FIELDS, 'label1', 'label4')}
            for source_id, row in enumerate(reader):
                code = int(split_lookup[source_id]); index = int(lookup[source_id])
                if code == 0 or (pass_id == 1 and code != 1):
                    continue
                split = 'train' if code == 1 else 'validation'
                if len(row) != len(headers):
                    raise ValueError(f'Malformed selected source row {source_id}')
                expected = original['classes'][int(arrays[split]['y'][index])]
                if row[positions['label4']] != expected:
                    raise ValueError('Source label and frozen label/row identity disagree')
                if (row[positions['label1']] == 'benign') != (expected == 'benign'):
                    raise ValueError('Binary and detailed label disagree')
                if pass_id == 2:
                    verified_rows[split] += 1
                offset = 0
                for field in PORT_FIELDS:
                    try:
                        ports = parse_ports(row[positions[field]])
                    except ValueError:
                        ports = None
                        if pass_id == 2:
                            invalid[f'{split}:{field}'] += 1
                            if len(invalid_examples) < 10:
                                invalid_examples.append({'split': split, 'source_row_id': source_id, 'field': field})
                    if pass_id == 1:
                        if ports:
                            frequencies[field][list(ports)] += 1
                    else:
                        width = len(vocabulary[field]) + 1
                        extra[split][index, offset:offset+width] = (np.nan if ports is None else port_vector(ports, vocabulary[field]))
                        offset += width
                if source_id and source_id % 100000 == 0:
                    print(f'Pass {pass_id}: reached source row {source_id}', flush=True)
        if pass_id == 1:
            vocabulary = {field: choose_vocabulary(counts) for field, counts in frequencies.items()}
            extra_names = [f'{f}__present_{p}' for f in PORT_FIELDS for p in vocabulary[f] + ['UNSEEN']]
            extra_names += DURATION_NAMES
            extra = {s: np.lib.format.open_memmap(prepared / f'extra_{s}.npy', mode='w+',
                dtype=np.float32, shape=(len(a['y']), len(extra_names))) for s, a in arrays.items()}
    for split, a in arrays.items():
        if verified_rows[split] != len(a['y']):
            raise ValueError('Selected source rows missing')
        counts = a['X'][:, [original['feature_names'].index(n) for n in COUNT_FIELDS]]
        extra[split][:, -5:] = duration_features(counts, a['window_start'], a['window_end'])
        extra[split].flush()
    transformation = {'version': 1, 'port_vocabulary': vocabulary, 'extra_feature_names': extra_names,
        'base_feature_names': original['feature_names'], 'target': 'label4',
        'vocabulary_fit': 'original training rows only; document frequency >=5; top32 per direction; deterministic tie order',
        'duration': 'window end minus window start; absolute timestamps excluded',
        'invalid_ports': dict(invalid), 'invalid_examples': invalid_examples,
        'source_label_row_alignment_verified': dict(verified_rows),
        'training_filter': 'Remove rows with all original 170 inputs equal to zero; preserve remaining labels',
        'validation_filter': 'none', 'test_loaded': False}
    write_json(prepared / 'feature_transform.json', transformation)
    report['transformation'] = transformation
    report['cleaned_training_rows'] = int(keep.sum())
    report['unchanged_validation_rows'] = len(val['y'])
    write_json(directory / 'audit.json', report)
    print(f'Prepared {keep.sum()} cleaned rows, {len(val["y"])} unchanged validation rows, {len(extra_names)} extra features', flush=True)


def cpu_smoothed_objective(labels, smoothing=.05):
    rows = np.arange(len(labels))
    def objective(margins, data):
        p = np.exp(margins - margins.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        hessian = np.maximum(2 * p * (1-p), 1e-6)
        gradient = p - smoothing/p.shape[1]
        gradient[rows, labels] -= 1-smoothing
        return gradient, hessian
    return objective


def fit(directory, arm, rounds=79, threads=2):
    import xgboost as xgb
    from framework.training.csv_monitoring import snapshot
    output = directory / 'runs' / arm
    if output.exists():
        raise FileExistsError('Preserve previous fit; select a new directory')
    output.mkdir(parents=True)
    manifest, arrays = load_local(directory)
    transform = json.loads((directory / 'prepared/feature_transform.json').read_text())
    names = list(manifest['feature_names'])
    data = {}
    for split, a in arrays.items():
        X = np.asarray(a['X'])
        if arm == 'cleaned_enriched':
            extras = np.load(directory / f'prepared/extra_{split}.npy', mmap_mode='r')
            X = np.column_stack([X, extras])
        y = np.asarray(a['y'])
        if split == 'train' and arm != 'original_control':
            mask = np.load(directory / 'prepared/training_keep.npy')
            X, y = X[mask], y[mask]
        data[split] = (X, y)
    if arm == 'cleaned_enriched':
        names += transform['extra_feature_names']
    prior = json.loads((BASELINE / 'result.json').read_text())
    config = prior['config']
    parameters = {k: config[k] for k in ('max_depth', 'learning_rate', 'max_bin', 'subsample',
        'colsample_bytree', 'colsample_bynode', 'min_child_weight', 'reg_lambda', 'reg_alpha', 'gamma')}
    parameters.update(objective='multi:softprob', num_class=len(manifest['classes']), seed=42,
                      base_score=.5, device='cpu', tree_method='hist', nthread=threads,
                      disable_default_eval_metric=True)
    write_json(output / 'request.json', {'arm': arm, 'parameters': parameters, 'rounds': rounds,
        'seed': 42, 'loss': 'label_smoothed_ce', 'smoothing': .05,
        'training_rows': len(data['train'][1]), 'validation_rows': len(data['validation'][1]),
        'selection': 'Fixed rounds from frozen incumbent; no new validation early-stopping search',
        'test_loaded': False, 'input_manifest_sha256': digest(directory / 'prepared/manifest.json'),
        'transform_sha256': digest(directory / 'prepared/feature_transform.json')})
    start = time.monotonic()
    training = xgb.QuantileDMatrix(data['train'][0], label=data['train'][1], max_bin=parameters['max_bin'], nthread=threads)
    validation = xgb.DMatrix(data['validation'][0], nthread=threads)
    truth = data['validation'][1]
    labels = data['train'][1]
    del data
    history = []
    class Progress(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if epoch % 10 == 0 or epoch+1 == rounds:
                prediction = model.predict(validation).argmax(1)
                entry = {'round': epoch+1, 'elapsed_seconds': time.monotonic()-start,
                         **snapshot(truth, prediction, len(manifest['classes']), manifest['benign_index'])}
                history.append(entry); write_json(output / 'progress.json', entry)
                print(f'{arm} round={epoch+1}/{rounds} accuracy={entry["accuracy"]:.6f} elapsed={entry["elapsed_seconds"]:.1f}s', flush=True)
            return False
    model = xgb.train(parameters, training, num_boost_round=rounds,
                      obj=cpu_smoothed_objective(labels), callbacks=[Progress()])
    checkpoint = output / 'model.ubj'; model.save_model(checkpoint)
    probability = model.predict(validation)
    reloaded = xgb.Booster(model_file=str(checkpoint))
    reloaded.set_param({'device': 'cpu', 'nthread': threads})
    np.testing.assert_allclose(reloaded.predict(validation), probability, atol=1e-7, rtol=0)
    prediction = probability.argmax(1)
    result = classification_metrics(truth, prediction, manifest['classes'], manifest['benign_index'])
    result['top_three_accuracy'] = float(np.mean((np.argsort(-probability, axis=1)[:, :3] == truth[:, None]).any(1)))
    np.save(output / 'validation_probabilities.npy', probability)
    np.save(output / 'validation_predictions.npy', prediction)
    full = {'status': 'completed', 'arm': arm, 'device': 'cpu', 'seed': 42,
            'metrics': result, 'history': history, 'rounds': rounds,
            'elapsed_seconds': time.monotonic()-start, 'checkpoint_reload_verified': True,
            'checkpoint_sha256': digest(checkpoint), 'feature_names': names,
            'test_loaded': False, 'scope': 'Single-seed exploratory data ablation; full unchanged validation; no deployment promotion'}
    write_json(output / 'result.json', full)
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_class','confusion_matrix')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'fit'])
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--arm', choices=['original_control', 'cleaned', 'cleaned_enriched'], default='original_control')
    parser.add_argument('--rounds', type=int, default=79)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.directory)
    else:
        fit(args.directory, args.arm, args.rounds, args.threads)


if __name__ == '__main__':
    main()
