"""Constrained training-only synthetic adaptations with exact donor lineage.

The neighbor methods are our context-constrained algorithm adaptations, not
unmodified imbalanced-learn estimators. SMOTENC reduces to continuous SMOTE
within groups whose categorical values are identical. Borderline-1 uses danger
parents; ADASYN allocates to parents by their opposite-class neighbor fraction.
CTGAN is the real optional ctgan.CTGAN CUDA implementation, never the old cGAN.
Synthetic inherited labels are not independently verified attack observations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

import numpy as np

from framework.training.augmentation import array_sha256, file_sha256, project_generated, validate_generated
from framework.training.research_features import validate_raw


DEFAULTS = {'method': 'none', 'max_extra_fraction': .2,
            'max_per_class_multiplier': 2, 'k_neighbors': 5, 'ctgan_epochs': 20}


def _settings(config):
    if set(config) - set(DEFAULTS):
        raise ValueError('Unsupported augmentation configuration; validation/test inputs are not accepted')
    settings = {**DEFAULTS, **config}
    if settings['method'] not in {'none', 'smotenc', 'borderline_smote', 'adasyn', 'ctgan'}:
        raise ValueError('Unsupported augmentation method')
    for key in ('max_extra_fraction', 'max_per_class_multiplier'):
        value = settings[key]
        if not np.isfinite(value) or value < (1 if key == 'max_per_class_multiplier' else 0):
            raise ValueError('Invalid augmentation budget')
    for key in ('k_neighbors', 'ctgan_epochs'):
        if isinstance(settings[key], bool) or not isinstance(settings[key], (int, np.integer)) or settings[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    return settings


def _allocate(weights, budget, capacities=None):
    """Largest-remainder exact allocation, bounded by integer capacities."""
    weights = np.asarray(weights, dtype=float)
    if capacities is None:
        capacities = np.full(len(weights), budget, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    number = min(int(budget), int(capacities[weights > 0].sum()))
    result = np.zeros(len(weights), dtype=np.int64)
    if number <= 0 or weights.sum() <= 0:
        return result
    ideal = weights / weights.sum() * number
    result = np.minimum(np.floor(ideal).astype(np.int64), capacities)
    remaining = number - int(result.sum())
    while remaining:
        eligible = np.flatnonzero((result < capacities) & (weights > 0))
        if not len(eligible):
            break
        order = eligible[np.argsort(-(ideal[eligible]-result[eligible]), kind='stable')]
        chosen = order[:remaining]
        result[chosen] += 1
        remaining -= len(chosen)
    return result


def _contexts(X, names):
    fixed = [i for i, n in enumerate(names) if '__present_' in n or
             n in {'network_interval-packets', 'log_interval-messages'}]
    masks = np.packbits(np.isnan(X), axis=1)
    context = np.nan_to_num(X[:, fixed], nan=0.).copy()
    context[context == 0] = 0  # Canonicalize signed zero in exact context keys.
    groups = {}
    for i in range(len(X)):
        key = masks[i].tobytes() + context[i].tobytes()
        groups.setdefault(key, []).append(i)
    return [np.asarray(ids, dtype=np.int64) for ids in groups.values()], fixed


def _bounds(X):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        low, high = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
    return np.nan_to_num(low, nan=0.), np.nan_to_num(high, nan=0.)


def _neighbor_rows(X, y, groups, names, settings, quotas, rng, distance_columns):
    from sklearn.neighbors import NearestNeighbors
    generated, labels, lefts, rights, fractions, skipped = [], [], [], [], [], []
    k = settings['k_neighbors']
    method = settings['method']
    classes = np.unique(y)
    for label, quota in zip(classes, quotas):
        if not quota:
            continue
        eligible = []
        for group_id, group in enumerate(groups):
            same = group[y[group] == label]
            if not len(same):
                continue
            reason = None
            if len(same) <= k:
                reason = 'insufficient same-class neighbors in fixed context'
            observed = np.flatnonzero(np.isfinite(X[group[0]]))
            columns = np.intersect1d(observed, distance_columns)
            if not len(columns):
                reason = 'no observed distance columns in fixed context'
            if reason:
                skipped.append({'class_label': int(label), 'context_group': group_id,
                                'training_rows': len(same), 'reason': reason})
                continue
            data = X[group][:, columns].astype(np.float64)
            # Train-only signed-log scaling reduces dominance by large counts.
            logged = np.sign(data) * np.log1p(np.abs(data))
            scale = np.std(logged, axis=0)
            scale[scale < 1e-9] = 1.
            distance = logged / scale
            positions = np.flatnonzero(y[group] == label)
            minority = distance[positions]
            neighbors = NearestNeighbors(n_neighbors=k+1, n_jobs=1).fit(minority).kneighbors(minority, return_distance=False)
            # Duplicate vectors may place self anywhere, so exclude explicitly.
            local_neighbors = np.array([row[row != i][:k] for i, row in enumerate(neighbors)])
            weights = np.ones(len(same))
            if method in {'borderline_smote', 'adasyn'}:
                m = min(10, len(group)-1)
                all_nn = NearestNeighbors(n_neighbors=m+1, n_jobs=1).fit(distance).kneighbors(minority, return_distance=False)
                majority_count = np.array([np.count_nonzero(y[group[row[row != pos][:m]]] != label)
                                           for pos, row in zip(positions, all_nn)])
                if method == 'borderline_smote':
                    weights = ((majority_count >= m/2) & (majority_count < m)).astype(float)
                else:
                    weights = majority_count / m
                if not weights.any():
                    skipped.append({'class_label': int(label), 'context_group': group_id,
                                    'training_rows': len(same), 'reason': 'no danger parents' if method == 'borderline_smote'
                                    else 'no opposite-class neighbors; ADASYN undefined'})
                    continue
            eligible.append((same, local_neighbors, weights))
        group_quotas = _allocate([float(item[2].sum()) for item in eligible], quota)
        low, high = _bounds(X[y == label])
        for (same, neighbors, weights), number in zip(eligible, group_quotas):
            if not number:
                continue
            if method == 'adasyn':
                parent_counts = _allocate(weights, number)
                base = np.repeat(np.arange(len(same)), parent_counts)
                rng.shuffle(base)
            else:
                base = rng.choice(len(same), size=number, p=weights/weights.sum())
            other = neighbors[base, rng.integers(0, k, size=number)]
            a, b = same[base], same[other]
            fraction = rng.random(number)
            donor = X[a]
            candidate = donor.astype(np.float64) + fraction[:, None] * (X[b].astype(np.float64)-donor)
            values = project_generated(candidate, donor, names, low, high)
            generated.append(values)
            labels.extend([label]*number)
            lefts.extend(a)
            rights.extend(b)
            fractions.extend(fraction)
    values = np.concatenate(generated) if generated else np.empty((0, X.shape[1]), dtype=np.float32)
    return values, np.asarray(labels, dtype=y.dtype), np.asarray(lefts, dtype=np.int64), np.asarray(rights, dtype=np.int64), np.asarray(fractions), skipped, {}


def _ctgan_rows(X, y, groups, names, settings, quotas, rng, output, seed):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Real CTGAN research generation requires CUDA; no CPU fallback')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    from ctgan import CTGAN
    import pandas as pd
    # Joint label/context categorical condition fixes protocol bits, missing mask
    # and intervals together; CTGAN only supports one sample condition column.
    contexts = np.empty(len(y), dtype=object)
    pools = {}
    for group_id, group in enumerate(groups):
        for label in np.unique(y[group]):
            ids = group[y[group] == label]
            token = f'{int(label)}:{group_id}'
            contexts[ids] = token
            pools[token] = ids
    frame = pd.DataFrame(np.nan_to_num(np.sign(X)*np.log1p(np.abs(X)), nan=0.), columns=names)
    condition = '__research_label_context'
    frame[condition] = contexts
    discrete = [n for n in names if '__present_' in n] + [condition]
    model = CTGAN(epochs=settings['ctgan_epochs'], batch_size=500, pac=10, cuda=True, verbose=False)
    model.set_random_state(seed)
    model.fit(frame, discrete_columns=discrete)
    if getattr(model, '_device', torch.device('cpu')).type != 'cuda':
        raise RuntimeError('CTGAN did not execute on CUDA')
    output.mkdir(parents=True, exist_ok=True)
    model.save(str(output / 'ctgan.pkl'))
    generated, labels, donors, skipped = [], [], [], []
    attempts, rejected, changed, diversity_rows = 0, 0, 0, []
    for label, quota in zip(np.unique(y), quotas):
        tokens = [token for token, ids in pools.items() if y[ids[0]] == label and len(ids) > settings['k_neighbors']]
        for token, ids in pools.items():
            if y[ids[0]] == label and quota and len(ids) <= settings['k_neighbors']:
                skipped.append({'class_label': int(label), 'training_rows': len(ids), 'reason': 'insufficient CTGAN donor context support'})
        amounts = _allocate([len(pools[t]) for t in tokens], quota)
        low, high = _bounds(X[y == label])
        for token, number in zip(tokens, amounts):
            if not number:
                continue
            sample = model.sample(int(number)*3, condition_column=condition, condition_value=token)
            attempts += len(sample)
            compatible = sample[condition].astype(str).to_numpy() == token
            raw = sample[names].to_numpy(dtype=np.float64)
            with np.errstate(over='ignore', invalid='ignore'):
                raw = np.sign(raw)*np.expm1(np.abs(raw))
            compatible &= np.isfinite(raw).all(axis=1)
            compatible &= (np.abs(raw) <= np.finfo(np.float32).max).all(axis=1)
            rejected += int((~compatible).sum())
            raw = raw[compatible][:int(number)]
            chosen = rng.choice(pools[token], len(raw), replace=True)
            donor = X[chosen]
            raw[np.isnan(donor)] = np.nan
            values = project_generated(raw, donor, names, low, high)
            changed += int(np.count_nonzero(~np.isclose(raw, values, equal_nan=True)))
            # Duplicated real observations are rejected, never labelled genuine diversity.
            real_hashes = {hashlib.sha256(row.tobytes()).digest() for row in X[pools[token]]}
            keep = np.array([hashlib.sha256(row.tobytes()).digest() not in real_hashes for row in values])
            rejected += int((~keep).sum())
            values, chosen = values[keep], chosen[keep]
            if len(values):
                generated.append(values)
                labels.extend([label]*len(values))
                donors.extend(chosen)
                diversity_rows.extend(hashlib.sha256(row.tobytes()).digest() for row in values)
    values = np.concatenate(generated) if generated else np.empty((0, X.shape[1]), dtype=np.float32)
    if not len(values):
        raise RuntimeError('CTGAN produced no accepted nonduplicate rows')
    unique_fraction = len(set(diversity_rows))/len(diversity_rows)
    if unique_fraction < .5:
        raise RuntimeError('CTGAN accepted output collapsed; augmentation rejected')
    donors = np.asarray(donors, dtype=np.int64)
    quality = {'backend': 'ctgan.CTGAN', 'gpu_training_verified': True,
               'device': str(model._device), 'attempted_rows': attempts, 'rejected_rows': rejected,
               'projection_changed_cells': changed, 'unique_fraction': unique_fraction,
               'quality_limitation': 'Schema validity and diversity do not establish attack semantics',
               'loss_values': model.loss_values.to_dict(orient='records')}
    return values, np.asarray(labels, dtype=y.dtype), donors, donors.copy(), np.full(len(donors), np.nan), skipped, quality


def augment_training(X_train, y_train, row_ids, feature_names, config, seed, output_dir, distance_columns=None):
    """Return all original training rows followed by bounded synthetic rows.

    Callers must supply only the immutable training partition. The interface
    deliberately accepts neither a validation nor a test array or label source.
    distance_columns selects raw feature indices/names for neighbor distances;
    it never changes generated schema or the fixed categorical context.
    """
    settings = _settings(config)
    names = list(feature_names)
    X = validate_raw(X_train, names)
    y, ids = np.asarray(y_train), np.asarray(row_ids)
    if y.ndim != 1 or ids.ndim != 1 or len(y) != len(X) or len(ids) != len(X) or not len(y):
        raise ValueError('Matching nonempty training features, labels and row IDs required')
    if not np.issubdtype(y.dtype, np.integer) or (y < 0).any():
        raise ValueError('Real training category labels must be nonnegative integers')
    if not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
        raise ValueError('Training row IDs must be unique integers')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh augmentation directory; refusing to overwrite provenance')
    groups, fixed = _contexts(X, names)
    if distance_columns is None:
        columns = np.array([i for i in range(len(names)) if i not in fixed], dtype=np.int64)
    else:
        columns = np.array([names.index(c) if isinstance(c, str) else c for c in distance_columns], dtype=np.int64)
        if not len(columns) or (columns < 0).any() or (columns >= len(names)).any():
            raise ValueError('Distance columns are outside the raw schema')
    classes, counts = np.unique(y, return_counts=True)
    capacities = np.minimum(counts.max()-counts, np.floor(counts*(settings['max_per_class_multiplier']-1))).astype(np.int64)
    quotas = _allocate(capacities, int(np.floor(len(y)*settings['max_extra_fraction'])), capacities)
    if settings['method'] == 'none':
        quotas[:] = 0
    rng = np.random.default_rng(seed)
    if settings['method'] == 'none':
        synth, target = np.empty((0, X.shape[1]), dtype=np.float32), np.empty(0, dtype=y.dtype)
        left = right = np.empty(0, dtype=np.int64)
        fractions, skipped, quality = np.empty(0), [], {}
    elif settings['method'] == 'ctgan':
        synth, target, left, right, fractions, skipped, quality = _ctgan_rows(X, y, groups, names, settings, quotas, rng, output, seed)
    else:
        synth, target, left, right, fractions, skipped, quality = _neighbor_rows(X, y, groups, names, settings, quotas, rng, columns)
    if not np.array_equal(y[left], target) or not np.array_equal(y[right], target):
        raise ValueError('Synthetic labels differ from training donor labels')
    audit = validate_generated(synth, X[left], names)
    if not audit['passed']:
        raise ValueError('Synthetic rows failed constraint audit')
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'synthetic_features.npy', synth)
    np.save(output / 'synthetic_labels.npy', target)
    # Store deterministic standalone .npy assets for hash identity; .npz ZIP
    # timestamps are not suitable for reproducibility fingerprints.
    np.save(output / 'parent_a_indices.npy', left)
    np.save(output / 'parent_b_indices.npy', right)
    np.save(output / 'interpolation_fraction.npy', fractions)
    np.savez(output / 'provenance.npz', parent_a_row_id=ids[left], parent_b_row_id=ids[right],
             parent_a_label=y[left], parent_b_label=y[right], generated_label=target,
             interpolation_fraction=fractions, label_is_real_observation=np.zeros(len(target), dtype=bool))
    assets = sorted([p for p in output.iterdir() if p.suffix in {'.npy', '.npz', '.pkl'}])
    actual = {'none': 'identity; all real training rows retained',
              'smotenc': 'our fixed-context SMOTENC adaptation: continuous same-label neighbor interpolation',
              'borderline_smote': 'our fixed-context Borderline-1 SMOTE adaptation',
              'adasyn': 'our fixed-context ADASYN adaptation with largest-remainder exact allocation',
              'ctgan': 'genuine ctgan.CTGAN CUDA conditioned on joint real label/donor context'}[settings['method']]
    metadata = {'config': settings, 'seed': int(seed), 'feature_names': names,
        'status': 'not_requested' if settings['method'] == 'none' else 'applied' if len(synth) else 'skipped',
        'training_rows': len(X), 'synthetic_rows': len(synth), 'new_independent_observations': 0,
        'validation_used': False, 'test_used': False, 'generated_labels': 'Inherited training donor category; not real observations or confirmed attacks',
        'actual_algorithm': actual, 'unmodified_imblearn_estimator': False,
        'constraints': ['same training category parents', 'identical protocol bits, missing masks and window interval context',
                        'integer nonnegative counts', 'min <= avg <= max', 'packet subsets <= packet total',
                        'protocol count metadata preserved from donor'],
        'training_features_sha256': array_sha256(X), 'training_labels_sha256': array_sha256(y),
        'training_row_ids_sha256': array_sha256(ids), 'distance_columns': columns.tolist(),
        'parent_provenance_sha256': array_sha256(np.column_stack((ids[left], ids[right], target))),
        'asset_hashes': {p.name: file_sha256(p) for p in assets}, 'skipped_groups': skipped,
        'requested_synthetic_counts': {str(int(c)): int(n) for c, n in zip(classes, quotas)},
        'synthetic_counts': {str(int(c)): int((target == c).sum()) for c in classes},
        'audit': audit, 'quality': quality}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    return np.concatenate((X, synth)), np.concatenate((y, target)), metadata
