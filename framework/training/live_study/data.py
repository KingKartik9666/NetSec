"""Measured packet-count projection and purged, train-only experiment roles."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from .protocol import ROOT, FEATURES, ROLES, atomic, digest, read, relative, verify

SOURCE = ROOT/'dataset/prepared/datasense_full_train'


def valid_counts(x):
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != len(FEATURES):
        raise ValueError('Exactly seven measured packet counts required')
    return (np.isfinite(x).all(1) & (x >= 0).all(1) & (x == np.floor(x)).all(1)
            & (x[:, 0] > 0) & (x[:, 1:] <= x[:, :1]).all(1))


def vector_keys(x):
    return [tuple(row) for row in np.asarray(x)]


def split_roles(x, y, start, end):
    """Chronological within source class; rare classes stay in fit, explicitly.

    Boundaries purge overlapping windows. Equal seven-count vectors belong to
    the earliest role globally, even when their source labels disagree.
    """
    roles = np.full(len(y), -1, np.int8)
    for label in np.unique(y):
        ids = np.flatnonzero(y == label)
        times = np.unique(start[ids])
        if len(times) < 30:
            roles[ids] = 0
            continue
        cuts = times[np.minimum((len(times)*np.array([.6, .75, .85])).astype(int), len(times)-1)]
        for role in range(4):
            keep = np.ones(len(ids), bool)
            if role: keep &= start[ids] >= cuts[role-1]
            if role < 3: keep &= end[ids] < cuts[role]
            roles[ids[keep]] = role
    owners = {}
    keys = vector_keys(x)
    for role in range(4):
        for i in np.flatnonzero(roles == role):
            previous = owners.setdefault(keys[i], role)
            if previous != role: roles[i] = -1
    return roles


def _extract(source, split, features):
    # No generic split CLI: callers can never request the sealed test arrays.
    if split not in ('train', 'validation'): raise ValueError('Held-out test is locked')
    arrays = {name: np.load(source/f'{name}_{split}.npy', mmap_mode='r')
              for name in ('X', 'y', 'row_ids', 'window_start', 'window_end')}
    x = arrays['X'][:, features].astype(np.float32)
    exact = arrays['window_end']-arrays['window_start'] == 10_000_000_000
    keep = exact & valid_counts(x)
    result = {name: (x[keep] if name == 'X' else np.asarray(a[keep])) for name, a in arrays.items()}
    return result, dict(source_rows=len(x), exact_ten_second_rows=int(exact.sum()),
                       accepted_rows=int(keep.sum()), rejected_exact_rows=int((exact & ~keep).sum()))


def prepare(study, source=SOURCE):
    study, source = Path(study), Path(source)
    folder = study/'data'; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'manifest.json').exists():
        existing = read(folder/'manifest.json'); verify(existing['array_hashes']); verify(existing['input_hashes'])
        if existing.get('preparation_version') != 2:
            raise ValueError('Prepared role protocol changed; preserve it and use a fresh preparation revision')
        return existing
    manifest = read(source/'manifest.json')
    if manifest['benign_index'] != 0: raise ValueError('Benign class must be index zero')
    indices = [manifest['feature_names'].index(name) for name in FEATURES]
    arrays, source_counts = _extract(source, 'train', indices)
    x, y, start, end = [arrays[k] for k in ('X','y','window_start','window_end')]
    roles = split_roles(x, y, start, end)
    n_classes = len(manifest['classes'])
    if len(np.unique(y[roles == 0])) != n_classes:
        raise ValueError('Every declared class must have a fitting example; do not invent labels')
    coverage = {}
    for number, name in enumerate(ROLES):
        labels = y[roles == number]
        if len(np.unique(labels != 0)) != 2:
            raise ValueError(f'{name} needs both benign and attack support')
        counts = np.bincount(labels, minlength=n_classes)
        coverage[name] = dict(rows=len(labels), class_counts=counts.tolist(),
            classes_without_support=[manifest['classes'][i] for i in np.flatnonzero(counts == 0)])
    for name, a in arrays.items(): np.save(folder/f'{name}_train.npy', a)
    np.save(folder/'roles.npy', roles)
    # Copying assessment inputs is preparation only; no model selection reads them.
    val, val_counts = _extract(source, 'validation', indices)
    seen = set(vector_keys(x))
    novel = np.array([key not in seen for key in vector_keys(val['X'])])
    for name, a in val.items(): np.save(folder/f'{name}_validation.npy', a)
    np.save(folder/'novel_validation.npy', novel)
    unique, groups = np.unique(x, axis=0, return_inverse=True)
    joint = np.zeros((len(unique), n_classes), np.int64)
    np.add.at(joint, (groups, y), 1)
    ambiguous = (joint > 0).sum(1) > 1
    # Inspectable actual CSV windows, not synthetic packet labels.
    with (folder/'csv_window_review.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(['source_row_id','label','role','start_ns','end_ns',*FEATURES,'labels_sharing_vector'])
        chosen = set()
        for i in range(len(y)):
            key = (int(y[i]), int(roles[i]))
            if key in chosen: continue
            chosen.add(key)
            labels = [manifest['classes'][j] for j in np.flatnonzero(joint[groups[i]])]
            writer.writerow([arrays['row_ids'][i], manifest['classes'][y[i]],
                ROLES[roles[i]] if roles[i] >= 0 else 'purged', start[i], end[i], *map(int, x[i]), '|'.join(labels)])
    inputs = [source/'manifest.json'] + [source/f'{name}_{split}.npy'
        for split in ('train','validation') for name in ('X','y','row_ids','window_start','window_end')]
    result = dict(preparation_version=2,feature_names=FEATURES, classes=manifest['classes'], benign_index=0,
        source=relative(source), training=source_counts, validation=val_counts, coverage=coverage,
        partition='class-conditional chronological 60/15/10/15; rare (<30 start times) fit only; strict overlap and cross-role vector purging',
        purged_training_rows=int((roles < 0).sum()), novel_validation_rows=int(novel.sum()),
        duplicate_validation_rows=int((~novel).sum()), novel_reference_scope='all accepted original training rows, including purged rows',
        unique_training_vectors=len(unique),
        conflicting_training_vectors=int(ambiguous.sum()),
        empirical_training_vector_accuracy_ceiling=float(joint.max(1).sum()/len(y)),
        warning='Seven header counts cannot identify every attack subtype. Within-capture CSV validation is not live Npcap accuracy. Source extractor equivalence remains to be established.',
        test_loaded=False, input_hashes={relative(p):digest(p) for p in inputs},
        array_hashes={relative(p):digest(p) for p in folder.glob('*.npy')})
    atomic(folder/'manifest.json', result)
    return result


def training_data(study):
    folder = Path(study)/'data'
    return (np.load(folder/'X_train.npy'), np.load(folder/'y_train.npy'), np.load(folder/'roles.npy'))
