"""Train-only input boundary and matched representation construction."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

TRAIN_FILES = frozenset(f'{name}_train.npy' for name in
                       ('X', 'y', 'row_ids', 'window_start', 'window_end'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    os.replace(temporary, path)


def verify_pins(pins, root):
    for name, expected in pins.items():
        if sha256(Path(root)/name) != expected:
            raise ValueError(f'Pinned input/source changed: {name}')


def load_training_array(directory, name):
    if name not in TRAIN_FILES:
        raise ValueError(f'File is outside the training-only allowlist: {name}')
    return np.load(Path(directory)/name, mmap_mode='r', allow_pickle=False)


def canonical(values):
    copy = np.array(values, dtype=np.float32, order='C', copy=True)
    if np.isinf(copy).any():
        raise ValueError('Infinite model inputs require explicit preprocessing')
    copy[np.isnan(copy)] = np.float32(np.nan)
    copy[copy == 0] = 0.
    return copy


def row_hashes(values):
    rows = canonical(values)
    return np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in rows],
                    dtype='S64')


def duplicate_columns(fitting):
    """Keep every constant; collapse exact nonconstant fitting duplicates only."""
    keep, groups, by_hash = [], [], {}
    for index in range(fitting.shape[1]):
        values = canonical(fitting[:, index])
        constant = np.all((values == values[0]) |
                          (np.isnan(values) & np.isnan(values[0])))
        if constant:
            keep.append(index)
            continue
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        if digest in by_hash:
            group = by_hash[digest]
            if not np.array_equal(fitting[:, group[0]], values, equal_nan=True):
                raise ValueError('Column hash collision')
            group.append(index)
        else:
            keep.append(index)
            group = [index]
            by_hash[digest] = group
            groups.append(group)
    return keep, [group for group in groups if len(group) > 1]


def feature_arms(fitting, names):
    target = 'network_tcp-flags_std_deviation'
    bitmasks = {f'network_{kind}-flags_{stat}' for kind in ('tcp', 'ip')
                for stat in ('avg', 'min', 'max', 'std_deviation')}
    if not bitmasks.issubset(names):
        raise ValueError('Expected flag-summary fields are missing')
    keep, groups = duplicate_columns(fitting)
    return {
        'all_features': list(range(len(names))),
        'drop_tcp_flag_std': [i for i, name in enumerate(names) if name != target],
        'drop_bitmask_summaries': [i for i, name in enumerate(names) if name not in bitmasks],
        'deduplicate_nonconstant': keep,
    }, [[names[i] for i in group] for group in groups]


def common_evaluation_mask(fitting, evaluation, arms):
    removed = np.zeros(len(evaluation), dtype=bool)
    counts = {}
    for name, columns in arms.items():
        collision = np.isin(row_hashes(evaluation[:, columns]),
                            row_hashes(fitting[:, columns]))
        counts[name] = int(collision.sum())
        removed |= collision
    return ~removed, dict(purged_union=int(removed.sum()), collisions_by_arm=counts)


def check_roles(labels, starts, ends, fit, evaluation):
    if not len(fit) or not len(evaluation):
        raise ValueError('Empty fitting or evaluation role')
    if np.intersect1d(fit, evaluation).size:
        raise ValueError('Fitting/evaluation row overlap')
    for label in np.intersect1d(labels[fit], labels[evaluation]):
        before = fit[labels[fit] == label]
        after = evaluation[labels[evaluation] == label]
        if ends[before].max() >= starts[after].min():
            raise ValueError(f'Temporal overlap or touching boundary for class {label}')
