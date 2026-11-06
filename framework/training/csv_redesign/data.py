"""CSV-only data preparation with fitting-only learning and visible exclusions.

Only the original training row whitelist is interpreted. The copied raw file is
streamed, selected raw/cleaned CSVs are archived, and every row keeps its original
label. No held-out test or original validation array is opened by this module.
"""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from experiment.audit_datasense_raw import LISTS, META
from experiment.datasense_integrity import clean_row
from experiment.datasense_quality import digest, selected_rows

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COPY = ROOT / 'dataset/experiments/csv_audit_redesign_20261030_v1'
DEFAULT_MODEL = ROOT / 'dataset/experiments/datasense_verified_clean_20261014_v2/model'
ROLES = ('fit', 'tune', 'blend', 'calibration', 'scenario_challenge')
VERSION = 1


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def _relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _verify(pins):
    for name, expected in pins.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if digest(path) != expected:
            raise ValueError(f'Pinned data changed: {name}')


def split_roles(labels, scenarios, starts, ends, observable, *, gap_ns=1_000_000_000,
                min_scenario_rows=30, min_remaining_rows=100, min_times=40):
    """Whole-scenario challenge, then purged temporal roles within each scenario.

    Metadata determine partitions before feature fitting. Scenario/device names
    are never predictors. Sparse scenarios stay in fitting; if temporal splitting
    would remove all observable fitting support for a class, its remaining
    scenarios stay fitting-only. The challenge is never silently moved into fit.
    """
    labels, scenarios, starts, ends, observable = map(
        np.asarray, (labels, scenarios, starts, ends, observable))
    if not all(a.ndim == 1 and a.shape == labels.shape for a in
               (scenarios, starts, ends, observable)) or np.any(ends <= starts):
        raise ValueError('Invalid aligned temporal metadata')
    roles = np.full(len(labels), -1, np.int8)
    challenge = []
    for label in np.unique(labels):
        ids = np.flatnonzero((labels == label) & observable)
        options = []
        for scenario in np.unique(scenarios[ids]):
            members = ids[scenarios[ids] == scenario]
            if len(members) >= min_scenario_rows and len(ids) - len(members) >= min_remaining_rows:
                options.append((int(starts[members].min()), int(scenario)))
        if len(np.unique(scenarios[ids])) > 1 and options:
            challenge.append(max(options)[1])
    for scenario in np.unique(scenarios):
        ids = np.flatnonzero(scenarios == scenario)
        if scenario in challenge:
            roles[ids] = 4
            continue
        times = np.unique(starts[ids])
        if len(times) < min_times:
            roles[ids] = 0
            continue
        cuts = times[(len(times) * np.array([.60, .75, .85])).astype(int)]
        for role in range(4):
            keep = np.ones(len(ids), bool)
            if role:
                keep &= starts[ids] >= cuts[role - 1]
            if role < 3:
                keep &= ends[ids] + gap_ns < cuts[role]
            roles[ids[keep]] = role
    fit_only_classes = []
    for label in np.unique(labels):
        if not np.any((labels == label) & observable & (roles == 0)):
            members = (labels == label) & ~np.isin(scenarios, challenge)
            if not np.any(members & observable):
                raise ValueError(f'No observable fitting class support for {label}')
            roles[members] = 0
            fit_only_classes.append(int(label))
    verify_intervals(roles, scenarios, starts, ends, gap_ns=gap_ns)
    return roles, {'challenge_scenarios': sorted(challenge),
                   'fit_only_class_fallbacks': fit_only_classes,
                   'temporal_gap_ns': int(gap_ns), 'min_temporal_start_times': min_times,
                   'challenge_min_rows': min_scenario_rows,
                   'challenge_min_remaining_rows': min_remaining_rows,
                   'rule': 'latest eligible whole scenario per class; remaining scenarios 60/15/10/15 chronology; strict boundary purge'}


def verify_intervals(roles, scenarios, starts, ends, *, gap_ns=1_000_000_000):
    """Reject cross-role overlapping or touching source windows, across devices."""
    for scenario in np.unique(scenarios):
        ids = np.flatnonzero((scenarios == scenario) & (roles >= 0))
        present = sorted(np.unique(roles[ids]))
        if 4 in present and len(present) != 1:
            raise ValueError('Scenario challenge overlaps a training role')
        for left, right in zip(present, present[1:]):
            a = ids[roles[ids] == left]
            b = ids[roles[ids] == right]
            if ends[a].max() + gap_ns >= starts[b].min():
                raise ValueError('Cross-role source interval overlap')


def vector_hashes(x, indices=None):
    """Canonical float32 hashes in bounded blocks; NaNs and signed zero agree."""
    hashes = np.empty(len(x), dtype='S64')
    for first in range(0, len(x), 2048):
        block = np.array(x[first:first + 2048], dtype=np.float32, copy=True)
        if indices is not None:
            block = np.ascontiguousarray(block[:, indices])
        block[block == 0] = 0
        block[np.isnan(block)] = np.float32(np.nan)
        if np.isinf(block).any():
            raise ValueError('Infinite encoded input')
        for offset, row in enumerate(block):
            hashes[first + offset] = hashlib.sha256(row.tobytes()).hexdigest().encode('ascii')
    return hashes


def quarantine_fit(x, y, roles, hard_invalid, no_signal, *, n_classes):
    fit = roles == 0
    eligible = fit & ~np.asarray(hard_invalid, bool) & ~np.asarray(no_signal, bool)
    hashes = vector_hashes(x)
    first_label, conflicting_keys = {}, set()
    for i in np.flatnonzero(eligible):
        key = bytes(hashes[i])
        previous = first_label.setdefault(key, int(y[i]))
        if previous != y[i]:
            conflicting_keys.add(key)
    conflicting = np.array([eligible[i] and bytes(hashes[i]) in conflicting_keys
                            for i in range(len(y))], bool)
    clean = eligible & ~conflicting
    counts = np.bincount(y[clean], minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError(f'Quality quarantine would remove fitting class support: {np.flatnonzero(counts == 0).tolist()}')
    return {'fit_clean': clean, 'conflicting_fit': conflicting,
            'conflicting_groups': len(conflicting_keys), 'hashes': hashes}


def select_redundancy(x, fit, names):
    """Only exact fitting constants/copies. Unknown-category flags are protected."""
    fit = np.asarray(fit, bool)
    if not fit.any() or x.shape[1] != len(names):
        raise ValueError('Feature selection requires aligned fitting observations')
    retained, constants, duplicates, buckets = [], {}, {}, {}
    for column, name in enumerate(names):
        vector = np.array(x[fit, column], dtype=np.float32, copy=True)
        protected = name.endswith('__present_UNSEEN')
        if not protected and (np.isnan(vector).all() or np.all(vector == vector[0])):
            constants[name] = None if np.isnan(vector).all() else float(vector[0])
            continue
        vector[vector == 0] = 0
        vector[np.isnan(vector)] = np.nan
        key = hashlib.sha256(vector.tobytes()).hexdigest()
        same = next((i for i in buckets.get(key, [])
                     if np.array_equal(x[fit, i], vector, equal_nan=True)), None)
        if not protected and same is not None:
            duplicates[name] = names[same]
        else:
            retained.append(column)
            buckets.setdefault(key, []).append(column)
    return {'retained': retained, 'constants': constants, 'duplicates': duplicates,
            'fitted_rows': int(fit.sum()), 'scope': 'fit-only exact redundancy; no correlation/MI cutoff',
            'unknown_flags_protected': [n for n in names if n.endswith('__present_UNSEEN')]}


def purge_duplicate_roles(x, roles, views):
    """Common evaluation cohort across feature ablations; fitting is unchanged.

    A row is excluded when any tested view duplicates an earlier-role row.
    Evaluation labels never participate. Same-role duplicates remain visible.
    """
    result = np.array(roles, dtype=np.int8, copy=True)
    keys = [vector_hashes(x, view) for view in views]
    owners = [dict() for _ in keys]
    for role in range(5):
        for i in np.flatnonzero(roles == role):
            if any(bytes(k[i]) in owner and owner[bytes(k[i])] != role
                   for k, owner in zip(keys, owners)):
                result[i] = -1
            # Include excluded rows as conservative earlier-role provenance.
            for k, owner in zip(keys, owners):
                owner.setdefault(bytes(k[i]), role)
    return result, (roles >= 0) & (result < 0)


def _cached_rows(path, keep=None):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if keep is None or keep[index]:
                yield row


def _coverage(labels, roles, classes):
    result = {}
    for code, name in enumerate(ROLES):
        counts = np.bincount(labels[roles == code], minlength=len(classes))
        result[name] = {'rows': int(counts.sum()), 'class_counts': counts.tolist(),
                        'classes_without_support': [classes[i] for i in np.flatnonzero(counts == 0)]}
    return result


def prepare(copy_root=DEFAULT_COPY, source_model=DEFAULT_MODEL):
    """Build and seal a new representation; completed revisions are immutable."""
    from .features import CsvEncoder

    copy_root, source_model = Path(copy_root).resolve(), Path(source_model).resolve()
    folder = copy_root / 'processed'
    if (folder / 'manifest.json').exists():
        existing = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if existing.get('preparation_version') != VERSION:
            raise ValueError('Preparation version changed; use a new dataset copy revision')
        _verify(existing['input_hashes']); _verify(existing['array_hashes'])
        _verify(existing['artifact_hashes'])
        return existing
    folder.mkdir(parents=True, exist_ok=True)
    source = copy_root / 'source/combined_dataset.csv'
    copy_manifest = copy_root / 'copy_manifest.json'
    copied = json.loads(copy_manifest.read_text(encoding='utf-8'))
    if source.stat().st_size != copied['bytes'] or digest(source) != copied['copy_sha256']:
        raise ValueError('Copied original CSV failed integrity')
    if copied['copy_sha256'] != copied['sha256']:
        raise ValueError('Copied CSV is not byte-identical to original receipt')
    old_manifest = json.loads((source_model / 'manifest.json').read_text(encoding='utf-8'))
    classes = old_manifest['classes']
    if old_manifest['benign_index'] != 0 or len(classes) != len(set(classes)):
        raise ValueError('Invalid original class mapping')
    originals = {name: np.load(source_model / f'{name}_train.npy', mmap_mode='r', allow_pickle=False)
                 for name in ('row_ids', 'y', 'window_start', 'window_end')}
    ids, y = originals['row_ids'], originals['y']
    n = len(ids)
    if any(a.shape != (n,) for a in originals.values()) or np.any(np.diff(ids) <= 0):
        raise ValueError('Original training whitelist is not aligned, sorted, unique')
    input_files = [source, copy_manifest, source_model / 'manifest.json'] + [
        source_model / f'{name}_train.npy' for name in originals]
    inputs = {_relative(p): digest(p) for p in input_files}
    receipt = folder / 'extraction.json'
    raw_cache = folder / 'selected_train.csv.gz'
    clean_cache = folder / 'cleaned_train.csv.gz'

    def progress(stage, rows=0):
        _json(folder / 'preparation_state.json', {'stage': stage, 'rows': int(rows),
              'total_rows': n, 'updated_at': time.time(), 'test_arrays_opened': False,
              'validation_arrays_opened': False})
        print(f'CSV preparation: {stage} {rows}/{n}', flush=True)

    if receipt.exists():
        extracted = json.loads(receipt.read_text(encoding='utf-8'))
        if extracted['input_hashes'] != inputs:
            raise ValueError('Incomplete preparation inputs changed')
        _verify(extracted['artifact_hashes'])
    else:
        progress('extracting original training whitelist')
        with source.open(encoding='utf-8-sig', newline='') as stream:
            header = next(csv.reader(stream))
        numeric = [field for field in header if field not in META + LISTS]
        # Strict feature allowlist is independently maintained by the encoder.
        from .features import NUMERIC_FIELDS
        if set(header) != set(META + LISTS + list(NUMERIC_FIELDS)) or set(numeric) != set(NUMERIC_FIELDS):
            raise ValueError('Unexpected raw source schema')
        hard, no_signal = np.zeros(n, bool), np.zeros(n, bool)
        scenarios, codes = [], np.empty(n, np.int32)
        scenario_map, event_counts = {}, Counter()
        raw_part, clean_part = (p.with_name(p.name + '.partial') for p in (raw_cache, clean_cache))
        with gzip.open(raw_part, 'wt', encoding='utf-8', newline='', compresslevel=1) as raw_out, \
             gzip.open(clean_part, 'wt', encoding='utf-8', newline='', compresslevel=1) as clean_out:
            fields = ['source_row_id', *header]
            raw_writer, clean_writer = (csv.DictWriter(s, fields, lineterminator='\n')
                                       for s in (raw_out, clean_out))
            raw_writer.writeheader(); clean_writer.writeheader()
            seen = 0
            for i, (source_id, row) in enumerate(selected_rows(source, ids)):
                clean, facts, _, events = clean_row(row, numeric, LISTS)
                if row['label4'] != classes[int(y[i])] or facts['start'] != originals['window_start'][i] \
                        or facts['end'] != originals['window_end'][i]:
                    raise ValueError(f'Original training label/window join failed at {source_id}')
                hard[i], no_signal[i] = facts['hard_invalid'], facts['no_signal']
                scenario = row['label_full']
                if scenario not in scenario_map:
                    scenario_map[scenario] = len(scenarios); scenarios.append(scenario)
                codes[i] = scenario_map[scenario]
                event_counts.update(event['rule'] for event in events)
                raw_writer.writerow({'source_row_id': source_id, **row})
                clean_writer.writerow({'source_row_id': source_id, **clean})
                seen += 1
                if seen % 25000 == 0:
                    progress('extracting original training whitelist', seen)
        if seen != n:
            raise ValueError('Selected row count changed')
        os.replace(raw_part, raw_cache); os.replace(clean_part, clean_cache)
        for name, a in [('hard_invalid', hard), ('no_signal', no_signal), ('scenario_codes', codes)]:
            np.save(folder / f'{name}.npy', a, allow_pickle=False)
        extracted = {'input_hashes': inputs, 'rows': n, 'scenario_names': scenarios,
                     'row_repair_event_counts': dict(event_counts),
                     'artifact_hashes': {_relative(p): digest(p) for p in
                        [raw_cache, clean_cache, folder / 'hard_invalid.npy',
                         folder / 'no_signal.npy', folder / 'scenario_codes.npy']}}
        _json(receipt, extracted)

    hard, no_signal, scenarios = [np.load(folder / f'{name}.npy', allow_pickle=False)
                                  for name in ('hard_invalid', 'no_signal', 'scenario_codes')]
    observable = ~hard & ~no_signal
    roles_before, split_report = split_roles(y, scenarios, originals['window_start'],
                                            originals['window_end'], observable)
    vocabulary_fit = (roles_before == 0) & observable
    progress('fitting new vocabulary', int(vocabulary_fit.sum()))
    encoder = CsvEncoder.fit(_cached_rows(raw_cache, vocabulary_fit))
    _json(folder / 'encoder.json', encoder.to_dict())
    progress('encoding new features')
    names = encoder.feature_names
    x = np.lib.format.open_memmap(folder / 'X_train.partial.npy', mode='w+', dtype=np.float32,
                                  shape=(n, len(names)))
    for i, row in enumerate(_cached_rows(raw_cache)):
        vector, facts, _ = encoder.encode_row(row)
        if facts['hard_invalid'] != hard[i] or facts['no_signal'] != no_signal[i]:
            raise ValueError('Encoder cleaning disagrees with archived row quality')
        x[i] = vector
        if (i + 1) % 25000 == 0:
            x.flush(); progress('encoding new features', i + 1)
    x.flush(); del x
    os.replace(folder / 'X_train.partial.npy', folder / 'X_train.npy')
    x = np.load(folder / 'X_train.npy', mmap_mode='r', allow_pickle=False)
    progress('quarantining fitting contradictions')
    quality = quarantine_fit(x, y, roles_before, hard, no_signal, n_classes=len(classes))
    progress('selecting fitting-only exact redundancy')
    selection = select_redundancy(x, quality['fit_clean'], names)
    selected = selection['retained']
    base = [i for i in selected if i in set(encoder.base_indices)]
    if not base:
        raise ValueError('No supported baseline features')
    progress('purging duplicate vectors across roles')
    roles_unfiltered, duplicate_purged = purge_duplicate_roles(x, roles_before, [base, selected])
    roles = roles_unfiltered.copy()
    roles[(roles == 0) & ~quality['fit_clean']] = -1
    verify_intervals(roles, scenarios, originals['window_start'], originals['window_end'])
    coverage = _coverage(y, roles, classes)
    for name in ('tune', 'blend', 'calibration'):
        counts = coverage[name]['class_counts']
        if not counts[0] or not sum(counts[1:]):
            raise ValueError(f'{name} has no benign/attack support after purging')
    for name, a in originals.items():
        np.save(folder / f'{name}_train.npy', a, allow_pickle=False)
    for name, a in [('roles', roles), ('roles_unfiltered', roles_unfiltered),
                    ('roles_before_purge', roles_before), ('fit_unfiltered', roles_before == 0),
                    ('fit_clean', quality['fit_clean']), ('conflicting_fit', quality['conflicting_fit']),
                    ('duplicate_purged', duplicate_purged), ('input_sha256', quality['hashes'])]:
        np.save(folder / f'{name}.npy', a, allow_pickle=False)
    _json(folder / 'feature_selection.json', selection)
    with gzip.open(folder / 'row_ledger.csv.gz', 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['source_row_id', 'class_index', 'scenario_code', 'initial_role', 'role',
                         'hard_invalid', 'no_signal', 'conflicting_fit', 'cross_role_duplicate'])
        for i in range(n):
            writer.writerow([ids[i], y[i], scenarios[i], roles_before[i], roles[i], int(hard[i]),
                             int(no_signal[i]), int(quality['conflicting_fit'][i]), int(duplicate_purged[i])])
    result = {'preparation_version': VERSION, 'created_at': time.time(), 'classes': classes,
              'benign_index': 0, 'rows': n, 'feature_names': names,
              'base_feature_indices': base, 'selected_feature_indices': selected,
              'feature_groups': encoder.feature_groups, 'scenario_names': extracted['scenario_names'],
              'coverage': coverage, 'unfiltered_coverage': _coverage(y, roles_unfiltered, classes),
              'partition': split_report, 'vocabulary_fitted_rows': int(vocabulary_fit.sum()),
              'feature_selection_fitted_rows': int(quality['fit_clean'].sum()),
              'quarantine': {'hard_invalid_fit': int(((roles_before == 0) & hard).sum()),
                             'no_signal_fit': int(((roles_before == 0) & no_signal).sum()),
                             'conflicting_fit_rows': int(quality['conflicting_fit'].sum()),
                             'conflicting_fit_groups': quality['conflicting_groups'],
                             'cross_role_duplicate_rows': int(duplicate_purged.sum()),
                             'temporal_boundary_rows': int((roles_before < 0).sum())},
              'row_repair_event_counts': extracted['row_repair_event_counts'],
              'source_copy_verified': True, 'labels_changed': False, 'validation_arrays_opened': False,
              'test_arrays_opened': False,
              'limitations': ['Existing row-local conservative repairs are reused, not newly discovered.',
                  'New fitting-only vocabulary and feature groups require matched retraining evidence.',
                  'Source validation is historically reused and remains unopened by this preparation.',
                  'Scenario challenge support is incomplete; all 84 labels remain in fitting.',
                  'Vocabulary is fit before contradictory-input quarantine; both use fitting rows only.',
                  'Duplicate purging uses both feature views for one common comparison cohort.'],
              'input_hashes': inputs,
              'array_hashes': {_relative(p): digest(p) for p in folder.glob('*.npy')},
              'artifact_hashes': {_relative(p): digest(p) for p in
                    [folder / 'encoder.json', folder / 'feature_selection.json', receipt,
                     folder / 'row_ledger.csv.gz', raw_cache, clean_cache]}}
    _verify(inputs)
    _json(folder / 'manifest.json', result)
    progress('complete', n)
    return result
