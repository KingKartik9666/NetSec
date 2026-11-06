"""Freeze the selected broad representation using original training rows only.

Validation is assembled by ``report.assess`` after model selection is frozen.
The historic seven-count and feature-selection manifests are never rewritten.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from framework.training.live_study.protocol import (
    ROOT, ROLES, atomic, digest, fingerprint, read, relative, verify,
)
from framework.training.live_features import ports
from framework.training.live_features.schema import core_names

FEATURE_STUDY = ROOT / 'artifacts/training/live_features_20261027_v1'
PREVIOUS = ROOT / 'artifacts/training/live_models_20261025_v1/data'
NUMERIC = ROOT / 'dataset/prepared/datasense_full_train'
CLEAN = ports.CLEAN
CLEAN_HASHES = CLEAN.parent / 'verification/output_hashes.json'
PROFILE = FEATURE_STUDY / 'training_profile.json'
EXPECTED_ROLE_ROWS = dict(zip(ROLES, (11716, 954, 945, 1637)))
WARNING = (
    'Exploratory within-capture original CSV validation, historically reused. '
    'Broad profile selection reused the training tune role. Port vocabulary was '
    'built from broader original training, not fold-local discovery. Results are '
    'not labelled live Npcap accuracy or publisher-extractor equivalence. '
    'HEM is a closed-set upstream-derived adaptation, not real HEM reproduction. '
    'No held-out test evaluation or deployment promotion.'
)


def unique_names(names):
    if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError('Feature names must be nonempty, unique strings')
    return {name: i for i, name in enumerate(names)}


def aligned_positions(source_ids, target_ids):
    """Exact, one-to-one identity join; positional ordering is never assumed."""
    source_ids, target_ids = np.asarray(source_ids), np.asarray(target_ids)
    for ids in (source_ids, target_ids):
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids):
            raise ValueError('Row IDs must be unique one-dimensional integer arrays')
    lookup = {int(value): i for i, value in enumerate(source_ids)}
    try:
        return np.asarray([lookup[int(value)] for value in target_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f'Required source row ID is absent: {error.args[0]}') from error


def load_arrays(folder, split):
    """There is deliberately no held-out-test loading route."""
    if split not in ('train', 'validation'):
        raise ValueError('Held-out test is locked')
    return {name: np.load(Path(folder) / f'{name}_{split}.npy', mmap_mode='r', allow_pickle=False)
            for name in ('X', 'y', 'row_ids', 'window_start', 'window_end')}


def validate_arrays(arrays, names, classes, *, exact_windows=True):
    unique_names(names)
    x, y = arrays['X'], arrays['y']
    count = len(y)
    if x.dtype != np.float32 or x.shape != (count, len(names)):
        raise ValueError('Feature array must be aligned float32 in the declared column order')
    if y.shape != (count,) or y.dtype.kind not in 'iu' or np.any((y < 0) | (y >= len(classes))):
        raise ValueError('Labels must be integer indexes in the declared class vocabulary')
    aligned_positions(arrays['row_ids'], arrays['row_ids'])
    for name in ('row_ids', 'window_start', 'window_end'):
        value = arrays[name]
        if value.shape != (count,) or value.dtype.kind not in 'iu':
            raise ValueError(f'{name} must contain aligned integer metadata')
    if exact_windows and np.any(arrays['window_end'] - arrays['window_start'] != 10_000_000_000):
        raise ValueError('Every admitted row must be an exact ten-second window')


def assert_metadata(source, target, positions):
    for name in ('row_ids', 'y', 'window_start', 'window_end'):
        if not np.array_equal(source[name][positions], target[name]):
            raise ValueError(f'Identity-joined {name} differs between source representations')


def select_columns(numeric, numeric_names, port_values, port_names, selected_names):
    numeric_index, port_index = unique_names(numeric_names), unique_names(port_names)
    unique_names(selected_names)
    if set(numeric_names) & set(port_names):
        raise ValueError('Numeric and port feature names overlap')
    if numeric.ndim != 2 or numeric.shape[1] != len(numeric_names):
        raise ValueError('Numeric feature shape differs from its names')
    if port_values.shape != (len(numeric), len(port_names)):
        raise ValueError('Port rows or columns differ from the numeric representation')
    if not np.all(np.isnan(port_values) | (port_values == 0) | (port_values == 1)):
        raise ValueError('Port indicators must preserve valid zero, one and unknown NaN')
    try:
        result = np.column_stack([
            numeric[:, numeric_index[name]] if name in numeric_index else port_values[:, port_index[name]]
            for name in selected_names
        ]).astype(np.float32)
    except KeyError as error:
        raise ValueError(f'Selected feature is unavailable: {error.args[0]}') from error
    if np.isinf(result).any():
        raise ValueError('Infinite features require an explicit new data revision')
    return result


def _training_pins(pins):
    # A historic manifest also lists validation artifacts: never hash/read those
    # while preparing training, even though their path strings are public metadata.
    return {name: sha for name, sha in pins.items()
            if 'validation' not in Path(name).name and '_test' not in Path(name).name}


def prepare(study, profile_path=PROFILE):
    """Return a live_study-compatible manifest for an immutable 252-column study."""
    study, profile_path = Path(study).resolve(), Path(profile_path).resolve()
    profile = read(profile_path)
    names, classes = profile['feature_names'], profile['classes']
    unique_names(names)
    if len(names) != 252 or profile.get('benign_index') != 0 or classes[0] != 'benign':
        raise ValueError('The frozen broad profile requires 252 features and benign index zero')
    if fingerprint(names) != profile['feature_schema_fingerprint']:
        raise ValueError('Frozen profile feature fingerprint differs')
    if _training_pins(profile['input_hashes']) != profile['input_hashes']:
        raise ValueError('Training profile cannot depend on validation or held-out-test files')
    verify(profile['input_hashes'])
    feature_study = ROOT / profile.get('feature_study_path', relative(FEATURE_STUDY))
    folder = study / 'data'
    if study in (feature_study.resolve(), PREVIOUS.parent.resolve()):
        raise ValueError('Preserve historical studies; use a separate broad training directory')
    if (folder / 'manifest.json').exists():
        existing = read(folder / 'manifest.json')
        if (existing.get('preparation_version') != 1 or existing.get('feature_names') != names
                or existing.get('profile_sha256') != digest(profile_path)):
            raise ValueError('Prepared profile differs; create an explicit new revision')
        verify(existing['array_hashes'])
        verify(existing['input_hashes'])
        return existing
    source_manifest = read(feature_study / 'data/manifest.json')
    protocol_path = feature_study / 'matched_assessment/protocol.json'
    protocol = read(protocol_path)
    if names != protocol['arms']['selected_plus_ports']:
        raise ValueError('Feature order differs from the matched selected_plus_ports arm')
    if protocol['manifest_hash'] != digest(feature_study / 'data/manifest.json'):
        raise ValueError('Feature-study manifest differs from its matched assessment')
    if source_manifest['classes'] != classes or source_manifest['feature_names'] != core_names():
        raise ValueError('Numeric source class or core-feature order differs')
    source_pins = dict(source_manifest['array_hashes'], **_training_pins(source_manifest['input_hashes']))
    verify(source_pins)
    numeric = load_arrays(feature_study / 'data', 'train')
    validate_arrays(numeric, source_manifest['feature_names'], classes)
    roles = np.load(feature_study / 'data/roles.npy', allow_pickle=False)
    if roles.shape != (len(numeric['y']),) or roles.dtype.kind not in 'iu' or not np.isin(roles, [-1, 0, 1, 2, 3]).all():
        raise ValueError('Preserved training roles are invalid')
    old_manifest = read(PREVIOUS / 'manifest.json')
    old_pins = _training_pins(old_manifest['array_hashes'])
    verify(old_pins)
    old = load_arrays(PREVIOUS, 'train')
    assert_metadata(old, numeric, aligned_positions(old['row_ids'], numeric['row_ids']))
    if not np.array_equal(roles, np.load(PREVIOUS / 'roles.npy', allow_pickle=False)):
        raise ValueError('Historical purged role ownership changed')
    coverage = {}
    for number, name in enumerate(ROLES):
        labels = numeric['y'][roles == number]
        counts = np.bincount(labels, minlength=len(classes))
        if len(labels) != EXPECTED_ROLE_ROWS[name]:
            raise ValueError(f'Preserved {name} row count differs')
        coverage[name] = dict(rows=len(labels), class_counts=counts.tolist(),
                              classes_without_support=[classes[i] for i in np.flatnonzero(counts == 0)])
    if coverage['fit']['classes_without_support']:
        raise ValueError('Every declared class requires real fitting support')
    clean_manifest = read(CLEAN / 'manifest.json')
    clean = load_arrays(CLEAN, 'train')
    validate_arrays(clean, clean_manifest['feature_names'], classes, exact_windows=False)
    if clean_manifest['classes'] != classes:
        raise ValueError('Cleaned class vocabulary differs')
    assert_metadata(clean, numeric, aligned_positions(clean['row_ids'], numeric['row_ids']))
    port_values, port_names, port_report = ports.join_port_features(feature_study)
    previous_port_path = feature_study / 'matched_assessment/port_profile.json'
    previous_port = read(previous_port_path)
    if (port_names != previous_port['selected_names'] or len(port_names) != 192
            or port_report['input_hashes'] != previous_port['input_hashes']):
        raise ValueError('Port data, vocabulary or fitting-only selection changed')
    if protocol['port_profile_hash'] != digest(previous_port_path):
        raise ValueError('Matched port profile changed')
    x = select_columns(numeric['X'], source_manifest['feature_names'], port_values, port_names, names)
    source_pins.update(old_pins)
    source_pins.update(port_report['input_hashes'])
    folder.mkdir(parents=True, exist_ok=True)
    arrays = dict(numeric, X=x)
    for name, value in arrays.items():
        np.save(folder / f'{name}_train.npy', value, allow_pickle=False)
    np.save(folder / 'roles.npy', roles, allow_pickle=False)
    small_files = [profile_path, feature_study / 'data/manifest.json', PREVIOUS / 'manifest.json',
                   NUMERIC / 'manifest.json', CLEAN / 'manifest.json', CLEAN_HASHES, ports.VOCABULARY,
                   protocol_path, previous_port_path]
    pins = dict(profile['input_hashes'])
    pins.update({relative(p): digest(p) for p in small_files})
    manifest = dict(preparation_version=1, profile='broad_live_selected252_v1', feature_names=names,
        classes=classes, benign_index=0, feature_schema_fingerprint=fingerprint(names),
        profile_path=relative(profile_path), profile_sha256=digest(profile_path), rows=len(x),
        feature_study_path=relative(feature_study), coverage=coverage, window_seconds=10,
        purged_training_rows=int((roles < 0).sum()),
        role_provenance='Exact historical seven-count IDs, labels, window bounds and purged roles retained',
        zero_and_missing_policy='Valid measured zero preserved; absent or censored values remain NaN, never filled as zero',
        novel_reference_scope='Final assessment retains the conservative historical seven-count novelty mask',
        preparation_source_hashes=source_pins,
        preparation_source_hash_policy='Verified when joined; each fit verifies frozen copied arrays and small provenance pins',
        input_hashes=pins, array_hashes={relative(p): digest(p) for p in folder.glob('*.npy')},
        warning=WARNING, validation_loaded=False, test_loaded=False,
        publisher_equivalence_verified=False, live_accuracy_verified=False, deployment_status='research_candidate_not_promoted')
    atomic(folder / 'manifest.json', manifest)
    return manifest
