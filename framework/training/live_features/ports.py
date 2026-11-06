"""Join the existing port representation by identity; never infer applications."""
from pathlib import Path

import numpy as np

from framework.training.live_study.protocol import ROOT, read, digest, relative, atomic
from .selection import remove_redundant

CLEAN = ROOT / 'dataset/experiments/datasense_verified_clean_20261014_v2/model'
VOCABULARY = ROOT / 'rust/netsec/crates/runtime/src/live/port_features.json'


def join_port_features(study):
    """Only original training data are accessed. Censored absence remains NaN."""
    study = Path(study)
    source = read(CLEAN / 'manifest.json')
    target = read(study / 'data/manifest.json')
    if source['classes'] != target['classes']:
        raise ValueError('Class order differs between numeric and cleaned representations')
    names = read(VOCABULARY)
    source_ids = np.load(CLEAN / 'row_ids_train.npy')
    target_ids = np.load(study / 'data/row_ids_train.npy')
    lookup = {int(value): i for i, value in enumerate(source_ids)}
    positions = np.array([lookup[int(value)] for value in target_ids])
    for field in ('y', 'window_start', 'window_end'):
        np.testing.assert_array_equal(np.load(CLEAN / f'{field}_train.npy')[positions],
                                      np.load(study / f'data/{field}_train.npy'))
    columns = [source['feature_names'].index(name) for name in names]
    x = np.load(CLEAN / 'X_train.npy', mmap_mode='r')[np.ix_(positions, columns)].copy()
    if not np.all(np.isnan(x) | (x == 0) | (x == 1)):
        raise ValueError('Port membership must preserve 0, 1 and unknown')
    roles = np.load(study / 'data/roles.npy')
    # Vocabulary is an existing broader-training control; selection here only
    # sees the fitting role. No minimum-frequency cutoff deletes rare signals.
    kept, removed = remove_redundant(x[roles == 0], names)
    kept = [i for i in kept if np.any(x[roles == 0, i] == 1)]
    report = dict(candidate_names=names, selected_names=[names[i] for i in kept],
                  removed=removed, selected_count=len(kept), candidate_count=len(names),
                  selection_scope='Fitting-role support, constants and exact duplicates only',
                  missing_policy='Observed token 1; complete absence 0; censored unknown NaN',
                  vocabulary_provenance='Existing full-training vocabulary: exploratory control, not a new fold-local vocabulary',
                  publisher_equivalence_verified=False, validation_loaded=False, test_loaded=False,
                  input_hashes={relative(p): digest(p) for p in (
                      CLEAN / 'manifest.json', CLEAN / 'X_train.npy', CLEAN / 'row_ids_train.npy',
                      CLEAN / 'y_train.npy', CLEAN / 'window_start_train.npy', CLEAN / 'window_end_train.npy', VOCABULARY)})
    return x[:, kept], [names[i] for i in kept], report
