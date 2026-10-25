"""Training-only chronological cross-fitting with window and duplicate purging."""
from __future__ import annotations

import numpy as np

from framework.data.research_data import prevent_duplicate_crossing


class IndexedRows:
    """Read columns or batches from a memory map without copying a full fold."""
    ndim = 2

    def __init__(self, values, indices):
        self.values = values
        self.indices = np.asarray(indices, dtype=np.int64)
        self.shape = (len(self.indices), values.shape[1])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, columns = key
            return self.values[self.indices[rows], columns]
        return self.values[self.indices[key]]


def forward_folds(labels, starts, ends, hashes):
    """Return two forward folds; role indices refer to the original train array.

    Early stopping precedes OOF prediction. Boundaries are within each class,
    matching the source's within-capture protocol, not a new device holdout.
    """
    y, s, e, h = map(np.asarray, (labels, starts, ends, hashes))
    if any(a.ndim != 1 or len(a) != len(y) for a in (y, s, e, h)):
        raise ValueError('Fold metadata must be aligned one-dimensional arrays')
    if not len(y) or np.any(e < s) or not np.isfinite(s).all() or not np.isfinite(e).all():
        raise ValueError('Invalid temporal intervals')
    folds = []
    for cutoff, upper in ((.6, .8), (.8, 1.)):
        codes = np.full(len(y), -1, dtype=np.int8)
        for c in np.unique(y):
            ids = np.flatnonzero(y == c)
            times = np.unique(s[ids])
            if len(times) < 10:
                raise ValueError(f'Insufficient distinct timestamps in class {c}')
            a, b = (times[int(len(times)*f)] for f in (cutoff-.1, cutoff))
            codes[ids[e[ids] < a]] = 0
            codes[ids[(s[ids] > a) & (e[ids] < b)]] = 1
            eligible = s[ids] > b
            if upper < 1.:
                eligible &= e[ids] < times[int(len(times)*upper)]
            codes[ids[eligible]] = 2
        codes, _ = prevent_duplicate_crossing(codes, h)
        fold = {name: np.flatnonzero(codes == i)
                for i, name in enumerate(('fit', 'early_stop', 'oof'))}
        if any(not len(ids) for ids in fold.values()):
            raise ValueError('Purging left an empty fold role')
        if not np.array_equal(np.unique(y[fold['fit']]), np.unique(y)):
            raise ValueError('Purging removed a fitting class')
        folds.append(fold)
    if np.intersect1d(folds[0]['oof'], folds[1]['oof']).size:
        raise ValueError('OOF observation identities overlap')
    return folds


def strict_fit_indices(indices, row_ids, strict_row_ids, labels, n_classes):
    """Apply the selected KAN arm without changing the shared OOF targets."""
    chosen = np.asarray(indices)[np.isin(np.asarray(row_ids)[indices], strict_row_ids)]
    if len(np.unique(np.asarray(labels)[chosen])) != n_classes:
        raise ValueError('Strict fitting arm lost class coverage')
    return chosen
