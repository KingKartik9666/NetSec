"""Train-fitted DataSense transforms; masks preserve unknown versus absent."""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.special import ndtri


def is_indicator(name: str) -> bool:
    return '__present_' in name or '__membership_incomplete' in name or name.startswith('quality_')


class TabMPreprocessor:
    def __init__(self, scaling='log_robust', n_bins=16):
        if scaling not in ('log_robust', 'quantile') or n_bins < 2:
            raise ValueError('Unsupported scaling or bin count')
        self.scaling, self.n_bins = scaling, int(n_bins)

    def fit(self, X, feature_names):
        if X.ndim != 2 or X.shape[1] != len(feature_names) or not len(X):
            raise ValueError('Training feature schema mismatch')
        if len(set(feature_names)) != len(feature_names):
            raise ValueError('Duplicate feature schema names')
        self.feature_names = list(feature_names)
        self.columns, self.embedded_indices, self.bins = [], [], []
        for i, name in enumerate(self.feature_names):
            values = np.asarray(X[:, i], dtype=np.float64)
            finite = values[np.isfinite(values)]
            flag = is_indicator(name)
            if flag and np.any((finite != 0) & (finite != 1)):
                raise ValueError(f'Nonbinary indicator: {name}')
            column = dict(name=name, flag=flag, missing=int(len(values)-len(finite)),
                          zero_count=int(np.sum(finite == 0)),
                          minimum=float(finite.min()) if finite.size else None,
                          maximum=float(finite.max()) if finite.size else None,
                          median=float(np.median(finite)) if finite.size and not flag else 0.,
                          center=0., scale=1., knots=[], quantiles=[])
            if finite.size and not flag:
                if self.scaling == 'log_robust':
                    logged = np.sign(finite)*np.log1p(np.abs(finite))
                    low, center, high = np.quantile(logged, [.25, .5, .75])
                    column['center'] = float(center)
                    # A zero IQR must not push distinct sparse training values into saturation.
                    extent = np.max(np.abs(logged-center))
                    column['scale'] = float(max(high-low, np.std(logged)*.1, extent/25., 1e-6))
                else:
                    qs = np.linspace(0, 1, 257)
                    knots, inverse = np.unique(np.quantile(finite, qs), return_inverse=True)
                    positions = np.bincount(inverse, weights=qs)/np.bincount(inverse)
                    column.update(knots=knots.tolist(), quantiles=positions.tolist())
            self.columns.append(column)
            if finite.size and not flag and column['minimum'] != column['maximum']:
                raw_bins = np.quantile(finite, np.linspace(0, 1, self.n_bins+1))
                # Preserve rare nonzero events when the mass at zero collapses quantiles.
                nonzero = finite[finite != 0]
                if column['zero_count'] > len(finite)/2 and len(nonzero):
                    raw_bins = np.unique(np.concatenate(([finite.min(),0.,finite.max()],
                        np.quantile(nonzero,np.linspace(0,1,max(2,self.n_bins-2))))))
                bins = np.unique(self._transform_column(raw_bins, column).astype('float32'))
                # Two-valued numerical fields retain their scalar path and missing mask.
                if len(bins) >= 3:
                    self.embedded_indices.append(i)
                    self.bins.append(bins.tolist())
        self.bypass_indices = [i for i in range(2*len(self.feature_names)) if i not in self.embedded_indices]
        return self

    def _transform_column(self, values, column):
        values = np.asarray(values, dtype=np.float64)
        clean = np.where(np.isfinite(values), values, column['median'])
        if column['flag']:
            return clean
        if self.scaling == 'log_robust':
            transformed = (np.sign(clean)*np.log1p(np.abs(clean))-column['center'])/column['scale']
            return np.clip(transformed, -30., 30.)  # Explicit model-side saturation, no source edits.
        if len(column['knots']) < 2:
            # Preserve deviations from training constants, rather than dropping the column.
            return np.clip(clean-column['median'], -30., 30.)
        uniform = np.interp(clean, column['knots'], column['quantiles'])
        return ndtri(np.clip(uniform, 1e-4, 1-1e-4))

    def transform(self, X, feature_names=None):
        if feature_names is not None and list(feature_names) != self.feature_names:
            raise ValueError('Input feature schema order differs from fitted schema')
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError('Input feature schema shape mismatch')
        result = np.empty((len(X), 2*X.shape[1]), dtype='float32')
        result[:, :X.shape[1]] = np.where(np.isfinite(X), X, 0.)
        result[:, X.shape[1]:] = ~np.isfinite(X)
        for i, column in enumerate(self.columns):
            values = np.asarray(X[:, i])
            if column['flag'] and np.any(np.isfinite(values) & (values != 0) & (values != 1)):
                raise ValueError(f'Nonbinary indicator: {column["name"]}')
            if not column['flag']:
                result[:, i] = self._transform_column(values, column)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite transformed values')
        return result

    def to_dict(self):
        return deepcopy(dict(version=2, scaling=self.scaling, n_bins=self.n_bins,
            feature_names=self.feature_names, columns=self.columns,
            embedded_indices=self.embedded_indices, bypass_indices=self.bypass_indices, bins=self.bins))

    @classmethod
    def from_dict(cls, state):
        if state['version'] != 2:
            raise ValueError('Unsupported preprocessing version')
        obj = cls(state['scaling'], state['n_bins'])
        for key in ('feature_names', 'columns', 'embedded_indices', 'bypass_indices', 'bins'):
            setattr(obj, key, deepcopy(state[key]))
        return obj
