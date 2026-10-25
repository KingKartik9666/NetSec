"""Train-only CPU transforms for the full CSV research study.

Imputation is used only to estimate mutual information. Inference preserves
unobserved measurements as NaN and observed zeroes as zero. Interval column
units are not verified, so no per-second rates are inferred from them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np


def validate_raw(X, feature_names):
    """Check ordered raw numeric schema, including observed binary flags."""
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError('Feature names must be nonempty and unique')
    if hasattr(X, 'columns') and list(X.columns) != list(feature_names):
        raise ValueError('Raw dataframe feature order differs from fitted schema')
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError('Raw feature shape differs from schema')
    if np.isinf(values).any():
        raise ValueError('Raw features contain infinite/out-of-range values')
    for i, name in enumerate(feature_names):
        if '__present_' in name:
            finite = values[np.isfinite(values[:, i]), i]
            if not np.isin(finite, [0, 1]).all():
                raise ValueError('Protocol presence flags must be zero, one or NaN')
    return values


def stratified_sample_indices(labels, cap=30000, seed=42):
    """Deterministic proportional sample with at least one row per class."""
    y = np.asarray(labels)
    if y.ndim != 1 or not len(y) or cap < 1:
        raise ValueError('Nonempty one-dimensional training labels and positive cap required')
    classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    if len(classes) > cap:
        raise ValueError('Sample cap cannot cover every training class')
    if len(y) <= cap:
        return np.arange(len(y), dtype=np.int64)
    quota = np.ones(len(classes), dtype=np.int64)
    remaining = cap - len(classes)
    ideal = (counts - 1) / (len(y) - len(classes)) * remaining
    quota += np.floor(ideal).astype(np.int64)
    for i in np.argsort(-(ideal - np.floor(ideal)), kind='stable')[:cap-int(quota.sum())]:
        quota[i] += 1
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(inverse == i), int(n), replace=False)
                                  for i, n in enumerate(quota)]))


@dataclass
class ResearchTransform:
    feature_names: list[str]
    config: dict
    selected_indices_: np.ndarray | None = field(default=None, init=False, repr=False)

    def _recipes(self):
        names = set(self.feature_names)
        recipes = []
        if not self.config.get('feature_engineering', False):
            return recipes
        total = 'network_packets_all_count'
        subsets = {'network_packets_src_count', 'network_packets_dst_count', 'network_fragmented-packets'}
        for name in self.feature_names:
            if total in names and (name in subsets or
                    (name.startswith('network_tcp-flags-') and name.endswith('_count'))):
                recipes.append(('research__' + name + '_fraction', name, total, 'fraction'))
            if name.endswith('_std_deviation') and name[:-14] + '_avg' in names:
                prefix = name[:-14]
                recipes.append(('research__' + prefix + '_cv', name, prefix + '_avg', 'cv'))
        return recipes

    def _engineer(self, X):
        recipes = self.recipes_
        if not recipes:
            return X.copy()
        result = np.full((len(X), len(self.feature_names) + len(recipes)), np.nan, dtype=np.float32)
        result[:, :len(self.feature_names)] = X
        lookup = {n: i for i, n in enumerate(self.feature_names)}
        for j, (_, numerator, denominator, kind) in enumerate(recipes):
            a, b = X[:, lookup[numerator]], X[:, lookup[denominator]]
            valid = np.isfinite(a) & np.isfinite(b) & (a >= 0) & (b > 0)
            if kind == 'fraction':
                valid &= a <= b
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                ratio = a[valid].astype(np.float64) / b[valid]
            valid_ids = np.flatnonzero(valid)
            bounded = np.isfinite(ratio) & (ratio <= np.finfo(np.float32).max)
            result[valid_ids[bounded], len(self.feature_names)+j] = ratio[bounded]
        return result

    def fit(self, X_train, y_train, seed=42):
        allowed = {'feature_selection', 'feature_engineering'}
        if set(self.config) - allowed:
            raise ValueError('Unsupported transform configuration')
        selection = self.config.get('feature_selection', 'none')
        if selection not in {'none', 'mi64', 'mi128'}:
            raise ValueError('Unsupported feature_selection')
        if not isinstance(self.config.get('feature_engineering', False), bool):
            raise ValueError('feature_engineering must be boolean')
        self.feature_names = list(self.feature_names)
        self.config = dict(self.config)
        X = validate_raw(X_train, self.feature_names)
        y = np.asarray(y_train)
        if y.ndim != 1 or len(y) != len(X) or not len(y):
            raise ValueError('Matching nonempty training labels required')
        self.recipes_ = self._recipes()
        self.all_names_ = self.feature_names + [r[0] for r in self.recipes_]
        self.discrete_features_ = np.array(['__present_' in n for n in self.all_names_], dtype=bool)
        self.selected_indices_ = np.arange(len(self.all_names_), dtype=np.int64)
        self.seed_ = int(seed)
        self.mi_sample_indices_ = np.empty(0, dtype=np.int64)
        self.mi_scores_ = None
        self.mi_imputation_ = None
        if selection != 'none':
            from sklearn.feature_selection import mutual_info_classif
            self.mi_sample_indices_ = stratified_sample_indices(y, 30000, seed)
            sampled = self._engineer(X[self.mi_sample_indices_])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                medians = np.nanmedian(sampled, axis=0)
            medians = np.nan_to_num(medians, nan=0.)
            for i in np.flatnonzero(self.discrete_features_):
                vals = sampled[np.isfinite(sampled[:, i]), i]
                medians[i] = float(np.mean(vals) > .5) if len(vals) else 0.
            self.mi_imputation_ = medians
            imputed = np.where(np.isnan(sampled), medians, sampled).astype(np.float64)
            # Singleton classes are retained in the fitted sample; sklearn's
            # continuous MI estimator excludes their undefined neighbor term.
            self.mi_scores_ = mutual_info_classif(imputed, y[self.mi_sample_indices_],
                discrete_features=self.discrete_features_, random_state=seed, n_jobs=1)
            number = min(int(selection[2:]), len(self.all_names_))
            ranked = np.argsort(-self.mi_scores_, kind='stable')[:number]
            self.selected_indices_ = np.sort(ranked)
        return self

    @property
    def raw_distance_columns(self):
        """Selected raw numeric dimensions for selection-before-resampling."""
        if self.selected_indices_ is None:
            raise RuntimeError('Transform has not been fitted')
        numeric = [i for i, name in enumerate(self.feature_names) if '__present_' not in name]
        selected = [int(i) for i in self.selected_indices_ if i in numeric]
        return selected or numeric

    @property
    def output_names(self):
        if self.selected_indices_ is None:
            raise RuntimeError('Transform must be fitted before inspecting output schema')
        return [self.all_names_[i] for i in self.selected_indices_]

    def transform(self, X):
        if self.selected_indices_ is None:
            raise RuntimeError('Transform has not been fitted')
        values = validate_raw(X, self.feature_names)
        return self._engineer(values)[:, self.selected_indices_].astype(np.float32, copy=False)
