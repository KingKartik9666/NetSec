"""CPU toy checks for train-only feature fitting and synthetic provenance."""
import importlib
import importlib.util
import json

import joblib
import numpy as np
import pytest


def module(name):
    path = 'framework.training.' + name
    assert importlib.util.find_spec(path) is not None, f'{path} has not been implemented'
    return importlib.import_module(path)


def test_identity_preserves_missing_zero_and_cpu_reload(tmp_path):
    cls = module('research_features').ResearchTransform
    names = ['network_packets_all_count', 'network_protocols_all__present_tcp']
    X = np.array([[0, 0], [8, 1], [np.nan, np.nan]], dtype=np.float32)
    fitted = cls(names, {}).fit(X, np.array([0, 1, 1]))
    np.testing.assert_array_equal(fitted.transform(X), X)
    joblib.dump(fitted, tmp_path / 'transform.joblib')
    np.testing.assert_array_equal(joblib.load(tmp_path / 'transform.joblib').transform(X), X)
    with pytest.raises(ValueError):
        fitted.transform(X[:, :1])
    with pytest.raises(ValueError):
        fitted.transform(np.array([[np.inf, 1]]))
    assert fitted.output_names == names
    assert fitted.raw_distance_columns == [0]


def test_semantic_ratios_require_observed_positive_denominators():
    cls = module('research_features').ResearchTransform
    names = ['network_packets_all_count', 'network_tcp-flags-syn_count',
             'network_packet-size_min', 'network_packet-size_avg',
             'network_packet-size_max', 'network_packet-size_std_deviation']
    X = np.array([[10, 4, 2, 4, 6, 2], [0, 0, 0, 0, 0, 0],
                  [np.nan, 4, 2, np.nan, 6, 2], [2, 4, 2, 4, 6, 2]], dtype=np.float32)
    fitted = cls(names, {'feature_engineering': True}).fit(X, [0, 0, 1, 1])
    result = fitted.transform(X)
    syn = fitted.output_names.index('research__network_tcp-flags-syn_count_fraction')
    cv = fitted.output_names.index('research__network_packet-size_cv')
    np.testing.assert_array_equal(result[:,:len(names)], X)
    assert result[0, syn] == pytest.approx(.4)
    assert result[0, cv] == pytest.approx(.5)
    assert np.isnan(result[1:, syn]).all()
    assert np.isnan(result[1:3, cv]).all()


def test_mi_train_only_deterministic_sample_keeps_rare_classes():
    cls = module('research_features').ResearchTransform
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30500, 66)).astype(np.float32)
    y = np.repeat([0, 1, 2], [15249, 15250, 1])
    X[:, 0] = y > 0
    X[::5, 1] = np.nan
    names = ['network_protocols_all__present_tcp'] + [f'numeric_{i}' for i in range(65)]
    fitted = cls(names, {'feature_selection': 'mi64'}).fit(X, y, seed=42)
    assert len(fitted.output_names) == 64
    assert fitted.discrete_features_[0]
    assert not fitted.discrete_features_[1:].any()
    assert len(fitted.mi_sample_indices_) == 30000
    assert set(y[fitted.mi_sample_indices_]) == {0, 1, 2}
    before = fitted.selected_indices_.copy()
    transformed = fitted.transform(np.full((3, 66), np.nan, dtype=np.float32))
    assert np.isnan(transformed).all()
    np.testing.assert_array_equal(fitted.selected_indices_, before)
    np.testing.assert_array_equal(fitted.mi_sample_indices_,
        module('research_features').stratified_sample_indices(y, 30000, 42))


@pytest.fixture
def parents():
    names = ['network_packets_all_count', 'network_tcp-flags-syn_count',
             'network_packet-size_min', 'network_packet-size_avg', 'network_packet-size_max',
             'network_protocols_all__present_tcp', 'network_interval-packets', 'unobserved']
    # Same-context minority beside majority supplies actual borderline examples.
    minority = [[10+i, 1, 1, 2, 4, 1, 10, np.nan] for i in range(6)]
    majority = [[10+i/4, 1, 1, 2, 4, 1, 10, np.nan] for i in range(20)]
    incompatible = [[60, 3, 1, 2, 4, 0, 5, np.nan], [70, 3, 1, 2, 4, 0, 6, np.nan]]
    X = np.array(majority + minority + incompatible, dtype=np.float32)
    y = np.array([0]*20 + [1]*6 + [2]*2, dtype=np.int32)
    return X, y, np.arange(100, 128), names


@pytest.mark.parametrize('method', ['none', 'smotenc', 'borderline_smote', 'adasyn'])
def test_augmentation_retains_originals_and_reproducible_training_provenance(parents, tmp_path, method):
    augment = module('research_augmentation').augment_training
    X, y, ids, names = parents
    config = {'method': method, 'max_extra_fraction': .2, 'max_per_class_multiplier': 2, 'k_neighbors': 3}
    Xa, ya, meta = augment(X, y, ids, names, config, 42, tmp_path / 'a')
    Xb, yb, repeat = augment(X, y, ids, names, config, 42, tmp_path / 'b')
    np.testing.assert_array_equal(Xa[:len(X)], X)
    np.testing.assert_array_equal(ya[:len(y)], y)
    np.testing.assert_array_equal(Xa, Xb)
    np.testing.assert_array_equal(ya, yb)
    assert len(Xa)-len(X) <= 5
    assert meta['synthetic_rows'] == len(Xa)-len(X)
    assert meta['validation_used'] is False and meta['test_used'] is False
    assert meta['new_independent_observations'] == 0
    assert meta['status'] == ('not_requested' if method == 'none' else 'applied')
    assert 'provenance.npz' in meta['asset_hashes']
    if method == 'none':
        assert sum(meta['requested_synthetic_counts'].values()) == 0
    if method != 'none':
        assert meta['synthetic_rows'] > 0
        assert meta['skipped_groups']
    provenance = np.load(tmp_path / 'a' / 'provenance.npz')
    lookup = {int(id): i for i, id in enumerate(ids)}
    for j, (left, right) in enumerate(zip(provenance['parent_a_row_id'], provenance['parent_b_row_id'])):
        a, b = lookup[int(left)], lookup[int(right)]
        assert y[a] == y[b] == ya[len(y)+j]
        np.testing.assert_array_equal(X[a, 5:], X[b, 5:])
        np.testing.assert_array_equal(Xa[len(X)+j, 5:], X[a, 5:])
        assert 0 <= Xa[len(X)+j, 1] <= Xa[len(X)+j, 0]
        assert Xa[len(X)+j, 2] <= Xa[len(X)+j, 3] <= Xa[len(X)+j, 4]
    assert meta['asset_hashes'] == repeat['asset_hashes']
    assert json.loads((tmp_path / 'a' / 'metadata.json').read_text())['synthetic_rows'] == meta['synthetic_rows']


def test_augmentation_rejects_bad_schema_ids_config_and_context(parents, tmp_path):
    augment = module('research_augmentation').augment_training
    X, y, ids, names = parents
    for config in [{'method': 'unknown'}, {'max_extra_fraction': -1}, {'k_neighbors': 0}, {'validation': X}]:
        with pytest.raises(ValueError):
            augment(X, y, ids, names, config, 42, tmp_path)
    with pytest.raises(ValueError):
        augment(X, y, np.zeros_like(ids), names, {}, 42, tmp_path)
    bad = X.copy()
    bad[0, 5] = .5
    with pytest.raises(ValueError):
        augment(bad, y, ids, names, {}, 42, tmp_path)


def test_ctgan_requires_actual_cuda_backend_without_cpu_fallback(parents, tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    augment = module('research_augmentation').augment_training
    X, y, ids, names = parents
    with pytest.raises(RuntimeError, match='CUDA'):
        augment(X, y, ids, names, {'method': 'ctgan'}, 42, tmp_path)


def test_no_compatible_neighbors_skips_explicitly_without_random_oversampling(tmp_path):
    augment = module('research_augmentation').augment_training
    X = np.array([[10, 1, 10], [20, 1, 10], [5, 0, 10], [6, 0, 20]], dtype=np.float32)
    y = np.array([0, 0, 1, 2])
    names = ['network_packets_all_count', 'network_protocols_all__present_tcp', 'network_interval-packets']
    result, target, meta = augment(X, y, np.arange(4), names,
        {'method': 'smotenc', 'max_extra_fraction': 1, 'k_neighbors': 1}, 42, tmp_path)
    np.testing.assert_array_equal(result, X)
    np.testing.assert_array_equal(target, y)
    assert meta['status'] == 'skipped'
    assert meta['synthetic_rows'] == 0
    assert all('insufficient' in item['reason'] for item in meta['skipped_groups'])


def test_dataframe_schema_order_and_invalid_feature_config_are_rejected():
    import pandas as pd
    cls = module('research_features').ResearchTransform
    X = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    fitted = cls(['a', 'b'], {}).fit(X, [0, 1])
    with pytest.raises(ValueError):
        fitted.transform(X[['b', 'a']])
    for config in [{'feature_selection': 'random'}, {'feature_engineering': 'yes'}, {'validation': True}]:
        with pytest.raises(ValueError):
            cls(['a', 'b'], config).fit(X, [0, 1])
