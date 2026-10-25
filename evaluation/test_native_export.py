"""Offline conversion must preserve decisions and avoid unsafe model formats."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_tensor_blob_preserves_dtype_shape_and_checks_range(tmp_path):
    from tools.native_export.export import TensorWriter, read_tensor
    values = np.array([[1.25, -2.5], [3.75, 9]], dtype=np.float64)
    with TensorWriter(tmp_path / 'weights.bin') as writer:
        tensor = writer.add('numbers', values)
    np.testing.assert_array_equal(read_tensor(tmp_path / 'weights.bin', tensor), values)
    bad = dict(tensor, offset_bytes=10**8)
    with pytest.raises(ValueError, match='range'):
        read_tensor(tmp_path / 'weights.bin', bad)


def test_sparse_forest_export_preserves_mixed_leaf_probabilities(tmp_path):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from tools.native_export.export import TensorWriter, export_forest, forest_oracle
    X = np.array([[0., np.nan], [1., 2.], [2., 3.], [3., 1.], [4., 3.], [5., 2.]], dtype=np.float32)
    y = np.array([0, 1, 2, 2, 1, 0])
    fitted = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                       ('classifier', RandomForestClassifier(n_estimators=3, max_depth=1, random_state=42))]).fit(X, y)
    with TensorWriter(tmp_path / 'weights.bin') as writer:
        description = export_forest(fitted, writer)
        tensors = dict(writer.tensors)
    actual = forest_oracle(X, description, tensors, tmp_path / 'weights.bin')
    np.testing.assert_allclose(actual, fitted.predict_proba(X), rtol=1e-14, atol=1e-14)


def test_validation_samples_are_stratified_without_test_access():
    from tools.native_export.export import sample_indices
    labels = np.repeat(np.arange(7), [2, 5, 8, 11, 4, 10, 9])
    selected = sample_indices(labels, 16)
    assert len(selected) == len(set(selected.tolist())) == 16
    assert set(labels[selected]) == set(labels)
    np.testing.assert_array_equal(selected, sample_indices(labels, 16))
