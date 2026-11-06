"""Prevent positional joins, unknown-to-zero conversion and assessment leakage."""
import numpy as np
import pytest

from framework.training.broad_live.data import aligned_positions, assert_metadata, select_columns, load_arrays
from framework.training.broad_live import report


def test_identity_join_reorders_and_rejects_duplicates_or_missing_rows():
    np.testing.assert_array_equal(aligned_positions(np.array([9, 2, 7]), np.array([7, 9])), [2, 0])
    with pytest.raises(ValueError, match='unique'):
        aligned_positions(np.array([2, 2]), np.array([2]))
    with pytest.raises(ValueError, match='absent'):
        aligned_positions(np.array([2, 3]), np.array([4]))


def test_join_checks_labels_and_window_metadata():
    target = dict(row_ids=np.array([7, 9]), y=np.array([1, 0]),
                  window_start=np.array([10, 20]), window_end=np.array([20, 30]))
    source = {k: v[::-1].copy() for k, v in target.items()}
    assert_metadata(source, target, np.array([1, 0]))
    source['y'][0] = 4
    with pytest.raises(ValueError, match='y differs'):
        assert_metadata(source, target, np.array([1, 0]))


def test_selected_column_order_preserves_zero_fraction_signed_and_unknown():
    numeric = np.array([[0, -.125], [2, .5]], dtype=np.float32)
    port = np.array([[np.nan, 1], [0, 0]], dtype=np.float32)
    values = select_columns(numeric, ['count', 'delta'], port, ['src', 'dst'], ['dst', 'delta', 'src', 'count'])
    np.testing.assert_equal(values, np.array([[1, -.125, np.nan, 0], [0, .5, 0, 2]], dtype=np.float32))
    with pytest.raises(ValueError, match='zero, one'):
        select_columns(numeric, ['count', 'delta'], port + .1, ['src', 'dst'], ['src'])


def test_test_arrays_have_no_loading_route(tmp_path):
    with pytest.raises(ValueError, match='locked'):
        load_arrays(tmp_path, 'test')


def test_incomplete_selection_fails_before_validation_is_opened(tmp_path, monkeypatch):
    from framework.training.live_study.protocol import atomic
    atomic(tmp_path / 'selected.json', {'branches': {}, 'hybrids': {}})
    def forbidden(*args, **kwargs):
        raise AssertionError('validation must remain unopened')
    monkeypatch.setattr(report, 'load_validation', forbidden)
    monkeypatch.setattr(report, 'verify_study', lambda _: ({'repeat_seeds': [43]}, {}))
    with pytest.raises(ValueError, match='frozen selections'):
        report.assess(tmp_path, tmp_path / 'output')


def test_csv_feature_values_preserve_fraction_and_unknown():
    assert report.csv_features(np.array([0, -.125, np.nan, 1.5], dtype=np.float32)) == [0., -.125, '', 1.5]
