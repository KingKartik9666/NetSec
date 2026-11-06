"""Leakage and provenance checks for the new CSV-only preparation."""
from __future__ import annotations

import numpy as np
import pytest

from framework.training.csv_redesign import data


def test_scenario_holdout_and_temporal_roles_do_not_overlap():
    # Two captures per class, two devices observing each interval. Device rows
    # must share their temporal role rather than crossing a row-random split.
    times = np.repeat(np.arange(100, dtype=np.int64) * 10_000_000_000, 2)
    starts = np.tile(times, 4) + np.repeat(np.arange(4) * 2_000_000_000_000, 200)
    ends = starts + np.tile([1_000_000_000, 15_000_000_000], 400)
    scenarios = np.repeat(np.arange(4), 200)
    labels = scenarios // 2
    roles, report = data.split_roles(labels, scenarios, starts, ends,
                                    np.ones(800, bool))
    assert report['challenge_scenarios'] == [1, 3]
    assert set(roles[scenarios == 1]) == {4}
    assert set(roles[scenarios == 3]) == {4}
    assert set(roles) == {-1, 0, 1, 2, 3, 4}
    data.verify_intervals(roles, scenarios, starts, ends, gap_ns=1_000_000_000)
    for scenario in [0, 2]:
        for left in range(3):
            a = ends[(scenarios == scenario) & (roles == left)]
            b = starts[(scenarios == scenario) & (roles == left + 1)]
            assert a.max() + 1_000_000_000 < b.min()


def test_interval_verifier_rejects_touching_across_devices():
    with pytest.raises(ValueError, match='overlap'):
        data.verify_intervals(np.array([0, 1]), np.array([7, 7]),
                              np.array([0, 5]), np.array([5, 10]), gap_ns=0)


def test_sparse_classes_remain_fitting_and_unsupported_roles_are_explicit():
    roles, report = data.split_roles(np.array([0, 0, 1]), np.array([0, 0, 1]),
                                    np.array([0, 10, 20]), np.array([1, 11, 21]),
                                    np.ones(3, bool))
    assert roles.tolist() == [0, 0, 0]
    assert report['challenge_scenarios'] == []


def test_quality_quarantine_uses_only_fit_labels_and_new_inputs():
    x = np.array([[1, 1], [2, 1], [2, 1], [5, 1], [1, 1], [0, 0]], np.float32)
    y = np.array([0, 0, 1, 1, 1, 0])
    roles = np.array([0, 0, 0, 0, 1, 0], np.int8)
    result = data.quarantine_fit(x, y, roles, np.zeros(6, bool),
                                 np.array([False] * 5 + [True]), n_classes=2)
    assert result['fit_clean'].tolist() == [True, False, False, True, False, False]
    assert result['conflicting_fit'].tolist() == [False, True, True, False, False, False]
    # A differently labelled tuning duplicate must not quarantine fitting row0.
    assert result['fit_clean'][0]
    np.testing.assert_array_equal(y, [0, 0, 1, 1, 1, 0])


def test_quality_filter_refuses_to_remove_last_class_support():
    with pytest.raises(ValueError, match='class support'):
        data.quarantine_fit(np.array([[1], [1]], np.float32), np.array([0, 1]),
                            np.array([0, 0]), np.zeros(2, bool),
                            np.zeros(2, bool), n_classes=2)


def test_feature_redundancy_fitted_only_on_fit_and_protects_unknown_flags():
    x = np.array([[1, 1, 0, np.nan, 0], [2, 2, 0, np.nan, 0],
                  [7, 8, 8, 9, 1]], np.float32)
    names = ['a', 'copy_a', 'constant', 'missing', 'protocol__present_UNSEEN']
    result = data.select_redundancy(x, np.array([True, True, False]), names)
    assert result['retained'] == [0, 4]
    assert result['duplicates'] == {'copy_a': 'a'}
    assert set(result['constants']) == {'constant', 'missing'}


def test_duplicate_purge_uses_union_of_views_and_preserves_labels():
    # Full features separate rows0/1, but base view does not. Both comparisons
    # must use the same eligible tuning rows.
    x = np.array([[1, 1], [1, 2], [3, 3], [4, 4], [3, 9]], np.float32)
    roles = np.array([0, 1, 1, 2, 4], np.int8)
    result, excluded = data.purge_duplicate_roles(x, roles, [[0], [0, 1]])
    assert result.tolist() == [0, -1, 1, 2, -1]
    assert excluded.tolist() == [False, True, False, False, True]
    assert roles.tolist() == [0, 1, 1, 2, 4]


def test_nonfinite_vectors_canonicalize_nan_and_signed_zero():
    x = np.array([[np.nan, -0.0], [np.nan, 0.0]], np.float32)
    result, _ = data.purge_duplicate_roles(x, np.array([0, 4]), [[0, 1]])
    assert result.tolist() == [0, -1]
