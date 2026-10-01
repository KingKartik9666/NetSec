import numpy as np

from framework.data.research_data import prevent_duplicate_crossing, stratified_cap, temporal_partitions


def test_overlapping_windows_do_not_cross_temporal_boundaries():
    labels = np.repeat([0, 1], 100)
    starts = np.tile(np.arange(100), 2)
    ends = starts + 4
    split, _ = temporal_partitions(labels, starts, ends)
    for label in [0, 1]:
        assert ends[(labels == label) & (split == 0)].max() < starts[(labels == label) & (split == 1)].min()
        assert ends[(labels == label) & (split == 1)].max() < starts[(labels == label) & (split == 2)].min()
    assert (split == -1).any()


def test_feature_duplicates_are_excluded_from_later_splits():
    split, count = prevent_duplicate_crossing(np.array([0, 1, 2, 1, -1]), np.array([5, 5, 7, 7, 8]))
    np.testing.assert_equal(split, [0, -1, -1, 1, -1])
    assert count == 2


def test_stratified_cap_retains_rare_class_and_is_deterministic():
    labels = np.array([0] * 1000 + [1] * 2 + [2])
    ids = np.arange(len(labels))
    a = stratified_cap(ids, labels, 30, 42)
    b = stratified_cap(ids, labels, 30, 42)
    np.testing.assert_equal(a, b)
    assert len(a) == 30 and set(labels[a]) == {0, 1, 2}
