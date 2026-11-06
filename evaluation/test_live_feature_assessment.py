"""Probability interpretation and missing-support stress must stay explicit."""
import numpy as np
import pytest

from framework.training.live_features.assessment import (
    probability_report, stress_missing_mss, wilson_interval,
)


def test_binary_and_family_scores_sum_probabilities_before_deciding():
    classes = ['benign', 'dos_tcp', 'dos_udp', 'recon_scan']
    # The most probable individual class is benign, while most probability is
    # assigned to attack and the combined DoS family is more probable still.
    report = probability_report(np.array([1]), np.array([[.4, .3, .3, 0.]]), classes)
    assert report['multiclass']['accuracy'] == 0
    assert report['family']['accuracy'] == 1
    assert report['binary']['recall'] == 1
    assert report['binary']['false_positive_rate'] is None
    assert report['binary']['benign_windows'] == 0


def test_missing_mss_stress_changes_only_zero_support_summaries():
    names = ['network_mss_avg', 'network_mss_max', 'network_mss_min',
             'network_mss_std_deviation', 'network_window-size_avg']
    source = np.array([[0, 0, 0, 0, 0], [1400, 1460, 1340, 60, 0]], np.float32)
    changed, affected = stress_missing_mss(source, names)
    assert affected == 1
    assert np.isnan(changed[0, :4]).all()
    np.testing.assert_array_equal(changed[1], source[1])
    assert changed[0, 4] == 0 and not np.isnan(source).any()


def test_zero_observed_false_alarms_does_not_mean_zero_upper_bound():
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0)
    assert .25 < high < .3
    assert wilson_interval(0, 0) is None
