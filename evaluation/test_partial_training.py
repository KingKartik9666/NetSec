"""Numerical and provenance checks for the optional partial-input CUDA study."""
import numpy as np
import pytest

from framework.training.partial_objectives import smoothed_derivatives, smoothed_loss


@pytest.mark.parametrize("smoothing", [0., .02, .05])
def test_smoothed_gradient_and_true_diagonal_match_finite_difference(smoothing):
    rng = np.random.default_rng(817)
    margins = rng.normal(size=(4, 5))
    labels = np.array([0, 4, 1, 3])
    weights = np.array([.2, 1., 2.3, 0.])
    gradient, diagonal, upper = smoothed_derivatives(margins, labels, weights, smoothing)
    numeric_gradient = np.empty_like(margins)
    numeric_diagonal = np.empty_like(margins)
    delta = 1e-4
    base = smoothed_loss(margins, labels, weights, smoothing)
    for index in np.ndindex(margins.shape):
        plus, minus = margins.copy(), margins.copy()
        plus[index] += delta
        minus[index] -= delta
        a = smoothed_loss(plus, labels, weights, smoothing)
        b = smoothed_loss(minus, labels, weights, smoothing)
        numeric_gradient[index] = (a-b)/(2*delta)
        numeric_diagonal[index] = (a-2*base+b)/delta**2
    np.testing.assert_allclose(gradient, numeric_gradient, atol=1e-8)
    np.testing.assert_allclose(diagonal, numeric_diagonal, atol=1e-6)
    np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=1e-14)
    assert np.all(upper >= diagonal)
    assert np.all(upper > 0)


def test_extreme_margins_are_finite_and_zero_smoothing_is_ce():
    z = np.array([[1000., -1000., 0.], [-1000., 1000., 0.]])
    y = np.array([1, 1])
    assert smoothed_loss(z, y, smoothing=0) == 2000.
    assert all(np.isfinite(a).all() for a in smoothed_derivatives(z, y))
    np.testing.assert_allclose(smoothed_derivatives(z + 9876, y)[0], smoothed_derivatives(z, y)[0])


@pytest.mark.parametrize("labels,weights,smoothing", [([3], None, .05), ([.5], None, .05), ([0], [-1], .05), ([0], None, 1.)])
def test_invalid_objective_inputs_are_rejected(labels, weights, smoothing):
    with pytest.raises(ValueError):
        smoothed_derivatives([[0., 1., 2.]], np.array(labels), weights, smoothing)


def test_masks_preserve_observed_values_and_missing_protocol_fields():
    from experiment.train_partial_xgboost import masked_live, random_partial
    features = ["a", "network_protocols_all__present_tcp", "network_protocols_all__present_udp", "b"]
    rows = np.arange(80, dtype=np.float32).reshape(20, 4)
    live = masked_live(rows, [0])
    np.testing.assert_array_equal(live[:, 0], rows[:, 0])
    assert np.isnan(live[:, 1:]).all()
    masked = random_partial(rows, features, .4, 17)
    observed = np.isfinite(masked)
    np.testing.assert_array_equal(masked[observed], rows[observed])
    np.testing.assert_array_equal(observed[:, 1], observed[:, 2])
    np.testing.assert_array_equal(masked, random_partial(rows, features, .4, 17))
    assert observed.any(axis=1).all()


def test_frozen_study_has_twelve_distinct_bounded_configurations():
    from experiment.train_partial_xgboost import configurations
    configs = configurations()
    assert len(configs) == len({c['id'] for c in configs}) == 12
    assert {c['warm_start'] for c in configs} == {True, False}
    assert {c['weight_strategy'] for c in configs} == {'none', 'sqrt_balanced', 'balanced'}
    assert {c['loss'] for c in configs} == {'native_ce', 'label_smoothed_ce'}
