"""Gradient descent: convergence, learning-rate behaviour, divergence, and edge cases."""
import numpy as np
import pytest

import data
from evaluate import closed_form_fit
from gradient_descent import (first_step_within, gradient_descent, learning_rate_sweep, losses,
                              standardize, unstandardize_params)
from model import loss


@pytest.fixture(scope="module")
def train():
    x, y = data.make_study_data()
    x_train, y_train, *_ = data.split_train_test(x, y)
    return x_train, y_train


def test_good_rate_loss_never_increases(train):
    x, y = train
    run = gradient_descent(x, y, 0.02, 1000)
    curve = np.array(losses(run))
    assert not run["diverged"]
    assert np.all(np.diff(curve) <= 1e-9)


def test_trajectory_records_every_step(train):
    x, y = train
    run = gradient_descent(x, y, 0.01, 50)
    assert run["steps_run"] == 50 and len(run["history"]) == 51
    assert run["history"][0][:3] == (0, 0.0, 0.0)


def test_zero_learning_rate_changes_nothing(train):
    x, y = train
    run = gradient_descent(x, y, 0.0, 25, w0=1.5, b0=-2.0)
    assert (run["w"], run["b"]) == (1.5, -2.0)
    assert len(set(losses(run))) == 1


def test_zero_steps_returns_the_start(train):
    x, y = train
    run = gradient_descent(x, y, 0.1, 0, w0=3.0, b0=4.0)
    assert run["steps_run"] == 0 and (run["w"], run["b"]) == (3.0, 4.0)


def test_negative_inputs_are_rejected(train):
    x, y = train
    with pytest.raises(ValueError):
        gradient_descent(x, y, -0.1, 10)
    with pytest.raises(ValueError):
        gradient_descent(x, y, 0.1, -1)


def test_too_large_rate_is_detected_as_divergence(train):
    x, y = train
    run = gradient_descent(x, y, 0.025, 1000)
    assert run["diverged"]
    assert run["steps_run"] < 1000                    # stopped early instead of overflowing
    assert np.all(np.diff(losses(run)) > 0)          # every step made it worse


def test_too_small_rate_is_slow_but_safe(train):
    x, y = train
    best = loss(x, y, *closed_form_fit(x, y))
    run = gradient_descent(x, y, 0.001, 1000)
    assert not run["diverged"]
    assert first_step_within(run, best) is None       # 1000 steps were not enough
    assert losses(run)[-1] < losses(run)[0]           # but it did go downhill


def test_sweep_orders_small_good_big(train):
    x, y = train
    best = loss(x, y, *closed_form_fit(x, y))
    rows = {r["rate"]: r for r in learning_rate_sweep(x, y, [0.001, 0.02, 0.025], 1000, best)}
    assert rows[0.001]["steps_to_target"] is None and not rows[0.001]["diverged"]
    assert rows[0.02]["steps_to_target"] is not None
    assert rows[0.025]["diverged"]


def test_standardising_speeds_up_descent(train):
    x, y = train
    best = loss(x, y, *closed_form_fit(x, y))
    raw_steps = first_step_within(gradient_descent(x, y, 0.02, 1000), best)
    xs, mean, std = standardize(x)
    scaled_steps = first_step_within(gradient_descent(xs, y, 0.1, 100), best)
    assert scaled_steps * 10 < raw_steps


def test_standardize_gives_mean_zero_spread_one_and_round_trips(train):
    x, y = train
    xs, mean, std = standardize(x)
    assert abs(xs.mean()) < 1e-12 and xs.std() == pytest.approx(1.0)
    w, b = unstandardize_params(2.0 * std, 3.0 + 2.0 * mean, mean, std)
    assert (w, b) == pytest.approx((2.0, 3.0))


def test_constant_feature_edge_case():
    # Every student studied 3 hours: the line can only learn the average score.
    x = np.array([3.0, 3.0, 3.0])
    y = np.array([1.0, 2.0, 6.0])
    with pytest.raises(ValueError):
        standardize(x)
    run = gradient_descent(x, y, 0.01, 5000)
    assert run["w"] * 3.0 + run["b"] == pytest.approx(y.mean(), abs=1e-6)
    assert losses(run)[-1] == pytest.approx(np.var(y), abs=1e-6)


def test_single_point_is_fitted_exactly():
    run = gradient_descent(np.array([2.0]), np.array([7.0]), 0.05, 2000)
    assert run["w"] * 2.0 + run["b"] == pytest.approx(7.0, abs=1e-6)
    assert losses(run)[-1] < 1e-10
