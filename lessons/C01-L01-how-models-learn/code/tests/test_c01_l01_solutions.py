"""The exercise solutions are correct (exercises.md / solutions.md)."""
import numpy as np
import pytest

import data
from evaluate import closed_form_fit
from model import mse_gradient, numerical_gradient
from solutions_c01_l01 import (buggy_descent, fixed_descent, level2_answers, level5_sweeps,
                               level7_compare, mse_gradient_no_numpy)


def test_level2_hand_numbers():
    a = level2_answers()
    assert a["predictions"] == [1.0, 3.0, 5.0] and a["errors"] == [0.0, 0.0, -3.0]
    assert a["mse"] == pytest.approx(3.0)
    assert (a["grad_w"], a["grad_b"]) == pytest.approx((-4.0, -2.0))
    assert (a["w1"], a["b1"]) == pytest.approx((2.4, 1.2))
    assert a["mse1"] == pytest.approx(4.4 / 3)


def test_level3_loop_gradient_matches_numpy_and_finite_difference():
    x, y = data.make_study_data()
    for w, b in [(0.0, 0.0), (4.0, 41.0), (-2.0, 7.5)]:
        loop = mse_gradient_no_numpy(list(x), list(y), w, b)
        assert loop == pytest.approx(mse_gradient(x, y, w, b), rel=1e-12)
        assert np.allclose(loop, numerical_gradient(x, y, w, b), rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        mse_gradient_no_numpy([], [], 0.0, 0.0)


def test_level4_buggy_loop_fails_and_fixed_loop_matches_closed_form():
    x, y = data.make_study_data()
    x_train, y_train, *_ = data.split_train_test(x, y)
    scaled = (x_train - x_train.mean()) / x_train.std()
    w_ls, b_ls = closed_form_fit(scaled, y_train)
    w_fix, b_fix = fixed_descent(list(scaled), list(y_train), 0.1, 200)
    assert (w_fix, b_fix) == pytest.approx((w_ls, b_ls), abs=1e-6)
    w_bug, b_bug = buggy_descent(list(scaled), list(y_train), 0.1, 200)
    assert not np.allclose([w_bug, b_bug], [w_ls, b_ls], atol=1.0)


def test_level5_measured_edge_agrees_with_curvature_bound():
    s = level5_sweeps()
    raw = {r["rate"]: r for r in s["raw"]}
    scaled = {r["rate"]: r for r in s["scaled"]}
    assert not raw[0.021]["diverged"] and raw[0.022]["diverged"]
    assert 0.021 < s["raw_bound"] < 0.022
    assert s["scaled_bound"] == pytest.approx(1.0)
    assert not scaled[0.99]["diverged"] and scaled[1.01]["diverged"]
    assert scaled[0.5]["steps_to_target"] == 1          # on a round bowl, rate 0.5 lands in a single step
    assert scaled[0.1]["steps_to_target"] == scaled[0.9]["steps_to_target"]


def test_level7_line_underfits_a_curve_and_squared_feature_fixes_it():
    r = level7_compare()
    assert r["quad_train"] < r["line_train"] / 2       # the line can't follow the curve even on training data
    assert r["quad_test"] < r["line_test"] / 2
