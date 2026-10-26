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


# ---------------------------------------------------------------- added in round 3 (exercises split by part)
from by_hand import nudge_table  # noqa: E402
from gradient_descent import gradient_descent  # noqa: E402
from model import loss  # noqa: E402
from solutions_c01_l01 import (buggy_loss_and_step, fixed_loss_and_step, gradient_descent_no_numpy,  # noqa: E402
                               mse_no_numpy, nudge_error_analysis, nudge_ratio_no_numpy, scaling_answers)


def test_part1_level3_loop_loss_and_gradient_match_numpy_and_the_nudge():
    x, y = data.make_study_data()
    for w, b in [(0.0, 0.0), (4.0, 41.0), (-2.0, 7.5)]:
        assert mse_no_numpy(list(x), list(y), w, b) == pytest.approx(loss(x, y, w, b), rel=1e-12)
        loop = mse_gradient_no_numpy(list(x), list(y), w, b)
        assert loop == pytest.approx(mse_gradient(x, y, w, b), rel=1e-12)
        assert np.allclose(loop, nudge_ratio_no_numpy(list(x), list(y), w, b, 1e-4), atol=0.01)
    assert nudge_ratio_no_numpy([1.0], [2.0], 0.0, 0.0, 0.5)[0] == pytest.approx(
        nudge_table([1.0], [2.0], 0.0, 0.0, [0.5])[0]["ratio"])
    with pytest.raises(ValueError):
        mse_no_numpy([], [], 0.0, 0.0)


def test_part1_level4_buggy_step_is_wrong_and_fixed_step_matches_the_lesson():
    hours, scores = list(data.TOY_HOURS), list(data.TOY_SCORES)
    assert buggy_loss_and_step(hours, scores, 5.0, 45.0, 0.1) == pytest.approx((-2.0, 4.2, 45.8))
    assert fixed_loss_and_step(hours, scores, 5.0, 45.0, 0.1) == pytest.approx((6.0, 5.8, 45.4))


def test_part1_level5_nudge_gap_is_h_times_mean_x_squared():
    a = nudge_error_analysis()
    assert round(a["exact_slope"]) == -435 and round(a["mean_x_squared"], 2) == 45.05
    assert [round(r["ratio"], 1) for r in a["rows"]] == [-390.2, -430.8, -434.8, -435.2]
    for r in a["rows"]:
        assert r["gap_over_h"] == pytest.approx(a["mean_x_squared"], rel=1e-6)


def test_part2_level2_scaling_by_hand():
    a = scaling_answers()
    assert (a["mean"], a["std"]) == (5.0, 2.0)
    assert a["scaled"] == [-1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 2.0]
    assert (a["w"], a["b"]) == (3.0, 55.0) and a["scaled_x_at"] == 1.0
    assert a["predict_scaled"] == a["predict_raw"] == 76.0


def test_part2_level3_list_loop_matches_the_lesson_loop():
    x, y = data.make_study_data()
    x_train, y_train, *_ = data.split_train_test(x, y)
    mine = gradient_descent_no_numpy(list(x_train), list(y_train), 0.01, 50)
    ref = gradient_descent(x_train, y_train, 0.01, 50)
    assert not mine["diverged"] and np.allclose(mine["history"], ref["history"], rtol=1e-9)
    big = gradient_descent_no_numpy(list(x_train), list(y_train), 0.025, 1000)
    assert big["diverged"] and big["steps_run"] == gradient_descent(x_train, y_train, 0.025, 1000)["steps_run"] == 27
    with pytest.raises(ValueError):
        gradient_descent_no_numpy([], [], 0.1, 1)


def test_part1_level1c_ratio_and_level5_nudge_bound():
    from facts import FACTS
    F = FACTS()
    assert round(F["downhill_w"] / F["downhill_b"], 1) == 7.2 and round(F["downhill_w"] / F["downhill_b"]) == 7
    bound = 0.01 / nudge_error_analysis()["mean_x_squared"]
    assert 0.0002 < bound < 0.00023                                    # "h below about 0.0002"
