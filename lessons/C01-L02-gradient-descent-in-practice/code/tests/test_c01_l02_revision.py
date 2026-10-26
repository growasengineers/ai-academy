"""Every number added by the clarity revision (revision-log.md) is computed and behaves as narrated."""
import numpy as np
import pytest

import data
from by_hand import (closed_form_work, descent_state_table, error_change, grid_search_size, nudge_table,
                     side_swings, square_nudge, update)
from facts import FACTS
from gradient_descent import gradient_descent


@pytest.fixture(scope="module")
def F():
    return FACTS()


@pytest.fixture(scope="module")
def train():
    x, y = data.make_study_data()
    x_train, y_train, *_ = data.split_train_test(x, y)
    return x_train, y_train


def test_adding_cancels_but_squaring_does_not(F):
    assert F["cancel_errors"] == [2, -2] and F["cancel_sum"] == 0
    assert F["cancel_squares"] == [4, 4] and F["cancel_sum_squares"] == 8


def test_grid_search_grows_as_a_power():
    assert grid_search_size(1000, 2) == 10 ** 6 and grid_search_size(1000, 3) == 10 ** 9
    with pytest.raises(ValueError):
        grid_search_size(0, 2)


def test_closed_form_work_example(F):
    assert closed_form_work(10, 3) == 90
    assert F["cf_features_squared"] == 10 ** 6 and F["cf_work"] == F["cf_rows"] * F["cf_features_squared"] == 10 ** 12


def test_bowl_points(F):
    assert round(F["bowl_flat_loss"]) == 5151 and F["bowl_flat_loss"] == F["start_loss"]
    assert F["bowl_best_loss"] == pytest.approx(F["best_train_loss"]) and F["frozen_b"] == 40.5


def test_nudge_table_settles_on_the_derivative(F):
    ratios = F["nudge_ratios"]
    assert [round(r, 1) for r in ratios] == [-390.2, -430.8, -434.8, -435.2]
    assert round(F["nudge_loss_before"]) == 1067 and round(F["nudge_loss_after_one"]) == 677
    gaps = [abs(r - F["slope_value"]) for r in ratios]
    assert all(a > b for a, b in zip(gaps, gaps[1:]))          # each smaller nudge gets closer
    assert gaps[-1] < 0.1 and round(F["slope_value"]) == -435   # settles on the analytic slope


def test_nudge_table_rejects_non_positive_nudges(train):
    with pytest.raises(ValueError):
        nudge_table(*train, 0.0, 40.0, [0.0])


def test_frozen_point_gradient_is_the_pair_of_nudge_slopes(F):
    assert round(F["frozen_slope_w"]) == -435 and round(F["frozen_slope_b"]) == -60


def test_square_grows_at_two_u():
    s = square_nudge(3.0, 0.01)
    assert round(s["nudged_square"], 4) == 9.0601 and round(s["rise"], 4) == 0.0601
    assert s["ratio"] == pytest.approx(6.0, abs=0.011) and round(s["ratio"]) == 2 * 3
    with pytest.raises(ValueError):
        square_nudge(3.0, -0.1)


def test_error_moves_x_times_for_w_and_once_for_b():
    dw, db = error_change(3.0, 5.0, 45.0, 0.01)
    assert dw == pytest.approx(0.03) and db == pytest.approx(0.01)


def test_quick_checks(F):
    assert F["perfect_slope_w"] == 0.0 and F["perfect_slope_b"] == 0.0
    assert update(0.0, 3.0, 0.1) == pytest.approx(-0.3) and F["quiz_change"] == -0.3


def test_toy_step_one_in_full(F):
    assert F["toy_preds1"] == [51.2, 57.0, 62.8]
    assert F["toy_errors1"] == [-1.8, 2.0, -0.2]
    assert F["toy_squares1"] == [3.24, 4.0, 0.04] and F["toy_sum_squares1"] == 7.28
    assert round(F["toy_sum_squares1"] / 3, 2) == round(F["toy_mse1"], 2) == 2.43
    # a step lowers the average, not every error: the middle student got worse
    assert F["toy_middle_error_before"] == 0 and F["toy_middle_error_after"] == 2.0


def test_toy_step_two(F):
    assert F["toy_err_x1"] == [-1.8, 4.0, -0.6] and F["toy_sum_err_x1"] == 1.6
    assert round(F["toy_grad_w1"], 2) == 1.07 and F["toy_grad_b1"] == 0.0 and F["toy_sum_err1"] == 0.0
    assert round(F["toy_step_w1"], 3) == 0.107 and round(F["toy_w2"], 2) == 5.69 and F["toy_b2"] == 45.4
    assert round(5.8 - 0.107, 2) == round(F["toy_w2"], 2)      # the spoken arithmetic is consistent
    assert F["toy_mse2"] < F["toy_mse1"] and round(F["toy_mse2"], 2) == 2.37


def test_first_run_state_table(F, train):
    rows = descent_state_table(*train, 0.01, 4)
    run = gradient_descent(*train, 0.01, 4)
    for row, hist in zip(rows, run["history"]):
        assert (row["w"], row["b"], row["loss"]) == pytest.approx(hist[1:])
    assert round(rows[1]["w"], 1) == 9.4 and round(rows[1]["b"], 2) == 1.41
    assert round(rows[2]["w"], 2) == 10.16 and round(rows[2]["b"], 2) == 1.63 and round(rows[2]["loss"], 1) == 225.3
    assert round(rows[3]["w"], 2) == 10.21 and round(rows[3]["b"], 2) == 1.74 and round(rows[3]["loss"]) == 224
    # w overshoots the true 5 after one step, then its slope turns positive and w comes back down
    assert rows[1]["w"] > data.TRUE_W
    assert rows[2]["slope_w"] < 0 < rows[3]["slope_w"] and rows[4]["w"] < rows[3]["w"]
    assert rows[3]["b"] < F["step3_b_ceiling"] == 2
    assert round(rows[0]["w"] - 0.01 * rows[0]["slope_w"], 1) == 9.4


def test_sweep_table_has_six_rates_and_only_the_largest_diverges(F):
    table = F["sweep_table"]
    assert F["n_sweep_rates"] == len(table) == 6 and F["n_compare_rates"] == 3
    assert [r["rate"] for r in table if r["diverged"]] == [0.025]
    reached = [r for r in table if r["steps_to_target"] is not None]
    assert [r["rate"] for r in reached] == [0.02]
    # ten times smaller steps, ten times as many of them, about the same place
    assert round(F["small_final_loss"]) == round(F["step100_loss"]) == 139


def test_divergence_bounces_wider_on_each_side(F):
    assert [round(v, 1) for v in F["bounce_w"][1:]] == [23.5, -7.0, 32.5, -18.8]
    right, left = F["bounce_right"], F["bounce_left"]
    assert right == sorted(right) and left == sorted(left, reverse=True)   # each landing farther out on its side
    assert F["big_losses_first"] == sorted(F["big_losses_first"])         # and higher up the bowl every step
    assert side_swings([1, -1, 2, -2], 0) == ([1, 2], [-1, -2])


def test_why_standardising_helps(F):
    assert round(F["scaled_min"]) == -2 and round(F["scaled_max"], 1) == 1.4 and round(F["scaled_max_abs"]) == 2
    assert F["scaled_mean"] == 0.0 and round(F["raw_w_lift"], 2) == 6.23


def test_converting_the_scaled_line_back_by_hand(F):
    ws, bs = round(F["scaled_w"], 2), round(F["scaled_b"], 2)
    assert (ws, bs) == (12.06, 70.64) and round(F["train_mean_score"], 2) == bs
    assert round(ws / round(F["hours_std"], 3), 2) == round(F["scaled_w_back"], 2) == 4.83
    assert round(F["w_times_mean"], 1) == 30.1
    assert round(bs - round(F["w_times_mean"], 1), 2) == round(F["scaled_b_back"], 2) == 40.54


def test_hundred_step_scaled_run_is_what_matches_to_eight_decimals(F):
    assert F["scaled_steps"] == 100 and F["scaled_steps_to_target"] == 29
    assert F["scaled_agree_decimals"] >= 8
