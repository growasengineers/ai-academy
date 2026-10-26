"""Every number added by round 4 of the learner review (revision-log.md) is computed and behaves as narrated."""
import numpy as np
import pytest

import data
from by_hand import closed_form_work, gradient_step_work, square_nudge
from evaluate import closed_form_fit, sklearn_pipeline_fit
from facts import FACTS
from gradient_descent import standardize


@pytest.fixture(scope="module")
def F():
    return FACTS()


@pytest.fixture(scope="module")
def split():
    x, y = data.make_study_data()
    return data.split_train_test(x, y)[:4]


def test_square_rule_on_a_second_example(F):
    # 5.01 squared is 25.1001: a rise of 0.1001 over a run of 0.01, about 10 = 2 * 5
    assert F["sq2_u"] == 5 and F["sq2_square"] == 25 and F["sq2_nudged"] == 5.01
    assert F["sq2_nudged_square"] == 25.1001 and F["sq2_rise"] == 0.1001
    assert F["sq2_ratio"] == pytest.approx(10.01) and round(F["sq2_ratio"]) == 2 * F["sq2_u"]
    assert square_nudge(5.0, 0.01)["ratio"] == pytest.approx(F["sq2_ratio"])


def test_why_multiply_the_two_rates(F):
    # the error moves 3 times as fast as w; the square moves 2 * (-3) = -6 times as fast as the error
    assert F["chain_square_rate"] == 2 * F["chain_error"] == -6.0
    assert F["chain_times_w"] * F["chain_square_rate"] == F["chain_rule_value"] == -18.0


def test_toy_squares_explain_why_the_loss_fell(F):
    # the middle square rose 0 -> 4, the outer two fell 9 -> 3.24 and 9 -> 0.04; the total fell 18 -> 7.28
    assert F["toy_squares"] == [9, 0, 9] and F["toy_squares1"] == [3.24, 4.0, 0.04]
    assert F["toy_squares1"][1] > F["toy_squares"][1]
    assert F["toy_sum_squares"] == 18 and F["toy_sum_squares1"] == 7.28 == round(sum(F["toy_squares1"]), 10)
    # step two: rate times slope, then the new w
    assert round(F["toy_rate"] * round(F["toy_grad_w1"], 2), 3) == round(F["toy_step_w1"], 3) == 0.107
    assert round(F["toy_w2"], 2) == 5.69


def test_hook_rate_is_four_times_smaller_than_the_hand_rate(F):
    assert F["hook_rate_ratio"] == 4.0 == round(F["toy_rate"] / F["big_rate"])
    assert F["big_rate"] < F["toy_rate"]


def test_centring_not_dividing_brings_hours_to_minus_two_to_one_point_four(F, split):
    x_train = split[0]
    # dividing alone keeps every hour positive (about 0.51 to 3.90) ...
    assert round(F["divide_only_min"], 2) == 0.51 and round(F["divide_only_max"], 2) == 3.90
    # ... centring then dividing gives about -1.98 to 1.41
    x_scaled, _, _ = standardize(x_train)
    assert round(float(x_scaled.min()), 2) == -1.98 and round(float(x_scaled.max()), 2) == 1.41
    assert round(F["scaled_min"]) == -2 and round(F["scaled_max"], 1) == 1.4


def test_ten_thousand_steps_still_cost_a_thousand_times_less(F):
    assert F["many_steps"] == 10_000
    assert F["many_steps_work"] == 10_000 * gradient_step_work(100, 1000) == 10 ** 9
    assert F["cf_over_many_steps"] == closed_form_work(10 ** 6, 1000) // 10 ** 9 == 1000
    assert F["cf_equations"] == F["cf_features"] + 1 > F["cf_features"]


def test_sklearn_pipeline_matches_closed_form(F, split):
    x_train, y_train, x_test, y_test = split
    pipe = sklearn_pipeline_fit(x_train, y_train, x_test, y_test)
    w, b = closed_form_fit(x_train, y_train)
    assert pipe["w"] == pytest.approx(w, abs=1e-9) and pipe["b"] == pytest.approx(b, abs=1e-9)
    assert F["pipeline_w"] == pytest.approx(F["closed_w"]) and F["pipeline_agree_decimals"] >= 8
    assert 0.0 < pipe["test_score"] <= 1.0 and np.isfinite(pipe["test_score"])


def test_level3_loop_gradient_gives_the_by_hand_toy_slopes(F):
    # exercises.md L3: without model.mse_gradient, check against the video's by-hand values -8 and -4
    from solutions_c01_l01 import mse_gradient_no_numpy
    gw, gb = mse_gradient_no_numpy(F["toy_hours"], F["toy_scores"], F["toy_w0"], F["toy_b0"])
    assert (gw, gb) == (pytest.approx(F["toy_grad_w"]), pytest.approx(F["toy_grad_b"]))
    assert (round(gw), round(gb)) == (-8, -4)
