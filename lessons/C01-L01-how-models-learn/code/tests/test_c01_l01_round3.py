"""Every number added by round 3 of the learner review (revision-log.md) is computed and behaves as narrated."""
import numpy as np
import pytest

import data
from by_hand import direction_reversals, gradient_step_work, squared_error_nudge, valley_floor_w
from facts import FACTS
from gradient_descent import gradient_descent, standardize
from model import mse_gradient


@pytest.fixture(scope="module")
def F():
    return FACTS()


@pytest.fixture(scope="module")
def train():
    x, y = data.make_study_data()
    x_train, y_train, *_ = data.split_train_test(x, y)
    return x_train, y_train


def test_chain_rule_checked_on_the_three_hour_student(F):
    # error -3; nudge w by 0.01 -> error -2.97, square 9 -> 8.8209; rise over run about -17.9; 2 * -3 * 3 = -18
    assert F["chain_score"] == 63 and F["chain_error"] == -3.0 and F["chain_error_nudged"] == -2.97
    assert F["chain_square"] == 9.0 and F["chain_square_nudged"] == 8.8209
    assert round(F["chain_sq_ratio"], 1) == -17.9 and F["chain_rule_value"] == -18.0
    # the ratio approaches the chain rule's value as the nudge shrinks
    gaps = [abs(squared_error_nudge(3.0, 63.0, 5.0, 45.0, h)["ratio"] - (-18.0)) for h in (0.1, 0.01, 0.001)]
    assert gaps == sorted(gaps, reverse=True) and gaps[-1] < 0.01
    with pytest.raises(ValueError):
        squared_error_nudge(3.0, 63.0, 5.0, 45.0, 0.0)


def test_first_run_path_is_one_leap_then_a_creep_not_a_zigzag(F, train):
    w = [row[1] for row in gradient_descent(*train, 0.01, 1000)["history"]]
    # w: 0 -> 9.40 -> 10.16 -> 10.21, then turns round once and falls a little every step
    assert [round(v, 2) for v in w[1:4]] == [9.4, 10.16, 10.21]
    assert F["walk_w_reversals"] == direction_reversals(w) == 1
    assert all(b < a for a, b in zip(w[3:], w[4:]))
    # after three steps, w sits on the valley floor: the best w for the b it has reached
    assert F["walk_leap_steps"] == 3 and F["walk_floor_gap"] < 0.02
    assert round(F["walk_creep_per_step"], 3) == 0.014
    assert round(w[100]) == 9 and round(w[1000], 2) == 5.19


def test_best_raw_rate_is_the_one_that_zigzags(F):
    assert [round(v, 2) for v in F["good_w_path"]] == [0.0, 18.81, 3.02, 16.18, 5.11, 14.32]
    assert F["good_w_reversals"] > 10


def test_valley_floor_is_where_the_w_slope_vanishes(train):
    for b in (0.0, 1.74, 40.5):
        w = valley_floor_w(*train, b)
        assert mse_gradient(*train, w, b)[0] == pytest.approx(0.0, abs=1e-9)


def test_direction_reversals_counts_turns():
    assert direction_reversals([0, 1, 2, 1, 0]) == 1
    assert direction_reversals([0, 2, 1, 3, 2]) == 3
    assert direction_reversals([5, 5, 5]) == 0


def test_raw_hours_cannot_take_the_scaled_rate(F):
    assert F["raw_fast_rate"] == F["scaled_rate"] == 0.1
    assert F["raw_fast_diverged"] and F["raw_fast_steps_run"] == 4


def test_scaled_hours_typical_size_is_one_at_most_about_two(F, train):
    x_scaled, _, _ = standardize(train[0])
    assert F["scaled_rms"] == 1.0 and float(np.mean(x_scaled ** 2)) == pytest.approx(1.0)
    assert round(F["scaled_max_abs"]) == 2


def test_last_sweep_row_is_the_hidden_one(F):
    assert F["n_sweep_shown_first"] == 5 == F["n_sweep_rates"] - 1
    assert F["sweep_table"][-1]["diverged"] and F["sweep_table"][-1]["rate"] == F["big_rate"]
    assert F["big_rate"] / F["good_rate"] == pytest.approx(1.25)                     # "a quarter bigger"


def test_downhill_arrow_is_minus_the_gradient(F):
    assert round(F["downhill_w"]) == 435 and round(F["downhill_b"]) == 60


def test_batch_step_cost(F):
    assert F["batch_rows"] == 100 and F["batch_work"] == 100 * F["cf_features"] == 100_000
    assert F["batch_work"] < F["cf_work"]
    with pytest.raises(ValueError):
        gradient_step_work(0, 10)
