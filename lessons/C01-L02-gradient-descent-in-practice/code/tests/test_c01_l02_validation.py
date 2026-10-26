"""Validation: from-scratch fit vs closed form and scikit-learn; the split; determinism."""
import numpy as np
import pytest

import data
from evaluate import (closed_form_fit, memorizer_predict, repeated_memorizer_experiment,
                      sklearn_fit, train_test_report)
from facts import FACTS
from gradient_descent import gradient_descent, standardize, unstandardize_params
from model import mse, mse_gradient


@pytest.fixture(scope="module")
def split():
    x, y = data.make_study_data()
    return data.split_train_test(x, y)


def test_dataset_is_synthetic_and_plausible():
    x, y = data.make_study_data()
    assert len(x) == data.N_STUDENTS
    assert x.min() >= data.HOURS_LOW and x.max() <= data.HOURS_HIGH
    assert y.min() >= 0 and y.max() <= 100             # plausible exam scores for this seed


def test_data_and_split_are_deterministic():
    a, b = data.make_study_data(), data.make_study_data()
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    s1, s2 = data.split_train_test(*a), data.split_train_test(*a)
    assert all(np.array_equal(p, q) for p, q in zip(s1, s2))


def test_different_seed_gives_different_data():
    assert not np.array_equal(data.make_study_data(seed=1)[0], data.make_study_data(seed=2)[0])


def test_split_has_no_overlap_and_covers_everything(split):
    *_, train_idx, test_idx = split
    assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())
    assert sorted(train_idx.tolist() + test_idx.tolist()) == list(range(data.N_STUDENTS))
    assert len(test_idx) == round(data.N_STUDENTS * data.TEST_FRACTION)


def test_split_rejects_bad_input():
    with pytest.raises(ValueError):
        data.split_train_test([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        data.split_train_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], test_fraction=1.5)
    with pytest.raises(ValueError):
        data.split_train_test([1.0], [1.0])


def test_from_scratch_matches_lstsq_and_sklearn(split):
    x_train, y_train, *_ = split
    xs, mean, std = standardize(x_train)
    run = gradient_descent(xs, y_train, 0.1, 100)
    w, b = unstandardize_params(run["w"], run["b"], mean, std)
    w_ls, b_ls = closed_form_fit(x_train, y_train)
    w_sk, b_sk = sklearn_fit(x_train, y_train)
    assert (w, b) == pytest.approx((w_ls, b_ls), abs=1e-6)
    assert (w_sk, b_sk) == pytest.approx((w_ls, b_ls), abs=1e-9)


def test_closed_form_is_where_the_gradient_vanishes(split):
    x_train, y_train, *_ = split
    w, b = closed_form_fit(x_train, y_train)
    assert np.allclose(mse_gradient(x_train, y_train, w, b), 0.0, atol=1e-9)


def test_closed_form_recovers_an_exact_line():
    x = np.array([0.0, 1.0, 2.0, 5.0])
    assert closed_form_fit(x, 3.0 * x - 2.0) == pytest.approx((3.0, -2.0))


def test_memorizer_is_perfect_on_training_data_but_worse_on_test(split):
    x_train, y_train, x_test, y_test, *_ = split
    assert len(set(x_train.tolist())) == len(x_train)  # no repeated hours, so memorising is exact
    assert mse(y_train, memorizer_predict(x_train, y_train, x_train)) == 0.0
    w, b = closed_form_fit(x_train, y_train)
    line = train_test_report(x_train, y_train, x_test, y_test, w, b)
    assert mse(y_test, memorizer_predict(x_train, y_train, x_test)) > line["test_mse"]
    assert line["test_mse"] > line["train_mse"]


def test_memorizer_loses_on_most_fresh_datasets():
    result = repeated_memorizer_experiment(n_datasets=200, first_seed=1000)
    assert result["memorizer_worse_count"] > 150
    assert result["memorizer_mean_test_mse"] > result["line_mean_test_mse"]


def test_facts_are_deterministic_and_consistent():
    f1, f2 = FACTS(), FACTS()
    assert f1 == f2
    assert f1["n_train"] + f1["n_test"] == f1["n_students"] and f1["overlap"] == 0
    assert f1["toy_mse"] == f1["toy_sum_squares"] / len(f1["toy_hours"])
    assert f1["big_diverged"] and not f1["small_reached"]
    assert f1["good_steps"] == f1["fastest_raw_steps"]
    assert f1["gradcheck_decimals"] >= 6 and f1["scaled_agree_decimals"] >= 6
    # the measured safe/unsafe rates agree with the curvature bound 2 / (largest curvature)
    assert f1["good_rate"] < f1["raw_max_stable_rate"] < f1["big_rate"]


def test_one_typo_drags_the_least_squares_line(split):
    from evaluate import outlier_refit

    x_train, y_train, *_ = split
    hi = int(np.argmax(x_train))
    w, b = closed_form_fit(x_train, y_train)
    w_bad, b_bad = outlier_refit(x_train, y_train, hi, 8.3)
    assert w_bad < w - 1.0                       # one bad score flattens the slope by more than a point per hour
    assert y_train[hi] != 8.3                    # the original data are untouched (refit works on a copy)
