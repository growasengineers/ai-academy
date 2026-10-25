"""The model, the loss and the gradient (SPEC phase 7: normal, edge, failure, expected behaviour)."""
import numpy as np
import pytest

import data
from model import loss, loss_curve, mse, mse_gradient, numerical_gradient, predict


def test_predict_is_a_line():
    assert predict(2.0, 5.0, 45.0) == pytest.approx(55.0)
    assert np.allclose(predict([1, 2, 3], 5, 45), [50, 55, 60])


def test_toy_mse_by_hand():
    # errors -3, 0, -3 -> squares 9, 0, 9 -> sum 18 -> mean 6 (the on-screen hand calculation)
    preds = predict(data.TOY_HOURS, 5.0, 45.0)
    assert np.allclose(preds - data.TOY_SCORES, [-3, 0, -3])
    assert mse(data.TOY_SCORES, preds) == pytest.approx(6.0)


def test_toy_gradient_by_hand():
    # dMSE/dw = 2/3 * (-3*1 + 0*2 - 3*3) = -8 ; dMSE/db = 2/3 * (-6) = -4
    grad_w, grad_b = mse_gradient(data.TOY_HOURS, data.TOY_SCORES, 5.0, 45.0)
    assert grad_w == pytest.approx(-8.0)
    assert grad_b == pytest.approx(-4.0)


def test_one_toy_step_lowers_the_loss():
    w1, b1 = 5.0 - 0.1 * -8.0, 45.0 - 0.1 * -4.0
    assert (w1, b1) == pytest.approx((5.8, 45.4))
    assert loss(data.TOY_HOURS, data.TOY_SCORES, w1, b1) < loss(data.TOY_HOURS, data.TOY_SCORES, 5.0, 45.0)


def test_mse_is_zero_for_perfect_predictions_and_never_negative():
    y = np.array([1.0, -2.0, 3.5])
    assert mse(y, y) == 0.0
    assert mse(y, y + np.array([1.0, -1.0, 0.5])) > 0.0


def test_squaring_stops_errors_cancelling():
    # errors +2 and -2 average to 0, but their squares average to 4
    assert mse([0.0, 0.0], [2.0, -2.0]) == pytest.approx(4.0)


def test_mse_rejects_empty_and_mismatched_inputs():
    with pytest.raises(ValueError):
        mse([], [])
    with pytest.raises(ValueError):
        mse([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        mse_gradient([], [], 0.0, 0.0)


@pytest.mark.parametrize("w,b", [(0.0, 0.0), (3.0, 40.5), (-7.0, 90.0), (12.0, -20.0)])
def test_gradient_matches_finite_differences(w, b):
    x, y = data.make_study_data()
    analytic = mse_gradient(x, y, w, b)
    numeric = numerical_gradient(x, y, w, b)
    assert np.allclose(analytic, numeric, rtol=1e-6, atol=1e-6)


def test_loss_curve_is_u_shaped_with_minimum_near_best_w():
    x, y = data.make_study_data()
    ws = np.linspace(0, 10, 101)
    curve = loss_curve(x, y, ws, 40.0)
    i = int(np.argmin(curve))
    assert 0 < i < len(ws) - 1                      # the lowest point is inside, not at an edge
    assert all(np.diff(curve[: i + 1]) < 0)          # falling on the left
    assert all(np.diff(curve[i:]) > 0)               # rising on the right


def test_loss_grid_matches_loss_and_is_lowest_near_closed_form():
    from evaluate import closed_form_fit
    from model import loss_grid

    x, y = data.make_study_data()
    w_star, b_star = closed_form_fit(x, y)
    ws = np.linspace(w_star - 2, w_star + 2, 21)     # w_star sits exactly in the middle column
    bs = np.linspace(b_star - 10, b_star + 10, 21)   # b_star sits exactly in the middle row
    grid = np.array(loss_grid(x, y, ws, bs))
    assert grid.shape == (len(bs), len(ws))
    assert grid[3, 7] == pytest.approx(loss(x, y, ws[7], bs[3]))
    assert np.unravel_index(np.argmin(grid), grid.shape) == (10, 10)
