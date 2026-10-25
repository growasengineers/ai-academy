"""Reference solutions for the coding exercises in exercises.md (levels 2, 3, 4, 5 and 7).

Try the exercises before reading this file. Each function is checked by
tests/test_c01_l01_solutions.py.
"""
import math

import numpy as np

import data
from evaluate import closed_form_fit
from gradient_descent import gradient_descent, learning_rate_sweep, standardize
from model import loss, mse

# ---------------------------------------------------------------- Level 2: by hand, then checked
L2_HOURS = [0.0, 1.0, 2.0]
L2_SCORES = [1.0, 3.0, 8.0]
L2_W, L2_B, L2_RATE = 2.0, 1.0, 0.1


def level2_answers():
    """The numbers a learner should get by hand for exercise 2 (computed, so the solution can't drift)."""
    from model import mse_gradient, predict

    preds = predict(L2_HOURS, L2_W, L2_B)
    grad_w, grad_b = mse_gradient(L2_HOURS, L2_SCORES, L2_W, L2_B)
    w1, b1 = L2_W - L2_RATE * grad_w, L2_B - L2_RATE * grad_b
    return {
        "predictions": preds.tolist(),
        "errors": (preds - np.array(L2_SCORES)).tolist(),
        "mse": loss(L2_HOURS, L2_SCORES, L2_W, L2_B),
        "grad_w": grad_w, "grad_b": grad_b,
        "w1": w1, "b1": b1,
        "mse1": loss(L2_HOURS, L2_SCORES, w1, b1),
    }


# ---------------------------------------------------------------- Level 3: gradient with plain loops
def mse_gradient_no_numpy(xs, ys, w, b):
    """dMSE/dw and dMSE/db using only Python lists and a for loop."""
    if len(xs) != len(ys) or len(xs) == 0:
        raise ValueError("xs and ys must be non-empty and the same length")
    sum_error_times_x = 0.0
    sum_error = 0.0
    for x, y in zip(xs, ys):
        error = (w * x + b) - y
        sum_error_times_x += error * x
        sum_error += error
    n = len(xs)
    return 2.0 / n * sum_error_times_x, 2.0 / n * sum_error


# ---------------------------------------------------------------- Level 4: the buggy loop and its fix
def buggy_descent(xs, ys, learning_rate, steps):
    """The broken loop from exercise 4 (kept here so the test can prove it is broken)."""
    w, b = 0.0, 0.0
    n = len(xs)
    for _ in range(steps):
        grad_w = sum(((w * x + b) - y) * x for x, y in zip(xs, ys))          # bug 3: missing 2/n
        w = w + learning_rate * grad_w                                        # bug 1: + instead of -
        grad_b = sum((w * x + b) - y for x, y in zip(xs, ys)) * 2.0 / n      # bug 2: uses the NEW w
        b = b - learning_rate * grad_b
    return w, b


def fixed_descent(xs, ys, learning_rate, steps):
    """The corrected loop: both slopes from the same (w, b), the 2/n factor, and a minus sign."""
    w, b = 0.0, 0.0
    n = len(xs)
    for _ in range(steps):
        grad_w = sum(((w * x + b) - y) * x for x, y in zip(xs, ys)) * 2.0 / n
        grad_b = sum((w * x + b) - y for x, y in zip(xs, ys)) * 2.0 / n
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
    return w, b


# ---------------------------------------------------------------- Level 5: learning-rate analysis
RAW_RATES = [0.005, 0.01, 0.015, 0.02, 0.021, 0.022, 0.025]
SCALED_RATES = [0.01, 0.1, 0.3, 0.5, 0.9, 0.99, 1.01, 1.1]


def level5_sweeps(steps=1000):
    """Sweep rates on raw and standardised hours; also return the curvature bound 2 / largest eigenvalue."""
    x, y = data.make_study_data()
    x_train, y_train, *_ = data.split_train_test(x, y)
    best = loss(x_train, y_train, *closed_form_fit(x_train, y_train))
    x_scaled, _mean, _std = standardize(x_train)

    def bound(feature):
        hessian = 2 * np.array([[np.mean(feature ** 2), np.mean(feature)], [np.mean(feature), 1.0]])
        return float(2 / np.linalg.eigvalsh(hessian)[-1])

    return {
        "raw": learning_rate_sweep(x_train, y_train, RAW_RATES, steps, best),
        "scaled": learning_rate_sweep(x_scaled, y_train, SCALED_RATES, steps, best),
        "raw_bound": bound(x_train),
        "scaled_bound": bound(x_scaled),
    }


# ---------------------------------------------------------------- Level 7: a curve a line can't follow
CURVE_SEED = 21


def make_curved_data(n=40, seed=CURVE_SEED):
    """Synthetic 'practice hours vs typing speed': fast gains at first, then levelling off."""
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0.0, 10.0, size=n)
    speed = 90.0 - 60.0 * np.exp(-0.4 * hours) + rng.normal(0.0, 2.0, size=n)
    return np.round(hours, 2), np.round(speed, 1)


def fit_with_features(columns, y):
    """Least squares on any list of feature columns plus an intercept column."""
    design = np.column_stack(list(columns) + [np.ones_like(columns[0])])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def level7_compare():
    """Train/test MSE of a straight line vs a line on [x, x squared], on the same split."""
    x, y = make_curved_data()
    x_train, y_train, x_test, y_test, *_ = data.split_train_test(x, y, seed=CURVE_SEED)
    w, b = closed_form_fit(x_train, y_train)
    line_train = mse(y_train, w * x_train + b)
    line_test = mse(y_test, w * x_test + b)
    c1, c2, c0 = fit_with_features([x_train, x_train ** 2], y_train)
    quad_train = mse(y_train, c1 * x_train + c2 * x_train ** 2 + c0)
    quad_test = mse(y_test, c1 * x_test + c2 * x_test ** 2 + c0)
    return {"line_train": line_train, "line_test": line_test,
            "quad_train": quad_train, "quad_test": quad_test,
            "line_rmse_train": math.sqrt(line_train), "quad_rmse_train": math.sqrt(quad_train)}


if __name__ == "__main__":
    print("level 2:", level2_answers())
    print("level 7:", level7_compare())
    sweeps = level5_sweeps()
    print("level 5 bounds:", sweeps["raw_bound"], sweeps["scaled_bound"])
    for name in ("raw", "scaled"):
        for row in sweeps[name]:
            print(name, row)
