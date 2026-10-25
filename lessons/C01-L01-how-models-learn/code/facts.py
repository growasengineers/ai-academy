"""FACTS(): every number the narration or the screen states, computed by the lesson's code.

Nobody types a value into the script or a scene. Segment `checks` evaluate expressions over
F = FACTS() and must reproduce each spoken phrase exactly, so if the code changes, the script
check fails until the words are updated.

Run `python3 facts.py` to print them all.
"""
import math

import numpy as np

import data
from evaluate import (closed_form_fit, memorizer_predict, outlier_refit, repeated_memorizer_experiment,
                      sklearn_fit, train_test_report)
from gradient_descent import (DIVERGENCE_FACTOR, first_step_within, gradient_descent, learning_rate_sweep, losses,
                              standardize, unstandardize_params)
from model import loss, mse, mse_gradient, numerical_gradient, predict

WALK_RATE = 0.01                 # the first full run we narrate step by step
WALK_STEPS = 1000
SHOW_STEP_A, SHOW_STEP_B = 3, 100   # the two in-between steps the narration stops at
SWEEP_RATES = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025]
SWEEP_STEPS = 1000
SMALL_RATE, GOOD_RATE, BIG_RATE = 0.001, 0.02, 0.025   # the three runs we compare on screen
SCALED_RATE = 0.1
SCALED_STEPS = 100
TOY_W0, TOY_B0, TOY_RATE = 5.0, 45.0, 0.1
PREDICT_HOURS = 7.0
OUTLIER_SCORE = 8.3     # storyboard A12: the top student's 83.5 mistyped as 8.3
SLOPE_AT_W = 0.0                 # where we draw the first tangent on the loss curve


def _decimals_of_agreement(a, b):
    """Largest d (up to 10) such that every pair agrees to within 10**-d."""
    diff = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
    if diff == 0.0:
        return 10
    return min(10, int(math.floor(-math.log10(diff))))


def FACTS():
    F = {}

    # ---- the synthetic dataset -------------------------------------------------------------
    x, y = data.make_study_data()
    x_train, y_train, x_test, y_test, train_idx, test_idx = data.split_train_test(x, y)
    F.update(n_students=len(x), n_train=len(x_train), n_test=len(x_test),
             test_fraction=data.TEST_FRACTION, seed=data.SEED,
             true_w=data.TRUE_W, true_b=data.TRUE_B, noise_sd=data.NOISE_SD,
             hours_low=data.HOURS_LOW, hours_high=data.HOURS_HIGH,
             score_min=float(y.min()), score_max=float(y.max()))
    lo, hi = int(np.argmin(x_train)), int(np.argmax(x_train))
    F.update(few_hours=float(x_train[lo]), few_score=float(y_train[lo]),
             many_hours=float(x_train[hi]), many_score=float(y_train[hi]))

    # ---- the by-hand example: three students, one candidate line, one step ---------------
    toy_pred = predict(data.TOY_HOURS, TOY_W0, TOY_B0)
    toy_err = toy_pred - data.TOY_SCORES
    toy_gw, toy_gb = mse_gradient(data.TOY_HOURS, data.TOY_SCORES, TOY_W0, TOY_B0)
    toy_w1, toy_b1 = TOY_W0 - TOY_RATE * toy_gw, TOY_B0 - TOY_RATE * toy_gb
    F.update(toy_hours=[int(v) for v in data.TOY_HOURS], toy_scores=[int(v) for v in data.TOY_SCORES],
             toy_w0=int(TOY_W0), toy_b0=int(TOY_B0), toy_preds=[int(v) for v in toy_pred],
             toy_errors=[int(v) for v in toy_err], toy_squares=[int(v) for v in toy_err ** 2],
             toy_sum_squares=int(np.sum(toy_err ** 2)),
             toy_mse=loss(data.TOY_HOURS, data.TOY_SCORES, TOY_W0, TOY_B0),
             toy_err_times_x=[int(v) for v in toy_err * data.TOY_HOURS],
             toy_sum_err_x=int(np.sum(toy_err * data.TOY_HOURS)), toy_sum_err=int(np.sum(toy_err)),
             toy_grad_w=toy_gw, toy_grad_b=toy_gb, toy_rate=TOY_RATE,
             toy_w1=round(toy_w1, 10), toy_b1=round(toy_b1, 10),
             toy_mse1=loss(data.TOY_HOURS, data.TOY_SCORES, toy_w1, toy_b1))

    # ---- the exact answer (closed form) and scikit-learn ---------------------------------
    w_star, b_star = closed_form_fit(x_train, y_train)
    w_sk, b_sk = sklearn_fit(x_train, y_train)
    best_loss = loss(x_train, y_train, w_star, b_star)
    F.update(closed_w=w_star, closed_b=b_star, best_train_loss=best_loss,
             sklearn_w=w_sk, sklearn_b=b_sk,
             sklearn_agree_decimals=_decimals_of_agreement([w_star, b_star], [w_sk, b_sk]),
             grad_at_best_w=mse_gradient(x_train, y_train, w_star, b_star)[0])

    # ---- one dimension first: the loss over w with b held at its best value --------------
    slope = mse_gradient(x_train, y_train, SLOPE_AT_W, b_star)[0]
    slope_numeric = numerical_gradient(x_train, y_train, SLOPE_AT_W, b_star)[0]
    F.update(slope_at_w=SLOPE_AT_W, slope_value=slope, slope_numeric=slope_numeric)

    # ---- gradient check on many random points -------------------------------------------
    rng = np.random.default_rng(0)
    analytic, numeric = [], []
    for _ in range(20):
        w, b = rng.uniform(-10, 10), rng.uniform(-50, 100)
        analytic.extend(mse_gradient(x_train, y_train, w, b))
        numeric.extend(numerical_gradient(x_train, y_train, w, b))
    F.update(gradcheck_points=20, gradcheck_decimals=_decimals_of_agreement(analytic, numeric))

    # ---- the first full run on raw hours, narrated step by step --------------------------
    walk = gradient_descent(x_train, y_train, WALK_RATE, WALK_STEPS)
    h = walk["history"]
    gw0, gb0 = mse_gradient(x_train, y_train, 0.0, 0.0)
    F.update(walk_rate=WALK_RATE, walk_steps=WALK_STEPS, show_step_a=SHOW_STEP_A, show_step_b=SHOW_STEP_B,
             divergence_factor=int(DIVERGENCE_FACTOR),
             start_loss=h[0][3], start_grad_w=gw0, start_grad_b=gb0,
             step1_w=h[1][1], step1_b=h[1][2], step1_loss=h[1][3],
             step3_w=h[SHOW_STEP_A][1], step3_b=h[SHOW_STEP_A][2], step3_loss=h[SHOW_STEP_A][3],
             step100_loss=h[SHOW_STEP_B][3], step100_w=h[SHOW_STEP_B][1], step100_b=h[SHOW_STEP_B][2],
             walk_final_w=walk["w"], walk_final_b=walk["b"], walk_final_loss=losses(walk)[-1])

    # ---- learning rate: too small, good, too large --------------------------------------
    sweep = learning_rate_sweep(x_train, y_train, SWEEP_RATES, SWEEP_STEPS, best_loss)
    by_rate = {row["rate"]: row for row in sweep}
    safe = [row for row in sweep if not row["diverged"] and row["steps_to_target"] is not None]
    fastest = min(safe, key=lambda row: row["steps_to_target"])
    F.update(sweep_rates=SWEEP_RATES, sweep_steps=SWEEP_STEPS,
             compare_rates=[SMALL_RATE, GOOD_RATE, BIG_RATE],
             small_rate=SMALL_RATE, small_final_loss=by_rate[SMALL_RATE]["final_loss"],
             small_reached=by_rate[SMALL_RATE]["steps_to_target"] is not None,
             good_rate=GOOD_RATE, good_steps=by_rate[GOOD_RATE]["steps_to_target"],
             big_rate=BIG_RATE, big_diverged=by_rate[BIG_RATE]["diverged"],
             big_stop_step=by_rate[BIG_RATE]["steps_run"], big_stop_loss=by_rate[BIG_RATE]["final_loss"],
             big_stop_loss_billions=int(by_rate[BIG_RATE]["final_loss"] // 1e9),
             fastest_raw_rate=fastest["rate"], fastest_raw_steps=fastest["steps_to_target"],
             diverging_rates=[row["rate"] for row in sweep if row["diverged"]])
    big = gradient_descent(x_train, y_train, BIG_RATE, SWEEP_STEPS)
    F.update(big_losses_first=[round(v) for v in losses(big)[:4]])
    # Curvature of the loss bowl (Hessian of MSE). Used in the outline and solutions only:
    # gradient descent on a quadratic is stable only for rates below 2 / (largest curvature).
    hess = 2 * np.array([[np.mean(x_train ** 2), np.mean(x_train)], [np.mean(x_train), 1.0]])
    eig = np.linalg.eigvalsh(hess)
    F.update(raw_max_stable_rate=float(2 / eig[-1]), raw_curvature_ratio=float(eig[-1] / eig[0]))

    # ---- the fix: standardise the feature ------------------------------------------------
    x_scaled, mean, std = standardize(x_train)
    scaled = gradient_descent(x_scaled, y_train, SCALED_RATE, SCALED_STEPS)
    w_back, b_back = unstandardize_params(scaled["w"], scaled["b"], mean, std)
    hess_s = 2 * np.array([[np.mean(x_scaled ** 2), np.mean(x_scaled)], [np.mean(x_scaled), 1.0]])
    eig_s = np.linalg.eigvalsh(hess_s)
    F.update(hours_mean=mean, hours_std=std, scaled_rate=SCALED_RATE, scaled_steps=SCALED_STEPS,
             scaled_steps_to_target=first_step_within(scaled, best_loss),
             scaled_final_loss=losses(scaled)[-1], scaled_w_back=w_back, scaled_b_back=b_back,
             scaled_agree_decimals=_decimals_of_agreement([w_back, b_back], [w_star, b_star]),
             scaled_max_stable_rate=float(2 / eig_s[-1]), scaled_curvature_ratio=float(eig_s[-1] / eig_s[0]),
             speedup=by_rate[GOOD_RATE]["steps_to_target"] / first_step_within(scaled, best_loss))

    # ---- generalisation: train vs test, and the memoriser trap ---------------------------
    report = train_test_report(x_train, y_train, x_test, y_test, w_star, b_star)
    memo_train = mse(y_train, memorizer_predict(x_train, y_train, x_train))
    memo_test = mse(y_test, memorizer_predict(x_train, y_train, x_test))
    repeat = repeated_memorizer_experiment()
    F.update(train_mse=report["train_mse"], test_mse=report["test_mse"],
             train_rmse=math.sqrt(report["train_mse"]), test_rmse=math.sqrt(report["test_mse"]),
             memo_train_mse=memo_train, memo_test_mse=memo_test,
             repeat_n=repeat["n_datasets"], repeat_line_mse=repeat["line_mean_test_mse"],
             repeat_memo_mse=repeat["memorizer_mean_test_mse"],
             repeat_memo_worse=repeat["memorizer_worse_count"],
             overlap=len(set(train_idx.tolist()) & set(test_idx.tolist())))

    # ---- one typo drags the line (storyboard A12, on screen only) -------------------------
    w_out, b_out = outlier_refit(x_train, y_train, hi, OUTLIER_SCORE)
    F.update(outlier_hours=float(x_train[hi]), outlier_true_score=float(y_train[hi]),
             outlier_wrong_score=OUTLIER_SCORE, outlier_w=w_out, outlier_b=b_out)

    # ---- a curve a line can't follow (storyboard A12 on screen; exercise 7) ----------------
    from solutions_c01_l01 import level7_compare
    curve = level7_compare()
    F.update(curve_line_train_mse=curve["line_train"], curve_line_test_mse=curve["line_test"],
             curve_quad_train_mse=curve["quad_train"], curve_quad_test_mse=curve["quad_test"])

    # ---- using the model ----------------------------------------------------------------
    F.update(predict_hours=PREDICT_HOURS, predict_score=float(predict(PREDICT_HOURS, w_star, b_star)))
    return F


if __name__ == "__main__":
    for key, value in FACTS().items():
        print(f"{key:28s} {value}")
