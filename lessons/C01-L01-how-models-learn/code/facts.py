"""FACTS(): every number the narration or the screen states, computed by the lesson's code.

Nobody types a value into the script or a scene. Segment `checks` evaluate expressions over
F = FACTS() and must reproduce each spoken phrase exactly, so if the code changes, the script
check fails until the words are updated.

Run `python3 facts.py` to print them all.
"""
import math

import numpy as np

import data
from by_hand import (closed_form_work, descent_state_table, direction_reversals, error_change,
                     gradient_step_work, grid_search_size, nudge_table, side_swings, square_nudge,
                     squared_error_nudge, update, valley_floor_w)
from evaluate import (closed_form_fit, memorizer_predict, outlier_refit, repeated_memorizer_experiment,
                      sklearn_fit, sklearn_pipeline_fit, train_test_report)
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
# Added by the clarity revision (revision-log.md):
CANCEL_ERRORS = [2, -2]          # two misses that cancel when added
GRID_POINTS = 1000               # values per parameter in the "try every pair" grid
NUDGES = [1.0, 0.1, 0.01, 0.001] # the nudge table at w = 0 (b frozen at its best value)
SQUARE_U, SQUARE_H = 3.0, 0.01   # chain rule step 1: how fast does a square grow at 3?
CHAIN_STUDENT, CHAIN_H = 2, 0.01 # chain rule step 2: the toy student with three hours, nudged by 0.01
QUIZ_SLOPE_B = 3.0               # think pause after the update rule: slope +3, rate 0.1
STATE_ROWS = 4                   # rows of the first-run state table (steps 0..4)
BOUNCE_STEPS = 4                 # landings of w narrated for the diverging rate
CF_ROWS, CF_FEATURES = 1_000_000, 1000   # the worked closed-form cost example (rows x features squared)
# Added by round 3 of the learner review (revision-log.md):
CREEP_FROM, CREEP_TO = 3, 10     # the first run's creep along the valley floor, measured over steps 3..10
BATCH_ROWS = 100                 # the worked gradient-descent cost example: one step on a batch of 100 rows
# Added by round 4 of the learner review (revision-log.md):
SQUARE_U2 = 5.0                  # chain rule step 1, second example: how fast does a square grow at 5?
MANY_STEPS = 10_000              # a long gradient-descent run, priced against the exact solve


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

    F.update(_revision_facts(x_train, y_train, w_star, b_star, F))
    F.update(_round3_facts(x_train, y_train, F))
    F.update(_round4_facts(x_train, y_train, x_test, y_test, F))
    return F


def _clean(v, nd=10):
    """Round away float noise (and turn -0.0 into 0.0) so spoken values are stable."""
    return round(float(v), nd) + 0.0


def _revision_facts(x_train, y_train, w_star, b_star, F):
    """Numbers added by the clarity revision (see revision-log.md): every by-hand moment, computed."""
    R = {}

    # ---- why squaring: +2 and -2 cancel when added, not when squared --------------------
    errs = np.array(CANCEL_ERRORS)
    R.update(cancel_errors=[int(v) for v in errs], cancel_sum=int(errs.sum()),
             cancel_squares=[int(v) for v in errs ** 2], cancel_sum_squares=int(np.sum(errs ** 2)))

    # ---- why not a grid: how many lines a grid search would score -----------------------
    R.update(grid_points=GRID_POINTS, grid_two=grid_search_size(GRID_POINTS, 2),
             grid_three=grid_search_size(GRID_POINTS, 3))
    R.update(cf_rows=CF_ROWS, cf_features=CF_FEATURES, cf_features_squared=CF_FEATURES ** 2,
             cf_work=closed_form_work(CF_ROWS, CF_FEATURES))

    # ---- the loss bowl: two marked points -----------------------------------------------
    R.update(bowl_flat_loss=loss(x_train, y_train, 0.0, 0.0), bowl_best_loss=loss(x_train, y_train, w_star, b_star),
             frozen_b=round(b_star, 1))

    # ---- the nudge table: rise over run settles on the derivative ------------------------
    table = nudge_table(x_train, y_train, SLOPE_AT_W, b_star, NUDGES)
    R.update(nudges=NUDGES, nudge_loss_before=table[0]["loss_before"], nudge_loss_after_one=table[0]["loss_after"],
             nudge_rises=[row["rise"] for row in table], nudge_ratios=[row["ratio"] for row in table])
    slope_w_frozen, slope_b_frozen = mse_gradient(x_train, y_train, SLOPE_AT_W, b_star)
    R.update(frozen_slope_w=slope_w_frozen, frozen_slope_b=slope_b_frozen)

    # ---- chain rule, by nudging ----------------------------------------------------------
    sq = square_nudge(SQUARE_U, SQUARE_H)
    R.update(sq_u=int(SQUARE_U), sq_square=int(sq["square"]), sq_h=SQUARE_H, sq_nudged=_clean(sq["nudged"]),
             sq_nudged_square=_clean(sq["nudged_square"]), sq_rise=_clean(sq["rise"]), sq_ratio=sq["ratio"])
    student_x = float(data.TOY_HOURS[CHAIN_STUDENT])
    dw, db = error_change(student_x, TOY_W0, TOY_B0, CHAIN_H)
    R.update(chain_hours=int(student_x), chain_h=CHAIN_H, chain_change_w=_clean(dw), chain_change_b=_clean(db),
             chain_times_w=_clean(dw / CHAIN_H), chain_times_b=_clean(db / CHAIN_H))

    # ---- quick checks (think pauses) ----------------------------------------------------
    perfect = mse_gradient(data.TOY_HOURS, TOY_W0 * data.TOY_HOURS + TOY_B0, TOY_W0, TOY_B0)
    R.update(perfect_slope_w=_clean(perfect[0]), perfect_slope_b=_clean(perfect[1]),
             quiz_slope_b=QUIZ_SLOPE_B, quiz_rate=TOY_RATE,
             quiz_change=_clean(update(0.0, QUIZ_SLOPE_B, TOY_RATE)))

    # ---- the toy example, step one in full and step two ---------------------------------
    w1, b1 = F["toy_w1"], F["toy_b1"]
    p1 = predict(data.TOY_HOURS, w1, b1)
    e1 = p1 - data.TOY_SCORES
    gw1, gb1 = mse_gradient(data.TOY_HOURS, data.TOY_SCORES, w1, b1)
    w2, b2 = update(w1, gw1, TOY_RATE), update(b1, gb1, TOY_RATE)
    R.update(toy_preds1=[_clean(v) for v in p1], toy_errors1=[_clean(v) for v in e1],
             toy_squares1=[_clean(v) for v in e1 ** 2], toy_sum_squares1=_clean(np.sum(e1 ** 2)),
             toy_err_x1=[_clean(v) for v in e1 * data.TOY_HOURS], toy_sum_err_x1=_clean(np.sum(e1 * data.TOY_HOURS)),
             toy_sum_err1=_clean(np.sum(e1)), toy_grad_w1=_clean(gw1), toy_grad_b1=_clean(gb1),
             toy_step_w1=_clean(TOY_RATE * gw1), toy_w2=_clean(w2), toy_b2=_clean(b2),
             toy_mse2=loss(data.TOY_HOURS, data.TOY_SCORES, w2, b2),
             toy_middle_error_before=F["toy_errors"][1], toy_middle_error_after=_clean(e1[1]))

    # ---- the first real run, one row per step -------------------------------------------
    states = descent_state_table(x_train, y_train, WALK_RATE, STATE_ROWS)
    R.update(state_table=[{k: _clean(v, 6) if k != "step" else v for k, v in row.items()} for row in states],
             step2_w=states[2]["w"], step2_b=states[2]["b"], step2_loss=states[2]["loss"],
             step1_slope_w=states[1]["slope_w"], step1_slope_b=states[1]["slope_b"],
             step3_slope_w=states[3]["slope_w"], step4_w=states[4]["w"],
             step3_b_ceiling=int(math.ceil(states[3]["b"])))

    # ---- learning-rate sweep: the full six-rate table -------------------------------------
    sweep = learning_rate_sweep(x_train, y_train, SWEEP_RATES, SWEEP_STEPS, F["best_train_loss"])
    R.update(sweep_table=[{"rate": r["rate"], "diverged": r["diverged"], "steps_to_target": r["steps_to_target"],
                           "final_loss": r["final_loss"]} for r in sweep],
             n_sweep_rates=len(SWEEP_RATES), n_compare_rates=len(F["compare_rates"]))

    # ---- divergence, one bounce at a time ------------------------------------------------
    big = gradient_descent(x_train, y_train, BIG_RATE, BOUNCE_STEPS)
    bounce_w = [row[1] for row in big["history"]]
    right, left = side_swings(bounce_w[1:], w_star)
    R.update(bounce_w=bounce_w, bounce_right=right, bounce_left=left, best_w_rounded=round(w_star, 1))

    # ---- why standardising helps --------------------------------------------------------
    x_scaled, mean, std = standardize(x_train)
    scaled = gradient_descent(x_scaled, y_train, SCALED_RATE, SCALED_STEPS)
    R.update(scaled_min=float(x_scaled.min()), scaled_max=float(x_scaled.max()),
             scaled_max_abs=float(np.max(np.abs(x_scaled))), scaled_mean=_clean(np.mean(x_scaled), 9),
             raw_w_lift=mean, scaled_w=scaled["w"], scaled_b=scaled["b"], train_mean_score=float(np.mean(y_train)),
             w_times_mean=F["scaled_w_back"] * mean)
    return R


def _round3_facts(x_train, y_train, F):
    """Numbers added by round 3 of the learner review (revision-log.md)."""
    R = {}

    # ---- the chain rule checked with numbers on the three-hour toy student ---------------
    student_x = float(data.TOY_HOURS[CHAIN_STUDENT])
    student_y = float(data.TOY_SCORES[CHAIN_STUDENT])
    c = squared_error_nudge(student_x, student_y, TOY_W0, TOY_B0, CHAIN_H)
    R.update(chain_score=int(student_y), chain_error=_clean(c["error"]), chain_error_nudged=_clean(c["nudged_error"]),
             chain_square=_clean(c["square"]), chain_square_nudged=_clean(c["nudged_square"]),
             chain_sq_rise=_clean(c["rise"]), chain_sq_ratio=_clean(c["ratio"], 6), chain_rule_value=_clean(c["chain"]))

    # ---- the path of the first run on the contour map: one leap, then a creep -------------
    walk = gradient_descent(x_train, y_train, WALK_RATE, WALK_STEPS)
    walk_w = [row[1] for row in walk["history"]]
    floor = valley_floor_w(x_train, y_train, walk["history"][SHOW_STEP_A][2])
    good = gradient_descent(x_train, y_train, GOOD_RATE, SWEEP_STEPS)
    good_w = [row[1] for row in good["history"]]
    R.update(walk_leap_steps=SHOW_STEP_A, walk_floor_w=floor, walk_floor_gap=abs(walk_w[SHOW_STEP_A] - floor),
             walk_w_reversals=direction_reversals(walk_w),
             walk_creep_per_step=(walk_w[CREEP_FROM] - walk_w[CREEP_TO]) / (CREEP_TO - CREEP_FROM),
             good_w_path=[_clean(v, 6) for v in good_w[:6]], good_w_reversals=direction_reversals(good_w))

    # ---- why the bigger rate needed the new units: rate 0.1 on raw hours ----------------
    raw_fast = gradient_descent(x_train, y_train, SCALED_RATE, SWEEP_STEPS)
    R.update(raw_fast_rate=SCALED_RATE, raw_fast_diverged=raw_fast["diverged"], raw_fast_steps_run=raw_fast["steps_run"])

    # ---- standardised hours: a typical scaled hour is one (root mean square), at most about two
    x_scaled, _mean, _std = standardize(x_train)
    R.update(scaled_rms=_clean(math.sqrt(np.mean(x_scaled ** 2)), 9))

    # ---- the six-rate table: the last row stays hidden until we run it ------------------
    R.update(n_sweep_shown_first=len(SWEEP_RATES) - 1)

    # ---- minus the gradient at the nudge-table spot (the downhill arrow) ----------------
    R.update(downhill_w=-F["frozen_slope_w"], downhill_b=-F["frozen_slope_b"])

    # ---- the cost of one gradient-descent step on a batch -------------------------------
    R.update(batch_rows=BATCH_ROWS, batch_work=gradient_step_work(BATCH_ROWS, CF_FEATURES))
    return R


def _round4_facts(x_train, y_train, x_test, y_test, F):
    """Numbers added by round 4 of the learner review (revision-log.md)."""
    R = {}

    # ---- the square rule on a second example: an error of five -------------------------
    sq = square_nudge(SQUARE_U2, SQUARE_H)
    R.update(sq2_u=int(SQUARE_U2), sq2_square=int(sq["square"]), sq2_nudged=_clean(sq["nudged"]),
             sq2_nudged_square=_clean(sq["nudged_square"]), sq2_rise=_clean(sq["rise"]), sq2_ratio=_clean(sq["ratio"], 6))

    # ---- why multiply: the square's own rate at the three-hour student (2 * error) --------
    R.update(chain_square_rate=_clean(2 * F["chain_error"]))

    # ---- Part 2's hook: the exploding rate against the rate used by hand ----------------
    R.update(hook_rate_ratio=_clean(F["toy_rate"] / F["big_rate"], 9))

    # ---- standardising: dividing alone keeps the hours positive; centring moves them ------
    x_scaled, mean, std = standardize(x_train)
    R.update(divide_only_min=float(np.min(x_train) / std), divide_only_max=float(np.max(x_train) / std))

    # ---- the cost of many gradient-descent steps, against the exact solve ----------------
    many = MANY_STEPS * F["batch_work"]
    R.update(many_steps=MANY_STEPS, many_steps_work=many, cf_over_many_steps=F["cf_work"] // many,
             cf_equations=F["cf_features"] + 1)

    # ---- the scikit-learn pipeline the narration describes (t17) -------------------------
    pipe = sklearn_pipeline_fit(x_train, y_train, x_test, y_test)
    R.update(pipeline_w=pipe["w"], pipeline_b=pipe["b"], pipeline_test_score=pipe["test_score"],
             pipeline_agree_decimals=_decimals_of_agreement([pipe["w"], pipe["b"]], [F["closed_w"], F["closed_b"]]))
    return R


if __name__ == "__main__":
    for key, value in FACTS().items():
        print(f"{key:28s} {value}")
