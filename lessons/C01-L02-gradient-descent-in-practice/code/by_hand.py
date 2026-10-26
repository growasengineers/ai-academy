"""The small calculations the video works through slowly, one line at a time.

Each function here backs a "by hand" moment in the narration: the nudge table that turns rise over run
into a derivative, the chain rule checked by nudging, a state table of the first descent steps, the
bouncing positions of a diverging run, and a few plain counts (grid sizes, closed-form work). They are
tiny on purpose, so a learner can recompute every number with a calculator.
"""
import numpy as np

from model import loss, mse_gradient, predict


def nudge_table(x, y, w, b, nudges):
    """Forward differences of the loss along w at (w, b): one row per nudge size h.

    Each row: h, loss_before, loss_after (at w + h), rise (the change in loss) and ratio = rise / h.
    As h shrinks, the ratio settles on the derivative dMSE/dw at w.
    """
    before = loss(x, y, w, b)
    rows = []
    for h in nudges:
        if h <= 0:
            raise ValueError("a nudge must be positive")
        after = loss(x, y, w + h, b)
        rows.append({"h": h, "loss_before": before, "loss_after": after,
                     "rise": after - before, "ratio": (after - before) / h})
    return rows


def square_nudge(u, h):
    """How fast does u squared grow at u? Nudge u by h and return the rise and rise over run (about 2u)."""
    if h <= 0:
        raise ValueError("a nudge must be positive")
    rise = (u + h) ** 2 - u ** 2
    return {"u": u, "square": u ** 2, "nudged": u + h, "nudged_square": (u + h) ** 2,
            "rise": rise, "ratio": rise / h}


def error_change(hours, w, b, h):
    """How much one student's prediction (and so their error) moves when w, or b, is nudged by h.

    Returns (change when w is nudged, change when b is nudged). The first is hours * h, the second is h,
    which is where the x and the 1 in the gradient formulas come from.
    """
    base = float(predict(hours, w, b))
    return float(predict(hours, w + h, b)) - base, float(predict(hours, w, b + h)) - base


def update(value, slope, rate):
    """One gradient-descent update of a single parameter: value minus rate times slope."""
    return value - rate * slope


def descent_state_table(x, y, rate, steps, w0=0.0, b0=0.0):
    """The state table the video fills one row per step: step, w, b, the two slopes AT that state, loss.

    Row k holds the parameters after k updates and the slopes that the next update will use.
    """
    w, b = float(w0), float(b0)
    rows = []
    for step in range(steps + 1):
        grad_w, grad_b = mse_gradient(x, y, w, b)
        rows.append({"step": step, "w": w, "b": b, "slope_w": grad_w, "slope_b": grad_b,
                     "loss": loss(x, y, w, b)})
        w, b = update(w, grad_w, rate), update(b, grad_b, rate)
    return rows


def side_swings(values, centre):
    """Split a bouncing sequence into its landings right of centre and left of centre (in order)."""
    right = [v for v in values if v > centre]
    left = [v for v in values if v < centre]
    return right, left


def grid_search_size(points_per_axis, n_params):
    """How many candidate models a grid search must score: points_per_axis to the power n_params."""
    if points_per_axis < 1 or n_params < 1:
        raise ValueError("need at least one point and one parameter")
    return points_per_axis ** n_params


def closed_form_work(rows, features):
    """The growth of the exact least-squares solve, rows times features squared (no constant factor)."""
    return rows * features ** 2


# ---- added in round 3 of the learner review (revision-log.md) ---------------------------------

def squared_error_nudge(hours, score, w, b, h):
    """Check the chain rule on one student: nudge w by h and watch their squared error.

    Returns the error before and after the nudge, both squares, the rise over run of the square, and the
    chain rule's prediction 2 * error * hours. The ratio approaches the prediction as h shrinks.
    """
    if h <= 0:
        raise ValueError("a nudge must be positive")
    error = float(predict(hours, w, b)) - score
    nudged = float(predict(hours, w + h, b)) - score
    return {"error": error, "nudged_error": nudged, "square": error ** 2, "nudged_square": nudged ** 2,
            "rise": nudged ** 2 - error ** 2, "ratio": (nudged ** 2 - error ** 2) / h,
            "chain": 2 * error * hours}


def valley_floor_w(x, y, b):
    """The best w for a fixed b: where the slope of the loss along w is zero (the valley floor at this b)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float((np.mean(x * y) - b * np.mean(x)) / np.mean(x ** 2))


def direction_reversals(values):
    """How many times a sequence turns around (an increase followed by a decrease, or the reverse)."""
    steps = [b - a for a, b in zip(values, values[1:]) if b != a]
    return sum(1 for a, b in zip(steps, steps[1:]) if (a > 0) != (b > 0))


def gradient_step_work(rows, features):
    """The work of one gradient step for a linear model: one multiply-add per feature per row in the batch."""
    if rows < 1 or features < 1:
        raise ValueError("need at least one row and one feature")
    return rows * features
