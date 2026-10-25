"""Gradient descent from scratch, plus feature standardisation.

The whole algorithm is four lines inside a loop:
    1. compute the gradient at the current (w, b)
    2. w = w - learning_rate * dMSE/dw
    3. b = b - learning_rate * dMSE/db
    4. repeat
Everything else here is record keeping so we can draw and check what happened.
"""
import math

import numpy as np

from model import loss, mse_gradient

# If the loss ever grows past this multiple of where it started, the run is going the wrong way
# for good: we stop and report divergence instead of letting numbers overflow to infinity.
DIVERGENCE_FACTOR = 1e6


def gradient_descent(x, y, learning_rate, steps, w0=0.0, b0=0.0):
    """Run `steps` updates of plain (full-batch) gradient descent on the MSE of a line.

    Returns a dict:
        w, b         the final parameters
        history      list of (step, w, b, loss), starting with step 0 = the starting point
        diverged     True if the loss blew up (then the run stops early)
        steps_run    how many updates actually happened
    """
    if learning_rate < 0:
        raise ValueError("learning_rate must be zero or positive")
    if steps < 0:
        raise ValueError("steps must be zero or positive")
    w, b = float(w0), float(b0)
    start_loss = loss(x, y, w, b)
    history = [(0, w, b, start_loss)]
    diverged = False
    for step in range(1, steps + 1):
        grad_w, grad_b = mse_gradient(x, y, w, b)
        # Both updates use the gradient from the SAME point, so compute both before changing either.
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
        current = loss(x, y, w, b)
        history.append((step, w, b, current))
        if not math.isfinite(current) or current > DIVERGENCE_FACTOR * max(start_loss, 1e-12):
            diverged = True
            break
    return {"w": w, "b": b, "history": history, "diverged": diverged, "steps_run": len(history) - 1}


def losses(result):
    """Just the loss column of a run's history (handy for plotting a loss curve)."""
    return [row[3] for row in result["history"]]


def first_step_within(result, target_loss, rel_tol=0.001):
    """First step whose loss is within rel_tol (0.1% by default) of target_loss; None if never."""
    for step, _w, _b, value in result["history"]:
        if value <= target_loss * (1 + rel_tol):
            return step
    return None


def learning_rate_sweep(x, y, rates, steps, target_loss, w0=0.0, b0=0.0):
    """Run gradient descent once per learning rate and summarise each run.

    Returns one dict per rate: rate, diverged, steps_to_target (None if never within 0.1%),
    final_loss, steps_run.
    """
    summary = []
    for rate in rates:
        run = gradient_descent(x, y, rate, steps, w0, b0)
        summary.append({
            "rate": rate,
            "diverged": run["diverged"],
            "steps_to_target": None if run["diverged"] else first_step_within(run, target_loss),
            "final_loss": losses(run)[-1],
            "steps_run": run["steps_run"],
        })
    return summary


def standardize(x):
    """Return (x_scaled, mean, std) with x_scaled = (x - mean) / std: centred on 0, spread 1."""
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std == 0.0:
        raise ValueError("cannot standardise a constant feature (its standard deviation is 0)")
    return (x - mean) / std, mean, std


def unstandardize_params(w_scaled, b_scaled, mean, std):
    """Convert a line learned on scaled x back to raw units.

    y = w_s * (x - mean) / std + b_s  =  (w_s / std) * x + (b_s - w_s * mean / std)
    """
    w = w_scaled / std
    b = b_scaled - w * mean
    return w, b
