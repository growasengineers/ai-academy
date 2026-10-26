"""The model (a straight line), its loss (mean squared error) and the loss's gradient.

    prediction      y_hat = w * x + b
    error           e_i   = y_hat_i - y_i
    loss            MSE   = (1/n) * sum(e_i ** 2)
    gradient        dMSE/dw = (2/n) * sum(e_i * x_i)
                    dMSE/db = (2/n) * sum(e_i)
"""
import numpy as np


def predict(x, w, b):
    """The line: one prediction per input. Works on a single number or a whole array."""
    return w * np.asarray(x, dtype=float) + b


def mse(y_true, y_pred):
    """Mean squared error: square every error (so signs cannot cancel), then average."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("MSE of zero examples is undefined")
    errors = y_pred - y_true
    return float(np.mean(errors ** 2))


def loss(x, y, w, b):
    """MSE of the line (w, b) on the data (x, y)."""
    return mse(y, predict(x, w, b))


def loss_curve(x, y, w_values, b):
    """Loss at each w in w_values with b held fixed: the U-shaped curve we draw first."""
    return [loss(x, y, w, b) for w in w_values]


def loss_grid(x, y, w_values, b_values):
    """Loss over a grid of (w, b): rows follow b_values, columns follow w_values (for a contour map)."""
    return [[loss(x, y, w, b) for w in w_values] for b in b_values]


def mse_gradient(x, y, w, b):
    """Return (dMSE/dw, dMSE/db): how fast the loss rises when w or b is nudged up.

    Derived with the chain rule on (w*x_i + b - y_i) ** 2:
    the outer square gives 2*e_i, and the inner part changes by x_i per unit of w and 1 per unit of b.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        raise ValueError("the gradient over zero examples is undefined")
    errors = predict(x, w, b) - y
    n = x.size
    grad_w = 2.0 / n * float(np.sum(errors * x))
    grad_b = 2.0 / n * float(np.sum(errors))
    return grad_w, grad_b


def numerical_gradient(x, y, w, b, h=1e-5):
    """Finite-difference estimate of the gradient, used only to CHECK mse_gradient.

    Central difference: (loss(w + h) - loss(w - h)) / (2h) approximates the slope at w.
    """
    grad_w = (loss(x, y, w + h, b) - loss(x, y, w - h, b)) / (2 * h)
    grad_b = (loss(x, y, w, b + h) - loss(x, y, w, b - h)) / (2 * h)
    return grad_w, grad_b
