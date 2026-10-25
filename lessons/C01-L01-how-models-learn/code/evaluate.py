"""Checking our work: closed-form least squares, scikit-learn, and a train/test comparison."""
import numpy as np

from model import mse, predict


def closed_form_fit(x, y):
    """Least-squares line straight from linear algebra (no iterations).

    We stack a column of x next to a column of ones, so that [x, 1] @ [w, b] = w*x + b,
    and numpy.linalg.lstsq returns the (w, b) that minimises the sum of squared errors.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([x, np.ones_like(x)])
    (w, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(w), float(b)


def sklearn_fit(x, y):
    """The same model fitted by scikit-learn's LinearRegression (expects a 2-D feature table)."""
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(np.asarray(x, dtype=float).reshape(-1, 1), np.asarray(y, dtype=float))
    return float(model.coef_[0]), float(model.intercept_)


def memorizer_predict(x_train, y_train, x_new):
    """A deliberately bad 'model' that memorises: answer with the score of the training student
    whose hours are closest. It is perfect on its own training data, which is exactly the trap."""
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    nearest = np.abs(x_new[:, None] - x_train[None, :]).argmin(axis=1)
    return y_train[nearest]


def repeated_memorizer_experiment(n_datasets=200, first_seed=1000):
    """Is the memoriser's worse test error just luck of one split? Repeat on many fresh datasets.

    Each dataset is generated and split with its own seed (same recipe as the lesson's data).
    Returns the average test MSE of the least-squares line and of the memoriser, and how many
    datasets the memoriser lost (test MSE strictly higher than the line's).
    """
    from data import make_study_data, split_train_test

    line_errors, memo_errors = [], []
    for seed in range(first_seed, first_seed + n_datasets):
        x, y = make_study_data(seed=seed)
        x_train, y_train, x_test, y_test, _, _ = split_train_test(x, y, seed=seed)
        w, b = closed_form_fit(x_train, y_train)
        line_errors.append(mse(y_test, predict(x_test, w, b)))
        memo_errors.append(mse(y_test, memorizer_predict(x_train, y_train, x_test)))
    line_errors, memo_errors = np.array(line_errors), np.array(memo_errors)
    return {
        "n_datasets": n_datasets,
        "line_mean_test_mse": float(line_errors.mean()),
        "memorizer_mean_test_mse": float(memo_errors.mean()),
        "memorizer_worse_count": int(np.sum(memo_errors > line_errors)),
    }


def outlier_refit(x_train, y_train, index, wrong_score):
    """Refit the least-squares line after replacing ONE training score (a typo) with wrong_score.

    Shows how squared error lets a single bad point drag the whole line. Works on a copy.
    """
    y_bad = np.array(y_train, dtype=float).copy()
    y_bad[index] = wrong_score
    return closed_form_fit(x_train, y_bad)


def train_test_report(x_train, y_train, x_test, y_test, w, b):
    """MSE of the line (w, b) on the data it learned from and on data it never saw."""
    return {
        "train_mse": mse(y_train, predict(x_train, w, b)),
        "test_mse": mse(y_test, predict(x_test, w, b)),
    }
