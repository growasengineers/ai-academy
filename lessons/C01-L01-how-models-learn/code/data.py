"""The lesson's data: a SYNTHETIC "study hours vs exam score" set, and a tiny hand-worked example.

Nothing here is real student data. We invent the data on purpose, so that we know the true
rule behind it and can check whether learning recovers it:

    score = TRUE_W * hours + TRUE_B + noise,    noise ~ Normal(mean 0, standard deviation NOISE_SD)

A fixed seed makes every run produce exactly the same numbers.
"""
import numpy as np

# The hidden rule we use to generate the data (the model never sees these numbers).
TRUE_W = 5.0          # each extra hour of study adds five points, on average
TRUE_B = 40.0         # the expected score of a student who studies zero hours
NOISE_SD = 5.0        # real scores scatter around the line; this is how much
N_STUDENTS = 40
HOURS_LOW, HOURS_HIGH = 1.0, 10.0
SEED = 14

TEST_FRACTION = 0.25  # one student in four is held back for testing
SPLIT_SEED = 14

# Three students for the by-hand example (small whole numbers so the arithmetic fits on screen).
TOY_HOURS = np.array([1.0, 2.0, 3.0])
TOY_SCORES = np.array([53.0, 55.0, 63.0])


def make_study_data(n=N_STUDENTS, seed=SEED, true_w=TRUE_W, true_b=TRUE_B, noise_sd=NOISE_SD):
    """Return (hours, scores) for n synthetic students, hours to two decimals, scores to one."""
    rng = np.random.default_rng(seed)
    hours = rng.uniform(HOURS_LOW, HOURS_HIGH, size=n)
    noise = rng.normal(0.0, noise_sd, size=n)
    scores = true_w * hours + true_b + noise
    # Rounding keeps the numbers readable on screen; it does not change the lesson.
    return np.round(hours, 2), np.round(scores, 1)


def split_train_test(x, y, test_fraction=TEST_FRACTION, seed=SPLIT_SEED):
    """Shuffle the row positions once, then cut: the first part trains, the rest tests.

    Returns (x_train, y_train, x_test, y_test, train_idx, test_idx). Every row lands in exactly
    one part, so the model is never tested on a student it trained on.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    n = len(x)
    n_test = int(round(n * test_fraction))
    if n_test == 0 or n_test == n:
        raise ValueError("not enough rows to make both a training part and a test part")
    order = np.random.default_rng(seed).permutation(n)
    test_idx = np.sort(order[:n_test])
    train_idx = np.sort(order[n_test:])
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx], train_idx, test_idx
