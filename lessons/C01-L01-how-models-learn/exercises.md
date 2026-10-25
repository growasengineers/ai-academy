# C01-L01 exercises: How machines learn from examples

Work through these in order. Solutions are in `solutions.md`. Read them only after you have an answer of your own.
The coding exercises use the lesson code in `code/`. Run everything from that folder.

Notation: ŷ = w·x + b, error e = ŷ − y, MSE = (1/n) Σ e², ∂MSE/∂w = (2/n) Σ e·x, ∂MSE/∂b = (2/n) Σ e,
update w ← w − η·∂MSE/∂w, b ← b − η·∂MSE/∂b.

---

## Level 1 · Recognition

**1a.** In ŷ = w·x + b for the study-hours model, say which symbol is the feature, the label's prediction, the
weight and the bias. Say which of them are *parameters* and which one is the *input*.

**1b.** In the update rule w ← w − η·∂MSE/∂w, what is η called, and is it a parameter or a hyperparameter? Why?

**1c.** Three loss curves (loss against step) are described below. Match each to "learning rate too small",
"good learning rate" and "learning rate too large":
- (i) falls steeply at first, then flattens close to the minimum;
- (ii) rises every step and grows without limit;
- (iii) falls in an almost straight, shallow line and is still far from the minimum at the end.

**Success criteria:** every symbol is named correctly. 1b names the learning rate as a hyperparameter and gives the
reason (we choose it; training does not learn it). All three matches in 1c are right.

---

## Level 2 · Application (by hand, then check)

Three students studied x = (0, 1, 2) hours and scored y = (1, 3, 8). Start from the line w = 2, b = 1.

1. Compute the three predictions, the three errors, and the MSE.
2. Compute ∂MSE/∂w and ∂MSE/∂b.
3. Take one gradient-descent step with η = 0.1. Give the new w and b, and the new MSE.
4. Did the loss go down? Which parameter changed more, and why?

**Success criteria:** every intermediate value is shown (predictions, errors, squares, sums). Your numbers match
`level2_answers()` in `code/solutions_c01_l01.py` (run `python3 solutions_c01_l01.py`). You answer 4 using the
signs and sizes of the two partial derivatives.

---

## Level 3 · Implementation

Write `mse_gradient_no_numpy(xs, ys, w, b)` using only Python lists and one `for` loop (no NumPy). It must return
`(dMSE/dw, dMSE/db)` and raise `ValueError` on empty or mismatched input.

Then verify it. On the lesson's synthetic data (`data.make_study_data()`), compare it at three different (w, b)
pairs with `model.mse_gradient` and with the finite-difference `model.numerical_gradient`.

**Success criteria:** agreement with `mse_gradient` to about 12 significant digits, and with the finite difference
to at least 6 decimal places, at every pair. The function raises on `[]`. A short note explains why
the finite difference is a good referee but a bad training method.

---

## Level 4 · Debugging

This loop is supposed to fit a line, but it doesn't. Find **three** bugs, explain what each one does to training,
and fix them. (It is `buggy_descent` in `code/solutions_c01_l01.py`.)

```python
def buggy_descent(xs, ys, learning_rate, steps):
    w, b = 0.0, 0.0
    n = len(xs)
    for _ in range(steps):
        grad_w = sum(((w * x + b) - y) * x for x, y in zip(xs, ys))
        w = w + learning_rate * grad_w
        grad_b = sum((w * x + b) - y for x, y in zip(xs, ys)) * 2.0 / n
        b = b - learning_rate * grad_b
    return w, b
```

**Success criteria:** all three bugs found, each with its effect on training. Your fixed loop, run on the
standardised training hours with η = 0.1 for 200 steps, matches `evaluate.closed_form_fit` on the same scaled data
to 6 decimal places. Bonus: say which bug a finite-difference gradient check would have caught.

---

## Level 5 · Analysis (experiment)

Using `gradient_descent.learning_rate_sweep`, run 1000 steps from (0, 0) on the training split:

- on **raw hours** with rates 0.005, 0.01, 0.015, 0.02, 0.021, 0.022, 0.025;
- on **standardised hours** with rates 0.01, 0.1, 0.3, 0.5, 0.9, 0.99, 1.01, 1.1.

1. For each feature version, find the largest rate that does not diverge and the rate that reaches the minimum in
   the fewest steps.
2. The lesson's outline says the stability limit is 2 ÷ (largest curvature of the bowl). Compute that number for
   both versions (`level5_sweeps()` returns it) and compare it with what you measured.
3. On standardised hours, rates 0.1 and 0.9 need exactly the same number of steps, and 0.5 needs only one. Explain
   why, using the update w′ ← w′ − η·2·(w′ − w′*) that holds on standardised data.
4. In two or three sentences: why is "just pick a bigger learning rate" not a fix for the raw feature?

**Success criteria:** a table of your results. The measured edge sits between the two rates around the computed
bound, for both versions. Part 3 is explained with the factor (1 − 2η). Part 4 refers to the steep and flat
directions of the valley.

---

## Level 6 · Design

For each scenario, choose **closed-form least squares** or **gradient descent**, list the preprocessing you would
do, and describe how you would test the result. Justify each choice in two or three sentences, citing the
trade-offs from the lesson.

- (a) 2,000 houses, 6 numeric features, one-off analysis on a laptop.
- (b) Many millions of rows arriving each day, too many to refit from scratch every time.
- (c) A small neural network that predicts the same target from 50 features.

**Success criteria:** (a) uses the closed form (or `LinearRegression`). (b) uses gradient descent on batches, with
feature scaling. (c) uses gradient descent, because there is no closed form. Every answer includes a held-out test
set that is split before any fitting, with no duplicates across the split. Scaling statistics come from the
training data only. Each answer has a sentence on how you would notice failure (rising loss, or a large
train/test gap).

---

## Level 7 · Transfer

New synthetic problem: "practice hours vs typing speed". Gains are fast at first and then level off
(`make_curved_data()` in `code/solutions_c01_l01.py`).

1. Plot it, or print it sorted by hours, and describe the shape.
2. Fit a straight line on the training split and report the training and test MSE.
3. Add a second feature, x², and fit ŷ = w₁·x + w₂·x² + b by least squares (`fit_with_features`). Report the
   training and test MSE again.
4. Is the straight line underfitting or overfitting? What evidence tells you?
5. Name one risk of adding more and more powers of x, and how you would detect it.

**Success criteria:** the line's training MSE stays high (underfitting). The x² model at least halves both
training and test MSE (`level7_compare()`). Part 4 uses *training* error as the evidence. Part 5 names
overfitting/memorising, with a growing train/test gap as the detector.
