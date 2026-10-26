# C01-L02 exercises (Part 2): gradient descent in code, learning rates and testing

These exercises cover Part 2 of the lesson: the descent loop in code, the gradient check, the learning rate,
standardising, the exact check and the test set. Part 1's exercises (the loss, the slopes and a step by hand) are in
`../C01-L01-how-models-learn/exercises.md`.

Work through them in order. Solutions are in `solutions.md`. Read them only after you have an answer of your own.
The coding exercises use the lesson code in `code/` (the same code as Part 1). Run everything from that folder.

Notation: ŷ = w·x + b, error e = ŷ − y, MSE = (1/n) Σ e², ∂MSE/∂w = (2/n) Σ e·x, ∂MSE/∂b = (2/n) Σ e,
update w ← w − η·∂MSE/∂w, b ← b − η·∂MSE/∂b, standardised feature x′ = (x − mean) / standard deviation.

---

## Level 1 · Recognition

**1a.** Three loss curves (loss against step) are described below. Match each to "learning rate too small",
"good learning rate" and "learning rate too large":
- (i) falls steeply at first, then flattens close to the minimum;
- (ii) rises every step and grows without limit;
- (iii) falls in an almost straight, shallow line and is still far from the minimum at the end.

**1b.** After standardising a feature, what are its mean and its standard deviation? What does a value of 0 on the
new scale mean?

**1c.** Which set of students is used to fit w and b, and which is used to judge the fitted line? Why must the
second one be split off before any training?

**Success criteria:** all three matches in 1a are right. 1b gives mean 0 and standard deviation 1, and says 0 is the
average student. 1c names the training set and the test set and says that test rows seen in training would turn the
test score into a training score in disguise.

---

## Level 2 · Application (by hand, then check)

Eight students studied x = (2, 4, 4, 4, 5, 5, 7, 9) hours.

1. Compute the mean and the standard deviation of the hours (divide by n = 8, as `np.std` does).
2. Standardise every value.
3. On the scaled hours, a line has weight w′ = 6 and bias b′ = 70. Convert it back to real hours: what are w and b?
4. Check your conversion: predict the score for 7 real hours with both lines.

**Success criteria:** every intermediate value is shown. Your numbers match `scaling_answers()` in
`code/solutions_c01_l01.py`. Part 3 uses the two reasons from the lesson: one scaled unit is one standard deviation
of real hours, and both lines must agree at the average student.

---

## Level 3 · Implementation

Write `gradient_descent_no_numpy(xs, ys, learning_rate, steps)` using only Python lists and loops. Start at
w = b = 0. Return a dict with the final `w` and `b`, a `history` of (step, w, b, loss) starting with step 0,
`diverged`, and `steps_run`. Stop and report divergence as soon as the loss is not finite or exceeds one million
times its starting value, exactly like the lesson's loop.

Check it against `gradient_descent.gradient_descent` on the raw training hours: at rate 0.01 for 50 steps, the two
histories must match; at rate 0.025, both must report divergence at the same step.

**Success criteria:** both slopes come from the same (w, b) before either changes. The histories match to about 9
significant digits. At rate 0.025 both loops stop at step 27. The function raises `ValueError` on empty input.

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
4. In two or three sentences: why is "pick a bigger learning rate" not a fix for the raw feature?

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
