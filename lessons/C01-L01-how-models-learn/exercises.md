# C01-L01 exercises (Part 1): loss, slopes and gradient descent by hand

These exercises cover Part 1 of the lesson: the model, the loss, the slope, the gradient and the update rule. Part 2's
exercises (learning rates, scaling, testing, design and transfer) are in
`../C01-L02-gradient-descent-in-practice/exercises.md`.

Work through them in order. Solutions are in `solutions.md`. Read them only after you have an answer of your own.
The coding exercises use the lesson code in `code/`. Run everything from that folder.

Notation: ŷ = w·x + b, error e = ŷ − y, MSE = (1/n) Σ e², ∂MSE/∂w = (2/n) Σ e·x, ∂MSE/∂b = (2/n) Σ e,
update w ← w − η·∂MSE/∂w, b ← b − η·∂MSE/∂b.

---

## Level 1 · Recognition

**1a.** In ŷ = w·x + b for the study-hours model, say which symbol is the feature, the label's prediction, the
weight and the bias. Say which of them are *parameters* and which one is the *input*.

**1b.** In the update rule w ← w − η·∂MSE/∂w, what is η called, and is it a parameter or a hyperparameter? Why?

**1c.** At some (w, b), the gradient of the loss is (−435, −60). Does gradient descent move w up or down? And b?
Which of the two moves further, and why?

**Success criteria:** every symbol is named correctly. 1b names the learning rate as a hyperparameter and gives the
reason (we choose it; training does not learn it). 1c says both go up, because the step is *minus* the gradient, and
w moves about seven times further than b because its slope is about seven times larger.

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

Using only Python lists and `for` loops (no NumPy), write:

- `mse_no_numpy(xs, ys, w, b)`, the mean squared error, and
- `mse_gradient_no_numpy(xs, ys, w, b)`, returning `(dMSE/dw, dMSE/db)` from the two formulas.

Both must raise `ValueError` on empty or mismatched input.

Then check them two ways on the lesson's synthetic data (`data.make_study_data()`), at three different (w, b) pairs:
against `model.loss` and `model.mse_gradient`, and against the lesson's nudging method, which divides the change in
your loss by a nudge of h = 0.0001, once for w and once for b.
(`model.loss` is the loss function shown in the video at `l17`–`l19`. `model.mse_gradient` is the lesson's NumPy
version of the same two slope formulas from `g19`–`g20`; this part doesn't show its code, and Part 2 walks through it
at `c1`–`c2b`. If you'd rather not use code you haven't seen, check the gradient against the nudge ratios and the
by-hand values from the video instead: on the three toy students with w = 5 and b = 45 it must give −8 and −4.)

**Success criteria:** agreement with `model.loss` and `model.mse_gradient` to about 12 significant digits. Agreement
with the nudge ratios to within 0.01 at every pair. Both functions raise on `[]`. A short note says why the nudge is
a good check but a slow way to get a gradient.

---

## Level 4 · Debugging

This function should return the MSE and the parameters after one gradient-descent step, but it doesn't. Find
**three** bugs, explain what each does, and fix them. (It is `buggy_loss_and_step` in `code/solutions_c01_l01.py`.)

```python
def buggy_loss_and_step(xs, ys, w, b, rate):
    n = len(xs)
    errors = [(w * x + b) - y for x, y in zip(xs, ys)]
    loss_value = sum(errors) / n
    grad_w = 2.0 / n * sum(e * x for e, x in zip(errors, xs))
    grad_b = 2.0 / n * sum(e * x for e, x in zip(errors, xs))
    w = w + rate * grad_w
    b = b - rate * grad_b
    return loss_value, w, b
```

**Success criteria:** all three bugs found, each with its effect. On the lesson's three students (hours 1, 2, 3;
scores 53, 55, 63) with w = 5, b = 45 and rate 0.1, your fixed function returns the lesson's numbers: loss 6, then
w = 5.8 and b = 45.4. Say what the buggy version returns on the same input, and which of its outputs should have
made you suspicious at once.

---

## Level 5 · Analysis (experiment)

The lesson's nudge table at w = 0 (b frozen at its best value) gave rise-over-run ratios of about −390.2, −430.8,
−434.8 and −435.2 for nudges of 1, 0.1, 0.01 and 0.001. The exact slope there is about −435.29.

1. For each nudge h, compute the gap between the ratio and the exact slope, and the gap divided by h
   (`nudge_error_analysis()` in `code/solutions_c01_l01.py` does it; try it yourself first with `by_hand.nudge_table`).
2. What do you notice about gap ÷ h? Compare it with the average of the squared training hours, mean(x²).
3. Explain why. Hint: with b fixed, the loss is a parabola in w. Expand mean((w + h)·x + b − y)² − mean(w·x + b − y)².
4. How small must h be for the ratio to be within 0.01 of the exact slope? Why does that make nudging a poor way to
   train a model with millions of parameters, even though it is a good way to check a formula?

**Success criteria:** a table of h, ratio, gap and gap ÷ h. gap ÷ h is the same number every time, mean(x²) ≈ 45.05,
and the expansion shows why: the extra term is exactly h·mean(x²). Part 4 gives h below about 0.0002 and argues from
the cost (an extra loss calculation per parameter per nudge).
