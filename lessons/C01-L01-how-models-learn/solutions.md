# C01-L01 worked solutions (Part 1)

Reasoning first, numbers second. Every number here is reproduced by `code/solutions_c01_l01.py` or by the lesson
code, and checked by `code/tests/test_c01_l01_solutions.py`. Part 2's solutions are in
`../C01-L02-gradient-descent-in-practice/solutions.md`.

---

## Level 1

**1a.** x is the feature (hours studied, the input). ŷ ("y hat") is the prediction of the label. w is the weight,
the slope in points per hour. b is the bias, the intercept in points at zero hours. w and b are the parameters
that training adjusts. x is the input and is never adjusted.

**1b.** η is the learning rate. It is a **hyperparameter**: we choose it before training, and gradient descent
does not learn it. w and b are parameters, which training does learn (Google MLCC "Hyperparameters", source S09).

**1c.** Both go **up**. The step is *minus* the learning rate times the gradient, so a negative slope makes
a parameter grow: w changes by −η·(−435) = +435η and b by +60η. w moves about **seven times** further than b, because
its slope is about seven times larger (435.3 ÷ 60.2 ≈ 7.2). This is the same spot as the lesson's nudge table (w = 0, b
frozen at about 40.5), where minus the gradient is (+435, +60): "increase w a lot and b a little".

---

## Level 2

Data x = (0, 1, 2), y = (1, 3, 8), w = 2, b = 1, n = 3.

1. Predictions ŷ = 2·x + 1 = (1, 3, 5). Errors e = ŷ − y = (0, 0, −3). Squares (0, 0, 9), sum 9, so
   **MSE = 9/3 = 3**.
2. Σ e·x = 0·0 + 0·1 + (−3)·2 = −6, so **∂MSE/∂w = (2/3)(−6) = −4**.
   Σ e = −3, so **∂MSE/∂b = (2/3)(−3) = −2**.
3. w = 2 − 0.1·(−4) = **2.4**. b = 1 − 0.1·(−2) = **1.2**.
   New predictions (1.2, 3.6, 6.0), new errors (0.2, 0.6, −2.0), squares (0.04, 0.36, 4.0), sum 4.4, so
   **MSE = 4.4/3 ≈ 1.47**.
4. Yes, the loss went down, from 3 to about 1.47. w changed twice as much as b (0.4 against 0.2), because its partial
   derivative is twice as large. The only miss is at x = 2, and the w slope multiplies each error by its x, so that
   miss counts double in the w direction. Both slopes are negative, so both parameters went up: the line was too low
   where it missed.

---

## Level 3

```python
def mse_no_numpy(xs, ys, w, b):
    if len(xs) != len(ys) or len(xs) == 0:
        raise ValueError("xs and ys must be non-empty and the same length")
    total = 0.0
    for x, y in zip(xs, ys):
        error = (w * x + b) - y
        total += error * error
    return total / len(xs)


def mse_gradient_no_numpy(xs, ys, w, b):
    if len(xs) != len(ys) or len(xs) == 0:
        raise ValueError("xs and ys must be non-empty and the same length")
    sum_error_times_x = 0.0
    sum_error = 0.0
    for x, y in zip(xs, ys):
        error = (w * x + b) - y
        sum_error_times_x += error * x
        sum_error += error
    n = len(xs)
    return 2.0 / n * sum_error_times_x, 2.0 / n * sum_error
```

The nudge check (`nudge_ratio_no_numpy`): `(mse_no_numpy(xs, ys, w + h, b) - mse_no_numpy(xs, ys, w, b)) / h`, and
the same with `b + h`.

**How it is verified:** `test_part1_level3_loop_loss_and_gradient_match_numpy_and_the_nudge` compares both functions
at (0, 0), (4, 41) and (−2, 7.5) with `model.loss` and `model.mse_gradient` (relative tolerance 1e-12), and the
gradient with the nudge ratios at h = 0.0001 (absolute tolerance 0.01). It also checks that empty input raises. If you didn't want to rely on `model.mse_gradient` (its code is shown only in Part 2), the by-hand toy
values are enough: on the three toy students at w = 5, b = 45 the loop gradient is (−8, −4), as in the video at
`g25`–`g26` (`test_level3_loop_gradient_gives_the_by_hand_toy_slopes`).

**Why the nudge is a check and not a trainer:** it needs no algebra, so it cannot repeat a mistake in our
derivation. But every slope costs an extra loss calculation over all the data, for every parameter, and the ratio is
only approximate (Level 5 measures how approximate). With millions of parameters, that is far too slow.

---

## Level 4

The three bugs:

1. **`loss_value = sum(errors) / n`** forgets to square. Positive and negative errors cancel, and the "loss" can even
   be negative. Fix: `sum(e * e for e in errors) / n`.
2. **`grad_b` multiplies each error by x**, copying w's formula. b's slope is the same formula with x replaced by
   one, so b moves by the wrong amount (here twice too far: −8 instead of −4). Fix: `2.0 / n * sum(errors)`.
3. **`w = w + rate * grad_w`** steps *up* the slope, so w moves the wrong way. Fix: `w - rate * grad_w`.

On the lesson's three students with w = 5, b = 45 and rate 0.1, the buggy function returns loss **−2**, w = **4.2**
and b = **45.8**. The fixed function (`fixed_loss_and_step`) returns loss **6**, w = **5.8** and b = **45.4**, the
lesson's numbers (`test_part1_level4_buggy_step_is_wrong_and_fixed_step_matches_the_lesson`).

The negative loss should raise the alarm at once: an average of squares can never be below zero. The w step is the
second clue. Both slopes are negative, so both parameters should go up, but w went down.

---

## Level 5

`nudge_error_analysis()` on the training split, at w = 0 with b frozen at its best value (exact slope ≈ −435.29):

| h | ratio | gap = ratio − exact | gap ÷ h |
|---|---|---|---|
| 1 | −390.24 | 45.05 | 45.05 |
| 0.1 | −430.78 | 4.505 | 45.05 |
| 0.01 | −434.84 | 0.4505 | 45.05 |
| 0.001 | −435.24 | 0.04505 | 45.05 |

1–2. gap ÷ h is the **same number every time**, and it equals mean(x²) ≈ 45.05, the average of the squared training
   hours (`test_part1_level5_nudge_gap_is_h_times_mean_x_squared`).
3. With b fixed, write e = w·x + b − y. Then ((w + h)·x + b − y)² = (e + h·x)² = e² + 2·h·e·x + h²·x². Average over the
   students and subtract mean(e²): the change in the loss is h·(2·mean(e·x)) + h²·mean(x²). The first term is h times
   the exact slope. Divide by h: ratio = slope + h·mean(x²). So the gap is exactly h·mean(x²). It shrinks in
   proportion to h, which is why each row in the lesson's table got ten times closer.
4. To get within 0.01 you need h·45.05 < 0.01, so h below about 0.0002. Nudging is a fine referee for a formula you
   derived, but as a training method it costs an extra pass over all the data for every parameter at every step,
   and it is never exact. The formula gives every slope from a single pass over the errors.
