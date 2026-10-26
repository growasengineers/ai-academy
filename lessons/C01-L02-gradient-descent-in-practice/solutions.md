# C01-L02 worked solutions (Part 2)

Reasoning first, numbers second. Every number here is reproduced by `code/solutions_c01_l01.py` or by the lesson
code (the same code as Part 1), and checked by `code/tests/test_c01_l02_solutions.py`. Part 1's solutions are in
`../C01-L01-how-models-learn/solutions.md`.

---

## Level 1

**1a.** (i) good rate: a fast early drop, then convergence near the minimum. (ii) too large: divergence, with each
overshoot landing higher (in the lesson, rate 0.025 on raw hours passed five billion by step 27). (iii) too small:
safe but slow (in the lesson, rate 0.001 was still at about 139 after 1000 steps, against a minimum of 15.76).

**1b.** Mean **0** and standard deviation **1**. A value of 0 is a student with exactly the average number of hours
(on our training set, 6.23 real hours); +1 is one standard deviation (about 2.5 hours) above average.

**1c.** w and b are fitted on the **training set** (30 students). The fitted line is judged on the **test set**
(10 students), which the model never saw. It is split off before any training, with no student in both sets: if
test rows leak into training, the model can tune itself to them, and the test score becomes a training score in
disguise (Google MLCC "Dividing the original dataset", source S11).

---

## Level 2

Hours x = (2, 4, 4, 4, 5, 5, 7, 9), n = 8.

1. Mean = 40 ÷ 8 = **5**. Deviations from the mean: (−3, −1, −1, −1, 0, 0, 2, 4); squares (9, 1, 1, 1, 0, 0, 4, 16),
   sum 32, so the variance is 32 ÷ 8 = 4 and the **standard deviation is 2**.
2. Scaled hours x′ = (x − 5) ÷ 2 = **(−1.5, −0.5, −0.5, −0.5, 0, 0, 1, 2)**.
3. One scaled unit is 2 real hours, so the weight per real hour is **w = 6 ÷ 2 = 3**. The average student (5 hours)
   sits at scaled 0, where the scaled line predicts 70. So 3 × 5 + b = 70, and **b = 70 − 15 = 55**.
4. At 7 hours: scaled x′ = (7 − 5) ÷ 2 = 1, so 6 × 1 + 70 = **76**. Real: 3 × 7 + 55 = **76**. They agree.

Checked by `test_part2_level2_scaling_by_hand` (`scaling_answers()`, which uses the lesson's `standardize` and
`unstandardize_params`).

---

## Level 3

```python
def gradient_descent_no_numpy(xs, ys, learning_rate, steps, divergence_factor=1e6):
    if len(xs) != len(ys) or len(xs) == 0:
        raise ValueError("xs and ys must be non-empty and the same length")
    w, b = 0.0, 0.0
    start = mse_no_numpy(xs, ys, w, b)
    history = [(0, w, b, start)]
    for step in range(1, steps + 1):
        grad_w, grad_b = mse_gradient_no_numpy(xs, ys, w, b)      # both slopes from the same spot
        w, b = w - learning_rate * grad_w, b - learning_rate * grad_b
        current = mse_no_numpy(xs, ys, w, b)
        history.append((step, w, b, current))
        if not math.isfinite(current) or current > divergence_factor * max(start, 1e-12):
            return {"w": w, "b": b, "history": history, "diverged": True, "steps_run": step}
    return {"w": w, "b": b, "history": history, "diverged": False, "steps_run": steps}
```

`mse_no_numpy` and `mse_gradient_no_numpy` are the plain-loop loss and gradient (Part 1's level 3 exercise; both are
in `code/solutions_c01_l01.py`).

**Evidence:** `test_part2_level3_list_loop_matches_the_lesson_loop`. At rate 0.01 for 50 steps the whole history
matches `gradient_descent.gradient_descent` (relative tolerance 1e-9). At rate 0.025 both loops report divergence
and stop at step **27**, the lesson's number. Empty input raises.

---

## Level 4

The three bugs:

1. **`w = w + learning_rate * grad_w`** has the wrong sign. It steps *uphill*, so the loss grows instead of
   shrinking. Fix: `w = w - learning_rate * grad_w`.
2. **`grad_b` is computed after `w` has changed**, so the two slopes come from different points. The update is no
   longer a gradient step. Fix: compute both slopes before updating either parameter.
3. **`grad_w` is missing the `2.0 / n` factor.** The step in w is then 2/n times too large relative to b, here
   15 times (n = 30 gives 2/n = 1/15). On its own, this rescales the effective learning rate for w and can tip it
   into divergence. Fix: multiply by `2.0 / n`.

Fixed version (`fixed_descent`):

```python
for _ in range(steps):
    grad_w = sum(((w * x + b) - y) * x for x, y in zip(xs, ys)) * 2.0 / n
    grad_b = sum((w * x + b) - y for x, y in zip(xs, ys)) * 2.0 / n
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b
```

**Evidence:** on standardised training hours with η = 0.1 for 200 steps, `fixed_descent` returns
w ≈ 12.0595, b ≈ 70.6367. The closed form on the same scaled data gives the same values to within 1e-6
(`test_level4_buggy_loop_fails_and_fixed_loop_matches_closed_form`). The buggy loop runs away: after only five
steps, w is already about −12,337.

**Which bugs a gradient check catches:** a finite-difference check of the *gradient function* catches bug 3 (a
wrong formula). Bugs 1 and 2 live in the *update loop*, not the gradient, so you catch them by watching the loss:
it should never rise at a small learning rate. The lesson's test `test_good_rate_loss_never_increases` does exactly
that.

---

## Level 5

Results from `level5_sweeps()` (1000 steps from (0, 0); "steps" means steps to get within 0.1 % of the minimum
training MSE, 15.76):

| raw rate | outcome | | scaled rate | outcome |
|---|---|---|---|---|
| 0.005 | not reached (loss 29.75) | | 0.01 | 315 steps |
| 0.01 | not reached (loss 16.68) | | 0.1 | 29 steps |
| 0.015 | not reached (loss 15.82) | | 0.3 | 7 steps |
| 0.02 | 874 steps | | 0.5 | 1 step |
| 0.021 | 832 steps (fastest) | | 0.9 | 29 steps |
| 0.022 | **diverged** | | 0.99 | 315 steps |
| 0.025 | **diverged** | | 1.01 | **diverged** |
| | | | 1.1 | **diverged** |

1. Raw hours: the largest safe rate is 0.021, which is also the fastest (832 steps). Standardised hours: the largest
   safe rate is 0.99, and the fastest is 0.5 (one step).
2. The MSE bowl is a quadratic with Hessian H = 2·[[mean(x²), mean(x)], [mean(x), 1]]. Gradient descent on a
   quadratic is stable only when η < 2/λ_max. For raw hours, 2/λ_max ≈ **0.0218**: between 0.021 (safe) and 0.022
   (diverges), exactly as measured. For standardised hours, mean(x′) = 0 and mean(x′²) = 1, so H = 2·I, λ_max = 2
   and the bound is **1.0**: between 0.99 and 1.01, again as measured. (The idea that the Hessian's eigenvalues set
   the scale of the learning rate is from Goodfellow et al., *Deep Learning*, §4.3.1, source S13.)
3. On standardised data the two parameters separate. Each one's distance to its optimum is multiplied by (1 − 2η)
   at every step. For η = 0.1 the factor is 0.8; for η = 0.9 it is −0.8. The size is the same, but the second one
   flips sign each step, so it zig-zags. Both shrink at the same rate, so both need 29 steps. For η = 0.5 the factor
   is 0, so a single step lands exactly on the minimum. Above η = 1 the factor's size exceeds 1, and every step
   grows the distance: divergence.
4. On raw hours the bowl's curvature differs by a factor of about 338 between its steepest and flattest directions
   (`raw_curvature_ratio` in `facts.py`). The rate must stay below about 0.0218 for the steep direction. In the flat
   direction each step then shrinks the remaining distance by only about 1 − 0.021·0.27 ≈ 0.994, which is why
   hundreds of steps are needed. A bigger rate would diverge first. Changing the units makes both curvatures equal,
   so no compromise is needed.

---

## Level 6

- **(a) Closed form** (`LinearRegression`, or `np.linalg.lstsq` with an intercept column). There are 2,000 rows and
  6 features, so the cost, which grows with rows × features² (scikit-learn user guide §1.1.1.2, S04), is trivial.
  The answer is exact, and there is no learning rate to tune. Scaling is optional for the fit but helps when
  comparing coefficients. **Testing:** split off a test set first (for example 25 %) and remove any duplicate rows
  across the split. Report train and test MSE. A large gap, or a test MSE far above the noise level, means something
  is wrong.
- **(b) Gradient descent on batches** (for example scikit-learn's `SGDRegressor`, which its guide recommends for
  problems with a large number of training samples, S05). Batches let you update without refitting on everything.
  **Preprocessing:** standardise with mean and standard deviation computed on training data only, and apply the
  same transform to new data. **Testing:** keep a test set from a later time period, untouched by training, and
  monitor the training loss for divergence (a rising loss means lower the learning rate).
- **(c) Gradient descent**, because a neural network has no closed-form solution, and nearly all of deep learning is
  trained with stochastic gradient descent (Goodfellow et al. §5.9, S14). **Preprocessing:** scale all 50 features.
  **Testing:** a held-out test set split before training, plus a gradient check on a small version of the model.
  Watch for overfitting: training error falling while test error rises.

---

## Level 7

`level7_compare()` on `make_curved_data()` (synthetic, seed 21, 30 train / 10 test):

| model | train MSE | test MSE |
|---|---|---|
| straight line ŷ = w·x + b | 38.56 | 25.99 |
| with x²: ŷ = w₁·x + w₂·x² + b | 9.13 | 1.60 |

1. Speed rises steeply over the first few hours, then flattens near 90: a curve that bends downward.
2. The line's training MSE is 38.56 (RMSE ≈ 6.2 words per minute) and its test MSE is 25.99. (Here the test MSE is
   lower than the training MSE; with only 10 test points this kind of variation is normal.)
3. Adding x² cuts training MSE to 9.13 and test MSE to 1.60, more than halving both
   (`test_level7_line_underfits_a_curve_and_squared_feature_fixes_it`).
4. The line is **underfitting**. The evidence is its *training* error: it stays high even though the line is the
   best possible line for this data (least squares). That matches the textbook definition: unable to obtain a
   sufficiently low training error (S14 §5.2). More optimisation cannot fix the wrong shape.
5. Adding many more powers (x³, x⁴, … x¹⁵) raises capacity until the model can memorise noise, like the memoriser in
   the lesson. You detect it with the train/test gap: training error keeps falling while test error starts rising.
   The fix is to choose model complexity on a validation set (a later lesson) and touch the test set only once at
   the end.
