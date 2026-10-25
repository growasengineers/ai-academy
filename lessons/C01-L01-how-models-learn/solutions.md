# C01-L01 worked solutions

Reasoning first, numbers second. Every number here is reproduced by `code/solutions_c01_l01.py` or by the lesson
code, and checked by `code/tests/test_c01_l01_solutions.py`.

---

## Level 1

**1a.** x is the feature (hours studied, the input). ŷ ("y hat") is the prediction of the label. w is the weight,
the slope in points per hour. b is the bias, the intercept in points at zero hours. w and b are the parameters
that training adjusts. x is the input and is never adjusted.

**1b.** η is the learning rate. It is a **hyperparameter**: we choose it before training, and gradient descent
does not learn it. w and b are parameters, which training does learn (Google MLCC "Hyperparameters", source S09).

**1c.** (i) good rate: a fast early drop, then convergence near the minimum. (ii) too large: divergence, with each
overshoot landing higher (in the lesson, rate 0.025 on raw hours passed five billion by step 27). (iii) too small:
safe but slow (in the lesson, rate 0.001 was still at about 139 after 1000 steps, against a minimum of 15.76).

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

**How it is verified:** `test_level3_loop_gradient_matches_numpy_and_finite_difference` compares it at (0, 0),
(4, 41) and (−2, 7.5) with `model.mse_gradient` (relative tolerance 1e-12) and with the central finite difference
(tolerance 1e-6). It also checks that empty input raises.

**Why the finite difference is a referee and not a trainer:** it needs no algebra, so it cannot repeat a mistake in
our derivation. But it costs two extra loss evaluations per parameter for every gradient, and it is only
approximate: too large an h gives truncation error, too small an h gives rounding error. With millions of
parameters, that cost is prohibitive.

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
