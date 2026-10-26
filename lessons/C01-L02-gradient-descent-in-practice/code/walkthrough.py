"""Run the whole lesson in order and print what happens at each stage.

    python3 walkthrough.py
"""
import data
from evaluate import closed_form_fit, memorizer_predict, sklearn_fit, train_test_report
from gradient_descent import gradient_descent, losses, standardize, unstandardize_params
from model import loss, mse, mse_gradient


def main():
    # 1. The data (synthetic: generated from a known line plus noise).
    hours, scores = data.make_study_data()
    x_train, y_train, x_test, y_test, _, _ = data.split_train_test(hours, scores)
    print(f"{len(hours)} synthetic students -> {len(x_train)} train, {len(x_test)} test")

    # 2. The by-hand example.
    print("toy MSE at w=5, b=45:", loss(data.TOY_HOURS, data.TOY_SCORES, 5, 45))
    print("toy gradient (dw, db):", mse_gradient(data.TOY_HOURS, data.TOY_SCORES, 5, 45))

    # 3. Gradient descent on raw hours, three learning rates.
    for rate in (0.001, 0.02, 0.025):
        run = gradient_descent(x_train, y_train, rate, 1000)
        status = "DIVERGED" if run["diverged"] else "ok"
        print(f"rate {rate:<6} {status:8s} steps {run['steps_run']:4d}  final loss {losses(run)[-1]:.4g}")

    # 4. Standardise the feature, descend, convert back to hours.
    x_scaled, mean, std = standardize(x_train)
    run = gradient_descent(x_scaled, y_train, 0.1, 100)
    w, b = unstandardize_params(run["w"], run["b"], mean, std)
    print(f"from scratch : w={w:.4f}  b={b:.4f}")
    print("closed form  : w={:.4f}  b={:.4f}".format(*closed_form_fit(x_train, y_train)))
    print("scikit-learn : w={:.4f}  b={:.4f}".format(*sklearn_fit(x_train, y_train)))

    # 5. Generalisation: train vs test, and why a perfect training score can lie.
    report = train_test_report(x_train, y_train, x_test, y_test, w, b)
    print(f"line      train MSE {report['train_mse']:.2f}  test MSE {report['test_mse']:.2f}")
    memo_train = mse(y_train, memorizer_predict(x_train, y_train, x_train))
    memo_test = mse(y_test, memorizer_predict(x_train, y_train, x_test))
    print(f"memoriser train MSE {memo_train:.2f}  test MSE {memo_test:.2f}")


if __name__ == "__main__":
    main()
