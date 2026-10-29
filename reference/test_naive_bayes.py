"""Gaussian naive Bayes: the Bend program against scikit-learn.

  reference/.venv/bin/python reference/test_naive_bayes.py

The Bend program (examples/classification/naive_bayes/main.bend) trains on the first
four fifths of the breast cancer dataset and scores the last fifth, with
the means and the population variances of every feature in every class.
This script fits scikit-learn's `GaussianNB` on the same split and
compares the accuracies.

No standardising is done on either side: a Gaussian is unchanged by an
affine change of one feature, so the two agree without it.
"""
import os
import re
import subprocess
import sys
import tempfile

from sklearn.naive_bayes import GaussianNB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR


def bend_result():
    """The accuracy, from the Bend program."""
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-nb-test")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "examples/classification/naive_bayes/main.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    return float(re.search(r"accuracy: ([0-9.eE+-]+)", out.stdout).group(1)), out.stdout


def sklearn_result():
    """The accuracy, from scikit-learn."""
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from test_knn import load

    x, y = load()
    split = len(x) * 4 // 5
    model = GaussianNB().fit(x[:split], y[:split])
    return float(model.score(x[split:], y[split:]))


def main():
    bend_accuracy, out = bend_result()
    sklearn_accuracy = sklearn_result()
    print(out.strip())
    print(f"sklearn: {sklearn_accuracy}")
    ok = True

    if bend_accuracy < 0.90:
        print(f"ERROR the accuracy is below 0.90: {bend_accuracy}")
        ok = False
    else:
        print(f"ok   the accuracy is at least 0.90: {bend_accuracy}")

    if abs(bend_accuracy - sklearn_accuracy) > 0.000001:
        print(f"ERROR the accuracy differs: {bend_accuracy} vs {sklearn_accuracy}")
        ok = False
    else:
        print(f"ok   the accuracy agrees: {bend_accuracy} vs {sklearn_accuracy}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
