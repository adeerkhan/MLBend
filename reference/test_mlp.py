"""The two-layer network: the Bend program against scikit-learn.

  reference/.venv/bin/python reference/test_mlp.py

The Bend program (examples/classification/mlp/main.bend) trains eight rectified hidden
units over the iris dataset for 400 full-batch steps of SGD, starting
from a fixed small pattern, and prints the loss before, the loss after
and the accuracy. This script checks the same shape of result with
scikit-learn's MLPClassifier: the loss must fall, the accuracy must be
at least 0.90, and the two accuracies must be within 0.06 of each other.

The two do not train the same way — Adam against plain SGD, another
initialisation — so what is compared is the result of the procedure, not
the path to it.
"""
import os
import re
import subprocess
import sys
import tempfile

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR


def bend_result():
    """The two losses and the accuracy, from the Bend program."""
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-mlp-test")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "examples/classification/mlp/main.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    text = out.stdout
    before = float(re.search(r"loss before: ([0-9.eE+-]+)", text).group(1))
    after = float(re.search(r"loss after:  ([0-9.eE+-]+)", text).group(1))
    accuracy = float(re.search(r"accuracy: ([0-9.eE+-]+)", text).group(1))
    return before, after, accuracy


def sklearn_result():
    """The same surface with scikit-learn: eight hidden units, standardised."""
    iris = load_iris()
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=0),
    )
    model.fit(iris.data, iris.target)
    return model.score(iris.data, iris.target)


def main():
    before, after, accuracy = bend_result()
    sklearn_accuracy = sklearn_result()
    ok = True

    if after >= before:
        print(f"ERROR the loss did not fall: {before} -> {after}")
        ok = False
    else:
        print(f"ok   the loss fell: {before} -> {after}")

    if accuracy < 0.90:
        print(f"ERROR the accuracy is below 0.90: {accuracy}")
        ok = False
    else:
        print(f"ok   the accuracy is at least 0.90: {accuracy}")

    if abs(accuracy - sklearn_accuracy) > 0.06:
        print(f"ERROR the accuracies are too far apart: {accuracy} vs {sklearn_accuracy}")
        ok = False
    else:
        print(f"ok   the accuracies agree: {accuracy} against {sklearn_accuracy}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
