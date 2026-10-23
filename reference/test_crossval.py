"""The cross-validation over the iris dataset: the Bend program against scikit-learn.

  reference/.venv/bin/python reference/test_crossval.py

The Bend program (examples/crossval/main.bend) fetches the iris dataset
from HuggingFace, standardises it with `mlbend-preprocess`, trains a
logistic regression with bend-ml's autodiff on four of every five
stratified folds, and scores the fifth with `mlbend-metrics`.

The folds here are not the same folds: scikit-learn shuffles each class
with numpy's generator, the Bend program shuffles with bend-kit-random's
xoshiro128**. What must agree is the accuracy of the whole procedure,
so the mean of the five folds is compared, with a tolerance that covers
the difference between two random partitions of a small dataset.
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

URL = "https://huggingface.co/datasets/mstz/iris/resolve/main/iris.data"
CLASSES = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}


def load():
    """The iris dataset, as a matrix and a label column, from the hub."""
    import urllib.request

    text = urllib.request.urlopen(URL, timeout=120).read().decode()
    rows, labels = [], []
    for line in text.split("\n"):
        if not line.strip():
            continue
        cells = line.split(",")
        rows.append([float(c) for c in cells[:-1]])
        labels.append(CLASSES[cells[-1].strip()])
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int64)


def bend_folds():
    """The accuracy of every fold, from the Bend program."""
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-crossval-test")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "examples/crossval/main.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    folds = [float(m) for m in re.findall(r"^fold \d+: ([0-9.eE+-]+)$", out.stdout, re.M)]
    mean = float(re.search(r"mean accuracy: ([0-9.eE+-]+)", out.stdout).group(1))
    return folds, mean


def sklearn_folds(x, y):
    """The accuracy of every fold, from scikit-learn.

    The scaler is fitted inside the fold (`Pipeline`), on the training rows
    only, as `examples/crossval` now does: standardising the whole matrix
    first would let the test rows into the means and the deviations.
    """
    scores = []
    for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(x[train], y[train])
        scores.append(float((model.predict(x[test]) == y[test]).mean()))
    return scores


def main():
    x, y = load()
    bend, bend_mean = bend_folds()
    sklearn = sklearn_folds(x, y)
    sklearn_mean = float(np.mean(sklearn))
    print(f"bend:    {[round(s, 4) for s in bend]} mean {bend_mean:.4f}")
    print(f"sklearn: {[round(s, 4) for s in sklearn]} mean {sklearn_mean:.4f}")
    ok = True
    if len(bend) != 5:
        print(f"ERROR the Bend program scored {len(bend)} folds, not 5")
        ok = False
    # two random partitions of 150 rows differ by a few rows of accuracy
    if abs(bend_mean - sklearn_mean) > 0.03:
        print(f"ERROR the mean accuracy differs: {bend_mean} against {sklearn_mean}")
        ok = False
    else:
        print(f"ok   the mean accuracy agrees: {bend_mean:.4f} against {sklearn_mean:.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
