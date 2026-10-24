"""The cross-validation over the iris dataset: the Bend program against scikit-learn.

  reference/.venv/bin/python reference/test_crossval.py

The Bend program (examples/crossval/main.bend) fetches the iris dataset
from HuggingFace, standardises it with `mlbend-preprocess`, trains a
logistic regression with bend-ml's autodiff on four of every five
stratified folds, and scores the fifth with `mlbend-metrics`.

The Bend program prints the fold of every row. The reference reads that
assignment, standardises inside each fold and runs the same 60 SGD steps,
so the two compare the same partition and the same model; the mean agrees
to F32 precision.
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

URL = "https://huggingface.co/datasets/mstz/iris/resolve/e99306d5df05d2ea66a188ea936037f6588a20e9/iris.data"
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
    out = subprocess.run([binary, "42"], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    folds = [float(m) for m in re.findall(r"^fold \d+: ([0-9.eE+-]+)$", out.stdout, re.M)]
    mean = float(re.search(r"mean accuracy: ([0-9.eE+-]+)", out.stdout).group(1))
    assignment = [int(v) for v in re.search(r"^folds: (.*)$", out.stdout, re.M).group(1).split()]
    return folds, mean, assignment


def sklearn_folds(x, y):
    """The adapter's reference: scikit-learn on its own stratified folds."""
    scores = []
    for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(x[train], y[train])
        scores.append(float((model.predict(x[test]) == y[test]).mean()))
    return scores


def main():
    x, y = load()
    bend, bend_mean, assignment = bend_folds()
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from experiment import reference_iris_crossval
    reference = reference_iris_crossval({"folds": assignment})
    print(f"bend:      {[round(s, 4) for s in bend]} mean {bend_mean:.4f}")
    print(f"reference: mean {reference['mean_accuracy']:.4f}")
    ok = True
    if len(bend) != 5:
        print(f"ERROR the Bend program scored {len(bend)} folds, not 5")
        ok = False
    if abs(bend_mean - reference["mean_accuracy"]) > 0.00001:
        print(f"ERROR the means differ: {bend_mean} against {reference['mean_accuracy']}")
        ok = False
    if abs(min(bend) - reference["fold_min"]) > 0.00001 or abs(max(bend) - reference["fold_max"]) > 0.00001:
        print("ERROR a fold's score differs")
        ok = False
    if ok:
        print("ok   the same folds and the same model agree")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
