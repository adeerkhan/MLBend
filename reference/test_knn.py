"""The k-nearest neighbours over the breast cancer dataset: Bend against scikit-learn.

  reference/.venv/bin/python reference/test_knn.py

The Bend program (examples/knn/main.bend) fetches the dataset from
HuggingFace, standardises every column, takes the first four fifths as
the training set and the last fifth as the test set, and lets the three
nearest neighbours vote. This script does the same with scikit-learn and
compares the accuracy.
"""
import os
import re
import subprocess
import sys

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR


def load():
    """The breast cancer dataset, as a matrix and a label column."""
    url = "https://huggingface.co/datasets/mstz/breast/resolve/main/cancer/train.csv"
    import urllib.request

    text = urllib.request.urlopen(url, timeout=120).read().decode()
    lines = [l for l in text.split("\n") if l.strip()]
    rows, labels = [], []
    for line in lines[1:]:
        cells = line.split(",")
        rows.append([float(c) for c in cells[:-1]])
        labels.append(float(cells[-1]))
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.float64)


def bend_knn():
    """The accuracy, from the Bend program."""
    binary = "/tmp/opencode/mlbend-knn-test"
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "examples/knn/main.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    accuracy = float(re.search(r"accuracy: ([0-9.eE+-]+)", out.stdout).group(1))
    return accuracy, out.stdout


def sklearn_knn():
    """The accuracy, from scikit-learn."""
    x, y = load()
    scaled = StandardScaler().fit_transform(x)
    split = len(x) * 4 // 5
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(scaled[:split], y[:split])
    return float(model.score(scaled[split:], y[split:]))


def main():
    bend_accuracy, out = bend_knn()
    sklearn_accuracy = sklearn_knn()
    print(out.strip())
    print(f"sklearn: {sklearn_accuracy}")
    # an F32 has about seven digits, and the two pipelines differ in the
    # order of the operations, so a small gap is expected
    if abs(bend_accuracy - sklearn_accuracy) > 0.02:
        print(f"ERROR the accuracy differs: {bend_accuracy} vs {sklearn_accuracy}")
        return 1
    print(f"ok   the accuracy agrees: {bend_accuracy} vs {sklearn_accuracy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
