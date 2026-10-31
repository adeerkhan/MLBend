"""Compares the metrics of mlbend-metrics with NumPy and scikit-learn.

  reference/.venv/bin/python reference/test_metrics.py

The Bend values are F32 and the references are f64, so the comparison is
within a relative 1e-5: F32 is not a real number and the two round
differently.
"""
import os
import subprocess
import sys

import numpy as np
from sklearn import metrics as sk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")

# the dataset of metrics/tests/ok.bend
TARGETS = [3.0, 0.5, 2.5, 7.0]
PREDICTIONS = [2.5, 1.0, 2.0, 8.0]
LABELS = [0, 1, 1, 0]
PREDICTED = [0, 0, 1, 1]

# the class probabilities of metrics/tests/ok.bend, one row per sample
PROBABILITIES = np.array([[0.7, 0.2, 0.1],
                          [0.1, 0.8, 0.1],
                          [0.2, 0.2, 0.6],
                          [0.6, 0.3, 0.1]], dtype=np.float32)

CLASSES = [0, 1]

TOL = 1e-5


def bend(*args):
    cmd = BEND.split() + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        sys.exit("ERROR: " + " ".join(cmd) + " failed:\n" + r.stdout + r.stderr)
    return r.stdout


def read_metrics(text):
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def main():
    got = read_metrics(bend("metrics/tests/ok.bend"))

    y = np.array(TARGETS, dtype=np.float64)
    p = np.array(PREDICTIONS, dtype=np.float64)
    want = {
        "mse": float(np.mean((y - p) ** 2)),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "accuracy": float(sk.accuracy_score(LABELS, PREDICTED)),
    }
    want["r2"] = float(sk.r2_score(TARGETS, PREDICTIONS))
    want["log_loss"] = float(sk.log_loss(LABELS, PROBABILITIES, labels=[0, 1, 2]))
    for a in ("macro", "weighted"):
        want[f"{a}_precision"] = float(sk.precision_score(LABELS, PREDICTED, labels=CLASSES, average=a, zero_division=0))
        want[f"{a}_recall"] = float(sk.recall_score(LABELS, PREDICTED, labels=CLASSES, average=a, zero_division=0))
        want[f"{a}_f1"] = float(sk.f1_score(LABELS, PREDICTED, labels=CLASSES, average=a, zero_division=0))
    matrix = sk.confusion_matrix(LABELS, PREDICTED, labels=CLASSES)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            want[f"confusion_{i}_{j}"] = float(matrix[i, j])

    for k in (0, 1):
        want[f"precision_{k}"] = float(sk.precision_score(LABELS, PREDICTED, pos_label=k, zero_division=0))
        want[f"recall_{k}"] = float(sk.recall_score(LABELS, PREDICTED, pos_label=k, zero_division=0))
        want[f"f1_{k}"] = float(sk.f1_score(LABELS, PREDICTED, pos_label=k, zero_division=0))

    bad = []
    for name, w in want.items():
        g = got.get(name)
        if g is None:
            bad.append(f"{name}: missing from the Bend output")
        elif abs(g - w) > TOL * max(1.0, abs(w)):
            bad.append(f"{name}: bend {g} != numpy {w}")
    extra = set(got) - set(want)
    if extra:
        bad.append("unexpected metrics in the Bend output: " + ", ".join(sorted(extra)))

    for line in bad:
        print("ERROR " + line)
    if bad:
        sys.exit(1)
    print(f"ok   {len(want)} metrics match NumPy and scikit-learn (tolerance {TOL})")


if __name__ == "__main__":
    main()
