"""Compares the report of mlbend-report with scikit-learn.

  reference/.venv/bin/python reference/test_report.py

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

# the dataset of report/tests/ok.bend: seven samples, three classes
# present and one absent
LABELS = [0, 0, 1, 1, 2, 2, 2]
PREDICTED = [0, 1, 1, 0, 2, 2, 0]
CLASSES = 4

TOL = 1e-5


def bend(*args):
    cmd = BEND.split() + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        sys.exit("ERROR: " + " ".join(cmd) + " failed:\n" + r.stdout + r.stderr)
    return r.stdout


def read_report(text):
    out = {"matrix": [], "classes": {}}
    in_matrix = False
    for line in text.splitlines():
        if line == "matrix:":
            in_matrix = True
            continue
        if in_matrix:
            cells = line.split()
            if cells and all(c.lstrip("-").isdigit() for c in cells):
                out["matrix"].append([int(c) for c in cells])
                continue
            in_matrix = False
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "class":
            out["classes"][int(parts[1])] = {
                "precision": float(parts[3]), "recall": float(parts[5]),
                "f1-score": float(parts[7]), "support": float(parts[9])}
        elif parts[0] in ("macro", "weighted"):
            out[f"{parts[0]} avg"] = {
                "precision": float(parts[2]), "recall": float(parts[4]),
                "f1-score": float(parts[6])}
        elif parts[0] == "accuracy":
            out["accuracy"] = float(parts[1])
    return out


def close(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def main():
    got = read_report(bend("report/tests/ok.bend"))
    y = np.array(LABELS)
    p = np.array(PREDICTED)
    labels = list(range(CLASSES))
    fail = 0

    want_matrix = sk.confusion_matrix(y, p, labels=labels).tolist()
    if got["matrix"] != want_matrix:
        print(f"ERROR the matrix differs: {got['matrix']} vs {want_matrix}")
        fail += 1
    else:
        print(f"ok   the matrix is scikit-learn's: {want_matrix}")

    report = sk.classification_report(y, p, labels=labels, zero_division=0,
                                      output_dict=True)
    checks = {}
    for k in labels:
        for m in ("precision", "recall", "f1-score", "support"):
            checks[f"class {k} {m}"] = (got["classes"][k][m], float(report[str(k)][m]))
    for m in ("precision", "recall", "f1-score"):
        checks[f"macro {m}"] = (got["macro avg"][m], float(report["macro avg"][m]))
        checks[f"weighted {m}"] = (got["weighted avg"][m], float(report["weighted avg"][m]))
    checks["accuracy"] = (got["accuracy"], float(sk.accuracy_score(y, p)))

    for name, (a, b) in checks.items():
        if close(a, b):
            print(f"ok   {name}: {a} vs {b}")
        else:
            print(f"ERROR {name}: {a} vs {b}")
            fail += 1

    if fail:
        print(f"{fail} checks differ")
        return 1
    print("the report matches scikit-learn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
