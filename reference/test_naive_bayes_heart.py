"""Gaussian naive Bayes on heart failure: the Bend program against scikit-learn.

  reference/.venv/bin/python reference/test_naive_bayes_heart.py

The same program as the breast cancer baseline, on `mstz/heart_failure`: 299
rows, twelve features, the label `is_dead`. The file writes `True`/`False`,
which the loader reads as 1 and 0, as the Bend datasets package does.
"""
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np
from sklearn.naive_bayes import GaussianNB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

URL = "https://huggingface.co/datasets/mstz/heart_failure/resolve/57c2e1a4b3bb2f28e766c6d3b10bc0ced972c857/death/train.csv"


def load():
    text = urllib.request.urlopen(URL, timeout=120).read().decode()
    lines = [l for l in text.split("\n") if l.strip()]
    def num(cell):
        if cell == "True":
            return 1.0
        if cell == "False":
            return 0.0
        return float(cell)
    data = np.array([[num(c) for c in line.split(",")] for line in lines[1:]], dtype=np.float64)
    return data[:, :-1], data[:, -1]


def bend_result():
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-nb-heart")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "examples/naive_bayes_heart/main.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    return float(re.search(r"accuracy: ([0-9.eE+-]+)", out.stdout).group(1)), out.stdout


def main():
    bend_accuracy, out = bend_result()
    x, y = load()
    split = len(x) * 4 // 5
    sklearn_accuracy = float(GaussianNB().fit(x[:split], y[:split]).score(x[split:], y[split:]))
    print(out.strip())
    print(f"sklearn: {sklearn_accuracy}")
    ok = abs(bend_accuracy - sklearn_accuracy) <= 0.000001
    print(("ok   " if ok else "ERROR ") + f"the accuracies agree: {bend_accuracy} vs {sklearn_accuracy}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
