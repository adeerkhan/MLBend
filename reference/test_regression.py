"""The linear regression: the Bend program against scikit-learn.

  reference/.venv/bin/python reference/test_regression.py

The Bend program (examples/regression/main.bend) standardises the wine
dataset, trains one linear layer with bendtorch's tape for 1000 steps at
a learning rate of 0.3, and prints the mean squared error before and
after the training. This script fits the same model with scikit-learn's
`LinearRegression` on the same standardised data and compares the two
errors: the loss must fall, and the two must agree to within one percent.
"""
import io
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np
from sklearn.linear_model import LinearRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

URL = "https://huggingface.co/datasets/mstz/wine/resolve/main/wine/train.csv"


def bend_result():
    """The two mean squared errors, from the Bend program."""
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-regression-test")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "examples/regression/main.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    text = out.stdout
    before = float(re.search(r"mse before: ([0-9.eE+-]+)", text).group(1))
    after = float(re.search(r"mse after:  ([0-9.eE+-]+)", text).group(1))
    return before, after


def load():
    """The wine dataset, as the features and the quality."""
    text = urllib.request.urlopen(URL, timeout=120).read().decode()
    lines = [l for l in text.split("\n") if l.strip()]
    rows = [[float(c) for c in line.split(",")] for line in lines[1:]]
    data = np.array(rows, dtype=np.float64)
    return data[:, :-2], data[:, -2]


def sklearn_mse():
    """The mean squared error of scikit-learn's least squares fit."""
    x, y = load()
    xs = (x - x.mean(axis=0)) / x.std(axis=0)
    ys = (y - y.mean()) / y.std()
    model = LinearRegression().fit(xs, ys)
    return float(np.mean((model.predict(xs) - ys) ** 2))


def main():
    before, after = bend_result()
    want = sklearn_mse()
    ok = True

    if after >= before:
        print(f"ERROR the loss did not fall: {before} -> {after}")
        ok = False
    else:
        print(f"ok   the loss fell: {before} -> {after}")

    if abs(after - want) > 0.01 * want:
        print(f"ERROR the errors are too far apart: {after} vs {want}")
        ok = False
    else:
        print(f"ok   the errors agree: {after} against {want}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
