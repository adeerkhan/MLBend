"""The transformers over bend-ml's matrix: the Bend program against scikit-learn.

  reference/.venv/bin/python reference/test_preprocess.py

The Bend program (preprocess/tests/ok.bend) standardises, min-maxes and
normalises a three-by-two matrix over bend-ml's `Mat`, and one-hot
encodes three labels. This script does the same with scikit-learn and
compares the numbers.
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]], dtype=np.float64)


def bend_numbers():
    """The Bend program's output, as a list of lists of numbers."""
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-preprocess-test")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "preprocess/tests/ok.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    got = {}
    for line in out.stdout.strip().split("\n"):
        name, _, values = line.partition(": ")
        got[name.strip()] = [float(v) for v in values.split()]
    return got


def same(name, bend, numpy):
    """Whether two lists of numbers agree to the precision of an F32."""
    if len(bend) != len(numpy):
        print(f"ERROR {name}: {len(bend)} numbers against {len(numpy)}")
        return False
    for a, b in zip(bend, numpy):
        if abs(a - b) > 1e-5:
            print(f"ERROR {name}: {a} against {b}")
            return False
    print(f"ok   {name}: {bend} against {list(numpy)}")
    return True


def main():
    got = bend_numbers()
    ok = True
    ok &= same("standardize", got["standardize"], StandardScaler().fit_transform(X).flatten())
    ok &= same("min_max", got["min_max"], MinMaxScaler().fit_transform(X).flatten())
    ok &= same("l1", got["l1"], Normalizer(norm="l1").fit_transform(X).flatten())
    ok &= same("l2", got["l2"], Normalizer(norm="l2").fit_transform(X).flatten())
    one_hot = np.eye(3)[[0, 2, 1]].flatten()
    ok &= same("one_hot", got["one_hot"], one_hot)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
