"""The no-leakage scaler: the Bend package against scikit-learn.

  reference/.venv/bin/python reference/test_pipeline.py

`mlbend-pipeline` fits on three fixed training rows and transforms them and
two test rows. This script fits scikit-learn's `StandardScaler` on the same
training rows and compares the means, the deviations and every transformed
number — the test rows included, which the scaler has never seen.
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

TRAIN = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]], dtype=np.float64)
TEST = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)


def bend_result():
    """The means, the deviations and the transformed rows, from Bend."""
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-pipeline-test")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "pipeline/tests/ok.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    text = out.stdout
    means = [float(v) for v in re.search(r"means: ([-0-9.eE+ ]+)", text).group(1).split()]
    stds = [float(v) for v in re.search(r"stds: ([-0-9.eE+ ]+)", text).group(1).split()]
    train = [[float(v) for v in line.split()]
             for line in re.search(r"train:\n(.*?)\ntest:", text, re.S).group(1).split("\n") if line.strip()]
    test = [[float(v) for v in line.split()]
            for line in re.search(r"test:\n(.*)", text, re.S).group(1).split("\n") if line.strip()]
    return means, stds, np.array(train), np.array(test)


def main():
    b_means, b_stds, b_train, b_test = bend_result()
    scaler = StandardScaler().fit(TRAIN)
    n_train = scaler.transform(TRAIN)
    n_test = scaler.transform(TEST)
    ok = True

    if not np.allclose(b_means, scaler.mean_, rtol=1e-5, atol=1e-5):
        print(f"ERROR the means differ: {b_means} vs {scaler.mean_.tolist()}")
        ok = False
    else:
        print(f"ok   the means agree: {b_means}")

    if not np.allclose(b_stds, scaler.scale_, rtol=1e-5, atol=1e-5):
        print(f"ERROR the deviations differ: {b_stds} vs {scaler.scale_.tolist()}")
        ok = False
    else:
        print(f"ok   the deviations agree: {b_stds}")

    if not np.allclose(b_train, n_train, rtol=1e-5, atol=1e-5):
        print(f"ERROR the training rows differ:\n{b_train}\nvs\n{n_train}")
        ok = False
    else:
        print("ok   the training rows agree")

    if not np.allclose(b_test, n_test, rtol=1e-5, atol=1e-5):
        print(f"ERROR the test rows differ:\n{b_test}\nvs\n{n_test}")
        ok = False
    else:
        print("ok   the unseen test rows agree")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
