"""Single-head self-attention: the Bend program against NumPy.

  reference/.venv/bin/python reference/test_attention.py

The Bend program (examples/attention/main.bend) runs one attention head
over four tokens of eight features with fixed weights. This script builds
the same tokens and weights with the same formulas and compares every
output number. F32 is single precision on both sides; the tolerance is the
rounding of one softmax pass.
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

TOKENS = 4
FEATURES = 8


def matrix(fn):
    return np.array([[fn(i, j) for j in range(FEATURES)] for i in range(FEATURES)], dtype=np.float32)


def tokens(fn):
    return np.array([[fn(i, j) for j in range(FEATURES)] for i in range(TOKENS)], dtype=np.float32)


def expected():
    x = tokens(lambda i, j: np.float32(((3 * i + 5 * j) % 7) * 0.1 - 0.2))
    wq = matrix(lambda i, j: np.float32(((2 * i + 3 * j) % 5) * 0.1 - 0.2))
    wk = matrix(lambda i, j: np.float32(((7 * i + 11 * j + 1) % 6) * 0.1 - 0.2))
    wv = matrix(lambda i, j: np.float32(((5 * i + 2 * j + 3) % 6) * 0.1 - 0.2))
    q = x @ wq
    k = x @ wk
    v = x @ wv
    scores = (q @ k.T) / np.float32(np.sqrt(8.0))
    scores = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(scores, dtype=np.float32)
    p = e / e.sum(axis=1, keepdims=True)
    return p @ v


def bend_result():
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-attention-test")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "examples/attention/main.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    rows = []
    for match in re.finditer(r"^row \d+: (.*)$", out.stdout, re.M):
        rows.append([float(v) for v in match.group(1).split()])
    checksum = float(re.search(r"checksum: ([0-9.eE+-]+)", out.stdout).group(1))
    return np.array(rows, dtype=np.float32), checksum


def main():
    got, checksum = bend_result()
    want = expected()
    ok = True

    if got.shape != want.shape:
        print(f"ERROR the shapes differ: {got.shape} vs {want.shape}")
        return 1
    diff = float(np.abs(got - want).max())
    print(f"max absolute difference: {diff:.3e}")
    if diff > 1e-5:
        print(f"ERROR the attention output differs by {diff}")
        print("bend:\n", got, "\nnumPy:\n", want)
        ok = False
    else:
        print(f"ok   every number agrees within 1e-5: {diff:.3e}")

    want_checksum = float(want.sum(dtype=np.float32))
    if abs(checksum - want_checksum) > 1e-4:
        print(f"ERROR the checksum differs: {checksum} vs {want_checksum}")
        ok = False
    else:
        print(f"ok   the checksum agrees: {checksum} against {want_checksum}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
