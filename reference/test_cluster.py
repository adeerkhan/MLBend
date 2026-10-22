"""The k-means package: the Bend test against NumPy.

  reference/.venv/bin/python reference/test_cluster.py

The Bend program (cluster/tests/ok.bend) clusters six fixed rows into
two clusters, starting from the first two rows, for two rounds. This
script does the same in NumPy and compares the labels, the sizes, the
inertia and the centres.
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

ROWS = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0],
                 [10.0, 10.0], [10.0, 11.0], [11.0, 10.0]], dtype=np.float32)
K = 2
ROUNDS = 2


def bend_result():
    """The label line, the sizes, the inertia and the centres, from Bend."""
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-cluster-test")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "cluster/tests/ok.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    text = out.stdout
    labels = [int(n) for n in re.search(r"labels: ([0-9 ]+)", text).group(1).split()]
    sizes = [int(n) for n in re.search(r"sizes: ([0-9 ]+)", text).group(1).split()]
    inertia = float(re.search(r"inertia: ([0-9.eE+-]+)", text).group(1))
    centres = [[float(x) for x in re.search(rf"centre {i}: ([0-9.eE+-]+) ([0-9.eE+-]+)", text).groups()]
               for i in range(K)]
    return labels, sizes, inertia, np.array(centres, dtype=np.float32)


def numpy_result():
    """The same algorithm in NumPy."""
    centres = ROWS[:K].copy()
    labels = np.zeros(len(ROWS), dtype=int)
    for _ in range(ROUNDS):
        d = ((ROWS[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        for i in range(K):
            members = ROWS[labels == i]
            if len(members):
                centres[i] = members.mean(axis=0)
    sizes = [int((labels == i).sum()) for i in range(K)]
    inertia = float(((ROWS - centres[labels]) ** 2).sum())
    return labels.tolist(), sizes, inertia, centres


def main():
    b_labels, b_sizes, b_inertia, b_centres = bend_result()
    n_labels, n_sizes, n_inertia, n_centres = numpy_result()
    ok = True

    if b_labels != n_labels:
        print(f"ERROR the labels differ: {b_labels} vs {n_labels}")
        ok = False
    else:
        print(f"ok   the labels agree: {b_labels}")

    if b_sizes != n_sizes:
        print(f"ERROR the sizes differ: {b_sizes} vs {n_sizes}")
        ok = False
    else:
        print(f"ok   the sizes agree: {b_sizes}")

    if abs(b_inertia - n_inertia) > 1e-3 * max(1.0, n_inertia):
        print(f"ERROR the inertia differs: {b_inertia} vs {n_inertia}")
        ok = False
    else:
        print(f"ok   the inertia agrees: {b_inertia} against {n_inertia}")

    if not np.allclose(b_centres, n_centres, rtol=1e-4, atol=1e-4):
        print(f"ERROR the centres differ: {b_centres.tolist()} vs {n_centres.tolist()}")
        ok = False
    else:
        print(f"ok   the centres agree: {b_centres.tolist()}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
