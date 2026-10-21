"""The k-means over the wine dataset: the Bend program against NumPy.

  reference/.venv/bin/python reference/test_kmeans.py

The Bend program (examples/kmeans/main.bend) fetches the wine dataset
from HuggingFace, standardises every column, starts from the first three
rows and runs ten rounds of Lloyd's algorithm. This script does the same
in NumPy and compares: the cluster sizes must be the same, and the
inertia must agree to the precision of an F32.
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


def bend_kmeans():
    """The inertia and the cluster sizes, from the Bend program."""
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-kmeans-test")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "examples/kmeans/main.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    inertia = float(re.search(r"inertia: ([0-9.eE+-]+)", out.stdout).group(1))
    sizes = [int(n) for n in re.search(r"cluster sizes: ([0-9 ]+)", out.stdout).group(1).split()]
    return inertia, sizes


def numpy_kmeans():
    """The inertia and the cluster sizes, from NumPy."""
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from kmeans_wine import kmeans, load, standardise

    inertia, sizes = kmeans(standardise(load()), 3, 10)
    return inertia, sizes


def main():
    bend_inertia, bend_sizes = bend_kmeans()
    numpy_inertia, numpy_sizes = numpy_kmeans()
    ok = True
    if bend_sizes != numpy_sizes:
        print(f"ERROR the cluster sizes differ: {bend_sizes} vs {numpy_sizes}")
        ok = False
    else:
        print(f"ok   the cluster sizes are the same: {bend_sizes}")
    # an F32 has about seven digits, so the inertia may differ in the fifth
    if abs(bend_inertia - numpy_inertia) > 1.0:
        print(f"ERROR the inertia differs: {bend_inertia} vs {numpy_inertia}")
        ok = False
    else:
        print(f"ok   the inertia agrees: {bend_inertia} vs {numpy_inertia}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
