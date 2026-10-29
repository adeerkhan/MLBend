"""PCA: the Bend program against scikit-learn.

  reference/.venv/bin/python reference/test_pca.py

The Bend program (examples/unsupervised/pca/main.bend) standardises the wine features,
builds the covariance matrix and runs a hundred power iterations for the
first principal component. This script does the same with scikit-learn's
`PCA` on the same standardised data and compares the explained variance
ratio.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR


def bend_ratio():
    """The explained variance ratio, from the Bend program."""
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-pca-test")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "examples/unsupervised/pca/main.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    return float(re.search(r"explained variance ratio: ([0-9.eE+-]+)", out.stdout).group(1))


def sklearn_ratio():
    """The same ratio, from scikit-learn."""
    sys.path.insert(0, os.path.join(ROOT, "reference"))
    from kmeans_wine import load, standardise
    from sklearn.decomposition import PCA

    return float(PCA(n_components=1).fit(standardise(load())).explained_variance_ratio_[0])


def main():
    bend = bend_ratio()
    want = sklearn_ratio()
    print(f"bend:    {bend}")
    print(f"sklearn: {want}")
    ok = abs(bend - want) <= 1e-3
    print(("ok   " if ok else "ERROR ") + f"the ratios agree: {bend} against {want}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
