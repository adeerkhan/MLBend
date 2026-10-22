"""The split of the rows: the Bend package against scikit-learn.

  reference/.venv/bin/python reference/test_split.py

The Bend package (split/main.bend) deals the rows of a class out over
the folds in turn, so every fold holds the same proportion of every
class — the rule scikit-learn's StratifiedKFold uses. The shuffle is
bend-kit-random's xoshiro128**, seeded.

The two programs do not agree on *which* rows go where: scikit-learn
shuffles with numpy's generator, the package with bend-kit-random's. So
the comparison is of the shape of the split, not of its exact contents:

- every fold holds the same number of rows;
- every fold holds the same proportion of every class;
- every row is in exactly one fold;
- the same seed gives the same split, and another seed gives another.
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

# 150 rows: three classes of fifty, as the iris dataset has.
CLASSES = [0] * 50 + [1] * 50 + [2] * 50
N_SPLITS = 5
SEED = 42


def bend_program():
    """A Bend program that prints the fold of every row."""
    labels = "".join(f"Con{{{c}n, " for c in CLASSES) + "Nil{}" + "}" * len(CLASSES)
    return f"""import Base
import mlbend-split@0.1.0.0/main.bend as Split

# The folds of the rows.
def folds_of() -> List<&2, Nat>:
  Split.folds({labels}, {SEED}, {N_SPLITS}n)

# The cells of a list, as text.
def cells_go(+xs: List<&2, Nat>, +acc: String) -> String:
  match xs:
    case Nil{{}}:
      acc
    case Con{{h, t}}:
      cells_go(t, String.append(acc, String.append(U32.show(U32.from_nat(h)), " ")))

# The cells of a list, as text.
def cells(+xs: List<&2, Nat>) -> String:
  cells_go(xs, "")

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(cells(folds_of()))
"""


def bend_folds(seed=SEED):
    """The fold of every row, from the Bend package."""
    src = bend_program().replace(f"{SEED}, {N_SPLITS}n", f"{seed}, {N_SPLITS}n")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "split_folds.bend")
        binary = os.path.join(d, "split_folds")
        with open(path, "w") as f:
            f.write(src)
        build = subprocess.run([BEND, "-o", binary, path], capture_output=True, text=True,
                               env=ENV, cwd=ROOT)
        if build.returncode != 0:
            raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
        out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
        if out.returncode != 0:
            raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
        return [int(x) for x in re.findall(r"\d+", out.stdout)]


def sklearn_folds():
    """The fold of every row, from scikit-learn."""
    y = np.array(CLASSES)
    folds = np.empty(len(y), dtype=int)
    for i, (_, test) in enumerate(StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                                  random_state=SEED).split(y, y)):
        folds[test] = i
    return list(folds)


def sizes(folds):
    """The number of rows in each fold."""
    return [folds.count(f) for f in range(N_SPLITS)]


def class_counts(folds, label):
    """The number of rows of one class in each fold."""
    return [sum(1 for f, c in zip(folds, CLASSES) if f == fold and c == label)
            for fold in range(N_SPLITS)]


def main():
    bend = bend_folds()
    sklearn = sklearn_folds()
    ok = True

    if len(bend) != len(CLASSES):
        print(f"ERROR the Bend program gave {len(bend)} folds for {len(CLASSES)} rows")
        return 1

    # every row is in exactly one fold, and every fold is one of the k
    if sorted(bend) != sorted(list(range(N_SPLITS)) * (len(CLASSES) // N_SPLITS)):
        print(f"ERROR the folds are not a partition: {sorted(bend)}")
        ok = False

    # the same seed gives the same split, and another seed gives another
    again = bend_folds()
    if again != bend:
        print("ERROR the same seed gave two different splits")
        ok = False
    other = bend_folds(seed=SEED + 1)
    if other == bend:
        print("ERROR another seed gave the same split")
        ok = False

    bend_sizes = sizes(bend)
    sklearn_sizes = sizes(sklearn)
    print(f"bend:    {bend_sizes}   sklearn: {sklearn_sizes}")
    if bend_sizes != sklearn_sizes:
        print(f"ERROR the fold sizes differ: {bend_sizes} against {sklearn_sizes}")
        ok = False

    for label in sorted(set(CLASSES)):
        bend_counts = class_counts(bend, label)
        sklearn_counts = class_counts(sklearn, label)
        print(f"class {label}: bend {bend_counts}  sklearn {sklearn_counts}")
        if bend_counts != sklearn_counts:
            print(f"ERROR the rows of class {label} are spread differently")
            ok = False

    if ok:
        print("ok   the split has the shape of scikit-learn's")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
