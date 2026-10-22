"""The logistic regression over bendtorch's tape, against bend-ml's autodiff.

  reference/.venv/bin/python reference/test_tape.py

Two programs train the same model on the same data: `examples/logistic`
(bend-ml's tensors and bend-ml's autodiff) and `examples/tape`
(bendtorch's tape). Both fetch the iris dataset from HuggingFace, both
standardise it with `mlbend-preprocess`, and both take a step with a
learning rate of 0.5. The two losses must agree: they are two
implementations of the same gradient step, over two different
libraries.
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


def bend_losses(program):
    """The two losses, from one of the programs."""
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-tape-test")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, program)],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit(f"{program} did not build:\n" + build.stdout + build.stderr)
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit(f"{program} did not run:\n" + out.stdout + out.stderr)
    before = float(re.search(r"loss before: ([0-9.eE+-]+)", out.stdout).group(1))
    after = float(re.search(r"loss after:\s+([0-9.eE+-]+)", out.stdout).group(1))
    return before, after


def main():
    ml_before, ml_after = bend_losses("examples/logistic/main.bend")
    tape_before, tape_after = bend_losses("examples/tape/tape.bend")
    ok = True
    if abs(ml_before - tape_before) > 1e-5:
        print(f"ERROR the loss before the step differs: {ml_before} vs {tape_before}")
        ok = False
    else:
        print(f"ok   the loss before the step agrees: {ml_before} vs {tape_before}")
    if abs(ml_after - tape_after) > 1e-5:
        print(f"ERROR the loss after the step differs: {ml_after} vs {tape_after}")
        ok = False
    else:
        print(f"ok   the loss after the step agrees: {ml_after} vs {tape_after}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
