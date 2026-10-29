"""The frozen comparison for the ViT: Bend's probe against the NumPy twin.

  reference/.venv/bin/python reference/vit_gate.py

Builds `reference/probe_vit_frozen.bend` (fixed initial weights, no training),
reads its three lines, and compares:

  eps    the model's output at t = 64 on the starting cloud — the *forward*
         gate (a weight-order or staging error shows up here at O(1))
  step   ONE ancestral step at t = 64, both sides fed Bend's own eps — the
         tight gate on the sampler's arithmetic and the token order
  ddim   the 16-step DDIM cloud, each side with its own eps — the loosest

It prints the three gaps and only then says whether they are inside the
tolerances, which are set from these measurements (the file records them).
Nothing here is in `make check` yet.
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

from test_diffusion_digits import BEND, ENV, F, noise_of, rev_step_eps

import twin_vit as twin

# measured first, then fixed: eps maxabs 1.19e-07, step 4.77e-07, ddim
# 6.38e-06 (float32, different summation orders over 64 terms). The eps is
# gated on maxabs, not on the element-wise relative gap: many of its numbers
# are near zero, where a relative gap says nothing.
TOL_EPS = 1e-5
TOL_STEP = 1e-5
TOL_DDIM = 1e-4


PROBE_BIN = "/tmp/mlbend-vit-frozen"
PROBE_SRC = ("reference/probe_vit_frozen.bend", "examples/diffusion/vit/main.bend")


def probe_binary():
    """The frozen probe's binary, rebuilt only when its sources are newer.

    The build is about six minutes here, so a cached binary is what makes
    the gate cheap enough to run in `make check`.
    """
    srcs = [os.path.join(ROOT, p) for p in PROBE_SRC]
    if os.path.exists(PROBE_BIN) and all(os.path.getmtime(PROBE_BIN) >= os.path.getmtime(p)
                                         for p in srcs):
        return PROBE_BIN
    build = subprocess.run([BEND, "-o", PROBE_BIN, os.path.join(ROOT, "reference/probe_vit_frozen.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the frozen probe did not build:\n" + build.stdout[-2000:] + build.stderr[-2000:])
    return PROBE_BIN


def run_probe():
    binary = probe_binary()
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the frozen probe did not run:\n" + out.stdout[-2000:] + out.stderr[-2000:])
    lines = {}
    for line in out.stdout.strip().split("\n"):
        if ": " in line:
            tag, rest = line.split(": ", 1)
            lines[tag] = np.array([float(v) for v in rest.split()], dtype=F)
    return lines


def rel(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    m = np.maximum(np.abs(a), np.abs(b))
    m = np.where(m > 0, m, 1.0)
    return float(np.max(np.abs(a - b) / m))


def maxabs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def main():
    bend = run_probe()
    p = twin.bend_init()

    twin_eps = twin.frozen_eps()
    start = twin.patch(noise_of(2000000, 4096).reshape(64, 64)).reshape(-1).astype(F)
    twin_step = rev_step_eps(start, 64, bend["eps"])       # shared eps: the sampler only
    own_step = rev_step_eps(start, 64, twin_eps)           # each side's own eps
    twin_ddim = twin.ddim_cloud(p)
    z = noise_of(7000000, 4096).reshape(64, 64).astype(F)
    twin_eta = twin.ddim_cloud(p, z=z)

    print(f"eps    bend vs twin        maxabs {maxabs(bend['eps'], twin_eps):.3g}  "
          f"(rel {rel(bend['eps'], twin_eps):.3g})   tol {TOL_EPS:g}  "
          f"{'ok' if maxabs(bend['eps'], twin_eps) <= TOL_EPS else 'FAIL'}")
    print(f"step   shared eps          maxabs {maxabs(bend['step'], twin_step):.3g}   "
          f"tol {TOL_STEP:g}  {'ok' if maxabs(bend['step'], twin_step) <= TOL_STEP else 'FAIL'}")
    print(f"step   each side's own eps maxabs {maxabs(bend['step'], own_step):.3g}   (report)")
    print(f"ddim   bend vs twin        maxabs {maxabs(bend['ddim'], twin_ddim):.3g}  "
          f"(rel {rel(bend['ddim'], twin_ddim):.3g})   tol {TOL_DDIM:g}  "
          f"{'ok' if maxabs(bend['ddim'], twin_ddim) <= TOL_DDIM else 'FAIL'}")
    print(f"eta=0  sampler noise is inert  maxabs {maxabs(twin_eta, twin_ddim):.3g}   "
          f"{'ok' if maxabs(twin_eta, twin_ddim) == 0 else 'FAIL'}")
    for name, v in (("bend", bend["ddim"]), ("twin", twin_ddim)):
        print(f"  {name} cloud: mean {float(v.mean()):+.4f}  std {float(v.std()):.4f}  "
              f"in [0,1] {int(((v >= 0) & (v <= 1)).sum())}/4096")


if __name__ == "__main__":
    main()
