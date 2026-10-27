"""The DDIM sampler against the ancestral one (issues #12 and #14).

  reference/.venv/bin/python reference/test_ddim.py

Two halves, both gated, and neither needs training:

1. The algebra. Over a grid of every t in 1..64 crossed with a grid of
   samples, noises and draws — with no model in the way — the step the
   ancestral loop takes and the step mlbend-generative's `ddim` takes at
   eta = 1 must be the same expression (they are: DDIM with eta = 1 is
   the DDPM posterior of Song et al., eq. 12, and sigma^2 is exactly
   beta-tilde_t). eta = 0 must add no noise at all.

2. The frozen probe. reference/probe_ddim.bend imports the example and
   runs both samplers with the *starting* weights (no training, so the
   numbers are a pure function of the code), each cloud printed as one
   line. The twin draws the same numbers and computes the same steps in
   float32, and the clouds are compared:

   anc          the ancestral loop over all 64 rungs, vs the twin
   ddim1        DDIM(eta = 1, stride 1): must reproduce the ancestral
                loop — the equivalence issue #14 buys, on Bend's side
   ddim0        DDIM(eta = 0, stride 4): the report's settings, vs the
                twin
   ddim0==...   eta = 0 reads no noise: the draw flag and the salt may
                not move the cloud (exact equality)
   ddim1!=...   eta = 1 does read the counter: another salt must move it

   The comparison is relative: the frozen model does not denoise, so the
   ancestral loop's 64 unstable steps take the cloud to ~8e15, where a
   f32 ulp is ~5e8. Both implementations drift by the same rounding
   amplified through those steps (the twin's own two paths differ by
   6.0e-6, which is where the tolerance comes from).

Nothing here is printed unpinned: every number below is a gate.
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

import test_diffusion as twin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

F = np.float32
T = twin.T

# measured on this grid before the gates were set: sigma^2 8.2e-08,
# one step 9.5e-07, the clouds 7.1e-06 (twin-vs-twin 6.0e-06) — each
# gate keeps ~16x headroom over what float32 rounding actually costs.
TOL_SIGMA2 = 1e-6
TOL_STEP = 5e-6
TOL_CLOUD = 1e-4

results = []


def gate(name, good, detail):
    results.append(bool(good))
    print(f"{'ok  ' if good else 'ERROR'} {name}: {detail}")


def rel(a, b):
    """The worst element-wise relative gap, in float64 so it is exact."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    m = np.maximum(np.abs(a), np.abs(b))
    m = np.where(m > 0, m, 1.0)
    return float(np.max(np.abs(a - b) / m))


# ---- the algebra: every t, a grid of inputs, no model ----

def grid():
    worst_sig2, worst_zero, worst_step = 0.0, 0.0, 0.0
    for t in range(1, T + 1):
        beta = twin.lin_beta_at(t)
        abar = twin.ALPHAS[t - 1]
        prev = twin.ALPHAS[t - 2] if t > 1 else F(1.0)
        posterior = F(beta * F(F(1.0) - prev) / F(F(1.0) - abar))
        sig1 = twin.sigma_of(abar, prev, 1.0)
        sig0 = twin.sigma_of(abar, prev, 0.0)
        worst_sig2 = max(worst_sig2, abs(float(sig1 * sig1) - float(posterior)))
        worst_zero = max(worst_zero, abs(float(sig0)))
        coef = F(beta / np.sqrt(F(1.0 - abar)).astype(F))
        scale = F(1.0 / np.sqrt(F(1.0 - beta)).astype(F))
        for x in [F(v) for v in (-3.0, -0.7, 0.0, 0.3, 1.5)]:
            for e in [F(v) for v in (-2.0, -0.4, 0.0, 1.1)]:
                for z in [F(v) for v in (-1.7, 0.0, 0.9)]:
                    anc = F(F(scale * F(x - F(coef * e))) + F(sig1 * z))
                    dd = twin.ddim(twin.step_of(abar, prev, 1.0), x, e, z)
                    worst_step = max(worst_step, abs(float(anc) - float(dd)))
    return worst_sig2, worst_zero, worst_step


# ---- the frozen probe ----

def run_probe():
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-ddim-probe")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "reference/probe_ddim.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the probe did not build:\n" + build.stdout + build.stderr)
    t0 = time.time()
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the probe did not run:\n" + out.stdout + out.stderr)
    return out.stdout, time.time() - t0


def clouds(out):
    """The probe's lines as float32 arrays, by their tag."""
    lines = {}
    for line in out.strip().split("\n"):
        if ": " in line:
            tag, rest = line.split(": ", 1)
            lines[tag] = rest
    def f32(tag):
        return np.array([float(v) for v in lines[tag].split()], dtype=F).astype(F)
    def ints(tag):
        return [int(v) for v in lines[tag].split()]
    return lines, f32, ints


def main():
    sig2, zero, step = grid()
    gate("sigma^2(eta=1) is the exact posterior", sig2 <= TOL_SIGMA2, f"worst {sig2:.3g} over t=1..{T}")
    gate("sigma^2(eta=0) adds no noise", zero == 0.0, f"worst {zero:.3g}")
    gate("one step: ancestral == DDIM(eta=1)", step <= TOL_STEP, f"worst {step:.3g}")

    out, seconds = run_probe()
    lines, f32, ints = clouds(out)
    gate("the probe", "settings" in lines, f"ran in {seconds:.1f}s")

    settings = lines.get("settings", "").split()
    gate("the report's settings", settings == ["0", "4"], f"eta, stride = {settings}")

    # the rungs: the twin's own strided walk, so the two lists are the
    # same walk in the same direction (issue #12)
    want4, want1 = twin.rungs(4), twin.rungs(1)
    t4, t1 = ints("rungts4"), ints("rungts1")
    r4, r1 = f32("rungs4"), f32("rungs1")
    gate("the rungs: 16 of 64, anchored at the top",
         t4 == want4 and len(r4) == len(t4) and np.array_equal(r4, twin.ALPHAS[np.array(want4)]),
         f"indices {t4[0]}..{t4[-1]} by -4, abar_64 = {float(r4[0]):.6g}")
    gate("the rungs at stride 1 walk all 64 in lockstep",
         t1 == want1 and len(r1) == len(t1) and np.array_equal(r1, twin.ALPHAS[np.array(want1)]),
         "64 rungs, 64 indices, abars exact")

    p = twin.init_params()
    twin_anc = twin.sample(p)
    twin_d1 = twin.ddim_sample(p, 1.0, 1, draw=True, salt=0)
    twin_d0 = twin.ddim_sample(p, 0.0, 4, draw=False)
    anc, d1, d1s, d0, d0z, d0s = (f32(t) for t in ("anc", "ddim1", "ddim1s",
                                                   "ddim0", "ddim0z", "ddim0s"))
    gate("the ancestral cloud vs the twin", rel(anc, twin_anc) <= TOL_CLOUD,
         f"worst relative {rel(anc, twin_anc):.3g}")
    gate("DDIM(eta=1, stride 1) reproduces the ancestral loop",
         rel(d1, anc) <= TOL_CLOUD, f"worst relative {rel(d1, anc):.3g} (issue #14)")
    gate("DDIM(eta=1, stride 1) vs the twin",
         rel(d1, twin_d1) <= TOL_CLOUD, f"worst relative {rel(d1, twin_d1):.3g}")
    gate("DDIM(eta=0, stride 4) vs the twin",
         rel(d0, twin_d0) <= TOL_CLOUD, f"worst relative {rel(d0, twin_d0):.3g}")
    gate("eta=0 reads no noise: the draw and the salt change nothing",
         np.array_equal(d0, d0z) and np.array_equal(d0, d0s),
         "ddim0 == ddim0z == ddim0s, bit for bit")
    gate("eta=1 reads the counter: another salt changes the cloud",
         not np.array_equal(d1, d1s),
         f"worst relative {rel(d1, d1s):.3g}")

    bad = results.count(False)
    if bad:
        print(f"\n{len(results) - bad}/{len(results)} checks passed")
        return 1
    print(f"\n{len(results)}/{len(results)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
