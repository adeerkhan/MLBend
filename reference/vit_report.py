"""Read a ViT probe run and report its loss, cloud and legibility.

  reference/.venv/bin/python reference/vit_report.py /tmp/opencode/vit_full.txt

`reference/probe_vit_model.bend` prints `loss:` every 25 steps, `final:<l>`,
and one `samples:` line of 4096 values in pixel order (the DDIM cloud). This
script prints what the milestone needs to be judged:

  loss        the curve's first, last and minimum
  cloud       mean, std, min, max, and how many of 4096 pixels are in [0,1]
              (the data's mean and std are 0.3076 and 0.3767)
  legibility  the best absolute correlation of each sample with any training
              digit, and the same statistic for random images (0.326) — the
              number the M1 gate (>= 0.70) and the M2 gate (>= 0.75) use

Read-only; it writes nothing.
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

from test_diffusion_digits import F, load


def legibility(samples, data, d=64):
    s = np.asarray(samples).reshape(-1, d).astype(np.float64)
    t = np.asarray(data).reshape(-1, d).astype(np.float64)
    s = s - s.mean(1, keepdims=True)
    s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-12)
    t = t - t.mean(1, keepdims=True)
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
    c = np.abs(s @ t.T).max(1)
    return float(c.mean()), float(c.min()), float(c.max())


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__.strip().splitlines()[2].strip())
    text = open(sys.argv[1]).read()
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", text, re.M)]
    final = re.search(r"^final:([0-9.eE+-]+)$", text, re.M)
    cloud = re.search(r"^samples: (.+)$", text, re.M)
    if not losses and not final:
        raise SystemExit("no loss lines: is the run finished? (its stdout is buffered)")
    print(f"loss lines: {len(losses)}")
    if losses:
        print(f"  first {losses[0]:.4f}  last {losses[-1]:.4f}  min {min(losses):.4f}")
    if final:
        print(f"  final {float(final.group(1)):.4f}")
    if cloud:
        v = np.array([float(x) for x in cloud.group(1).split()], dtype=F)
        unit = int(((v >= 0) & (v <= 1)).sum())
        print(f"cloud: n {len(v)}  mean {v.mean():+.4f}  std {v.std():.4f}  "
              f"min {v.min():+.4f}  max {v.max():+.4f}  in [0,1]: {unit}/{len(v)}")
        m, lo, hi = legibility(v, load())
        print(f"legibility: mean {m:.3f}  min {lo:.3f}  max {hi:.3f}   "
              f"(random 0.326, MLP 0.408, ViT torch 0.809, UNet 0.926)")
        print(f"  M1 gate (>= 0.70): {'PASS' if m >= 0.70 else 'FAIL'}")
    else:
        print("no samples line (the run may have stopped before sampling)")


if __name__ == "__main__":
    main()
