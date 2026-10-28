"""Turn a ViT run's output into the evidence a milestone is judged on.

  reference/.venv/bin/python reference/vit_results.py /tmp/opencode/vit_ln.txt

Reads the example's stdout (buffered until it exits) and writes

  results/diffusion-vit/run.json          the numbers, one object
  results/diffusion-vit/loss.svg          the loss curve
  results/diffusion-vit/grid.svg          the 64 DDIM samples, 8x8 each
  results/diffusion-vit/REPORT.md         the report card

and prints the same numbers, so a run can be quoted without the files.
The legibility statistic is `vit_report`'s: the best absolute correlation
of each sample with any of the 1024 training digits, averaged. Random
images score 0.326, the MLP 0.408, torch's causal ViT 0.774, the
reference's UNet 0.926.
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

from test_diffusion_digits import load
from vit_report import legibility

OUT = os.path.join(ROOT, "results", "diffusion-vit")


def parse(text):
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", text, re.M)]
    final = re.search(r"^final(?:_loss)?: ([0-9.eE+-]+)$", text, re.M)
    vals = {}
    for k in ("data_mean", "sample_mean", "sample_std"):
        m = re.search(rf"^{k}: ([0-9.eE+-]+)$", text, re.M)
        if m:
            vals[k] = float(m.group(1))
    cloud = re.search(r"^samples: (.+)$", text, re.M)
    if not losses or not cloud:
        raise SystemExit("no loss lines or no samples: is the run finished? (stdout is buffered)")
    return losses, float(final.group(1)) if final else losses[-1], vals, \
        np.array([float(v) for v in cloud.group(1).split()], dtype=np.float32)


def loss_svg(path, losses):
    w, h, left, top, bottom = 640, 220, 60, 20, 40
    pw, ph = w - left - 20, h - top - bottom
    lo, hi = min(losses), max(losses)
    hi = hi if hi > lo else lo + 1.0
    pts = []
    for i, v in enumerate(losses):
        x = left + pw * i / max(1, len(losses) - 1)
        y = top + ph * (1.0 - (v - lo) / (hi - lo))
        pts.append(f"{x:.1f},{y:.1f}")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'viewBox="0 0 {w} {h}">',
           f'<rect width="{w}" height="{h}" fill="#101418"/>',
           f'<polyline fill="none" stroke="#5ee6c8" stroke-width="2" points="{" ".join(pts)}"/>',
           f'<text x="{left}" y="{top - 6}" fill="#98a2b3" font-family="monospace" '
           f'font-size="12">loss {losses[0]:.4f} -&gt; {losses[-1]:.4f} '
           f'(min {min(losses):.4f}), {len(losses)} points</text>',
           f'<text x="{left}" y="{h - 14}" fill="#98a2b3" font-family="monospace" '
           f'font-size="11">step 0</text>',
           f'<text x="{w - 90}" y="{h - 14}" fill="#98a2b3" font-family="monospace" '
           f'font-size="11">step {len(losses) - 1}</text>', "</svg>"]
    open(path, "w").write("\n".join(out) + "\n")


def grid_svg(path, cloud):
    """64 samples, each 8x8, in one 8x8 grid; pixel value -> grey."""
    cell, gap, digits, per_row = 8, 2, 64, 8
    side = digits // per_row
    w = per_row * (side * cell + gap) + gap
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{w}" '
           f'viewBox="0 0 {w} {w}">', f'<rect width="{w}" height="{w}" fill="#0b0f13"/>']
    for d in range(digits):
        ox = gap + (d % per_row) * (side * cell + gap)
        oy = gap + (d // per_row) * (side * cell + gap)
        for i in range(64):
            v = float(np.clip(cloud[d * 64 + i], 0.0, 1.0))
            shade = int(round(255 * v))
            out.append(f'<rect x="{ox + (i % 8) * cell}" y="{oy + (i // 8) * cell}" '
                       f'width="{cell}" height="{cell}" fill="rgb({shade},{shade},{shade})"/>')
    out.append("</svg>")
    open(path, "w").write("\n".join(out) + "\n")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__.strip().splitlines()[2].strip())
    text = open(sys.argv[1]).read()
    losses, final, vals, cloud = parse(text)
    data = load()
    DATA_MEAN, DATA_STD = float(data.mean()), float(data.std())
    m, lo, hi = legibility(cloud, data)
    unit = int(((cloud >= 0) & (cloud <= 1)).sum())
    run = {
        "task": "diffusion-vit",
        "program": "examples/diffusion_vit/main.bend",
        "steps": len(losses) - 1,
        "first_loss": losses[0],
        "final_loss": final,
        "loss_min": min(losses),
        "legibility": {"mean": m, "min": lo, "max": hi},
        "cloud": {"n": int(len(cloud)), "mean": float(cloud.mean()),
                  "std": float(cloud.std()), "min": float(cloud.min()),
                  "max": float(cloud.max()), "in_unit": unit},
        "data": {"mean": DATA_MEAN, "std": DATA_STD},
        "architecture": {"patches": "2x2", "tokens": 16, "features": 20,
                         "dim": 64, "heads": 1, "blocks": 3, "attention": "causal",
                         "residual": "pre-LN", "params": 100736},
    }
    if vals:
        for k, v in vals.items():
            run[k] = v
    os.makedirs(OUT, exist_ok=True)
    json.dump(run, open(os.path.join(OUT, "run.json"), "w"), indent=2)
    loss_svg(os.path.join(OUT, "loss.svg"), losses)
    grid_svg(os.path.join(OUT, "grid.svg"), cloud)

    card = f"""# diffusion-vit — report card

The Bend ViT denoiser over the first 1024 optdigits digits: `examples/diffusion_vit/main.bend`,
sampled by the 16-step DDIM walk with the clamped start.

| | value |
|---|---|
| first loss | {losses[0]:.6f} |
| final loss | {final:.6f} (minimum {min(losses):.6f}) |
| legibility (mean, min, max) | **{m:.3f}**, {lo:.3f}, {hi:.3f} |
| cloud mean / std | {run['cloud']['mean']:+.4f} / {run['cloud']['std']:.4f} (data {DATA_MEAN} / {DATA_STD}) |
| pixels in [0,1] | {unit}/{len(cloud)} |

Legibility is the best absolute correlation of each sample with any training
digit, averaged. Random images score 0.326, the digits MLP 0.408, torch's
*causal* ViT 0.774, the vendored lucidrains UNet 0.926. The M2 gate is 0.75.

![loss](loss.svg)

![samples](grid.svg)

Architecture: 2x2 patches, 16 tokens of 20 features (4 pixels, 8 time, 8
position), width 64, one head, three pre-LN blocks, one 4-value head per
patch, 100,736 parameters, every layer BendTorch's. BendTorch's `Attention`
is causal (ISSUE-011), so the 16 tokens are attended as a sequence; the
torch arm of the comparison runs the same mask.
"""
    open(os.path.join(OUT, "REPORT.md"), "w").write(card)
    print(json.dumps({k: run[k] for k in ("first_loss", "final_loss", "legibility", "cloud")},
                     indent=2))


if __name__ == "__main__":
    main()
