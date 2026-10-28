"""The lucidrains reference on the same digits, for comparing against Bend.

  reference/.venv-lucidrains/bin/python reference/lucidrains_compare.py            # 3000 steps
  STEPS=200 reference/.venv-lucidrains/bin/python reference/lucidrains_compare.py  # a smoke run

`reference/lucidrains/` is a verbatim copy of
github.com/lucidrains/denoising-diffusion-pytorch at the commit in its
VENDORED.md. This script does not modify the library: it trains the same
denoiser object our `examples/diffusion_digits/main.bend` trains (noise
prediction, T = 64, linear schedule) on the same 1024 optdigits rows, so the
two implementations can be compared on one dataset and one schedule.

What it prints, in the order the comparison needs it:

  1. the schedule: the reference's linear beta vs the one the Bend package's
     `lin_beta_at` builds, and the resulting alpha-bar at the top rung;
  2. the training curve, at the same 1/100/200 cadence the Bend example prints;
  3. the two clouds (DDIM at every 4th rung, and the ancestral loop over all
     64), their statistics, and the same statistics Bend's run.json holds.

Writes results/lucidrains-diffusion/: loss.tsv, grid.svg (data left, DDIM
right, drawn with the cell offset fixed -- see issues-hit.md), run.json.

The reference is a reference: the Bend layer does not import it, and
`make check` does not run it. It is here to answer "what does the paper's
implementation do here" with a program instead of a memory.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
sys.path.insert(0, os.path.join(ROOT, "reference", "lucidrains"))

import torch
from torch.optim import Adam

from denoising_diffusion_pytorch import GaussianDiffusion, Unet

from test_diffusion_digits import D, N, STEPS, load

# ---- the two schedules, side by side ------------------------------------

T = 64


def reference_betas(timesteps=T):
    """linear_beta_schedule(timesteps) from the vendored library, in numpy."""
    scale = 1000.0 / timesteps
    return np.linspace(scale * 0.0001, scale * 0.02, timesteps, dtype=np.float64)


def bend_betas(timesteps=T):
    """`lin_beta_at(t, steps)` from mlbend-generative, t = 1..timesteps."""
    return np.array([0.0001 + 0.2799 * (t / timesteps) for t in range(1, timesteps + 1)],
                    dtype=np.float64)


def print_schedules():
    ref, bend = reference_betas(), bend_betas()
    ref_abar = np.cumprod(1.0 - ref)
    bend_abar = np.cumprod(1.0 - bend)
    print("the schedules (both T = 64, linear)")
    print(f"  beta first        reference {ref[0]:.7f}   bend {bend[0]:.7f}")
    print(f"  beta last         reference {ref[-1]:.7f}   bend {bend[-1]:.7f}")
    print(f"  alpha-bar at T    reference {ref_abar[-1]:.6e}   bend {bend_abar[-1]:.6e}")
    worst = float(np.max(np.abs(ref - bend) / np.maximum(ref, bend)))
    print(f"  worst beta gap    {worst:.3%} (relative)")
    return ref_abar[-1], bend_abar[-1]


# ---- the run ------------------------------------------------------------

def grid_svg(path, data, samples):
    """The digits grid. The cell offset is `c * (cell/8)`; the Bend example's
    renderer adds a stray `c*cell//8*8//8` term that makes cells overlap."""
    cell, pad, rows = 14, 2, 8
    px = cell / 8
    w = 2 * rows * cell + 3 * pad
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{w}" viewBox="0 0 {w} {w}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<text x="4" y="10" font-family="sans-serif" font-size="9">'
             'data (left) vs lucidrains DDIM samples (right)</text>']

    def draw(img, ox, oy):
        img = np.clip(np.asarray(img, dtype=np.float64).reshape(8, 8), 0, 1)
        for r in range(8):
            for c in range(8):
                v = int(round(255 * (1 - float(img[r, c]))))
                parts.append(f'<rect x="{ox + c * px:.1f}" y="{oy + r * px:.1f}" '
                             f'width="{px:.1f}" height="{px:.1f}" fill="rgb({v},{v},{v})"/>')

    for i in range(rows * rows):
        ox = pad + (i % rows) * cell
        oy = pad + 12 + (i // rows) * cell
        draw(data[i] if i < len(data) else np.zeros(64), ox, oy)
        draw(samples[i] if i < len(samples) else np.zeros(64), pad + rows * cell + pad + (i % rows) * cell, oy)
    parts.append("</svg>")
    open(path, "w").write("\n".join(parts))


def main():
    steps = int(os.environ.get("STEPS", STEPS))
    dim = int(os.environ.get("DIM", 32))
    batch = int(os.environ.get("BATCH", 128))
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 4)))

    print_schedules()

    x0 = load().astype(np.float32)                     # (1024, 64) in [0, 1]
    x = torch.from_numpy(x0.reshape(N, 1, 8, 8))       # the library normalizes to [-1, 1]

    model = Unet(dim=dim, dim_mults=(1, 2), channels=1, flash_attn=False)
    diffusion = GaussianDiffusion(model, image_size=8, timesteps=T,
                                  sampling_timesteps=16, objective="pred_noise",
                                  beta_schedule="linear")
    params = sum(p.numel() for p in model.parameters())
    print(f"\nthe model: Unet(dim={dim}, dim_mults=(1, 2)) -- {params} parameters")
    print(f"  alpha-bar of the library's own tensor: {float(diffusion.alphas_cumprod[-1]):.6e}")

    opt = Adam(model.parameters(), lr=2e-4)
    losses = []
    t0 = time.time()
    for i in range(steps):
        idx = torch.randint(0, N, (batch,))
        loss = diffusion(x[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == steps:
            print(f"loss: {float(loss):.6g}")
    seconds = time.time() - t0
    print(f"\ntrained {steps} steps in {seconds:.1f}s ({seconds / steps * 1000:.0f} ms/step)")

    model.eval()
    # `diffusion` is the DDIM object (sampling_timesteps = 16); the ancestral
    # loop needs sampling_timesteps == timesteps, so it gets its own wrapper
    # over the same trained weights
    ddpm = GaussianDiffusion(model, image_size=8, timesteps=T, sampling_timesteps=T,
                             objective="pred_noise", beta_schedule="linear")
    with torch.no_grad():
        ddim = diffusion.sample(batch_size=64).squeeze(1).numpy().reshape(-1, D)
        ancestral = ddpm.sample(batch_size=64).squeeze(1).numpy().reshape(-1, D)

    from vit_report import legibility as leg_stat, nearest_mse

    data = load()
    def stats(a):
        m, lo, hi = leg_stat(a, data)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max()),
                "in_unit": int(((a >= 0) & (a <= 1)).sum()),
                "n": int(a.size), "legibility": m, "legibility_min": lo,
                "legibility_max": hi, "nearest_mse": nearest_mse(a, data),
                "samples": [float(z) for z in np.asarray(a).reshape(-1)]}

    sd, sa = stats(ddim), stats(ancestral)
    print("\nthe clouds (lucidrains, 64 digits x 64 pixels)")
    for name, s in (("ddim", sd), ("ancestral", sa)):
        print(f"  {name:10s} mean {s['mean']:+.4f}  std {s['std']:.4f}  "
              f"min {s['min']:+.4f}  max {s['max']:+.4f}  in [0,1]: {s['in_unit']}/4096")
        print(f"  {name:10s} legibility {s['legibility']:.3f}  "
              f"nearest-digit mse {s['nearest_mse']:.4f}")

    print("\nthe same statistics, Bend's digits run (results/diffusion-digits/run.json)")
    bend = json.load(open(os.path.join(ROOT, "results", "diffusion-digits", "run.json")))["bend"]
    print(f"  ancestral  mean {bend['sample_mean']:+.4f}  std {bend['sample_std']:.4f}")
    print(f"  ddim       mean {bend['ddim_mean']:+.4f}  std {bend['ddim_std']:.4f}")
    print(f"  data       mean {bend['data_mean']:+.4f}  std {bend['data_std']:.4f}")

    res = os.path.join(ROOT, "results", "lucidrains-diffusion")
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, "loss.tsv"), "w") as f:
        f.write("step\tloss\n")
        for i, v in enumerate(losses):
            f.write(f"{i}\t{v}\n")
    grid_svg(os.path.join(res, "grid.svg"), x0[:64], ddim)
    json.dump({"created_at": datetime.now(timezone.utc).isoformat(),
               "reference": "lucidrains/denoising-diffusion-pytorch (vendored, reference/lucidrains)",
               "steps": steps, "batch": batch, "seconds": seconds,
               "model": {"unet_dim": dim, "params": params},
               "schedule": {"reference_beta_first": float(reference_betas()[0]),
                            "reference_beta_last": float(reference_betas()[-1]),
                            "reference_abar_T": float(diffusion.alphas_cumprod[-1]),
                            "bend_beta_first": float(bend_betas()[0]),
                            "bend_beta_last": float(bend_betas()[-1]),
                            "bend_abar_T": float(np.cumprod(1.0 - bend_betas())[-1])},
               "loss": {"first": losses[0], "final": losses[-1]},
               "starting_noise": "the library's own draw (sample() takes no image); "
                                 "the Bend and torch arms share noise_of(2000000, 4096)",
               "lucidrains": {"ddim": sd, "ancestral": sa},
               "bend": bend},
              open(os.path.join(res, "run.json"), "w"), indent=2)
    print(f"\nwrote {res}/loss.tsv, grid.svg, run.json")


if __name__ == "__main__":
    main()
