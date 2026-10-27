"""Which factor keeps the digits denoiser at MSE 1.0? A controlled ablation.

  reference/.venv-lucidrains/bin/python reference/digits_ablation.py

The Bend digits example is a 72 -> 64 -> 64 -> 64 SiLU MLP (64 pixel values
plus 8 time-embedding values in, 64 noise values out), trained full-batch by
Adam at lr 1e-3 for 3000 steps. Its loss sits at the "predict nothing" value
of 1.0 (ISSUE-001), while the vendored reference reaches 0.106 in 300 steps.
Three differences could explain it, and this script changes them one at a
time on the *same* architecture, data and optimizer:

  init      bend  : uniform [-0.2, +0.5] (the example's `value`)
            torch : PyTorch's default Linear init (zero-mean, 1/sqrt(fan_in))
  range     [0,1] : raw pixels, as the example feeds them
            [-1,1]: centred, as the reference normalizes them
  schedule  bend  : beta = 1e-4 + 0.2799*t/T (mlbend-generative)
            ref   : linspace(1000/T*1e-4, 1000/T*0.02, T) (the reference)

It prints the first loss, the loss at step 300, and the mean of the last 100
steps for each variant, so the cause is a number rather than a theory. Nothing
here is a gate; it exists so ISSUE-001 can be closed with evidence.
"""
import os
import sys

import numpy as np
import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

from test_diffusion_digits import N, D, load

T = 64
STEPS = 3000
LR = 1e-3

FREQS = (0.001, 0.01, 0.1, 1.0)


def temb(t):
    """mlbend-generative's temb_of(t, 64, 4): sin/cos of tau*f, p = 0..3."""
    tau = t / T
    out = []
    for f in FREQS:
        out += [np.sin(tau * f), np.cos(tau * f)]
    return np.array(out, dtype=np.float32)


def bend_betas():
    return np.array([0.0001 + 0.2799 * (t + 1) / T for t in range(T)], dtype=np.float32)


def ref_betas():
    scale = 1000.0 / T
    return np.linspace(scale * 0.0001, scale * 0.02, T, dtype=np.float32)


def make_model(init):
    model = nn.Sequential(nn.Linear(D + 8, 64), nn.SiLU(),
                          nn.Linear(64, 64), nn.SiLU(),
                          nn.Linear(64, D))
    if init == "bend":
        with torch.no_grad():
            for layer in model:
                if isinstance(layer, nn.Linear):
                    # the example's value(i, j, seed) = (k*0.1) - 0.2, k in 0..6
                    w = layer.weight
                    k = (torch.arange(w.numel()) % 7).reshape(w.shape).float()
                    layer.weight.copy_(k * 0.1 - 0.2)
                    layer.bias.copy_(torch.zeros_like(layer.bias))
    return model


def run(init, rng, schedule, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x0 = load()
    if rng == "m11":
        x0 = 2.0 * x0 - 1.0
    x0 = torch.from_numpy(x0.astype(np.float32))
    betas = bend_betas() if schedule == "bend" else ref_betas()
    abar = np.cumprod(1.0 - betas).astype(np.float32)
    tembs = torch.from_numpy(np.stack([temb(t) for t in range(1, T + 1)]))

    model = make_model(init)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    losses = []
    for i in range(steps):
        idx = torch.randint(0, N, (128,))
        t = torch.randint(0, T, (128,))
        eps = torch.randn(128, D)
        a = torch.from_numpy(abar[t.numpy()]).unsqueeze(1)
        b = torch.sqrt(1.0 - a)
        xt = a * x0[idx] + b * eps
        inp = torch.cat([xt, tembs[t]], dim=1)
        loss = ((model(inp) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return losses


def main():
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 4)))
    x0 = load()
    print(f"data: {N} digits, {D} pixels, mean {x0.mean():.4f} std {x0.std():.4f}")
    print(f"{'variant':34s} {'init':6s} {'range':6s} {'sched':5s} "
          f"{'loss@0':>9s} {'loss@300':>9s} {'last100':>9s}")
    for init, rng, sched in (
        ("bend", "01", "bend"),      # the example as it stands
        ("torch", "01", "bend"),
        ("torch", "m11", "bend"),
        ("torch", "m11", "ref"),
        ("bend", "m11", "ref"),
    ):
        losses = run(init, rng, sched)
        name = f"{init} / {rng} / {sched}"
        print(f"{name:34s} {init:6s} {rng:6s} {sched:5s} "
              f"{losses[0]:9.4f} {losses[300]:9.4f} {np.mean(losses[-100:]):9.4f}")


if __name__ == "__main__":
    main()
