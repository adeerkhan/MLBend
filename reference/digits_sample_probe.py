"""Does clamping x0 bound the cloud? One trained MLP, two samplers, four ways.

  reference/.venv-lucidrains/bin/python reference/digits_sample_probe.py

The Bend digits run now learns (first loss 0.868, final 0.0417 after the init
fix) but its sampled cloud is 95x too wide (std 35.8 against the data's 0.377),
while the vendored reference's cloud sits at 0.34. The reference clamps the
predicted x0 to its data range before forming the posterior mean, and the Bend
samplers do not. This script trains the example's own MLP once and samples it
two ways, each with and without that clamp, so the clamp's effect is a number
rather than an inference from two different models.

x0 = (x_t - sqrt(1 - abar) eps) / sqrt(abar)          the model's implied start
clamp to [lo, hi]                                      what the reference does
mean = coef1 * x0 + coef2 * x_t,  sigma = sqrt(beta_tilde)
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

from digits_ablation import D, N, T, bend_betas, make_model, temb, LR
from test_diffusion_digits import load, legibility

STEPS = 3000
SAMPLES = 64


def train_model():
    torch.manual_seed(0)
    np.random.seed(0)
    x0 = torch.from_numpy(load().astype(np.float32))
    betas = bend_betas()
    abar = np.cumprod(1.0 - betas).astype(np.float32)
    tembs = torch.from_numpy(np.stack([temb(t) for t in range(1, T + 1)]))
    model = make_model("torch")
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    for _ in range(STEPS):
        idx = torch.randint(0, N, (128,))
        t = torch.randint(0, T, (128,))
        eps = torch.randn(128, D)
        a = torch.from_numpy(abar[t.numpy()]).unsqueeze(1)
        xt = a * x0[idx] + torch.sqrt(1.0 - a) * eps
        loss = ((model(torch.cat([xt, tembs[t]], dim=1)) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model, torch.from_numpy(abar), tembs


@torch.no_grad()
def sample(model, abar, tembs, ddim, clamp, lo=0.0, hi=1.0, salt=0):
    torch.manual_seed(1000 + salt)
    x = torch.randn(SAMPLES, D)
    abar_t = torch.from_numpy(abar.numpy())
    positions = list(range(T - 1, -1, -1))
    if ddim:
        positions = positions[3::4] + [0]
    prev_pos = positions[1:] + [-1]
    for pos, below in zip(positions, prev_pos):
        t = torch.full((SAMPLES,), pos, dtype=torch.long)
        eps = model(torch.cat([x, tembs[t]], dim=1))
        a = abar_t[pos].reshape(1, 1)
        a_prev = abar_t[below].reshape(1, 1) if below >= 0 else torch.ones(1, 1)
        x0 = (x - torch.sqrt(1.0 - a) * eps) / torch.sqrt(a)
        if clamp:
            x0 = x0.clamp(lo, hi)
        beta = torch.from_numpy(1.0 - (abar_t[pos] / a_prev).numpy()).reshape(1, 1)
        if ddim:
            c = torch.sqrt(torch.clamp(1.0 - a_prev - 0.0, min=0.0))
            x = torch.sqrt(a_prev) * x0 + c * eps
        else:
            coef1 = beta * torch.sqrt(a_prev) / (1.0 - a)
            coef2 = torch.sqrt(1.0 - beta) * (1.0 - a_prev) / (1.0 - a)
            sig = torch.sqrt(torch.clamp(beta * (1.0 - a_prev) / (1.0 - a), min=0.0))
            noise = torch.randn_like(x) if pos > 0 else torch.zeros_like(x)
            x = coef1 * x0 + coef2 * x + sig * noise
    return x.numpy().reshape(-1)


def main():
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 4)))
    model, abar, tembs = train_model()
    data = load()
    print(f"data        mean {data.mean():+.4f}  std {data.std():.4f}")
    for name, ddim, clamp in (("ddpm  no clamp", False, False),
                              ("ddpm  clamped ", False, True),
                              ("ddim  no clamp", True, False),
                              ("ddim  clamped ", True, True)):
        v = sample(model, abar, tembs, ddim, clamp)
        unit = int(((v >= 0) & (v <= 1)).sum())
        corr = legibility(v, load())[0] if clamp else float("nan")
        print(f"{name}  mean {v.mean():+9.4f}  std {v.std():9.4f}  "
              f"in [0,1]: {unit}/4096  legibility {corr:.3f}")


if __name__ == "__main__":
    main()
