"""Is a ViT-style denoiser enough for 8x8 digits? A design probe before porting.

  reference/.venv-lucidrains/bin/python reference/digits_vit_probe.py

The dense MLP in `examples/diffusion/digits` learns noise (final loss 0.042)
but generates mush: its samples' best correlation with any training digit is
0.408, against 0.326 for random images, while the reference's UNet reaches
0.926. The gap is spatial structure. `Conv2d` does not exist in BendTorch yet
(issue #15), so the route we can take today is attention: 2x2 patches, a
transformer encoder, LayerNorm and residual connections — all of which
BendTorch has.

This probe asks the only question that matters before porting: does that
architecture reach legible digits on this data inside the same 3000-step
budget? It reports the same legibility statistic and the cloud's statistics,
next to the numbers already recorded for the MLP and the UNet.

It is NOT a gate and NOT the Bend arm; it is the design decision, measured.
"""
import json
import os
import sys

import numpy as np
import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

from digits_ablation import bend_betas, temb
from test_diffusion_digits import D, N, T, load

STEPS = int(os.environ.get("STEPS", 3000))
BATCH = int(os.environ.get("BATCH", 128))
DIM = 64
HEADS = int(os.environ.get("HEADS", 4))
BLOCKS = 3
LR = float(os.environ.get("LR", 3e-3))
PATCH = 2                      # 8x8 pixels -> 4x4 = 16 tokens
TOKENS = (8 // PATCH) ** 2
PIX = PATCH * PATCH


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(DIM)
        self.attn = nn.MultiheadAttention(DIM, HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM, 2 * DIM), nn.GELU(), nn.Linear(2 * DIM, DIM))

    def forward(self, x):
        h = self.ln1(x)
        mask = None
        if CAUSAL:
            mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool,
                                         device=x.device), 1)
        x = x + self.attn(h, h, h, need_weights=False, attn_mask=mask)[0]
        return x + self.mlp(self.ln2(x))


POS_MODE = os.environ.get("POS", "learned")
# BendTorch's only attention is causal (ISSUE-011), so the arm that represents
# the Bend model in the comparison runs with the same mask.
CAUSAL = os.environ.get("CAUSAL") == "1"
# DEVICE=cuda on the card; the arm is 64 wide, so on the CPU 3 threads beat 16
DEVICE = os.environ.get("DEVICE", "cpu")
# TRAIN=bend reproduces the Bend example's loop instead of the usual one:
# one timestep for the whole batch, the digit chunk rotating once per sweep,
# the step index seeding the noise. Everything else stays the same, so a
# difference in the cloud is attributable to the loop.
BEND_LOOP = os.environ.get("TRAIN") in ("bend", "bend-rand")


def pos_features():
    """Fixed sinusoidal positions, as 8 values per token (Bend can carry a
    position only as part of the row: BendTorch has no per-row table)."""
    out = []
    for p in range(TOKENS):
        row = []
        for f in (0.001, 0.01, 0.1, 1.0):
            row += [np.sin(p * f), np.cos(p * f)]
        out.append(row)
    return torch.tensor(out, dtype=torch.float32)


class Vit(nn.Module):
    def __init__(self):
        super().__init__()
        extra = 8 if POS_MODE == "features" else 0
        self.embed = nn.Linear(PIX + 8 + extra, DIM)   # pixels + time (+ position)
        self.pos = nn.Parameter(torch.zeros(1, TOKENS, DIM))
        self.blocks = nn.ModuleList(Block() for _ in range(BLOCKS))
        self.ln = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, PIX)

    def forward(self, patches, te):
        te = te.unsqueeze(1).expand(-1, patches.shape[1], -1)
        if POS_MODE == "features":
            pf = pos_features().to(patches.device).unsqueeze(0).expand(patches.shape[0], -1, -1)
            x = self.embed(torch.cat([patches, te, pf], dim=-1))
        else:
            x = self.embed(torch.cat([patches, te], dim=-1)) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def patchify(x):
    """(B, 64) -> (B, 16, 4), row-major 2x2 patches."""
    b = x.shape[0]
    v = x.reshape(b, 4, 2, 4, 2)          # rows/2, 2, cols/2, 2
    return v.permute(0, 1, 3, 2, 4).reshape(b, TOKENS, PIX)


def unpatchify(p):
    b = p.shape[0]
    v = p.reshape(b, 4, 4, 2, 2).permute(0, 1, 3, 2, 4)
    return v.reshape(b, D)


def legibility(samples, data):
    s = samples.reshape(len(samples), 64).astype(np.float64)
    d = data.reshape(len(data), 64).astype(np.float64)
    s = s - s.mean(1, keepdims=True); s /= np.linalg.norm(s, axis=1, keepdims=True) + 1e-12
    d = d - d.mean(1, keepdims=True); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    c = np.abs(s @ d.T).max(1)
    return c.mean(), c.min(), c.max()


def dev():
    return torch.device(DEVICE)


def start_noise(n=64):
    """The starting cloud every arm uses: noise_of(2000000, 4096), n x D.

    The Bend example starts its samplers from exactly these numbers, so the
    comparison holds the starting noise fixed instead of drawing its own.
    """
    from test_diffusion_digits import noise_of
    v = noise_of(2000000, 4096).astype(np.float32)
    if n * D > v.size:
        v = np.tile(v, 1 + n * D // v.size)
    return torch.from_numpy(v[: n * D].reshape(n, D).copy()).to(dev())


@torch.no_grad()
def sample_ddpm(model, abar, tembs, n=64, seed=1000):
    """The 64-step ancestral walk, Bend's rev_step arithmetic: clamp the
    predicted start to [0,1], then the posterior mean c1*x0 + c2*x."""
    model.eval()
    torch.manual_seed(seed)
    x = start_noise(n).clone()
    beta = torch.from_numpy(bend_betas().astype(np.float32)).to(dev())
    for t in range(T - 1, -1, -1):
        tt = torch.full((n,), t, dtype=torch.long, device=dev())
        eps = unpatchify(model(patchify(x), tembs[tt]))
        a = abar[t].reshape(1, 1)
        ap = abar[t - 1].reshape(1, 1) if t > 0 else torch.ones(1, 1)
        bt = beta[t].reshape(1, 1)
        c1 = bt * torch.sqrt(ap) / (1 - a)
        c2 = torch.sqrt(1 - bt) * (1 - ap) / (1 - a)
        sa, sm = torch.sqrt(a), torch.sqrt(1 - a)
        x0 = ((x - sm * eps) / sa).clamp(0, 1)
        sig = torch.sqrt(bt * (1 - ap) / (1 - a))
        z = torch.randn(n, D, device=dev()) if t > 0 else torch.zeros(n, D, device=dev())
        x = c1 * x0 + c2 * x + sig * z
    return x.detach().cpu().numpy()


@torch.no_grad()
def sample(model, abar, tembs, n=64):
    model.eval()
    torch.manual_seed(1000)
    x = start_noise(n).clone()
    # the reference's rungs (lucidrains' linspace(-1, T-1, steps+1)): every 4th
    # from T-1 down to 3, and the last step lands on abar = 1. The old
    # `[3::4] + [0]` gave 60..0 plus a duplicated 0 - a schedule the reference
    # never uses, which made the DDIM column not like-for-like.
    pos = list(range((T - 1) % 4, T, 4))[::-1]
    for i, p in enumerate(pos):
        below = pos[i + 1] if i + 1 < len(pos) else -1
        t = torch.full((n,), p, dtype=torch.long, device=dev())
        eps = unpatchify(model(patchify(x), tembs[t]))
        a = abar[p].reshape(1, 1)
        ap = abar[below].reshape(1, 1) if below >= 0 else torch.ones(1, 1)
        x0 = ((x - torch.sqrt(1 - a) * eps) / torch.sqrt(a)).clamp(0, 1)
        eps = (x - torch.sqrt(a) * x0) / torch.sqrt(1 - a)
        x = torch.sqrt(ap) * x0 + torch.sqrt(torch.clamp(1 - ap, min=0.0)) * eps
    return x.detach().cpu().numpy()


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(int(os.environ.get("THREADS", max(1, min(16, os.cpu_count() or 4)))))
    x0 = torch.from_numpy(load().astype(np.float32)).to(dev())
    abar = torch.from_numpy(np.cumprod(1.0 - bend_betas()).astype(np.float32)).to(dev())
    tembs = torch.from_numpy(np.stack([temb(t) for t in range(1, T + 1)])).to(dev())
    model = Vit().to(dev())
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ViT: dim {DIM}, {HEADS} heads, {BLOCKS} blocks, {TOKENS} tokens of {PIX} px "
          f"-- {n_params} parameters; pos {POS_MODE}, causal {CAUSAL}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.999))
    losses = []
    for s in range(STEPS):
        if BEND_LOOP:
            chunk = (s // 64) % (N // BATCH)
            if os.environ.get("TRAIN") == "bend-rand":
                # ONE t for the whole batch, drawn at random, random chunk:
                # the true isolator for "periodic" against "per-image t"
                chunk = int(torch.randint(0, N // BATCH, (1,)))
                tt = int(torch.randint(0, T, (1,)))
                t = torch.full((BATCH,), tt, dtype=torch.long, device=dev())
                idx = torch.arange(chunk * BATCH, (chunk + 1) * BATCH, device=dev())
            else:
                t = torch.full((BATCH,), s % T, dtype=torch.long, device=dev())
                idx = torch.arange(chunk * BATCH, (chunk + 1) * BATCH, device=dev())
            torch.manual_seed(s)
            eps = torch.randn(BATCH, D, device=dev())
        else:
            idx = torch.randint(0, N, (BATCH,)).to(dev())
            t = torch.randint(0, T, (BATCH,)).to(dev())
            eps = torch.randn(BATCH, D, device=dev())
        a = abar[t].unsqueeze(1)
        xt = a * x0[idx] + torch.sqrt(1 - a) * eps
        out = unpatchify(model(patchify(xt), tembs[t]))
        loss = ((out - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    print(f"loss: first {losses[0]:.4f}  last100 {np.mean(losses[-100:]):.4f}")
    data = load()
    clouds = {"ddim": sample(model, abar, tembs),
              "ddpm": sample_ddpm(model, abar, tembs)}
    from vit_report import nearest_mse
    report = {}
    for name, smp in clouds.items():
        m, lo, hi = legibility(smp, data)
        unit = int(((smp >= 0) & (smp <= 1)).sum())
        report[name] = {"legibility": m, "legibility_min": lo, "legibility_max": hi,
                        "cloud": {"mean": float(smp.mean()), "std": float(smp.std()),
                                  "min": float(smp.min()), "max": float(smp.max()),
                                  "in_unit": unit, "n": int(smp.size)},
                        "nearest_mse": nearest_mse(smp, data),
                        "samples": [float(z) for z in smp.reshape(-1)]}
        print(f"{name}: cloud mean {smp.mean():+.4f}  std {smp.std():.4f}  "
              f"in [0,1] {unit}/{smp.size}")
        print(f"{name}: nearest-digit |corr| mean {m:.3f}  min {lo:.3f}  max {hi:.3f}  "
              f"mse {report[name]['nearest_mse']:.4f}")
    if len(sys.argv) > 1:
        out = {"arm": "torch-vit", "params": n_params, "steps": STEPS, "batch": BATCH,
               "lr": LR, "heads": HEADS, "pos": POS_MODE, "causal": CAUSAL,
               "device": DEVICE, "train_loop": os.environ.get("TRAIN", "torch"),
               "loss": {"first": losses[0], "final": losses[-1]},
               "curve": losses,
               "nearest_mse": None, "per_sampler": report}
        os.makedirs(os.path.dirname(os.path.abspath(sys.argv[1])), exist_ok=True)
        json.dump(out, open(sys.argv[1], "w"), indent=2)
        print(f"wrote {sys.argv[1]}")


if __name__ == "__main__":
    main()
