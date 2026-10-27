"""Is a ViT-style denoiser enough for 8x8 digits? A design probe before porting.

  reference/.venv-lucidrains/bin/python reference/digits_vit_probe.py

The dense MLP in `examples/diffusion_digits` learns noise (final loss 0.042)
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
import os
import sys

import numpy as np
import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

from digits_ablation import bend_betas, temb
from test_diffusion_digits import D, N, T, load

STEPS = 3000
BATCH = 128
DIM = 64
HEADS = 4
BLOCKS = 3
LR = 3e-3
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
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


POS_MODE = os.environ.get("POS", "learned")


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


@torch.no_grad()
def sample(model, abar, tembs, n=64):
    model.eval()
    torch.manual_seed(1000)
    x = torch.randn(n, D)
    pos = list(range(T - 1, -1, -1))[3::4] + [0]
    for i, p in enumerate(pos):
        below = pos[i + 1] if i + 1 < len(pos) else -1
        t = torch.full((n,), p, dtype=torch.long)
        eps = unpatchify(model(patchify(x), tembs[t]))
        a = abar[p].reshape(1, 1)
        ap = abar[below].reshape(1, 1) if below >= 0 else torch.ones(1, 1)
        x0 = ((x - torch.sqrt(1 - a) * eps) / torch.sqrt(a)).clamp(0, 1)
        eps = (x - torch.sqrt(a) * x0) / torch.sqrt(1 - a)
        x = torch.sqrt(ap) * x0 + torch.sqrt(torch.clamp(1 - ap, min=0.0)) * eps
    return x.numpy()


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 4)))
    x0 = torch.from_numpy(load().astype(np.float32))
    abar = torch.from_numpy(np.cumprod(1.0 - bend_betas()).astype(np.float32))
    tembs = torch.from_numpy(np.stack([temb(t) for t in range(1, T + 1)]))
    model = Vit()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ViT: dim {DIM}, {HEADS} heads, {BLOCKS} blocks, {TOKENS} tokens of {PIX} px "
          f"-- {n_params} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.999))
    losses = []
    for s in range(STEPS):
        idx = torch.randint(0, N, (BATCH,))
        t = torch.randint(0, T, (BATCH,))
        eps = torch.randn(BATCH, D)
        a = abar[t].unsqueeze(1)
        xt = a * x0[idx] + torch.sqrt(1 - a) * eps
        out = unpatchify(model(patchify(xt), tembs[t]))
        loss = ((out - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    print(f"loss: first {losses[0]:.4f}  last100 {np.mean(losses[-100:]):.4f}")
    smp = sample(model, abar, tembs)
    m, lo, hi = legibility(smp, load())
    unit = int(((smp >= 0) & (smp <= 1)).sum())
    print(f"cloud: mean {smp.mean():+.4f}  std {smp.std():.4f}  in [0,1] {unit}/4096")
    print(f"nearest-digit |corr|: mean {m:.3f}  min {lo:.3f}  max {hi:.3f}   "
          f"(MLP 0.408, UNet 0.926, random 0.326)")


if __name__ == "__main__":
    main()
