"""TEM-lite: the paper's shape at a size BendTorch can train.

16x16 images, 4x4 patches, 16 tokens, width 64, two blocks, causal attention,
learned position, additive class label, two energy heads. Everything here maps
to a BendTorch layer, unlike the 64x64 TEM arm (full attention, RoPE, adaLN).

  reference/.venv-lucidrains/bin/python papers/TEM/lite.py --mode score
  reference/.venv-lucidrains/bin/python papers/TEM/lite.py --mode cd

Both arms are written here so the Bend port has an exact reference to match.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import load

OUT = os.path.join(HERE, "results")
DIM, PATCH, GRID, HEADS = 64, 4, 4, 4
NTOK = GRID * GRID


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(DIM)
        self.qkv = nn.Linear(DIM, 3 * DIM)
        self.proj = nn.Linear(DIM, DIM)
        self.n2 = nn.LayerNorm(DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.SiLU(),
                                 nn.Linear(4 * DIM, DIM))

    def forward(self, x):
        b, n = x.shape[0], x.shape[1]
        qkv = self.qkv(self.n1(x)).reshape(b, n, 3, HEADS, DIM // HEADS)
        q, k, v = qkv.unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(out.transpose(1, 2).reshape(b, n, DIM))
        return x + self.mlp(self.n2(x))


class Head(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.n = nn.LayerNorm(DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM, DIM), nn.SiLU(),
                                 nn.Linear(DIM, 1, bias=bias))

    def forward(self, x):
        return self.mlp(self.n(x.mean(1))).squeeze(-1)


class Lite(nn.Module):
    def __init__(self, score=False):
        super().__init__()
        self.score = score
        self.embed = nn.Linear(PATCH * PATCH * 3, DIM)
        self.blocks = nn.ModuleList([Block() for _ in range(2)])
        self.heads = nn.ModuleList([Head(False), Head(False)])
        self.shead = nn.Linear(DIM, PATCH * PATCH * 3)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def pos_code(self, device):
        t = torch.arange(NTOK, device=device, dtype=torch.float32)[:, None]
        f = 1.0 / (10000 ** (torch.arange(0, DIM // 2, device=device,
                                         dtype=torch.float32) / (DIM // 2)))
        a = t * f
        return torch.cat([a.sin(), a.cos()], 1)[None]

    def label_code(self, y):
        k = y.float()[:, None]
        f = torch.arange(DIM // 2, device=y.device, dtype=torch.float32) / (DIM // 2)
        a = k * f * torch.pi
        return torch.cat([a.sin(), a.cos()], 1)[:, None, :]

    def patches(self, x):
        b = x.shape[0]
        return (x.reshape(b, 3, GRID, PATCH, GRID, PATCH)
                .permute(0, 2, 4, 1, 3, 5).reshape(b, NTOK, -1))

    def forward(self, x, y):
        z = (self.embed(self.patches(x)) + self.pos_code(x.device)
             + self.label_code(y))
        outs = []
        for i, block in enumerate(self.blocks):
            z = block(z)
            if i < len(self.heads):
                outs.append(self.heads[i](z))
        if self.score:
            s = self.shead(z).reshape(-1, GRID, GRID, 3, PATCH, PATCH)
            return s.permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, GRID * PATCH, GRID * PATCH)
        return torch.stack(outs, 1)

    def energy(self, x, y):
        return self(x, y).sum(1)


def data16():
    tr, ty = load("train")
    te, ey = load("test")
    def prep(x):
        t = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 127.5 - 1
        return F.interpolate(t, size=(16, 16), mode="bilinear", align_corners=False)
    return prep(tr), torch.from_numpy(ty).long(), prep(te), torch.from_numpy(ey).long()


def grid(samples, path):
    from PIL import Image
    x = ((samples.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()
    n, s = 8, 16
    out = np.zeros((n * s, n * s, 3), np.uint8)
    for i in range(min(n * n, len(x))):
        out[(i // n) * s:(i // n + 1) * s, (i % n) * s:(i % n + 1) * s] = x[i]
    Image.fromarray(out).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cd", "score"], default="score")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--step-size", type=float, default=0.05)
    ap.add_argument("--sample-steps", type=int, default=60)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=250)
    a = ap.parse_args()

    torch.manual_seed(0)
    tr, ty, te, ey = data16()
    model = Lite(score=(a.mode == "score"))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.999))
    if a.mode == "cd":
        bi = torch.randint(0, len(tr), (1024,))
        buffer = (tr[bi] + 0.1 * torch.randn(1024, 3, 16, 16)).clamp(-1, 1)
    curve, t0 = [], time.time()

    for step in range(a.steps):
        idx = torch.randint(0, len(tr), (a.batch,))
        x, y = tr[idx], ty[idx]
        if a.mode == "score":
            eps = torch.randn_like(x)
            pred = model(x + a.sigma * eps, y)
            loss = F.mse_loss(pred, -eps / a.sigma)
        else:
            bidx = torch.randint(0, 1024, (a.batch,))
            neg = buffer[bidx].clone()
            for _ in range(9):
                with torch.enable_grad():
                    neg = (neg + 0.01 * torch.randn_like(neg)).requires_grad_(True)
                    g = torch.autograd.grad(model.energy(neg, y).sum(), neg)[0]
                neg = (neg - a.step_size * g).detach().clamp(-1, 1)
            pos_e = model.energy(x, y)
            neg_e = model.energy(neg, y)
            loss = ((pos_e ** 2).mean() + (neg_e ** 2).mean()
                    + pos_e.mean() - neg_e.mean())
            buffer[bidx] = neg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        curve.append({"step": step, "loss": float(loss.detach())})
        if step % a.log_every == 0:
            print(f"{step:5d}  loss {float(loss): .4f}  "
                  f"{(time.time() - t0) / (step + 1):.3f}s/step", flush=True)

    model.eval()
    torch.manual_seed(1)
    sel = torch.randint(0, len(te), (a.samples,))
    x = torch.randn(a.samples, 3, 16, 16).clamp(-1, 1)
    y = ey[sel]
    if a.mode == "score":
        alpha = a.sigma ** 2
        for _ in range(a.sample_steps):
            with torch.no_grad():
                s = model(x, y)
            x = (x + alpha * s + (2 * alpha) ** 0.5 * torch.randn_like(x)).clamp(-1, 1)
    else:
        for _ in range(a.sample_steps):
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                g = torch.autograd.grad(model.energy(x, y).sum(), x)[0]
            x = (x - a.step_size * g + 0.01 * torch.randn_like(x)).clamp(-1, 1)

    tag = f"lite-{a.mode}"
    grid(x, os.path.join(OUT, f"{tag}-samples.png"))
    np.save(os.path.join(OUT, f"{tag}-samples.npy"), x.numpy())
    nn = torch.cdist(x.reshape(len(x), -1), te.reshape(len(te), -1)).min(1).values
    rec = {"tag": tag, "mode": a.mode, "steps": a.steps,
           "params": sum(p.numel() for p in model.parameters()),
           "loss_first": curve[0]["loss"], "loss_last50": float(np.mean([c["loss"] for c in curve[-50:]])),
           "nn_test_mean": float(nn.mean()), "seconds": time.time() - t0,
           "curve": curve[::max(1, len(curve) // 200)]}
    json.dump(rec, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=2)
    print(json.dumps({k: rec[k] for k in ("mode", "loss_first", "loss_last50",
                                          "nn_test_mean", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
