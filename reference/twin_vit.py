"""The NumPy twin of the Bend ViT denoiser (issue #16, milestone M2).

  reference/.venv/bin/python reference/twin_vit.py

A mirror of `reference/probe_vit_model.bend`, in float32, at the same
operation order:

  20 features per token (4 patched pixels, 8 time, 8 position)
  embed      LinearNB 20 -> 64          y = x @ W.T          (no bias)
  block x 3  LayerNorm -> attention -> residual
             LayerNorm -> 64 -> 128 -> 64 GELU -> residual
  attention  q, k, v each x @ W.T, one head,
             scores = (q @ k.T) * 0.125, softmax, out = scores @ v,
             then x @ Wo.T
  LayerNorm  y = ((x - mean) * rstd) * g + b, rstd = 1/sqrt(var + eps)
  GELU       0.5x (1 + tanh(0.7978845608 (x + 0.044715 x^3)))
  head       LinearNB 64 -> 4

The weights are named as in the file: embed, then per block (att q/k/v/out,
norms, mlp), then head. `from_flat` takes the flat list a Bend `Layer.dump`
writes, in the loader's order, so a run of the Bend probe and a run of this
twin can be compared without trusting either side's initialisation.

Nothing here is a gate yet: the gates land with `reference/test_diffusion_vit.py`.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

F = np.float32
W, HI, PX, FE, TK = 64, 128, 4, 20, 16
SCALE = F(0.125)
EPS = F(1e-5)


def patch(x):
    """64 pixels -> 64 values in token order (16 tokens of 4, 2x2 patches)."""
    idx = []
    for k in range(16):
        r, c = k // 4, k % 4
        b = 16 * r + 2 * c
        idx += [b, b + 1, b + 8, b + 9]
    return x[..., idx]


def unpatch(p):
    out = np.zeros_like(p)
    for i in range(64):
        k = 4 * (i // 16) + ((i % 8) // 2)
        d = 2 * ((i // 8) % 2) + (i % 2)
        out[..., i] = p[..., 4 * k + d]
    return out


def layernorm(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    r = F(1.0) / np.sqrt(v + EPS)
    return F(F(F(x - m) * r) * g + b)


def gelu(x):
    return F(F(F(0.5) * x) * F(F(1.0) + np.tanh(F(F(0.7978845608028654) * F(x + F(F(0.044715) * x ** 3))))))


def attention(x, wq, wk, wv, wo):
    q = F(x @ wq.T)
    k = F(x @ wk.T)
    v = F(x @ wv.T)
    s = F(F(q @ k.swapaxes(-1, -2)) * SCALE)
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    a = F(e / e.sum(-1, keepdims=True))
    return F(F(a @ v) @ wo.T)


def block(x, w):
    h = layernorm(x, w["ln1g"], w["ln1b"])
    x = F(x + attention(h, w["wq"], w["wk"], w["wv"], w["wo"]))
    h = layernorm(x, w["ln2g"], w["ln2b"])
    m = F(h @ w["w1"].T)
    m = gelu(m)
    m = F(m @ w["w2"].T)
    return F(x + m)


def forward(weights, tokens):
    """tokens: (B, 16, 20) float32 -> (B, 16, 4) noise."""
    x = F(tokens @ weights["embed"].T)
    for i in range(3):
        x = block(x, {k: v for k, v in weights[f"b{i}"].items()})
    y = layernorm(x, weights["head_ln_g"], weights["head_ln_b"])
    return F(y @ weights["head"].T)


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
    """A shape check with deterministic pseudo-weights until the loader order
    is wired: 64 digits of 20 features in, 64 x 16 x 4 out."""
    rng = np.random.default_rng(0)
    w = {"embed": F(rng.standard_normal((W, FE)) * 0.1),
         "head": F(rng.standard_normal((PX, W)) * 0.1),
         "head_ln_g": np.ones(W, dtype=F), "head_ln_b": np.zeros(W, dtype=F)}
    for i in range(3):
        w[f"b{i}"] = {
            "wq": F(rng.standard_normal((W, W)) * 0.1),
            "wk": F(rng.standard_normal((W, W)) * 0.1),
            "wv": F(rng.standard_normal((W, W)) * 0.1),
            "wo": F(rng.standard_normal((W, W)) * 0.1),
            "w1": F(rng.standard_normal((HI, W)) * 0.1),
            "w2": F(rng.standard_normal((W, HI)) * 0.1),
            "ln1g": np.ones(W, dtype=F), "ln1b": np.zeros(W, dtype=F),
            "ln2g": np.ones(W, dtype=F), "ln2b": np.zeros(W, dtype=F),
        }
    tokens = F(rng.standard_normal((64, TK, FE)))
    out = forward(w, tokens)
    print(f"twin forward: {tokens.shape} -> {out.shape}, finite {np.isfinite(out).all()}, "
          f"rms {float(np.sqrt((out ** 2).mean())):.4f}")
    x = F(rng.standard_normal(64))
    print(f"patch roundtrip exact: {np.array_equal(unpatch(patch(x)), x)}")


if __name__ == "__main__":
    main()
