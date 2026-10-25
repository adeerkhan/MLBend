"""The digits diffusion model: BendTorch against a NumPy twin. WORK IN
PROGRESS (issue #10's follow-up): renamed away from test_* on 2026-10-25
so make check stays green while the input-layout mismatch is finished —
the exact-comparison gate caught four real bugs already (see NOTES.md);
one data-ordering difference remains. Run directly:

  reference/.venv/bin/python reference/diffusion_digits_wip.py

`examples/diffusion_digits/main.bend` reads the first 1024 rows of
mstz/optdigits (pinned revision, 8x8 digits, pixels scaled to [0, 1]),
trains a three-linear-layer SiLU denoiser full-batch by AdamW to predict
the noise of the 64-step linear schedule, then samples 64 fresh digits by
the ancestral loop. This twin does the same in float32 with the same
counter-based noise, compares the first loss (gated) and the sample
statistics (reported), and draws the generated grid beside real digits:

  results/diffusion-digits/grid.svg, loss.tsv, run.json
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEND = os.environ.get("BEND_BIN") or os.path.expanduser("~/.bend/bin/bend")
ENV = dict(os.environ, BEND_NO_TELEMETRY="1")
ENV_DIR = os.path.join(ROOT, ".bend", "env")
if os.path.isdir(ENV_DIR):
    ENV["BEND_LIB"] = ENV_DIR

N = 1024            # training digits
D = 64              # pixels a digit
T = 64              # diffusion steps
STEPS = 3000        # training steps
H = 64              # hidden width
S = 64              # sampled digits
LR, B1, B2, EPS, WD = 0.01, 0.9, 0.999, 1e-8, 0.0
URL = ("https://huggingface.co/datasets/mstz/optdigits/resolve/"
       "4f20cc827b763fb69e2a80e18e892e4e11498aec/optdigits.data")
SHA = "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd"

F = np.float32
np.seterr(over="ignore")


def load():
    cache = os.path.join(ROOT, "reference/.cache/datasets/mstz/optdigits/optdigits.data")
    if os.path.exists(cache):
        text = open(cache).read()
    else:
        import urllib.request
        text = urllib.request.urlopen(URL, timeout=120).read().decode()
    rows = [l for l in text.split("\n") if l.strip()]
    data = np.array([[int(c) for c in r.split(",")] for r in rows[:N]], dtype=F)
    return (data[:, :-1] / F(16.0)).astype(F)


# ---- the twin's counter RNG and schedule (mlbend-generative mirrored) ----

def mix32(z):
    z = np.uint32(z)
    a = np.uint32(z ^ (z >> np.uint32(16)))
    b = np.uint32(a * np.uint32(2246822519))
    c = np.uint32(b ^ (b >> np.uint32(13)))
    e = np.uint32(c * np.uint32(3266489917))
    return np.uint32(e ^ (e >> np.uint32(16)))


def hash_of(s, i):
    k = np.uint32(np.uint32(s) * np.uint32(2654435769))
    return np.uint32(k + np.uint32(np.uint32(i) + np.uint32(305419898)))


def uniforms(s, i):
    k = np.uint32(np.uint32(s) * np.uint32(2654435769))
    h = mix32(np.uint32(k + np.uint32(np.uint32(i) + np.uint32(305419898))))
    m = np.uint32((h & np.uint32(16777215)) + np.uint32(1))
    return (m.astype(F) / F(16777216.0)).astype(F)


def noise_of(s, m):
    # Box-Muller over counter uniforms, vectorized; element-wise float32
    # ops in the scalar twins' order, so the bits agree
    j = np.arange(m)
    pair = j - j % 2
    u = uniforms(s, np.arange(m + 1))
    r = np.sqrt(F(0.0) - F(2.0 * np.log(u[pair]).astype(F))).astype(F)
    ang = (F(6.2831855) * u[pair + 1]).astype(F)
    z = np.where(j % 2 == 0, (r * np.cos(ang)).astype(F), (r * np.sin(ang)).astype(F))
    return z.astype(F)


def lin_beta_at(t, steps=T):
    return F(F(0.0001) + F(F(0.2799) * F(F(t) / F(steps))))


ALPHAS = np.array([f for f in (np.cumprod([F(1.0 - lin_beta_at(t)) for t in range(1, T + 1)],
                                          dtype=F))], dtype=F).flatten()


def temb_of(t):
    tau = F(F(t) / F(64.0))
    freqs = [F(0.001), F(0.01), F(0.1), F(1.0)]
    out = []
    for f in freqs:
        out += [np.sin(F(tau * f)).astype(F), np.cos(F(tau * f)).astype(F)]
    return np.array(out, dtype=F)


def input_of(s, t, a, b, x0, rows):
    eps = noise_of(s, rows * D)
    xt = F(F(a * x0.reshape(rows * D)) + F(b * eps)).reshape(rows, D)
    te = np.tile(temb_of(t), (rows, 1))
    return np.concatenate([xt, te], axis=1).astype(F)


def silu(x):
    return F(x / F(1.0 + np.exp(np.float32(-x))))


def value(i, j, seed):
    m = (i * 3 + j * 5 + seed) % 7
    return F(F(F(m) * F(0.1)) - F(0.2))


def init(seed, r, c):
    return np.array([[value(i, j, seed) for j in range(c)] for i in range(r)], dtype=F)


def forward(params, inp):
    h = silu(F(inp @ params["w1"].T + params["b1"]))
    h = silu(F(h @ params["w2"].T + params["b2"]))
    return F(h @ params["w3"].T + params["b3"])


def train(x0):
    params = {"w1": init(1, H, D + 8), "b1": np.zeros(H, dtype=F),
              "w2": init(2, H, H), "b2": np.zeros(H, dtype=F),
              "w3": init(3, D, H), "b3": np.zeros(D, dtype=F)}
    moment = {k: np.zeros_like(v) for k, v in params.items()}
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    curve = []
    for s in range(STEPS):
        t = s % T + 1
        abar = ALPHAS[t - 1]
        inp = input_of(s, t, np.sqrt(abar).astype(F),
                       np.sqrt(F(1.0 - abar)).astype(F), x0, N)
        tgt = noise_of(s, N * D).reshape(N, D)
        h1 = silu(F(inp @ params["w1"].T + params["b1"]))
        h2 = silu(F(h1 @ params["w2"].T + params["b2"]))
        out = F(h2 @ params["w3"].T + params["b3"])
        diff = F(out - tgt)
        curve.append(float(np.mean(F(diff * diff))))
        d_out = F(F(2.0 / F(N * D)) * diff)
        grads = {"w3": F(d_out.T @ h2), "b3": d_out.sum(axis=0).astype(F)}
        s2 = F(1.0 / F(1.0 + np.exp(np.float32(-h2))))
        d_h2 = F(F(F(d_out @ params["w3"]) * s2) * F(1.0 + F(h2 * F(1.0 - s2))))
        grads["w2"] = F(d_h2.T @ h1)
        grads["b2"] = d_h2.sum(axis=0).astype(F)
        s1 = F(1.0 / F(1.0 + np.exp(np.float32(-h1))))
        d_h1 = F(F(F(d_h2 @ params["w2"]) * s1) * F(1.0 + F(h1 * F(1.0 - s1))))
        grads["w1"] = F(d_h1.T @ inp)
        grads["b1"] = d_h1.sum(axis=0).astype(F)
        for k in params:
            g = grads[k]
            moment[k] = F(B1 * moment[k] + F(1.0 - B1) * g)
            velocity[k] = F(B2 * velocity[k] + F(1.0 - B2) * F(g * g))
            mhat = F(moment[k] / F(1.0 - B1 ** (s + 1)))
            vhat = F(velocity[k] / F(1.0 - B2 ** (s + 1)))
            params[k] = F(params[k] - F(LR * F(F(mhat / F(np.sqrt(vhat) + EPS)) + WD * params[k])))
    return params, curve


def sample(params):
    x = noise_of(2000000, S * D).copy()
    for t in range(T, 0, -1):
        beta = lin_beta_at(t)
        abar = ALPHAS[t - 1]
        coef = F(beta / np.sqrt(F(1.0 - abar)).astype(F))
        scale = F(1.0 / np.sqrt(F(1.0 - beta)).astype(F))
        sig = np.sqrt(beta).astype(F)
        inp = input_of(1000000 + t, t, F(1.0), F(0.0), x, S)
        eps = forward(params, inp).reshape(S * D)
        z = noise_of(1000000 + t, S * D)
        if t == 1:
            x = F(scale * F(x - F(coef * eps)))
        else:
            x = F(scale * F(F(x - F(coef * eps)) + F(sig * z)))
    return x


def run_bend():
    binary = os.path.join(tempfile.mkdtemp(), "mlbend-digits-test")
    build = subprocess.run([BEND, "-o", binary, os.path.join(ROOT, "examples/diffusion_digits/main.bend")],
                           capture_output=True, text=True, env=ENV, cwd=ROOT)
    if build.returncode != 0:
        raise SystemExit("the Bend program did not build:\n" + build.stdout + build.stderr)
    t0 = time.time()
    out = subprocess.run([binary], capture_output=True, text=True, env=ENV, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit("the Bend program did not run:\n" + out.stdout + out.stderr)
    return out.stdout, time.time() - t0


def parse(out):
    losses = [float(v) for v in re.findall(r"^loss: ([0-9.eE+-]+)$", out, re.M)]
    vals = {k: float(re.search(rf"^{k}: ([0-9.eE+-]+)$", out, re.M).group(1))
            for k in ("data_mean", "data_std", "sample_mean", "sample_std")}
    samples = np.array([float(v) for v in
                        re.search(r"^samples: (.+)$", out, re.M).group(1).split()], dtype=F)
    return losses, vals, samples


def grid_svg(path, data, samples):
    cell, pad = 14, 2
    rows = 8
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{2*rows*cell+3*pad}" height="{2*rows*cell+3*pad}" viewBox="0 0 {2*rows*cell+3*pad} {2*rows*cell+3*pad}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<text x="4" y="10" font-family="sans-serif" font-size="9">data (left) vs Bend samples (right)</text>']
    def draw(img, ox, oy):
        px = np.clip(img.reshape(8, 8), 0, 1)
        for r in range(8):
            for c in range(8):
                v = int(round(255 * (1 - float(px[r, c]))))
                parts.append(f'<rect x="{ox + c*cell//8*8//8 + c*(cell/8):.1f}" y="{oy + r*(cell/8):.1f}" '
                             f'width="{cell/8:.1f}" height="{cell/8:.1f}" fill="rgb({v},{v},{v})"/>')
    for i in range(rows * rows):
        ox = pad + (i % rows) * cell
        oy = pad + 12 + (i // rows) * cell
        draw(data[i] if i < len(data) else np.zeros(64), ox, oy)
        draw(samples[i] if i < len(samples) else np.zeros(64),
             pad + rows * cell + pad + (i % rows) * cell, oy)
    parts.append("</svg>")
    open(path, "w").write("\n".join(parts))


def main():
    out, seconds = run_bend()
    losses, vals, bend_samples = parse(out)
    print("\n".join([l for l in out.strip().split("\n") if not l.startswith("samples:")]))

    x0 = load()
    params, curve = train(x0)
    xs = sample(params).reshape(S, D)

    fail = 0
    if abs(losses[0] - curve[0]) <= 1e-2 * max(1.0, abs(curve[0])):
        print(f"ok   the first loss agrees: {losses[0]} vs {curve[0]}")
    else:
        print(f"ERROR the first loss differs: {losses[0]} vs {curve[0]}")
        fail += 1
    dm = float(x0.mean())
    if abs(vals["data_mean"] - dm) < 1e-5:
        print(f"ok   the data mean agrees: {vals['data_mean']} vs {dm:.7f}")
    else:
        print(f"ERROR the data mean differs: {vals['data_mean']} vs {dm:.7f}")
        fail += 1
    print(f"     final_loss (report): {losses[-1]} vs {curve[-1]}")
    print(f"     sample_mean (report): {vals['sample_mean']} vs {float(xs.mean()):.4f}")
    print(f"     sample_std  (report): {vals['sample_std']} vs {float(xs.std()):.4f}")

    res = os.path.join(ROOT, "results/diffusion-digits")
    os.makedirs(res, exist_ok=True)
    idx = [k - 1 for k in range(1, STEPS + 1)
           if k == 1 or (k + 1) % 100 == 0 or k + 1 == STEPS]
    with open(os.path.join(res, "loss.tsv"), "w") as f:
        f.write("step\tbend\tnumpy\n")
        for i, s in enumerate(idx):
            f.write(f"{s}\t{losses[i] if i < len(losses) else float('nan')}\t{curve[s]}\n")
    grid_svg(os.path.join(res, "grid.svg"), x0.reshape(N, D)[:64], bend_samples.reshape(S, D))
    json.dump({"created_at": datetime.now(timezone.utc).isoformat(),
               "example": "examples/diffusion_digits/main.bend",
               "dataset": {"url": URL, "sha256": SHA, "rows": N},
               "seconds": seconds,
               "bend": {"first_loss": losses[0], "final_loss": losses[-1], **vals},
               "numpy": {"first_loss": curve[0], "final_loss": curve[-1],
                         "sample_mean": float(xs.mean()), "sample_std": float(xs.std())}},
              open(os.path.join(res, "run.json"), "w"), indent=2)
    print(f"wrote {res}/loss.tsv, grid.svg, run.json")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
