# diffusion-vit — a transformer denoiser for 8×8 digits

The same task as `examples/diffusion_digits` (noise prediction over the first
1024 optdigits rows, T = 64, the linear schedule of `mlbend-generative`, 16-step
DDIM with the clamped start), but the denoiser is a transformer instead of a
dense MLP.

    bendenv run -- bend examples/diffusion_vit/main.bend
    ./scripts/vit.sh m1            # build + train in its own session, with a log
    ./scripts/vit.sh m1-results    # -> results/diffusion-vit/

## The model

| | |
|---|---|
| patch | 2×2, so 16 tokens per 8×8 image |
| features | 20 per token: 4 pixels, 8 of time (`t/64 × {0.001, 0.01, 0.1, 1}` sin/cos), 8 of position |
| width | 64, one head, three blocks |
| block | `x + att(ln(x))`, then `x + mlp(ln(x))` — pre-LN, both halves |
| head | one 4-value linear per patch |
| parameters | 100,736 |
| layers | BendTorch's only: `LinearNB`, `LayerNorm`, `Attention`, `GELU`, `Residual`, `Seq` |

Two facts shaped it, both in `issues-hit.md`:

- **The residual must wrap the norm** (`ISSUE-010`). Written
  `Seq(ln, Residual(att))` it computes `ln(x) + att(ln(x))`, which discards the
  residual stream at every half-block; the model then trains to a plateau and
  the NumPy twin disagreed with Bend at O(1) until this was found.
- **BendTorch's `Attention` is causal** (`ISSUE-011`): query `i` reads keys
  `0..i` and the layer takes no mask. The 16 tokens are therefore attended as a
  sequence — token 0 sees only itself, token 15 sees all — and the PyTorch arm
  of the comparison runs the same mask to be the same architecture.

Two more, about training: the ViT needs lr 3e-3 rather than the MLP's 1e-2
(`ISSUE-012`), and the minibatch must rotate through the sixteen 64-digit
chunks rather than being the same 64 digits every step (`ISSUE-013`).

## What it measures

`run()` trains, then samples **both** ways from the one model on the one
starting cloud (`samples:` the 16-step DDIM, `samples_ddpm:` the 64-step
ancestral), and prints `data_mean`, `sample_mean`, `sample_std`. The legibility
statistic is the mean best absolute correlation of a sample with any training
digit (random images 0.326, the digits MLP 0.408).

The three arms, same data, same starting noise, same sampler step counts:

| arm | DDIM 16 | ancestral 64 | DDIM MSE | ancestral MSE |
|---|---|---|---|---|
| torch-ViT, causal, 1 head, 3000 steps | 0.786 | 0.779 | 0.0602 | 0.0633 |
| lucidrains UNet, 3000 steps | 0.932 | 0.940 | 0.0202 | 0.0183 |
| **Bend-ViT, 1200 steps** (in `results/diffusion-vit/`) | **0.341** | **0.342** | 0.210 | 0.199 |

The Bend row is a **1200-step run, a quarter short of the comparison's 3000**,
and it has not converged: 0.341 sits just above random (0.326). The pipeline is
complete end to end — loss curve, both clouds, statistics, nearest-digit MSE —
and the number is what it is. The 3000-step run is `./scripts/vit.sh m1`
(STEPS=3000); when it lands, `m1-results` rewrites the card and
`./scripts/vit.sh compare` rewrites the page.

![loss](../results/diffusion-vit/loss.svg)

![samples](../results/diffusion-vit/grid.svg)
