# Issues hit

Running log for the diffusion work, kept as issues are encountered. Every
entry in **Issues** was reproduced and checked before it was written here; the
things still being tested sit in **Hypotheses** until a measurement promotes or
kills them, and the things that looked like issues but were not are in
**Checked and dismissed** so they are not re-investigated.

Each entry says what happened, why it happened, how it showed up, and which
library or component carries the cause. `ours` means MLBend (this repository);
a package name means the published Bend package; `lucidrains` means the
vendored reference in `reference/lucidrains/`.

The reference is `github.com/lucidrains/denoising-diffusion-pytorch` at
`c8155b1dc622e3f1ef0d53dc6e5a570ca105bcec`, copied verbatim (see
`reference/lucidrains/VENDORED.md`). It is run by
`reference/lucidrains_compare.py` on the same 1024 optdigits rows, the same
T = 64 and the same noise-prediction task as the Bend example.

---

## ISSUE-001 — the digits denoiser learned nothing (ours, high) — FIXED

**What.** `examples/diffusion_digits/main.bend` trained a 3-layer SiLU MLP for
3000 full-batch AdamW steps and ended at MSE **1.0258** (NumPy twin
**1.0746**). The target is unit-variance noise, so predicting nothing scores
exactly **1.0**: both arms were at or worse than the mean.

**Why.** `value(i, j, seed)` returned `k*0.1 - 0.2` for `k ∈ 0..6`, i.e. a
uniform init on **[−0.2, +0.5]** with mean **+0.100**. The first layer is 72
wide, so the layer adds `72 × 0.100 × 0.329 ≈ +2.4` to every pre-activation
(measured: z1 mean **+2.36**, rms 2.43); SiLU is roughly linear there
(h1 mean +2.16), and each following layer multiplies that common mode by
`fan_in × 0.100 = 6.4` — z2 **+13.8**, the output **+88.4** rms. Against a
target of rms 1.0 that is the measured first loss, **8062.6895** (the
reference's is 0.963). Adam then had
to undo a 10⁴× scale with lr 0.01, and 3000 steps were not enough — it landed
on the mean predictor and stayed.

**How it showed up.** The loss dropped 8062.7 → ~18 by step 98 and then sat
between 0.99 and 1.06 for the remaining 2900 steps. Downstream, the sampled
cloud was meaningless: `sample_mean` 1.2e17, `sample_std` 1.075e19, 0 of 4096
pixels inside [0,1].

**Component.** Ours: `examples/diffusion_digits/main.bend`, the weight
initialiser. Not BendTorch, not the schedule — the NumPy twin plateaued at the
same 1.0.

**Verified by.** Three independent measurements:
1. The failure itself: `results/diffusion-digits/run.json`, and the twin's
   `numpy_answer()`.
2. A controlled ablation of our exact MLP, optimizer and data
   (`reference/digits_ablation.py`), one factor at a time — the init was the
   only factor that mattered:

   | init | range | schedule | loss@0 | loss@300 | last 100 |
   |---|---|---|---|---|---|
   | bend `[-0.2, 0.5]` | [0,1] | bend | 3480.03 | 62.20 | **1.3766** |
   | torch zero-mean | [0,1] | bend | 1.0138 | 0.3211 | **0.1455** |
   | torch zero-mean | [−1,1] | bend | 1.0133 | 0.3637 | 0.1755 |
   | torch zero-mean | [−1,1] | ref | 1.0134 | 0.3659 | 0.1787 |
   | bend `[-0.2, 0.5]` | [−1,1] | ref | 1297.93 | 29.23 | **1.1827** |

   (3000 steps, batch 128, lr 1e-3, identical data and seeds.)
3. The fix in the real example: first loss **1.0039** (the twin reads 1.003894329, delta 2.9e-08), final **0.0417** — see
   ISSUE-005 for the sampler that then had to be fixed too.

**Fix.** `value()` is now a 97-value hash on [−1, 1) with zero mean, scaled by
`1/sqrt(fan_in)` at the row builder — PyTorch's `nn.Linear` init made
deterministic, which is what the reference uses. The NumPy twin mirrors it.

---

## ISSUE-002 — the two "linear schedules" are not the same schedule (ours, medium) — OPEN

**What.** `mlbend-generative`'s `lin_beta_at(t, 64)` is
`1e-4 + 0.2799·t/64`: β = **0.0044734 … 0.28**, ᾱ₆₄ = **4.085385e-05**.
The reference's `linear_beta_schedule(64)` is
`linspace(1000/T·1e-4, 1000/T·0.02, T)`: β = **0.0015625 … 0.3125**,
ᾱ₆₄ = **1.223548e-05**. Worst per-step relative gap: **65.07%**.

**Why.** Two conventions both called "the linear schedule": DDPM's 1000-step
schedule rescaled by `1000/T`, and a directly parameterised 0 → 0.28 ramp.
Neither is wrong; they are not interchangeable.

**How it showed up.** Only when the two were printed side by side
(`reference/lucidrains_compare.py`, "the schedules"). No gate could see it,
because each arm was compared with its own mirror.

**Component.** Ours (`mlbend-generative`); the reference defines the other
convention.

**Verified by.** The script above, and the library's own
`diffusion.alphas_cumprod[-1]` printed in the same run.

**Status.** Open. The ablation says it does not explain ISSUE-001 (0.1755 vs
0.1455), so this is a fidelity decision, not a bug fix: adopt the reference's
rescale, or document ours as a different schedule.

---

## ISSUE-003 — the digits grid draws each cell 1.7× too wide (ours, medium) — FIXED

**What.** `grid_svg` in `reference/test_diffusion_digits.py` computes a pixel's
x as `ox + c*cell//8*8//8 + c*(cell/8)`. The stray first term is
`floor(1.75·c)`, so with `cell = 14` the last sub-column starts **24.25 px**
into a **14 px** cell. The y offset has no such term.

**Why.** A leftover from an earlier layout; it duplicates part of the intended
`c*(cell/8)`.

**How it showed up.** Reading the generated SVG while checking the digits
plot: the data cells smear into their neighbours.

**Component.** Ours (`reference/test_diffusion_digits.py`).

**Verified by.** The code, and a re-render keeping only `c*(cell/8)`
(`reference/lucidrains_compare.py:grid_svg`), which aligns.

**Status.** Fixed: the offset is `ox + c*(cell/8)`; the committed `grid.svg`
and the example's `plot.svg` were regenerated with it.

---

## ISSUE-004 — the vendored package does not expose its version (lucidrains, low) — WORKED AROUND

**What.** `version.py` defines `__version__`; `__init__.py` does not import
it, so `d.__version__` raises `AttributeError`.

**Why.** `__init__` re-exports classes from every submodule but not the
version string.

**How it showed up.** The first import smoke test tried to print it and failed.

**Component.** `lucidrains/denoising-diffusion-pytorch` (vendored, unmodified).

**Verified by.** The `AttributeError`, and `version.py` containing
`__version__` while `__init__.py` never mentions it.

**Status.** Not patched (the copy stays verbatim). Provenance is the commit in
`VENDORED.md`, which is also what `results/lucidrains-diffusion/run.json`
records.

---

## ISSUE-005 — the sampler has no x̂₀ clamp, so the cloud explodes (ours, high) — FIXED

**What.** After ISSUE-001 the digits model learned (final loss 0.0417, better
than the reference's 0.078) but its sampled cloud was **std 35.8** with mean
−1.39, against the data's mean 0.3076 / std 0.3767 — and 0 of 4096 pixels
inside [0,1].

**Why.** The reverse step formed its mean directly from the raw ε̂. The
vendored reference reconstructs the start and **clamps it to the data range**
before forming the posterior mean (`p_mean_variance` with `clip_denoised=True`,
and `ddim_sample` with `clip_x_start=True`, re-deriving ε̂ from the clamped
start). Without that clamp one bad prediction compounds over 64 steps; with it
the sample cannot leave the data range.

**How it showed up.** The training loss was healthy while the samples were not,
which is what pointed away from the trainer and at the sampler.

**Component.** Ours: `examples/diffusion_digits/main.bend` (and its twin).

**Verified by.** A controlled probe with **one** trained MLP, sampling it four
ways (`reference/digits_sample_probe.py`, same weights, same schedule, same
seeds):

| sampler | clamp | mean | std | in [0,1] |
|---|---|---|---|---|
| DDPM | no | −2.5074 | **32.7934** | 123/4096 |
| DDPM | yes | +0.2960 | **0.3005** | 4096/4096 |
| DDIM | no | −0.7978 | **17.7724** | 194/4096 |
| DDIM | yes | +0.3899 | **0.3614** | 4096/4096 |

(data: mean 0.3076, std 0.3767)

**Fix.** Both steps now reconstruct `x0 = (x − √(1−ᾱ)·ε̂)/√ᾱ`, clamp it to
[0,1], and form the mean from it; DDIM re-derives ε̂ from the clamped start
(the reference's `rederive_pred_noise`). The twin mirrors both.

---

## ISSUE-006 — both NumPy twins took the SiLU derivative at the wrong point (ours, high) — FIXED

**What.** The two NumPy twins (`reference/test_diffusion_digits.py`,
`reference/test_diffusion.py`) computed the SiLU backward as
`sigmoid(h) * (1 + h*(1 - sigmoid(h)))` where `h` is the **activated** output.
BendTorch's `silu_d(dy, x)` takes `x` = the **pre-activation**; the derivative
is `s(z)*(1 + z*(1-s(z)))`, `s = sigmoid(z)`. The twins kept only `h = silu(z)`,
so their gradient was wrong wherever `z ≠ silu(z)`.

**Why.** The forward was written first (`h = silu(z)`), and the backward was
then written against `h`, which is the value in scope — the pre-activation was
never stored.

**How it showed up.** It could not show up until ISSUE-001 was fixed: while both
arms sat at the mean predictor the gradients are ~0, so a wrong gradient and a
right one agree. After the init fix, **from an identical first loss**
(1.0038943 vs 1.003894329, delta 6e-9) Bend reached **0.044** in 3000 steps
and the twin stayed at **1.031**; its sampled cloud (0.5018/0.3982) only
looked sane because of the ISSUE-005 clamp.

**Component.** Ours: the two twins' training loops. Not Bend, not BendTorch —
Bend's layer was right, which is exactly why the arms diverged.

**Verified by.** The identical first loss with divergent finals (above); the
same bug in both twins, found by reading the backward after the divergence;
and BendTorch's own `silu_d` in `.bend/env/.../main.bend`, which takes the
pre-activation as `x`.

**Fix.** Both twins now keep `z1`, `z2` and evaluate the derivative there.
The gated first loss is unchanged (one step, no update), which is why this was
invisible to `make check`.



- **H3 — the data range ([0,1] vs [−1,1]) is the cause.** Dismissed by the
  ablation: reference init on [0,1] scored 0.1455, on [−1,1] 0.1755 — the
  same order, and worse centred. Kept as a fidelity option only.
- **H4 — the time embedding as 8 concatenated input features is the cause.**
  Not supported: the ablation kept that layout and learned fine once the init
  was fixed. The reference injects the embedding as per-block scale/shift; a
  fidelity question for the model, not the cause of the failure.
- **Nondeterminism of Bend training.** Two runs of the same binary (before
  these fixes) were bit-identical: final loss 1.0258427, same sample line. The
  old report card's different numbers came from a pre-`2f2631b` code state.

## ISSUE-007 — `bend file.bend` overflows its stack on a 32k-element list (bendlang/bend, high) — UPSTREAM

**What.** The plain run path dies with `bend: memory fault (machine stack
overflow?)`, exit 1, no source location, on a twelve-line program:

```bend
import Base
import mlbend-generative@0.1.1.0/main.bend as G

def main() -> IO(Unit):
  xs = G.noise_of(7n, 32768n)
  do IO<Unit>:
    IO.print(Nat.show(List.length(&2, F32, xs)))
    return Unit{}
```

**Where the line is.** 4096, 8192 and **16384** run and print their count;
**32768** and 65536 crash. Same on Bend **2.0.38 and 2.0.39**. The same file
built with `bend … -o prog` and then run prints 65536 on both versions.

**Why.** The run path's stack is fixed, and each list element is built by a
Box–Muller call whose frame is large enough that ~32k of them do not fit; the
native build has no such limit (the guide: the compiled model has no C stack;
each def is a segment of a flat state machine). It is frame size, not depth
alone: a 65536-element **Nat** list summed by recursion on the same path is
fine.

**How it showed up.** Running `examples/diffusion_digits/main.bend` the
documented way. `--check-only` passes and `-o` runs the full 3000-step
training, so it looked like a fault in the program; it is not — it is the run
path, and the message names no file and no line.

**Component.** `bendlang/bend`, the `bend <file.bend>` (check, then run) path.

**Verified by.** The program above on both versions, the 16384/32768
threshold, the `-o` contrast at 65536, and the Nat-list contrast. Filed as
[bendlang/bend#1894](https://github.com/bendlang/bend/issues/1894).

---

## Checked and dismissed

- **`accelerate` "no matching distribution" on Python 3.14.** pip said
  `from versions: none`, which looked like a hard incompatibility. PyPI's
  metadata says `requires_python >=3.10.0`, and a retry installed
  `accelerate 1.15.0` on 3.14 unchanged — the failure was a transient index
  interruption, not the library.
- **The forward dot product's four partial sums (7 ulps from NumPy).** I
  measured Bend's `eps` 5.34e-05 (7 ulps at magnitude 95) from the twin's,
  traced it to `dotr` in `bend-ml-tensor-array`, and intended to file it. On
  inspection it is not a defect: the four accumulators are documented at the
  top of `dotr` as the design, and `Mat.matmul_precise` — the "precise product
  for long dots" — only re-blocks sums **longer than 1024**, so a 72-term dot
  is four-accumulator in both variants. A few ulps is what any two summation
  orders give; it is a measurement to set tolerances with (it is why the
  sampler gate shares the model output), not a bug. Not filed.

---

## ISSUE-008 — no convolution and no group norm, so no UNet (BendTorch, high) — UPSTREAM

**What.** BendTorch exports no `Conv2d`, no `GroupNorm` and no `BatchNorm`
(constructor list in the issue). A UNet — Conv2d + GroupNorm + attention, the
reference's denoiser — cannot be expressed, so image diffusion has no natural
model.

**Why it matters, measured.** Same data, task, schedule and 3000 steps; the
samples are scored by the best absolute correlation with any training digit
(random images: 0.326):

| denoiser | params | nearest-digit \|corr\| |
|---|---|---|
| the reference's UNet | 825k | **0.926** (min 0.813) |
| our dense 72→64→64→64 MLP | 13k | **0.408** (min 0.239) |

The MLP has the *lower* training loss (0.042 against 0.078) and still
generates mush: dense layers over a flattened digit carry no spatial
structure. That is the gap.

**Component.** `BendTorch` (layers). Not a defect — an absent capability.

**Status.** Filed as [nuxyel/BendTorch#15](https://github.com/nuxyel/BendTorch/issues/15).
Interim route: a ViT-style denoiser from `Attention` + `LayerNorm`, which do
exist; `Conv2d` later. **Measured** (`reference/digits_vit_probe.py`): 16
tokens of 2×2 patches, dim 64, 4 heads, 3 blocks, 103k parameters, 3000 steps
— nearest-digit `|corr|` **0.798** (min 0.676), cloud **0.260/0.299**,
4096/4096 in [0,1]. The route works; the port is
[adeerkhan/MLBend#16](https://github.com/adeerkhan/MLBend/issues/16).
