# Handoff

One file, overwritten at the end of every session. It is the literal memory
context: a fresh agent reads this first, then the issue it names, then
`AGENTS.md`.

**Write it before you stop.** An agent that ends a session without writing this
leaves the next one blind.

## 2026-10-28 — the ViT denoiser: the residual, causality, the minibatch

**Where things stand.** `examples/diffusion_vit/main.bend` is the ViT denoiser
(2×2 patches, 16 tokens of 20 features, width 64, one head, three pre-LN
blocks, one 4-value head per patch, 100,736 parameters, every layer
BendTorch's). It trains on 64-digit chunks, and prints `loss:`, `final_loss:`,
`data_mean:`, `sample_mean:`, `sample_std:`, `samples:` (16-step DDIM) and
`samples_ddpm:` (64-step ancestral) from the one trained model and the one
starting cloud. The twin is `reference/twin_vit.py`; the frozen comparison is
`reference/vit_gate.py`; the gate file is `reference/vit_gates.py` (deliberately
not `test_*` until `results/diffusion-vit/run.json` exists — rename it in the
same commit as the results, because `check_all` collects every `test_*.py`).
Everything else is committed on `main`; the tree is clean.

**Four issues, all found by measurement.**

- **ISSUE-010 (fixed, the big one).** `AttBlk` was `Seq(LayerNorm,
  Residual(Att))`, i.e. `ln(x) + att(ln(x))` — the residual on the *normalized*
  value, no identity path. The five-checkpoint ladder (`emb0`, `ln0`, `att0`,
  `blk0`, `body0` in `reference/probe_vit_frozen.bend`) showed staging, embed,
  norm and attention all identical to the twin and `blk0` not; with Bend's own
  wiring the twin reproduces `blk0` to **7.5e-08**. The block is now
  `Residual(x, Seq(ln, f))`, parameter order and count unchanged.
- **ISSUE-011 (upstream, not fixed).** BendTorch's only `Attention` is
  **causal**: `qrow` builds `i+1` scores, `prow` softmaxes exactly those, no
  mask argument. The 16 tokens attend as a sequence; the torch arm runs the
  same mask. Every earlier ViT number in `NOTES.md` (0.809, 0.798, 0.778) was
  bidirectional and is labelled so.
- **ISSUE-012 (fixed).** The ViT inherited the digits MLP's lr 1e-2; torch,
  same architecture: 1e-2 → 0.721, 3e-3 → 0.782, 3e-3/128 → 0.774. The example
  runs `CAdam{0.003, …}`.
- **ISSUE-013 (fixed).** `step_in` took the *first* 64 digits on every step
  (a 16× smaller task than the torch arm's fresh 64 of 1024). The loss said so
  (0.0055 at step 275 against torch's 0.0608 after 3000); the batch is now
  chunk `s mod 16` of sixteen chunks.

**The loss curve is a sawtooth and that is the timestep, not divergence.** The
example draws one t for the whole batch (`t = 1 + s mod 64`), so a loss printed
at low t is ~1.0 by construction and at high t ~0.002. Three runs at lr 1e-3,
2e-3 and 3e-3 spike at the same printed steps — that is the tell. So the
per-step loss is not comparable with the torch arm's (a t per image); the
cloud is.

**Measured, all on the same 1024 optdigits rows, the same starting cloud
`noise_of(2000000, 4096)`, the same 3000 steps, nearest-digit |corr| and
nearest-digit MSE:**

| arm | DDIM 16 | ancestral 64 | DDIM MSE | ancestral MSE |
|---|---|---|---|---|
| torch-ViT (causal, 1 head, lr 3e-3, batch 64) | 0.786 | 0.779 | 0.0602 | 0.0633 |
| lucidrains UNet | 0.932 | 0.940 | 0.0202 | 0.0183 |

The Bend arm's row was still training when this was written; `reference/vit_results.py
<log>` writes `results/diffusion-vit/{run.json,loss.svg,grid.svg,REPORT.md}`,
and `reference/compare_vit.py` gathers all three arms into
`results/compare/{compare.json,index.html}`.

**The twin's gates, measured** (`reference/vit_gate.py`): frozen `eps` maxabs
**1.19e-07**, one ancestral step **4.77e-07**, the DDIM cloud **6.38e-06**,
eta = 0 exactly **0**. Tolerances are set from those numbers (1e-5, 1e-5,
1e-4), on maxabs rather than an element-wise relative gap (many `eps` values
are near zero). `reference/vit_gates.py` turns them into a gate plus the
twin's first loss against the run's and legibility ≥ 0.75.

**PyTorch on a 64-wide model wants few threads:** 21 minutes on 16 threads, 4
on 3. `THREADS` is an env var in both torch arms. The UNet arm needs its
`torch.randn` draw replaced for the two sampling calls to start from the same
cloud (the vendored file is untouched).

**What is running / next, in order.**

1. The M1 run (lr 3e-3, `examples/diffusion_vit/main.bend`, ~100 min on this
   CPU) and two stability variants (lr 1e-3 and 2e-3 with `CAdamW` wd 0.01,
   built from `_lr1e3.bend` / `_lr2e3.bend` copies — **delete those two files**
   once the run is read; they are not part of the example).
2. `reference/vit_results.py /tmp/opencode/vit_m3.txt` → the results card; if
   legibility ≥ 0.75, commit it with the gate rename.
3. `git mv reference/vit_gates.py reference/test_diffusion_vit.py`, run it
   (rebuilds the frozen probe, ~6 min + 5 min run).
4. `reference/compare_vit.py` → `results/compare/`.
5. `make check` (started this session; check `/tmp/opencode/make_check_final.log`).
6. `docs/agents/handoff.md` is this file.

**Blocked, with the evidence recorded.** M5 (CIFAR-10): the installed
`bendtorch@1.0.1.0` has no `torch/gpu.bend`, so `Attention` has no device
mirror; a ViT step is ~1.5 s at 1024 tokens here, and CIFAR is 4× the rows —
10+ hours on the CPU. M6 (the reference UNet in Bend): BendTorch#15
(`Conv2d`/`GroupNorm`) is open with no comments, re-checked today.

**Upstream.** bendlang/bend#1894, nuxyel/BendTorch#15, nuxyel/bend-ml#26 all
open, no comments; bendlang/bend PR #1820 `mergeable: clean`, waiting on the
maintainer.

**DoD (6) is settled and recorded in `NOTES.md`**: our schedule
(`1e-4 + 0.2799·t/T`) is a *different* schedule from the reference's
`linspace(1000/T·1e-4, 1000/T·0.02, T)`, kept and named; the 2d example has no
clamp and needs none (cloud 0.599/0.424 against the data's 0.866/0.494), with
the reason written down.
