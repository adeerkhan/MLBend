# Handoff

One file, overwritten at the end of every session. It is the literal memory
context: a fresh agent reads this first, then the issue it names, then
`AGENTS.md`.

**Write it before you stop.** An agent that ends a session without writing this
leaves the next one blind.

## 2026-10-30 — PR #17 merged; the corpus is renamed and the README is real

**Where things stand.** `main` is green, **`make check` 92/92** on the
merged tree. PR #17 landed the structure: thirteen flat examples became four
areas, every path followed, and the local folder is now
`~/github/bend/bend-lab` with `origin` at `adeerkhan/bend-lab`. The README is
rewritten around `docs/assets/bend-hero.png`, the measured three-arm table and the real
layout; `examples/README.md` is a 13-task index instead of a stale second copy.

**The one red, and its fix (ISSUE-019).** `examples/diffusion/2d` could not be
imported: Bend refuses an import segment that starts with a digit, and
`probe_ddim.bend` imports its example. The directory is now
`examples/diffusion/two_moons`; the task id stays `diffusion-2d`. The nine
generated example commands and the result links the move had broken were
regenerated, and the ViT example README now carries the committed 0.787/0.811.

**The plumbing split, and the fix (2026-10-30).** A step was **543 ms**, with
77% in input and target construction and `init_params` at 25.3 s. The builders
appended into a growing list (O(n²)) and the tokeniser scanned each digit with
`at()` 64 times. Cons + one reverse and a linear 16-pixel tokeniser bring a
step to **131 ms** (40% forward, 54% loss and backward, 5% build) and
`init_params` to **6 ms**, with the full row list and the first three losses
bit-identical. `results/PLUMBING.md` and `bench/plumbing/main.bend`.

**ISSUE-020 found and fixed (2026-10-30).** `mix_img` returned 4033 values for
64×64, so the last digit in every batch was its first pixel and 63 zeros. The
fix repeats each image's `(a, b)` pair for its 64 pixels and mixes three flat
lists in step. Retrained 3000 steps: legibility **0.7875** (was 0.7829),
ancestral **0.8110** (was 0.8070), final loss 0.0712 (was 0.0896). The gate
passes at a 1.15e-4 first-loss delta.

**Flow matching (2026-10-30).** `examples/diffusion/flow_matching` runs
rectified flow on the DDPM example's two moons, normalised to [-1, 1]:
3 -> 128 -> 128 -> 2 with SiLU, full batch, 6000 AdamW steps, Euler 32.
Last-100 loss **0.8859** (the probe measured 0.8662), sample mean/std
**+0.049 / -0.048 · 0.610 / 0.675** against the data's 0 / 0 · 0.577 / 0.659,
and **112/128** inside the box. `make flow` rebuilds it and rewrites
`results/flow-matching/`. The frozen probe and its NumPy twin are still owed.

**The measured state (unchanged, 2026-10-29).** The diffusion objective's
M1–M4 are done on measured numbers; M5 (CIFAR) and M6 (UNet) are not.

| | DDIM 16 | ancestral 64 | DDIM MSE | ancestral MSE |
|---|---|---|---|---|
| **Bend-ViT, per-image t, 3000 steps** | **0.787** | **0.811** | 0.0598 | 0.0561 |
| torch-ViT, same architecture | 0.790 | 0.779 | 0.0583 | 0.0633 |
| lucidrains UNet | 0.932 | 0.940 | 0.0202 | 0.0183 |

Cloud mean/std +0.2957 / 0.3395 against the data's 0.3076 / 0.3767. The gate is
`reference/test_diffusion_vit.py` (six checks; first-loss tolerance 1e-3 from a
measured 1.27e-4 delta). Evidence: `results/diffusion-vit/{run.json, REPORT.md,
loss.svg, grid.svg}` (run.json carries the loss `curve` and both samplers'
samples) and `results/compare/{compare.json, index.html}` — torch-vit.json also
carries its curve, so both arms can be plotted from the JSONs alone.

**The finding that closed M1: one timestep per image, not per batch** — worth
**+0.36**. Isolation table, all 3000 steps, same model: one t per batch
0.359/0.344 (deterministic), 0.392/0.441 (random), 0.409 (Weyl); one t per image
0.770/0.778. Implemented as `mix_img` (per-image a/b), `rows_img` (per-image
8-value time embedding) and `t_img`. **The first version of that table was
miswired** (the "random" arm drew t per image) — ISSUE-016 keeps the correction
above the table.

**The GPU lane: it runs; the no-speed result was plumbing-bound**
(ISSUE-018).
`bendtorch@1.1.1.0` + `bendtorch-gpu@1.1.1.0` are in `bend.toml` on branch
**`lane/gpu`**; the ViT there uses eight `LinearNBF` and one `F.Attention`. The
binary links `libnvrtc`, `nvidia-smi` shows the process with **104 MiB on the
card**, the frozen values are **bit-identical** to the CPU lane — and the wall
time was unchanged (5m44 both ways) against the old data path. After the
builder fix a step is 131 ms and 95% of it is forward and backward, so the
card is worth re-measuring.

## What is next, in order

1. **Second migration** — package sources under `packages/`, `reference/` under
   `apparatus/`; `bend.toml`'s members, the intra-package imports,
   `check_all.py`'s package list and the Makefile. One commit, `make check`
   behind it.
2. **Flow matching's frozen gate**: a `probe_flow.bend` at the fixed init and
   a `twin_flow.py` for one forward and one Euler step, the way the ViT gates
   work. Then flowers at batch 32 / 11k steps, dataset pinned by sha256.
3. **`FullAttention`** — the PR we accepted on BendTorch#17: separate def, share
   `qrow`/`prow` with the key length, its own three laws over all t,
   `demo/ops_test.bend` + `reference/test_ops.py` against
   `F.scaled_dot_product_attention`, `Attention`'s bits untouched. Then
   `Im2col` (#15: `Im2col(n,h,w,c,k,s,p)`, gather backward, device bangs).
4. **CIFAR-10** only behind (1), a real step time and a re-measured device lane.
5. **Ledger hygiene**: 22 entries — 12 fixed, **0 open**, 1 upstream, 9 other.
   Collapse resolved-and-reported entries to one line in an archive; ISSUE-014
   and ISSUE-018 keep their corrections above the original text on purpose.

## Open threads upstream

- **bendlang/bend#1894** — closed as a known limit; PR 1921 carries the
  diagnostic (`Refs`), nothing owed.
- **PR 1820** — poll period now 64 (our `6b9c405f`); waiting on the maintainers;
  we cannot re-request review there (no permission).
- **BendTorch#15** — design agreed; `Im2col` offered, after `FullAttention`.
- **BendTorch#17** — our issue; we took the `FullAttention` PR.
- **bend-ml#26** — laws landed on `devel` (`ASet.*`); our **PR #27** (64x64 test
  half) is open; the corollary law is next.
- **BendTorch `make gpu`** — its demo step calls `reference/.venv/bin/python`,
  absent from the checkout (Error 127); a one-line fix we could PR.

## Traps paid for this session

`Nat` fail-stops past 2^48-1 (use `U32` for mixers); a `match` cannot scrutinise
a computed value, and a binder cannot be matched after a later one; `List`s are
affine unless the parameter is `+`; stdout is block-buffered (use `stdbuf -oL`
and `setsid`, as `scripts/vit.sh` does); linking `libnvrtc` is not proof of
execution — `nvidia-smi --query-compute-apps` is; a GPU run is only faster
if the work is matmul and not plumbing; an import path segment must start
with a letter, so a directory named `2d` cannot be imported (`two_moons` can);
and `List.append` into a growing list is O(n²), as is an `at()` scan inside a
loop — cons and reverse once, and walk the list linearly.
