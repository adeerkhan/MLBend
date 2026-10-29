# Handoff

One file, overwritten at the end of every session. It is the literal memory
context: a fresh agent reads this first, then the issue it names, then
`AGENTS.md`.

**Write it before you stop.** An agent that ends a session without writing this
leaves the next one blind.

## 2026-10-30 — PR #17 merged; the corpus is renamed and the README is real

**Where things stand.** `main` is `22259d7`, **`make check` 92/92** on the
merged tree. PR #17 landed the structure: thirteen flat examples became four
areas, every path followed, and the local folder is now
`~/github/bend/bend-lab` with `origin` at `adeerkhan/bend-lab`. The README is
rewritten around `bend-hero.png`, the measured three-arm table and the real
layout; `examples/README.md` is a 13-task index instead of a stale second copy.

**The one red, and its fix (ISSUE-019).** `examples/diffusion/2d` could not be
imported: Bend refuses an import segment that starts with a digit, and
`probe_ddim.bend` imports its example. The directory is now
`examples/diffusion/two_moons`; the task id stays `diffusion-2d`. The nine
generated example commands and the result links the move had broken were
regenerated, and the ViT example README now carries the committed 0.783/0.807.

**The measured state (unchanged, 2026-10-29).** The diffusion objective's
M1–M4 are done on measured numbers; M5 (CIFAR) and M6 (UNet) are not.

| | DDIM 16 | ancestral 64 | DDIM MSE | ancestral MSE |
|---|---|---|---|---|
| **Bend-ViT, per-image t, 3000 steps** | **0.783** | **0.807** | 0.0618 | 0.0569 |
| torch-ViT, same architecture | 0.790 | 0.779 | 0.0583 | 0.0633 |
| lucidrains UNet | 0.932 | 0.940 | 0.0202 | 0.0183 |

Cloud mean/std +0.3007 / 0.3450 against the data's 0.3076 / 0.3767. The gate is
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

**The GPU lane: it runs, and it does not make this faster** (ISSUE-018).
`bendtorch@1.1.1.0` + `bendtorch-gpu@1.1.1.0` are in `bend.toml` on branch
**`lane/gpu`**; the ViT there uses eight `LinearNBF` and one `F.Attention`. The
binary links `libnvrtc`, `nvidia-smi` shows the process with **104 MiB on the
card**, the frozen values are **bit-identical** to the CPU lane — and the wall
time is unchanged (5m44 both ways). The time goes to **plumbing**: seven
`init_params()`, 1024 rows built per forward, and the list conversions.

## What is next, in order

1. **Second migration** — package sources under `packages/`, `reference/` under
   `apparatus/`; `bend.toml`'s members, the intra-package imports,
   `check_all.py`'s package list and the Makefile. One commit, `make check`
   behind it.
2. **Split a training step's seconds** — `init_params`, one `rows_go` build, one
   `Mat.from_list`, one forward — before any CIFAR decision. If plumbing
   dominates, the fix is to build rows once per shape and reuse; that is also
   what would make a bigger model affordable. Highest-value measurement left.
3. **`examples/flow_matching/main.bend`**: 2D first, on the lane (`F.LinearNBF`,
   build gate `ldd | grep libnvrtc`), against `reference/flow_probe.py`'s
   targets (loss 2.1253 -> 0.8662; samples mean [-0.111, 0.150] / std
   [0.546, 0.745] against the data's [-0.002, -0.000] / [0.583, 0.741]; 465/512
   in the box). Then flowers at batch 32 / 11k steps, dataset pinned by sha256.
4. **`FullAttention`** — the PR we accepted on BendTorch#17: separate def, share
   `qrow`/`prow` with the key length, its own three laws over all t,
   `demo/ops_test.bend` + `reference/test_ops.py` against
   `F.scaled_dot_product_attention`, `Attention`'s bits untouched. Then
   `Im2col` (#15: `Im2col(n,h,w,c,k,s,p)`, gather backward, device bangs).
5. **CIFAR-10** only behind (1) and a real step time.
6. **Ledger hygiene**: 21 entries — 11 fixed, **0 open**, 1 upstream, 9 other.
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
if the work is matmul and not plumbing; and an import path segment must start
with a letter, so a directory named `2d` cannot be imported (`two_moons` can).
