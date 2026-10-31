# Rectified flow at 64x64 on Oxford Flowers

A flow model in Bend on BendTorch: noise to a flower in 32 Euler steps, one
row of 64 pixels per token, so the image is read in raster order and nothing
indexes backwards.

```
make doctor                       # what this machine can compile
make flow64 PRESET=medium         # build inside a cgroup, run, report
```

## The data

512 flowers at 64x64, one grayscale byte per pixel, made from the pinned Oxford
Flowers parquet by `reference/flowers_export.py`:

| file | bytes | sha256 |
|---|---|---|
| `flowers64-512-gray.u8` | 2,097,152 | `6b9c4c66…` |
| `flowers256-512-gray.u8` | 33,554,432 | `c58eab33…` |

The 64x64 RGB export the first version read is 12,288 bytes per image; the
tokeniser takes one byte per pixel, so it was reading a scrambled raster until
the grayscale conversion. `papers/TEM/data.py` is where the parquet's hash and
the reduction are recorded.

## The model

`config.bend` fixes the shapes before the compile, because a Bend program's
sizes are type-level constants: images per step, width, resolution, steps and
Euler count. `scripts/flow-config.py` writes it for a named preset, and
`scripts/system-profile.py` says which presets the machine can pay for.

| preset | images/step | width | rows | compile |
|---|---|---|---|---|
| `tiny` | 1 | 64 | 64 | 926 MB, 19 s |
| `medium` | 2 | 128 | 128 | 926 MB, 19 s |
| `full` | 8 | 128 | 512 | 926 MB, 19 s |
| `hi` | 1 | 128 | 256 (256x256) | 940 MB, 18 s |

The compile number does not move with the model, which is the finding recorded
in `docs/NOTES.md`: a `match` on three values cost 4.8 GB of codegen where the
same function matching two costs 6 MB. `scripts/bend-guard.sh` runs every build
in a memory cgroup with swap off, so a runaway compile is killed instead of
taking the desktop with it.

## Measured

All runs: 2000 steps, 32 Euler steps, 28 cores. `legibility` is the best
absolute correlation of a sample with any training flower; random images score
0.048.

| model | attention | final loss | legibility |
|---|---|---|---|
| 1 block, 128 wide | causal | 0.328 | 0.419 |
| 1 block, 128 wide | full | 0.328 | 0.427 |
| 3 blocks, 128 wide | causal | 0.325 | **0.450** |

Two things follow from the table and both were stated before the runs. Full
attention is the right layer for an image (with row-tokens, causal attention
lets row 40 see rows 0..40 only), but at 64 tokens and one block it is worth
about 0.008 legibility for 2.2x the time: the restriction was not what limited
the model. Depth was: three blocks are worth 0.031.

Samples are blurred flowers: the colour and the layout are right, the detail is
not. That is what a 1-3 block model at 64 tokens on 512 images should look like,
and it is not comparable to a paper's FID.

## Not done

- **A held-out split.** Every run above trained on all 512 images, so
  legibility can only measure memory of the batch. The reporter has
  `--holdout N`; the training run on 500 images is the missing half.
- **256x256.** It builds (940 MB) but the builders advance on the literal
  `64n`, so a 256 preset needs a builder that never names the resolution.
- **Full attention in the GPU lane.** `FullAttention` is in `torch/main.bend`;
  the device lane still has the causal-only `Attention`.
