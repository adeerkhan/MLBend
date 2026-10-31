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
| 3 blocks, 128 wide | full | **0.316** | 0.418 |

Full attention is the right layer for an image: with row-tokens, causal
attention lets row 40 see rows 0..40 only. At one block it is worth 0.008
legibility for 2.2x the time, so the restriction was not what limited the
model; depth was, at 0.031. At three blocks full attention gives the best loss
of the four and the lowest legibility of the pair, the same inversion the
2-image and 8-image runs showed. A training-set legibility proxy cannot rank
models, which is what the holdout below is for.

Samples are blurred flowers: the colour and the layout are right, the detail is
not. That is what a 1-3 block model at 64 tokens on 512 images should look like,
and it is not comparable to a paper's FID.

## What it learned, and what it memorised

The 500/12 run (three blocks, causal, 2000 steps) separates the two:

```
legibility  trained 0.529    held out 0.227    baseline 0.032    gap +0.301
```

The samples reach 0.227 against unseen flowers, seven times the random baseline,
so something transfers. They reach 0.529 against the training flowers, so more
than half of that resemblance is memory. **Every number in the table above is a
training-set number and therefore an upper bound**; 0.227 is the honest figure
for images the model has not seen.

That makes 512 images the binding constraint rather than the step count: more
steps on the same corpus would raise the first number and leave the second
roughly where it is.

## Not done

- **A per-class held-out split.** Every run above trained on all 512 images, so
  legibility measures memory of the batch. The parquet is grouped by class (the
  first 512 are classes 0..11), so "the last 12" is one class, not a random
  sample; the 500/12 split measures unseen images of a seen class and is
  recorded as such. A proper split takes one image per class and needs a
  re-export.
- **256x256.** It builds (940 MB) but the builders advance on the literal
  `64n`, so a 256 preset needs a builder that never names the resolution.
- **Full attention in the GPU lane.** `FullAttention` is in `torch/main.bend`;
  the device lane still has the causal-only `Attention`.
