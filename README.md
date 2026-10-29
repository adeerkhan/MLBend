<p align="center">
  <img src="bend-hero.png" alt="bend-lab: machine learning, built in Bend" width="100%">
</p>

# bend-lab

**Machine learning, built in Bend.** I write models from scratch in
[Bend](https://github.com/bendlang/bend) and train them on
[BendTorch](https://github.com/nuxyel/BendTorch). This is an experiment corpus,
not a library. bend-ml owns the tensors and the laws, BendTorch owns the
training, and this repository uses them, measures them and writes down what
happened.

Every number below comes from a program in this repository. I check each one
three ways: against a NumPy twin written at the same operation order, against
an outside implementation where one exists (scikit-learn, PyTorch, the
lucidrains UNet), and against an earlier version of itself through frozen
probes. When they disagree, the disagreement is the finding. If I did not
measure a number, I do not write it down.

<p align="center">
  <a href="https://github.com/adeerkhan/bend-lab/actions/workflows/ci.yml"><img src="https://github.com/adeerkhan/bend-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

## What it shows

The diffusion track is complete through a ViT denoiser: 2×2 patches, 16 tokens,
three pre-LN blocks, 100,736 parameters, 3000 steps on the first 1024 optdigits
images. Legibility is the mean best absolute correlation of a sample with any
training digit. Random images score 0.326, the digits MLP scores 0.408, and the
gate is 0.75.

| arm (same data, sampler and starting noise) | DDIM 16 | ancestral 64 | DDIM MSE | ancestral MSE |
|---|---:|---:|---:|---:|
| **Bend-ViT** | **0.783** | **0.807** | 0.0618 | 0.0569 |
| PyTorch, same architecture | 0.790 | 0.779 | 0.0583 | 0.0633 |
| lucidrains UNet (reference) | 0.932 | 0.940 | 0.0202 | 0.0183 |

The Bend model is within 0.007 of the PyTorch twin on DDIM and ahead of it on
the ancestral sampler, with the same 4096 starting samples. Its cloud mean and
standard deviation are 0.3007 / 0.3450 against the data's 0.3076 / 0.3767, and
4082 of 4096 pixels land in [0, 1].

The gap closed in the training loop, not in the architecture. Bend drew one
timestep per batch while the reference draws one per image, worth +0.36
legibility. The evidence is in `results/diffusion-vit/`, and the three-arm page
is `results/compare/index.html`.

## Layout

```
examples/          what is run: one directory per task, with main.bend, README.md and plot.svg
  classification/    knn, logistic, mlp, naive_bayes, naive_bayes_heart, crossval
  diffusion/         two_moons, digits, vit
  transformers/      attention, periodic
  unsupervised/      kmeans, pca
{metrics,hf,datasets,preprocess,split,select,cluster,pipeline,report,generative}/
                   the ten packages published to BendHub
reference/         what proves it: twins, probes, gates, comparisons, the task registry
results/           what it proved: run.json, REPORT.md, plots, the three-arm compare page
docs/              PLAN, AUDIT, the agent handoff
bench/             compute-only benchmarks
```

The rule for the tree: the package directories are what I publish, `examples/`
is what I run, `reference/` is what proves it, `results/` is what it proved.
Moving the packages under `packages/` and the verification layer under
`apparatus/` is the next structural step. Those moves are mechanical but touch
`bend.toml` and many paths, so they go in one commit with `make check` behind
them.

## Quickstart

Bend 2.0.38, Lean 4.34.0, clang and the Python environment, one command:

```sh
make setup                                               # toolchain, lock, environment
bendenv run -- bend examples/diffusion/vit/main.bend     # train and sample the ViT
make check                                               # the gate: proofs, kernels, twins, comparisons
```

`make check` is the gate. It checks the proofs and the kernel with Lean, runs
the programs that must be refused, and compares every measured number with a
NumPy or scikit-learn reference. On a cold machine it builds each program once,
and that is the slow part.

## The corpus

| example | data | what is checked | result |
|---|---|---|---|
| `diffusion/vit` | optdigits, 1024 × 8×8 | legibility of 4096 samples, three arms | **0.783 / 0.807** (torch 0.790 / 0.779) |
| `diffusion/digits` | optdigits, 1024 × 8×8 | first loss against the twin, cloud statistics | 1.0039 (twin agrees to 2.9e-8) |
| `diffusion/two_moons` | two moons, 128 × 2 | loss curve, cloud mean and std | 152.95 → 0.065 |
| `transformers/periodic` | periodic 0…7, 32 × 8 | loss and next-token accuracy | 2.359 → 0.002, accuracy 1.00 |
| `transformers/attention` | synthetic 4 × 8 | every output against NumPy | checksum 1.76295, Δ 3.3e-9 |
| `classification/crossval` | iris, 150 × 4 | five stratified folds, no leakage | 0.9307 ± 0.006 (sklearn 0.9307) |
| `classification/knn` | breast cancer, 683 × 9 | accuracy on the same split | 0.978102 (sklearn 0.978102) |
| `classification/naive_bayes` | breast cancer, 683 × 9 | accuracy on the same split | 0.985401 (sklearn 0.985401) |
| `classification/naive_bayes_heart` | heart failure, 299 × 12 | accuracy on the same split | 0.950000 (sklearn 0.950000) |
| `classification/mlp` | iris, 150 × 4 | accuracy, 400 SGD steps | 0.973333 (sklearn 0.973333) |
| `classification/logistic` | iris, 150 × 4 | the loss over four steps | 1.0986 → 1.0672 |
| `unsupervised/kmeans` | wine, 6497 × 11 | inertia and cluster sizes | 50645.64, [2384, 990, 3123] |
| `unsupervised/pca` | wine, 6497 × 11 | explained variance ratio | 0.275443 (sklearn 0.275443) |

`make results` writes the full table, with the delta and the tolerance of every
task, to `results/RESULTS.md`. The dataset, seed, hub hashes and git state of
each run are in `results/latest/`.

## How a result gets here

1. An example in `examples/<area>/<task>/main.bend` trains and prints its
   numbers.
2. A twin in `reference/` mirrors it in NumPy at the same operation order.
3. A gate in `reference/test_<task>.py` compares the two and runs in
   `make check`.
4. A result in `results/<task>/` records the run, the report card and the plots.

A number that was not measured does not go in a README, a report card or a
result.

Run one task, or the whole corpus, through the experiment layer:

```sh
reference/.venv/bin/python reference/experiment.py --list
reference/.venv/bin/python reference/experiment.py --task iris-crossval
```

Each run writes a report card and a manifest: the git commit, the Bend version,
every package at its hub hash, the machine, the seed and the tolerances.

## The GPU lane

`bend.toml` pins `bendtorch@1.1.1.0` and `bendtorch-gpu@1.1.1.0`. The ViT builds
with eight `LinearNBF` and one `F.Attention`, links `libnvrtc`, and executes on
an RTX 5060. `nvidia-smi` shows the process with 104 MiB on the card, and the
frozen values are bit-identical to the CPU lane.

It buys no speed yet. It is 5m44 either way, because the time goes to plumbing:
seven `init_params()` calls, 1024 rows rebuilt per forward, and the
`Mat.to_list` conversions. That turns CIFAR from a CUDA question into a data
path question. The measurement that decides it is next: one training step split
into `init_params`, one `rows_go` build, one `Mat.from_list` and one forward.

## Status

`make check` is 92/92. Diffusion is complete through the ViT, with the gate at
0.75 and the Bend model measured at 0.783 and 0.807. Flow matching has a
measured probe and needs its 2D example. CIFAR-10 waits on the plumbing
measurement and a real step time. The UNet in Bend waits on `Im2col` and
`GroupNorm` upstream. Upstream, `FullAttention` for BendTorch and `Im2col` come
next.

## Licence

MIT. See `LICENSE`.
