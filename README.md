# bend-lab

Experiments in machine learning, written in [Bend](https://github.com/bendlang/bend) and run on
[BendTorch](https://github.com/nuxyel/BendTorch). Models are trained from scratch in Bend, every
number is compared against a NumPy twin and — where one exists — against the reference
implementation (`lucidrains/denoising-diffusion-pytorch`, PyTorch, `nn.Conv2d`), and every finding is
written down with a date in [`issues-hit.md`](issues-hit.md).

This is not another ML library: the libraries are separate (`bend-ml`, `BendTorch`, and the packages
in [`packages/`](packages/)). This is where they are **used, measured and shown**.

## Layout

```
examples/     what is run — one directory per task: main.bend, README.md, plot.svg
  classification/   knn, logistic, mlp, naive_bayes, naive_bayes_heart, crossval
  diffusion/        2d, digits, vit
  transformers/     attention, periodic
  unsupervised/     kmeans, pca
{metrics,hf,datasets,preprocess,split,generative,cluster}/   the sources published to the hub
reference/    what proves it — twins, probes, gates, comparisons, the task registry, run scripts
results/      what it proved — run.json, REPORT.md, plots, per task
docs/         PLAN, AUDIT, notes, handoff
bench/        compute-only benchmarks
```

The rule: **the package directories are what you publish, `examples/` is what you run, `reference/`
is what proves it, `results/` is what it proved.** (Moving the packages under `packages/` and the
verification layer under `apparatus/` is the next structural step; the moves are mechanical but touch
`bend.toml`'s members and every path, so they go in their own commit with `make check` behind them.)

## Quickstart

```sh
make setup                                  # the Bend env, the lock, the two Python venvs
bendenv run -- bend examples/diffusion/vit/main.bend     # train + sample the ViT denoiser
make check                                  # 92 checks: proofs, kernels, gates, twins, comparisons
```

## The corpus

| task | data | what is measured | result |
|---|---|---|---|
| `diffusion/2d` | two moons, 128 points | loss, cloud statistics | in its README |
| `diffusion/digits` | 1024 optdigits, 8×8 | loss, cloud, nearest-digit \|corr\| | 0.408 (MLP) |
| `diffusion/vit` | the same 1024 digits | the above, three arms | **Bend 0.783 / 0.807**, torch 0.790 / 0.779, UNet 0.932 / 0.940 |
| `transformers/attention` | a periodic sequence | loss, accuracy | in its README |
| `transformers/periodic` | the same | loss, accuracy | in its README |
| `classification/*` | iris, breast cancer, heart | accuracy, reports | in each README |
| `unsupervised/{kmeans,pca}` | wine, 13 features | inertia, cluster sizes, explained variance | in each README |

## How a task is built

1. **an example** in `examples/<area>/<task>/main.bend` that trains and prints its numbers;
2. **a twin** in `apparatus/` — a NumPy mirror at the same operation order;
3. **a gate** in `apparatus/test_<task>.py`, comparing the two, wired into `make check`;
4. **a result** in `results/<area>/<task>/` — `run.json`, `REPORT.md`, plots;
5. **a line in `issues-hit.md`** for every surprise, with the date and the measurement.

A number that was not measured does not go in a README, a report card or a ledger entry.

## Where the numbers come from

Every headline figure is written by a program in this repository and stored under `results/` (with
the loss curve and, for diffusion, the samples themselves). The gates run in `make check`; the
comparisons against PyTorch and against the reference UNet are in `apparatus/` and are run by hand,
because they need a separate environment.

## Status

`make check`: **92/92**. The diffusion track is complete through the ViT denoiser (0.783 DDIM /
0.807 ancestral, gate at 0.75), with the three-arm comparison and the ledger current. Open:
CIFAR-10 (a compute decision, measured), the UNet (waiting on `Conv2d`/`GroupNorm` upstream), and
flow matching (probe measured, example next).

## Licence

MIT. See [`LICENSE`](LICENSE).
