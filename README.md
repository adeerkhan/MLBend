<p align="center">
  <img src="bend-hero.png" alt="bend-lab: machine learning, built in Bend" width="100%">
</p>

# bend-lab

**Machine learning, built in Bend.** Models are written from scratch in
[Bend](https://github.com/bendlang/bend) and trained on
[BendTorch](https://github.com/nuxyel/BendTorch). The repository is an
experiment corpus, not a library. bend-ml owns the tensors and the laws,
BendTorch owns the training, and this repository uses them, measures them and
records the results.

Every number comes from a program in the repository and is checked three ways:
against a NumPy twin written at the same operation order, against an outside
implementation where one exists (scikit-learn, PyTorch, the lucidrains UNet),
and against an earlier version of itself through frozen probes. When the three
disagree, the disagreement is the result. A number that was not measured is not
written down.

<p align="center">
  <a href="https://github.com/adeerkhan/bend-lab/actions/workflows/ci.yml"><img src="https://github.com/adeerkhan/bend-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

## Results

![Legibility by arm](docs/figures/legibility.webp)

The diffusion track is complete through a ViT denoiser: 2×2 patches, 16 tokens,
three pre-LN blocks, 100,736 parameters and 3000 steps on the first 1024
optdigits images. Legibility is the mean best absolute correlation of a sample
with any training digit. Random images score 0.326, the digits MLP scores 0.408,
and the gate is 0.75.

| arm (same data, sampler and starting noise) | DDIM 16 | ancestral 64 | DDIM MSE | ancestral MSE |
|---|---:|---:|---:|---:|
| **Bend-ViT** | **0.787** | **0.811** | 0.0598 | 0.0561 |
| PyTorch, same architecture | 0.790 | 0.779 | 0.0583 | 0.0633 |
| lucidrains UNet (reference) | 0.932 | 0.940 | 0.0202 | 0.0183 |

The Bend model is within 0.003 of the PyTorch twin on DDIM and ahead of it on
the ancestral sampler, with the same 4096 starting samples. Its cloud mean and
standard deviation are 0.2957 / 0.3395 against the data's 0.3076 / 0.3767, and
4087 of 4096 pixels land in [0, 1].

The gap closed in the training loop, not in the architecture. Bend drew one
timestep per batch while the reference draws one per image, worth +0.36
legibility. The evidence is in `results/diffusion-vit/`, and the three-arm
comparison is `results/compare/index.html`.

## Diffusion

![ViT loss](docs/figures/loss.webp)

The loss falls from 1.009 to 0.071 over 3000 steps, with a minimum of 0.058.
The frozen probe pins the forward pass, one ancestral step, the 16-step DDIM
cloud and the eta=0 invariance to the NumPy twin; the legibility gate is 0.75.
The same task also runs on a dense MLP over real digits, and on two moons with
an ancestral sampler and DDIM.

![DDIM samples](docs/figures/samples.webp)

## Flow matching

![Rectified flow loss](docs/figures/flow.webp)

Rectified flow replaces the diffusion chain with one vector field and an Euler
ODE: `x_t = noise*(1-t) + data*t`, target `v = data - noise`, MSE loss, and a
sample integrated over 32 steps of `linspace(0, 1)`. The 2D example uses three
linear layers with SiLU, full batch, 6000 AdamW steps, and reaches a last-100
loss of 0.886. Its samples have mean 0.049 / -0.048 and standard deviation
0.610 / 0.675 against the data's 0.000 / 0.000 and 0.577 / 0.659, with 112 of
128 inside the data's box. A frozen probe and NumPy twin for this example are
still to be added.

## Where the time goes

![One training step before and after](docs/figures/plumbing.webp)

A training step used to take 543 ms, of which 418 ms built the input and the
target, and `init_params` cost 25.3 s. The builders appended into a growing
list, which is O(n²), and the tokeniser scanned each digit. Rebuilt with cons
and one reverse, and with a linear tokeniser, a step takes 131 ms (7 build, 53
forward, 71 loss, backward and optimizer) and `init_params` takes 6 ms. The
values are bit-identical. The step is now compute-bound, so the device lane is
worth re-measuring.

## Layout

```
examples/          what is run: one directory per task, with main.bend, README.md and plot.svg
  classification/    knn, logistic, mlp, naive_bayes, naive_bayes_heart, crossval
  diffusion/         two_moons, flow_matching, digits, vit
  transformers/      attention, periodic
  unsupervised/      kmeans, pca
{metrics,hf,datasets,preprocess,split,select,cluster,pipeline,report,generative}/
                   the ten packages published to BendHub
reference/         what proves it: twins, probes, gates, comparisons, the task registry
results/           what it proved: run.json, REPORT.md, plots, the three-arm comparison
docs/              PLAN, AUDIT, the agent handoff, the generated figures
papers/            paper comparisons: the claim, the reduced protocol, the two arms
bench/             compute-only benchmarks
```

The rule for the tree: the package directories are what gets published,
`examples/` is what runs, `reference/` is what proves it, and `results/` is what
it proved. Moving the packages under `packages/` and the verification layer
under `apparatus/` is the next structural step. Those moves touch `bend.toml`
and many paths, so they go in one commit with `make check` behind them.

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
which is the slow part.

## The corpus

| example | data | what is checked | result |
|---|---|---|---|
| `diffusion/vit` | optdigits, 1024 × 8×8 | legibility of 4096 samples, three arms | **0.787 / 0.811** (torch 0.790 / 0.779) |
| `diffusion/digits` | optdigits, 1024 × 8×8 | first loss against the twin, cloud statistics | 1.0039 (twin agrees to 2.9e-8) |
| `diffusion/two_moons` | two moons, 128 × 2 | loss curve, cloud mean and std | 152.95 → 0.065 |
| `diffusion/flow_matching` | two moons, 128 × 2 | last-100 loss, sample mean/std, box | 0.886, 112/128 in the box |
| `transformers/periodic` | periodic 0…7, 32 × 8 | loss and next-token accuracy | 2.359 → 0.002, accuracy 1.00 |
| `transformers/attention` | synthetic 4 × 8 | every output against NumPy | checksum 1.76295, delta 3.3e-9 |
| `classification/crossval` | iris, 150 × 4 | five stratified folds, no leakage | 0.9307 ± 0.006 (sklearn 0.9307) |
| `classification/knn` | breast cancer, 683 × 9 | accuracy on the same split | 0.978102 (sklearn 0.978102) |
| `classification/naive_bayes` | breast cancer, 683 × 9 | accuracy on the same split | 0.985401 (sklearn 0.985401) |
| `classification/naive_bayes_heart` | heart failure, 299 × 12 | accuracy on the same split | 0.950000 (sklearn 0.950000) |
| `classification/mlp` | iris, 150 × 4 | accuracy, 400 SGD steps | 0.973333 (sklearn 0.973333) |
| `classification/logistic` | iris, 150 × 4 | the loss over four steps | 1.0986 → 1.0672 |
| `unsupervised/kmeans` | wine, 6497 × 11 | inertia and cluster sizes | 50645.64, [2384, 990, 3123] |
| `unsupervised/pca` | wine, 6497 × 11 | explained variance ratio | 0.275443 (sklearn 0.275443) |

`make results` writes the full table, with the delta and tolerance of every
task, to `results/RESULTS.md`. The dataset, seed, hub hashes and git state of
each run are in `results/latest/`.

## Method

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
every package at its hub hash, the machine, the seed and the tolerances. The
README figures are generated from those result files with `make figures`.

## The GPU lane

`bend.toml` pins `bendtorch@1.1.1.0` and `bendtorch-gpu@1.1.1.0`. The ViT builds
with eight `LinearNBF` and one `F.Attention`, links `libnvrtc`, and executes on
an RTX 5060. `nvidia-smi` shows the process with 104 MiB on the card, and the
frozen values are bit-identical to the CPU lane.

The earlier 5m44 either way was measured against the data path, not the device.
After the builder fix the step is 131 ms and 95% of it is forward and backward,
so the device lane is worth re-measuring. The split is in
`results/PLUMBING.md`.

## Papers

Paper comparisons live in `papers/`, one folder per paper, with the paper's
claims, the reduced protocol used here, and the Python and Bend arms. The first
entry, TEM, is a negative result: the 64x64 contrastive protocol collapses to a
constant energy, and the folder records the diagnosis and the blocked Bend
port. Paper numbers are never presented as reproduced.

## Status

`make check` is 92/92. Diffusion is complete through the ViT, with the gate at
0.75 and the Bend model measured at 0.787 and 0.811. Flow matching has a
measured 2D example, with its frozen gate still to be added. CIFAR-10 waits on
a real step time and a re-measured device lane. The UNet in Bend waits on
`Im2col`; GroupNorm has landed upstream, and a faithful reference arm also
needs the full and linear attention variants.

## Licence

MIT. See `LICENSE`.
