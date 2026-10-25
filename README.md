# MLBend

**The experiment layer for machine learning in [Bend 2](https://bend-lang.com):**
a dataset in, a reproducible report out, over any Bend trainer.

[bend-ml](https://github.com/nuxyel/bend-ml) owns the tensors and the laws,
[BendTorch](https://github.com/nuxyel/BendTorch) owns the training. MLBend owns
the *experiment* around them: the data, the split, the transformations, the
model selection, the metrics, and the evidence that a number is the number it
claims to be. It writes no tensor, no autodiff and no optimizer of its own.

What Bend changes is the ground under all of it: a fitted object is a **value**,
the **shape** is in the type, the structural claims are **laws** the kernel
checks, and every run can carry a **manifest** — dataset revision, seed, split,
package hashes, hardware — so a second person can reproduce it.

## The layers

| Layer | Repository | Owns |
|---|---|---|
| primitives | [bend-ml](https://github.com/nuxyel/bend-ml) | tensors, shapes, type-level laws, the BPE tokenizer; `autograd` is frozen |
| training | [BendTorch](https://github.com/nuxyel/BendTorch) | the layers, the losses, the optimizers, `Train` |
| experiment | **this repository** | datasets, splits, transformers, model selection, metrics, reports, benchmarks |

A model enters MLBend through a **task**, not through a new engine. That is the
boundary: gradient-trained models come from BendTorch or bend-ml's autograd;
MLBend composes them, cross-validates them, scores them and records the run.
The one kind of model it keeps is the gradient-free baseline — k-means, naive
Bayes, k-nearest neighbours — because a baseline needs no trainer and the
harness needs one.

## The design, in five rules

1. **Values, not objects.** A fitted transformer is a value: `scaler.transform(y)`,
   no `self`, nothing mutated. Bend's affine rule makes this the only shape a
   program can have.
2. **Shapes in the types.** `Mat<r, c>` for the data; a model over `c` features is
   `Model<c, k>`; a pipeline with the wrong shape does not compile.
3. **Structure proved, numbers tested.** Sizes, partitions and compositions are
   laws the kernel checks; `F32` results are compared with NumPy and
   scikit-learn. `F32` is not a real number, so no value is ever proved.
4. **Effects at the edges only.** Fetching a dataset, reading a file, seeding a
   generator. Everything after the edge is pure and reproducible.
5. **Every number has a manifest.** A score without the dataset revision, the
   seed, the split and the package hashes is a claim, not a result.

## The packages

| Package | What it is | Status |
|---|---|---|
| [`mlbend-metrics`](metrics) | MSE, MAE, RMSE, accuracy, precision, recall, F1, and the counts they rest on | v0.1: written, checked, one law proved |
| [`mlbend-preprocess`](preprocess) | standardize, min-max, one-hot, L1 and L2 normalization | v0.1: written, checked, published |
| [`mlbend-split`](split) | a seeded, deterministic split of the rows, stratified by label | v0.1.1: written, checked, published |
| [`mlbend-select`](select) | k-fold cross-validation on top of the split | v0.1: written, checked, published |
| [`mlbend-cluster`](cluster) | k-means, functional: a fixed number of rounds, one cluster per row | v0.1: written, checked, published |
| [`mlbend-pipeline`](pipeline) | fit on the training rows only: the scaler of one fold, applied to both sides | v0.1: written, checked, published |
| [`mlbend-report`](report) | the confusion matrix and the per-class report: precision, recall, F1, support, macro and weighted averages | v0.1: written, checked, two laws proved |
| [`mlbend-generative`](generative) | the generative skeleton: a counter RNG (mix32 and its inverse, Box-Muller), the cosine and linear schedules, the time embedding | v0.1: written, checked, four laws proved |
| [`mlbend-datasets`](datasets) | the datasets of scikit-learn, pinned revisions and an authoritative local cache (`MLBEND_DATA`) | v0.1.6: pinned and strict, published |
| [`mlbend-huggingface`](hf) | a HuggingFace client: the dataset page, the file tree, a file's bytes, the JSON rows | v0.1.1: written, checked, published |

The ten packages live here, in one repository, as the workspace in
`bend.toml`; there is no repository per package. `make release PKG=<dir>`
publishes one to BendHub and records it under the tag `<name>-v<version>`.
The package manager, `bendenv`, is the one piece that stays in its own
repository ([bend-env](https://github.com/adeerkhan/bend-env)).

## The experiment layer

[`reference/experiment.py`](reference/experiment.py) is the first product of
the layer. A **task** is a dataset, a program and the numbers it must produce;
the runner builds and runs the program, parses its output, computes the same
numbers with NumPy or scikit-learn, and writes a **report card** as JSON.

```console
$ reference/.venv/bin/python reference/experiment.py --list
attention-parity    examples/attention/main.bend           single-head attention forward, against PyTorch
transformer-periodic examples/transformer/main.bend        a one-block causal transformer, trained on BendTorch
diffusion-2d        examples/diffusion/main.bend           a DDPM denoiser, trained and sampled on two moons
cancer-knn          examples/knn/main.bend                 three-nearest-neighbours over breast cancer
cancer-naive-bayes  examples/naive_bayes/main.bend         Gaussian naive Bayes over breast cancer
heart-naive-bayes   examples/naive_bayes_heart/main.bend   the same baseline on heart failure
iris-crossval       examples/crossval/main.bend            logistic regression over five stratified folds
iris-mlp            examples/mlp/main.bend                 a two-layer rectified network over iris
wine-kmeans         examples/kmeans/main.bend              k-means over wine, three clusters
wine-pca            examples/pca/main.bend                 the first principal component of wine

$ reference/.venv/bin/python reference/experiment.py --task cancer-knn
task: cancer-knn  (three-nearest-neighbours over breast cancer, four fifths train)
git: main ...
bend: 2.0.38   run: 1.4 s

metric                      bend     reference       delta  ok
accuracy                0.978102      0.978102   1.022e-08  ok

PASS
report: results/raw/cancer-knn-....json
```

The corpus is not only classical: `attention-parity` runs a single-head
self-attention block and checks every output number against PyTorch,
`transformer-periodic` trains a one-block causal transformer with the real
BendTorch, and `diffusion-2d` and `diffusion-digits` train a DDPM denoiser
and sample it — over two moons, and over the first 1024 rows of real
`mstz/optdigits` digits (`results/diffusion-digits/grid.svg` shows real
digits beside the samples). Every
"random" draw is a pure function of a counter (`mlbend-generative`), so the
NumPy twin draws the same numbers (`results/diffusion/` holds the scatter
beside the loss curve).

**The GPU lane.** Bend programs reach the GPU through bangs (`!`), the
kernels of `bend-ml-tensor-gpu`, and `bend -o` builds the device program
when `CUDA_HOME` names a toolkit with `nvrtc.h` (NVRTC compiles it at build
time; three small RPMs unpacked user-locally are enough — the recipe and the
`libnvrtc.so.13` loader failure it fixes are in NOTES.md, 2026-10-26).
bendtorch ships the drop-in (`torch/gpu.bend`); our
examples use the proved CPU layers on purpose. Verified on an RTX 5060:
the linear layers give bit-identical results on the GPU, and the
exponential-taking layers differ only in CUDA's exp ulps — exactly
bendtorch's documented policy.

The manifest records the git commit and whether the tree was dirty, the Bend
version, every `mlbend-*`, `bend-ml-*` and `bendtorch-*` package at its hub
hash, the machine, the seeds, the tolerances, the dataset's pinned revision and
expected SHA-256, the bytes actually read, the **build time**, the **run time**
(minimum of the repeats), the **fetch** and the **reference time**. The runner fills `reference/.cache/datasets` with the
exact bytes it hashes and sets `MLBEND_DATA`, so a warm run measures compute;
the fetch is recorded separately and is 0 on a warm cache. The build time is
Bend's compiler and is reported separately. `--wandb` logs the same
manifest to **Weights & Biases** (project `mlbend`), offline when there is no
API key, so a run never blocks on the network.

```console
$ make experiment TASK=cancer-knn        # one report card, saved under results/
$ make experiment TASK=cancer-knn TRAINER=numpy   # the same task, NumPy as the trainer
$ make experiments                       # every task
$ make results                           # rebuild results/RESULTS.md
make bench                             # the compute-only kernel benchmark
$ reference/.venv/bin/python reference/experiment.py --task iris-crossval --wandb
```

## The results

The reports are committed, not thrown away:

- [`results/RESULTS.md`](results/RESULTS.md) — the latest report card of every
  task: dataset, Bend number, Python number, delta, milliseconds and speed
  ratio.
- [`results/transformer/`](results/transformer/) — the trained transformer's
  loss curve (`loss.svg`), its step-by-step table (`loss.tsv`) and the run
  manifest (`run.json`).
- [`results/selection/`](results/selection/) — the kNN model-selection curve (k = 1..15).
- [`results/report/`](results/report/) — the classification report and confusion matrix.
- [`results/BENCH.md`](results/BENCH.md) — the compute-only kernel benchmark
  (one matrix-product workload, Bend against NumPy), with GFLOP/s.
- `results/latest/<task>.json` and `results/raw/` — the full manifests and the
  timestamped history.

The task timings are warm-cache compute: both sides read the cached bytes. A
stochastic task (the five-fold cross-validation) is run over several seeds and
reported as mean ± sample deviation; its reference reads the fold assignment
the program printed, so the two compare the same partition and the same model.
The kernel benchmark has no dataset and no IO. `make experiments` and `make bench` regenerate them.

## Checks

```bash
make setup        # the pinned Bend 2.0.38, Lean 4.34.0 and the Python venv
make check        # the proofs, the kernel (--verdict), the rejected programs, the NumPy comparison
make laws         # every law statement of every package
make experiment TASK=cancer-knn   # one task, one report card
make experiments                 # every task
```

`make check` is the gate before a commit; CI runs it on every push.
`make experiments` is the gate for the layer itself: a task whose Bend number
drifts from its reference turns red, and `make results` writes the comparison
table that is committed.

## What is proved, and what is trusted

[`docs/AUDIT.md`](docs/AUDIT.md) says what each package proves, what it only
tests, and what it trusts. The short version: the counts are proved, the `F32`
numbers are tested against NumPy and scikit-learn, and every experiment reports
both sides of the comparison.

## For an agent

Start at [`AGENTS.md`](AGENTS.md): the rules, the read order, the commands.
[`docs/agents/handoff.md`](docs/agents/handoff.md) says where the work is;
[`docs/agents/domain.md`](docs/agents/domain.md) is the vocabulary and the map;
[`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md) is how the
issues carry the work. The roadmap is [`docs/PLAN.md`](docs/PLAN.md).

## License

MIT. Everything in this repository is MIT, and a package published to BendHub
carries its own `LICENSE` next to the entry file.
