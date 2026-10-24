# Plan: the experiment layer for Bend ML

## Why

The Bend ML ecosystem now has its primitives and its trainer:

- **[bend-ml](https://github.com/nuxyel/bend-ml)** — tensors, shapes, the
  type-level laws, the BPE tokenizer. Its `tensor` and `autograd` are frozen;
  `tensor-array` and `bpe` are alive.
- **[BendTorch](https://github.com/aaaxn/bendtorch)** — the tape, the layers,
  the losses, the optimizer, the MNIST run.

Neither owns the question a person actually asks: *“what does this model score
on this data, and can I reproduce that number?”* That is the experiment layer,
and it is what MLBend is for. The rule that keeps it distinct:

> **MLBend writes no tensor, no autodiff and no optimizer. A model enters
> through an adapter, not through a new engine.**

The layer is: datasets, splits, transformations, model selection, metrics,
reports and manifests — the part of scikit-learn that is `datasets`,
`model_selection`, `preprocessing` and `metrics`, deliberately without the
estimators' internals.

## The layers

| Layer | Repository | Owns | Does not own |
|---|---|---|---|
| primitives | bend-ml | tensors, shapes, laws, tokenizer | training, evaluation |
| training | BendTorch | tape, layers, losses, optimizer | data, splits, reports |
| experiment | **MLBend** | data, splits, transformations, selection, metrics, manifests, benchmarks | tensors, autodiff, optimizers |

## What the layer must do

1. **A task** is a dataset, a split, a program and the numbers that program must
   produce, with a tolerance for each.
2. **An adapter** builds and runs the program and parses its output. A new
   trainer should become a task without touching the runner.
3. **A reference** computes the same numbers with NumPy or scikit-learn.
4. **A report card** compares the two and says pass or fail.
5. **A manifest** records the git commit (and whether the tree was dirty), the
   Bend version, every package at its hub hash, the machine, the seed and the
   tolerances.
6. **A report sink** writes the JSON and, with `--wandb`, logs the same manifest
   to Weights & Biases.

## Status (2026-10-23)

The substrate is written and published: `mlbend-metrics`, `mlbend-preprocess`,
`mlbend-split`, `mlbend-select`, `mlbend-cluster`, `mlbend-datasets`,
`mlbend-huggingface`. `mlbend-cluster` proved `cluster_covers`, and CI certified
it with the kernel (63/63 checks).

The experiment layer exists as [`reference/experiment.py`](../reference/experiment.py):

| task | program | reference |
|---|---|---|
| `iris-crossval` | logistic regression, five stratified folds | scikit-learn `LogisticRegression` + `StratifiedKFold` |
| `iris-mlp` | two rectified layers, 400 SGD steps | scikit-learn `MLPClassifier(8)` |
| `wine-regression` | linear regression, 1000 steps on BendTorch's tape | scikit-learn `LinearRegression` |
| `wine-kmeans` | k-means, three clusters, ten rounds | the same algorithm in NumPy |
| `cancer-naive-bayes` | Gaussian naive Bayes, four fifths train | scikit-learn `GaussianNB` |
| `cancer-knn` | three nearest neighbours, four fifths train | scikit-learn `KNeighborsClassifier` |

Every run writes `reports/<task>-<stamp>.json`; `--wandb` logs it. The report
carries the dataset, the seed, the git state, the Bend version, the hub hashes,
the machine and the metric table.

## The roadmap

### 1. The pipeline, without leakage — written (2026-10-23)

`examples/crossval` standardised the whole matrix before the folds were made:
the scaling had seen the test rows. `mlbend-pipeline@0.1.0.0` fixes that:
`fit` takes the training rows and nothing else, `transform` applies the fitted
means and deviations to any rows, and `fold` does both for one split. The law
`pipeline_preserves_rows` proves one output row per input row; the numbers are
compared with scikit-learn's `StandardScaler` and with
`Pipeline(StandardScaler, LogisticRegression)` on the same folds. `crossval`
now uses it and its mean accuracy is unchanged (0.9333).

The design decided while building it:

- `fit(transform, train_rows)` consumes train rows only;
- `transform(fitted, test_rows)` is pure;
- the cross-validated score must match scikit-learn's `Pipeline` on the same
  folds, or the difference must be explained by the folds.

It is a package because the rows are lists and the counts are clean. The next
transformer (min-max, one-hot) either joins it behind the same `fit`/`transform`
shape or the package grows a chain; the law is the stop criterion either way.

### 2. The adapter protocol

Write down what a task may be: a program with a fixed output grammar, or a
Python entry point that drives a trainer and prints the same metrics. Add
adapters for BendTorch and for bend-ml's autograd, so the same task can be run
by either and the report says which. No task code should change when a trainer
does.

### 3. Model selection

`mlbend-select` walks the folds. The next step is the search: a grid or random
search over a parameter value, nested cross-validation, and per-fold tables.
The acceptance is scikit-learn's `GridSearchCV` on the same folds within the
same tolerance the single model already meets.

### 4. Reports

The metrics exist; the *report* does not. Classification: confusion matrix,
per-class precision/recall/F1 with support, macro and weighted averages;
regression: R², MAE, RMSE, residual quantiles. The numbers must match
scikit-learn's `classification_report` and `confusion_matrix`. Whether this is
a Bend package (`mlbend-report`) or a runner-side renderer depends on how much
of the format is pure — the metric arithmetic is pure and belongs in Bend.

### 5. The deep-learning track — training on the real BendTorch

The layer is trainer-agnostic, so a deep model is another task. There are two
now:

- `attention-parity`: a single-head self-attention block (Q, K, V, scaled dot
  product, row softmax, the weighted sum of V) over four tokens, written in
  plain Bend and compared number by number with PyTorch (1.5e-8). It is the
  forward pass; the weights are fixed.
- `transformer-periodic`: a one-block causal transformer trained on
  `bendtorch@1.0.1.0` — embedding, RMSNorm, single-head causal attention,
  output projection, RMSNorm, linear head — with AdamW for 200 steps on
  next-token prediction over the periodic sequence 0..7. The loss falls from
  2.3591 to 0.0021; a NumPy twin starts at 2.3591 and ends at 0.1351. The loss
  curve, the table and the manifest are in `results/transformer/`.

**The foundation correction (2026-10-24).** The earlier "missing softmax" was
true only of the old `bendtorch-core@0.1.0.0` line; the maintained
`bendtorch@1.0.1.0` has `Attention`, `RMSNorm`, `Embedding`, `CrossEntropy`,
`AdamW` and `Train`. MLBend now depends on the real package and pins Bend
2.0.38. The old examples will be ported or retired; no new model is built on
the dead line.

### 6. The corpus and the scoreboard

Grow the tasks beyond the six: more datasets from `mlbend-datasets`, the same
model under different seeds, the two trainers on one task. Then publish the
results as a checked-in table with the manifest beside each row. The point is
not a leaderboard for its own sake: it is the evidence that the ecosystem's
numbers are real and repeatable.

### 7. Weights & Biases

`--wandb` logs one run today. Next: a project per task family, the manifest as
the run config, the report JSON as an artifact, and sweeps over the parameters
the selection step defines. Offline mode stays the default without an API key so
CI never depends on the network.

## What we deliberately do not build

- No tensors, autodiff, layers or optimizers: those are bend-ml's and
  BendTorch's.
- No `mlbend-linear` as a trainer. Ordinary least squares, ridge and logistic
  regression arrive as adapters over BendTorch or bend-ml, not as a second
  implementation.
- No `mlbend-linalg`: the tensors and the lemmas are already published.
- Gradient-free baselines only (`mlbend-cluster`, naive Bayes, kNN): they need
  no trainer, and the harness needs something to beat.

## Stop criteria

- A task is done when its Bend number and its reference are within the declared
  tolerance, the report is reproducible from the manifest, and `make
  experiments` is green.
- A package is done when its law is proved (`--verdict`), its numbers match
  NumPy/scikit-learn, its README explains the claim in plain language, and
  `make check` is green.
- A claim that cannot be proved yet is an issue and a `NOTES.md` entry, not a
  package.

## Open questions

- **The Bend/Python boundary.** The runner is Python because it must call the
  compiler, the reference libraries and W&B. Which parts of the experiment —
  the metric arithmetic, the fold bookkeeping, the report format — belong in
  law-checked Bend packages instead? The rule so far: pure and structural in
  Bend, IO and orchestration in Python.
- **The manifest schema.** `mlbend.experiment/1` is a first draft. What must
  version 2 carry — the dataset file hash, the compiler's build, the exact
  command line?
- **Network datasets.** HuggingFace revisions can move. A manifest names the
  repository and the split; should it pin the file's SHA-256 as well?
- **The folds.** Two libraries shuffle with different generators, so equality
  is procedural, not row-by-row. Is a shared seeded split (`mlbend-split`)
  enough to make the comparison exact across trainers?
