# Plan: the experiment layer for Bend ML

## Why

The Bend ML ecosystem now has its primitives and its trainer:

- **[bend-ml](https://github.com/nuxyel/bend-ml)** — tensors, shapes, the
  type-level laws, the BPE tokenizer. Its `tensor` and `autograd` are frozen;
  `tensor-array` and `bpe` are alive.
- **[BendTorch](https://github.com/nuxyel/BendTorch)** — the layers, the
  losses, the optimizers, `Train`, the MNIST and tiny-Llama runs.

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
| training | BendTorch | layers, losses, optimizers, `Train` | data, splits, reports |
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
| `wine-kmeans` | k-means, three clusters, ten rounds | the same algorithm in NumPy |
| `wine-pca` | the first principal component, power iteration | scikit-learn `PCA` |
| `cancer-naive-bayes` | Gaussian naive Bayes, four fifths train | scikit-learn `GaussianNB` |
| `heart-naive-bayes` | the same on heart failure | scikit-learn `GaussianNB` |
| `cancer-knn` | three nearest neighbours, four fifths train | scikit-learn `KNeighborsClassifier` |
| `attention-parity` | single-head attention forward | NumPy |
| `transformer-periodic` | a one-block causal transformer, 200 AdamW steps | a NumPy twin |

Every run writes `reports/<task>-<stamp>.json`; `--wandb` logs it. The report
carries the dataset, the seed, the git state, the Bend version, the hub hashes,
the machine and the metric table.

## The roadmap

### 1. The pipeline, without leakage — written (2026-10-23)

`examples/classification/crossval` standardised the whole matrix before the folds were made:
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

**The classification half is done (2026-10-25, `mlbend-report@0.1.0.0`).**
`mlbend-report` is the Bend package the plan asked about: the confusion matrix
(`cell`, `matrix`) and the per-class report — precision, recall, F1, support,
with the macro and the weighted averages — over `Nat` labels, resting on
`mlbend-metrics`' counts and its `zero_division=0` convention. Two laws are
proved (`matrix_rows`, `row_length`: the counted forms of the matrix build the
shape they claim); the numbers match scikit-learn's `confusion_matrix` and
`classification_report` within a relative 1e-5 (`reference/test_report.py`),
and `cancer-knn` prints the report beside its accuracy, checked by
`reference/test_knn.py`. The debt — the cells add up to the number of samples
— is the value-dependent wall of issues #1 and #8, recorded in the package's
README and issue #9. The regression half (R², MAE, RMSE beside residual
quantiles) is still open: the metrics exist, and the renderer stays
runner-side until a task needs it.

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
- `mlbend-generative@0.1.1.0` (2026-10-27, issues #10 and #12): the counter
  RNG (murmur3's fmix32 and its closed-form inverse, Box-Muller over 24-bit
  uniforms), the cosine and the linear schedules, the sinusoidal time
  embedding, and now the DDIM algebra — the posterior's σ, the five
  coefficients of the step, the element-wise fold, and the strided rung walk
  (`strided_from`, `groups_of`) that samples every k-th rung. Six length
  laws are proved (`noise_length`, `alphas_length`, `lin_alphas_length`,
  `temb_length`, `strided_length`, `ddim_length`), and every number is
  mirrored bit for bit by the NumPy twins. The bijection of mix32 is tested,
  not proved: the debt of issue #10.
- `diffusion-2d` (2026-10-25, extended 2026-10-27 with issue #12): a DDPM
  denoiser — three linear layers with SiLU — trained by AdamW to predict the
  noise of a 64-step linear-β forward process over two moons, then sampled
  two ways: the ancestral loop over all 64 rungs, and DDIM over every 4th
  rung (16 model calls instead of 64; at η = 0 a step reads no noise at all,
  so the cloud is a pure function of the model and the starting noise). The
  first training loss is the gate: every "random" number is a pure function
  of a counter (murmur3's fmix32, Box-Muller over 24-bit uniforms), so the
  NumPy twin draws the same numbers and the two agree to **1.5e-5**
  (152.954 against 152.954). Two sampling bugs went with the new sampler
  (issues #14 and #15): the step divided by the *cosine* ᾱ while the
  schedule is linear, and its noise was √β_t *inside* the 1/√α scale
  instead of the posterior's σ *outside* it. The final loss, both clouds'
  statistics and the scatter (`results/diffusion/`) are report-mode: the
  optimizers round differently and a 64-step loop amplifies a ulp. The
  finding the task cost: model parameters are affine (they hold arrays), so
  a sampling loop cannot reuse them — `Layer.fwd` returns them in
  `Fwd{p, y, s}` and the loop threads them through a record, the pattern
  bendtorch's own eval loop uses.
- `diffusion-digits`: the same task over real data — the first
  1024 rows of mstz/optdigits (8×8 digits, pinned revision, read through
  `csv_of`), a 64-wide SiLU body over a digit's 64 pixels plus 8 time-embedding
  values, 3000 full-batch AdamW steps, then 64 fresh digits two ways (the
  ancestral loop and DDIM at every 4th rung, issue #12). The exact gate caught
  four real bugs (σ_t, the per-row noise-index reset, the `drop` semantics, the
  65th-pixel consumption), a fifth in the staged input build (a pixel skipped
  per row, fixed in `2f2631b`), and the sampler work added two more (issues
  #13 and #14). The denoiser's own failure — it sat at the "predict nothing"
  loss of 1.0 — was the initialiser: a positive-mean `[-0.2, +0.5]` started
  every unit of the 72-wide first layer carried a +2.4 common mode that each
  later layer multiplied by 6.4, so the output started 88× the target (first
  loss **8062.7** against the vendored reference's 0.96), and the 95× cloud expansion was the
  missing clamp of the predicted start. Both are fixed and measured against
  `lucidrains/denoising-diffusion-pytorch` (the local issue ledger, ISSUE-001 and
  ISSUE-005): first loss **1.0039**, final **0.0417**, ancestral cloud
  **0.385/0.355**, DDIM **0.415/0.406**, against the data's 0.308/0.377, and
  both clouds' statistics are report-mode.

- `diffusion-vit` (2026-10-28): the same 1024 digits, but the denoiser is a
  transformer — 2×2 patches, 16 tokens of 20 features (4 pixels, 8 of time, 8
  of position), width 64, one head, three pre-LN blocks, one 4-value head per
  patch, 100,736 parameters, all of them BendTorch's layers. Two facts came
  out of building it: the residual must wrap the norm (`Residual(x,
  Seq(ln, att))` = x + att(ln(x)); written the other way the residual is taken
  on the normalized value and the block loses its identity path — ISSUE-010),
  and BendTorch's `Attention` is **causal**, so the 16 tokens attend as a
  sequence and the torch arm of the comparison runs the same mask (ISSUE-011).
  The learning rate is 3e-3, not the MLP's 1e-2 (ISSUE-012: 0.721 against
  0.782 in torch). The task prints both samplers' clouds from the one trained
  model, which is what `reference/compare_vit.py` needs to hold the model and
  the starting noise fixed and vary only the sampler.

**The foundation correction (2026-10-24).** The earlier "missing softmax" was
true only of the old `bendtorch-core@0.1.0.0` line; the maintained
`bendtorch@1.0.1.0` has `Attention`, `RMSNorm`, `Embedding`, `CrossEntropy`,
`AdamW` and `Train`. MLBend now depends on the real package and pins Bend
2.0.38. The three old examples were retired on 2026-10-24; no new model is built
on the dead line.

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
