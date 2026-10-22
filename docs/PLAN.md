# Plan: the classic-ML surface, natively in Bend 2

## Why

scikit-learn's surface is the one everybody knows: an estimator is fitted and then
applied, a transformer fits and transforms, a pipeline chains them, model
selection cross-validates, and the metrics score the result. Bend can offer the
same surface with two things scikit-learn cannot: the shapes are in the types,
and the structural claims are laws the kernel checks.

MLBend is the classic half of the ecosystem. bend-ml has the tensors, the
autodiff and the models; MLBend has the estimators, the transformers, the
pipelines and the metrics, and builds on bend-ml's packages rather than
repeating them.

## The packages, in order

Each package is one file, published to BendHub under `mlbend-*`, and imports
only Base and the packages below it.

| v | Package | What it is | The law it earns |
|---|---|---|---|
| 0.1 | `mlbend-metrics` | MSE, MAE, RMSE, accuracy, precision, recall, F1, and the counts they rest on | the count of the samples is the length of the label list |
| 0.2 | `mlbend-datasets` | the `load_*` surface of scikit-learn: iris, breast cancer, diabetes, on top of `bend-kit-csv` | the loader reads the rows the file has (over the bytes) |
| 0.3 | `mlbend-preprocess` | standardize, min-max, one-hot, L1 and L2 normalization | one-hot is a unit vector; the standardizer's output has mean 0 and variance 1 (over exact arithmetic) |
| 0.4 | `mlbend-split` | a seeded, deterministic train/test split, stratified by label | every row of the dataset is placed in a fold, so the folds partition the rows |
| 0.5 | `mlbend-select` | k-fold cross-validation on top of the split | the folds partition the data, like the split |
| 0.6 | `mlbend-linear` | ordinary least squares, ridge, logistic regression | the residual of the fit is orthogonal to the design |
| 0.7 | `mlbend-cluster` | k-means, functional (a fixed number of iterations) | every point is in exactly one cluster, and the assignment is a function of the point |
| 0.8 | `mlbend-pipeline` | a typed chain of transformers and estimators | the chain's type is the chain of the steps' types |


**v0.1 is written** (2026-10-20): the metrics, the counts, one law proved by the
Lean kernel, the numbers against NumPy and scikit-learn. The rest of the table
is the plan; the open issues carry the debt of each step.

**No `mlbend-linalg`.** The base layer already exists and is published:
`bend-kit-csv` (an RFC 4180 reader), `bend-kit-bytes`, `bend-kit-f64` (a soft
float, the numbers scikit-learn defaults to), `bend-kit-int` (U64, I64),
`bend-kit-lemmas` (proved arithmetic) and `bend-kit-parse`; bend-ml has the
tensors (`bend-ml-tensor`, `bend-ml-tensor-array`) and the reverse-mode
autodiff (`bend-ml-autograd`). MLBend writes the *machine-learning* surface and
imports the rest: a second `Mat` and a second set of lemmas would be the third
copy of the same thing in the ecosystem.

**The one gap in the base layer**: Base cannot read a number from a string
(bendlang/bend#1789), and the kit's f64 says so outright (*"Decimal text is not
here"*). The data layer needs it, so v0.2 carries a small decimal parser until
Base has one, and drops it when it does.

## The order, and why

- **Metrics first.** Every later package scores itself with them, and they are
  pure, so they can be proved and tested before anything else exists.
- **The containers before the estimators.** `mlbend-linalg` gives the shapes; the
  estimators are written against `Mat<r, c>` and their mistakes become type
  errors.
- **The split before the models.** A deterministic split (a pure PRNG in the
  package) is what makes every later number reproducible; the partition law is
  the first law that says something about the *data* rather than about arithmetic.
- **The models last.** Each one is a few dozen lines once the containers and the
  split exist, and each earns a law (orthogonality for OLS, one cluster per
  point for k-means).
- **The pipeline and the datasets at the end.** They are the scikit-learn
  surface proper, and they need the estimators to exist.

## How the work carries

The plan is here; the *process* is in [`AGENTS.md`](../AGENTS.md) and
[`docs/agents/`](../agents): one issue per work item, work on `devel`, `make
check` before every push, `NOTES.md` for what was measured and what failed, and
`docs/agents/handoff.md` for where we are.

## Stop criteria

- A package is done when its law is proved (the kernel says so), its numbers
  match NumPy and scikit-learn, and its README says in plain language what it
  proves and what it only tests.
- A claim that cannot be proved yet is **not** in the package: it goes into
  `NOTES.md` as debt, with the experiment that failed, and comes back when the
  machinery to prove it exists.
- v0.9 is done when the iris dataset goes through standardize → logistic
  regression → cross-validation, with the accuracy matching scikit-learn, and
  the README shows the numbers.

## Open questions

- **The containers.** bend-ml's `Mat` over a flat `Array<F32>` (fast, its
  capacity is a claim) or its list-based `Mat` (simple, its length is a claim)?
  For the tabular sizes scikit-learn deals with the list version is enough; for
  anything that grows, the flat one. Measured when v0.3 needs it.
- **The numbers.** `F32` (Base) or `F64` (`bend-kit-f64`, a soft float, and
  slow)? scikit-learn is f64; the kit's f64 has no square root and no
  transcendentals, so the estimators that need them (the softmax, the logistic)
  would be F32 anyway. One precision is a rule; two is a headache.
- **The proved core.** Over `Nat` (as bend-ml's nat-lemmas does) or over a
  rational type? Rational arithmetic would make the standardization laws
  provable as stated, at the price of a new numeric tower.
- **The PRNG.** `splitmix64` over `U32`, seeded by a `Nat`. It has to be pure
  (a seeded generator is a function, not an effect) for the split to be a law.
  **Answered (2026-10-22): `bend-kit-random@0.1.0.1` on the hub** — xoshiro128**,
  pure (a draw returns the value beside the next state), with unbiased ranges
  and Fisher-Yates shuffles. It is someone else's package, so the split uses it
  rather than a generator of our own.
