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
| 0.2 | `mlbend-linalg` | `Vec<n>`, `Mat<r, c>` over lists, with the inner-product lemmas proved | the dot product is symmetric; the sum of a list is its length times its head |
| 0.3 | `mlbend-preprocess` | standardize, min-max, one-hot, L1 and L2 normalization | one-hot is a unit vector; the standardizer's output has mean 0 and variance 1 (over exact arithmetic) |
| 0.4 | `mlbend-split` | a seeded, deterministic train/test split, stratified by label | the split is a partition: the sizes add up and no index is in both |
| 0.5 | `mlbend-linear` | ordinary least squares, ridge, logistic regression | the residual of the fit is orthogonal to the design |
| 0.6 | `mlbend-cluster` | k-means, functional (a fixed number of iterations) | every point is in exactly one cluster, and the assignment is a function of the point |
| 0.7 | `mlbend-pipeline` | a typed chain of transformers and estimators | the chain's type is the chain of the steps' types |
| 0.8 | `mlbend-datasets` | the fetchers (iris, breast cancer, california housing) as effects | none: an effect, tested on both lanes |
| 0.9 | `mlbend-select` | k-fold cross-validation on top of `mlbend-split` | the folds partition the data, like the split |

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

- **The containers.** `mlbend-linalg` over lists, or bend-ml's `Mat` over a flat
  `Array<F32>`? The list version is simple and its length is a claim; the array
  version is fast and its capacity is a claim. v0.2 measures both before
  choosing.
- **The proved core.** Over `Nat` (as bend-ml's nat-lemmas does) or over a
  rational type? Rational arithmetic would make the standardization laws
  provable as stated, at the price of a new numeric tower.
- **The PRNG.** `splitmix64` over `U32`, seeded by a `Nat`. It has to be pure
  (a seeded generator is a function, not an effect) for the split to be a law.
