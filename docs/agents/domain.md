# The domain

One context: this file is the map of the packages, the vocabulary and the
design rules. An agent that reads it knows what a "transformer" means here and
what the shape rules are, before it reads a line of Bend.

## The vocabulary

| Word | What it means here |
|---|---|
| **metric** | a function of two equally long sequences that answers a number (and why it could not) |
| **transformer** | a fitted value that changes the data: `s = Scaler.fit(x)`, then `s.transform(y)` |
| **estimator** | a fitted value that predicts: `m = Linear.fit(x, y)`, then `m.predict(x2)` |
| **pipeline** | a chain of transformers and an estimator, whose types are checked |
| **the counts** | the `Nat` arithmetic every metric rests on (`count`, `count_true`, `count_pred`, `count_both`, `count_right`) |
| **the shape** | the `r` and `c` of a `Mat<r, c>`: a type parameter, erased at run time, checked by the checker |
| **law** | a statement the kernel proves; the proof is the `def` right below it |
| **debt** | a claim that is not proved yet; it lives in an issue, not in a package |

## The packages

| Package | What it is | The shape it uses |
|---|---|---|
| `mlbend-metrics` | the evaluation surface: MSE, MAE, RMSE, accuracy, precision, recall, F1, and the counts | `List<&2, F32>` and `List<&2, Nat>` (the values may be read twice) |
| `mlbend-linalg` (v0.2) | `Vec<n>`, `Mat<r, c>` and the inner-product lemmas | `Mat<r, c>` |
| `mlbend-preprocess` (v0.3) | standardize, min-max, one-hot, L1 and L2 normalization | `Mat<r, c>` |
| `mlbend-split` (v0.4) | a seeded, deterministic train/test split | indices over `Mat<r, c>` |
| `mlbend-linear` (v0.5) | ordinary least squares, ridge, logistic regression | `Model<c, k>` |
| `mlbend-cluster` (v0.6) | k-means, functional | `Vec<n>` |
| `mlbend-pipeline` (v0.7) | a typed chain of the above | the chain's types |
| `mlbend-datasets` (v0.8) | the fetchers, as effects | `Mat<r, c>` |
| `mlbend-select` (v0.9) | k-fold cross-validation | over the split |

The order and the stop criteria are in `docs/PLAN.md`.

## The design rules, and why each one is there

1. **An estimator is a value, not an object.** `m = Linear.fit(x, y)` and then
   `m.predict(x2)`. Bend's affine rule makes the mutating shape impossible, and
   the value shape is better anyway: a fitted model can be passed, stored and
   threaded, and the *type* of the model says what it was fitted to.
2. **The shape is in the type.** `Mat<r, c>`: the dimensions are erased type
   parameters, so they cost nothing at run time, but the checker verifies them
   in every operation. A model applied to the wrong number of features does not
   compile.
3. **Structure is proved, numbers are tested.** Sizes, partitions and
   compositions are laws; `F32` values are compared with NumPy and scikit-learn.
   `F32` is not a real number, so no value is ever proved.
4. **Effects are at the edges.** Fetching data, reading a file, seeding a
   generator. The estimators are pure, so a run is reproducible and a proof is
   about the code, not about the world.
5. **An error is a value.** A metric answers `Done{value}` or `Fail{...}`; a
   length that does not match is a `Mismatch`, and an empty input is `Empty`.
6. **A seeded generator is a function, not an effect.** The split is a law
   ("the split is a partition"), so the generator has to be pure: `splitmix64`
   over `U32`, seeded by a `Nat`.

## The types the packages share

```python
# the counts: over Nat, so that they can be proved
type Bad is Data:            # why a metric could not answer
  Mismatch{expected: Nat, got: Nat}
  Empty{}

type Sum is Data:            # a sum and the number of terms beside it
  Sum{n: Nat, s: F32}
```

A package defines its own types next to its own operations; a type that two
packages need moves to the package below them.

## The conventions

- A package is one file, published to BendHub under `mlbend-*`, importing only
  Base and the packages below it, at pinned versions.
- The entry file is `<pkg>/main.bend`; the tests are `<pkg>/tests/`; the
  reference comparisons are in `reference/`.
- The public names of a package are plain (`mse`, `accuracy`), the way
  `bend-ml-nat-lemmas` names them (`add_comm`, `length_append`).
- A file's first lines say what the package is, how to import it, and what is
  proved and what is tested.
