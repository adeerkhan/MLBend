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
| **pipeline** | a transformer fitted on the training rows only, then applied to both sides of a fold |
| **the counts** | the `Nat` arithmetic every metric rests on (`count`, `count_true`, `count_pred`, `count_both`, `count_right`) |
| **the shape** | the `r` and `c` of a `Mat<r, c>`: a type parameter, erased at run time, checked by the checker |
| **law** | a statement the kernel proves; the proof is the `def` right below it |
| **debt** | a claim that is not proved yet; it lives in an issue, not in a package |
| **task** | a dataset, a split, a program and the numbers it must produce, with tolerances |
| **adapter** | the code that builds and runs a task's program and parses its output; the trainer enters here |
| **reference** | the same numbers computed with NumPy or scikit-learn |
| **report card** | the metric table: the Bend number, the reference number, the delta, pass or fail |
| **manifest** | the run's provenance: git commit and dirtiness, Bend version, hub hashes, machine, seed, tolerances |

## The packages

| Package (`directory`) | What it is | The shape it uses |
|---|---|---|
| `mlbend-metrics` (`metrics/`) | the evaluation surface: MSE, MAE, RMSE, accuracy, precision, recall, F1, and the counts | `List<&2, F32>` and `List<&2, Nat>` (the values may be read twice) |
| `mlbend-preprocess` (`preprocess/`) | standardize, min-max, one-hot, L1 and L2 normalization | `Mat<r, c>` |
| `mlbend-split` (`split/`) | a seeded, deterministic split of the rows, stratified by label | indices over `Mat<r, c>` |
| `mlbend-select` (`select/`) | k-fold cross-validation on top of the split | indices over `Mat<r, c>` |
| `mlbend-datasets` (`datasets/`) | the datasets of scikit-learn, fetched from HuggingFace or read from the `MLBEND_DATA` cache | `Table` |
| `mlbend-cluster` (`cluster/`) | k-means, functional: the centres after a fixed number of rounds | `List<&2, F32>` rows |
| `mlbend-pipeline` (`pipeline/`) | fit on the training rows only, transform both sides of a fold | `List<&2, List<&2, F32>>` rows |
| `mlbend-report` (`report/`) | the confusion matrix and the per-class report: precision, recall, F1, support, macro and weighted averages | `List<&2, Nat>` labels |
| `mlbend-huggingface` (`hf/`) | the HuggingFace client the datasets package reads through | bytes and JSON |

All of them are one repository: each package is a directory beside this file,
and `bend.toml` at the root is the workspace that lists them.

The training library is **`bendtorch@1.0.1.0`** (`nuxyel/BendTorch`), which has
`Attention`, `RMSNorm`, `Embedding`, `CrossEntropy`, `AdamW` and `Train`. The old `bendtorch-core@0.1.0.0` line and its three examples (`tape`,
`regression`, `stacks`) were retired on 2026-10-24; nothing imports it. There is no
`mlbend-linalg`: the tensors and the lemmas are `bend-ml`'s and the kit's.
There is no `mlbend-linear` trainer: linear, ridge and logistic regression enter
as adapters over BendTorch or bend-ml, not as a second implementation.

The order and the stop criteria are in `docs/PLAN.md`.

## The design rules, and why each one is there

1. **A fitted thing is a value, not an object.** `s.transform(y)`, `m.predict(x2)`.
   Bend's affine rule makes the mutating shape impossible, and the value shape
   is better anyway: a fitted transformer or model can be passed, stored and
   threaded, and its *type* says what it was fitted to.
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
7. **No trainer here.** No tensor, no autodiff, no optimizer, no second
   `linear`/`ridge`/`logistic`. A model enters through an adapter; the only
   models in this repository are gradient-free baselines.
8. **A number carries its manifest.** A score without the dataset, seed, split,
   versions and machine is not a result.

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
