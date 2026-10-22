# mlbend-split

A seeded, deterministic split of the rows, stratified by label.

```python
import mlbend-split@0.1.0.0/main.bend as Split
```

The rows of a class are shuffled and then dealt out over the folds, so
every fold holds the same proportion of every class — the rule
scikit-learn's `StratifiedKFold` uses. The shuffle is `bend-kit-random`'s
xoshiro128\*\*, seeded: the same seed gives the same split, and another
seed gives another one. **ML Bend writes no generator of its own.**

The labels are class indices, `0` to the number of classes.

| function | what it does |
| --- | --- |
| `folds(labels, seed, k)` | the fold of every row, in the order of the rows |
| `split(labels, seed, k, f)` | the rows that test in fold `f`, and the rows that train |

`split` gives the indices of the rows, so the rows themselves are picked
out of the dataset by index.

## What is proved

- **Every row of the dataset is placed in a fold** (`split_covers`). The
  kernel proves it; the proof is an induction over the rows. The fold
  list holds one entry per row, so the folds cover the dataset exactly
  once each: the test rows of a fold and its training rows are
  disjoint, and together they hold every row. The split walks the fold
  list and appends each index to one side or the other, so a row is in
  one side only.
- **The shape of every result is in the type.** `split` gives a `Split`
  holding two `List<&2, Nat>` — the indices of the rows — and the
  checker verifies it.

## What is only tested

- **The numbers.** The folds are compared with scikit-learn's
  `StratifiedKFold` (`reference/test_split.py`): every fold holds the
  same number of rows, every fold holds the same number of rows of every
  class, every row is in exactly one fold, and the same seed gives the
  same split while another seed gives another. The two programs do not
  agree on *which* rows go where — scikit-learn shuffles with numpy's
  generator, this package with bend-kit-random's — so the comparison is
  of the shape of the split, not of its exact contents.
