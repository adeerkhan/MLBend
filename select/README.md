# mlbend-select

k-fold cross-validation on top of the split.

```python
import mlbend-select@0.1.0.0/main.bend as Select
```

The folds are the ones `mlbend-split` makes, so the two agree by
construction: the same seed and the same labels give the same folds.
This package adds the walk over the folds — the rows of each fold, the
training rows of each fold, and the sizes — so a model can be fitted and
scored fold by fold without writing the loop again.

| function | what it does |
| --- | --- |
| `folds(labels, seed, k)` | the fold of every row, in the order of the rows |
| `fold_of(fs, f)` | the rows that test in fold `f`, and the rows that train |
| `folds_of(labels, seed, k)` | the rows of every fold |
| `fold_size(fs, f)` | the number of rows in fold `f` |
| `fold_sizes(fs, k)` | the number of rows in each fold |

`fs` is the fold list `folds` gives, so the folds are computed once and
asked of as many times as needed.

## What is proved

- **Every row of the dataset is in exactly one fold** (`select_covers`).
  Bend's checker proves it; the proof is an induction over the rows. The
  fold list holds one entry per row, so the folds cover the dataset
  exactly once each: the test rows of a fold and its training rows are
  disjoint, and together they hold every row.
- **The shape of every result is in the type.** A fold is a `Fold`
  holding two `List<&2, Nat>` — the indices of the rows — and the
  checker verifies it.

## What the verdict says

The Lean kernel's verdict for this package is a **refusal to certify**:
`mlbend-split` depends on `bend-kit-random`, which carries two defs that
call the operating system (`entropy` and `from_os`). The verdict
refuses a book that holds any foreign code, and it cannot tell that
nothing here calls them. So the law above is *checked* by Bend's
checker, not certified by the kernel.

## What is only tested

- **The numbers.** The folds are compared with scikit-learn's
  `StratifiedKFold` (`reference/test_select.py`): every fold holds the
  same number of rows, every fold holds the same number of rows of every
  class, every row is in exactly one fold, and the same seed gives the
  same split while another seed gives another. The two programs do not
  agree on *which* rows go where — scikit-learn shuffles with numpy's
  generator, this package with bend-kit-random's — so the comparison is
  of the shape of the split, not of its exact contents.
