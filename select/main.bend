# mlbend-select: k-fold cross-validation on top of the split.
#
#   import mlbend-select@0.1.0.0/main.bend as Select
#
# The folds are the ones `mlbend-split` makes, so the two agree by
# construction: the same seed and the same labels give the same folds.
# The package adds the walk over the folds — the rows of each fold, the
# training rows of each fold, and the sizes — so a model can be fitted
# and scored fold by fold without writing the loop again.
#
# What is proved: every row of the dataset is in exactly one fold
# (`select_covers`) — the fold list holds one entry per row, so the
# folds cover the dataset exactly once each. What is tested: that the
# fold sizes and the class counts agree with scikit-learn's
# `StratifiedKFold` (reference/test_select.py).

import Base
import mlbend-split@0.1.1.0/main.bend as Split

# The rows of one fold: the ones that test and the ones that train.

type Fold is Data:
  Fold{test: List<&2, Nat>, train: List<&2, Nat>}

# LAW (select_covers): every row of the dataset is in exactly one fold —
# the fold list holds one entry per row, so the folds cover the dataset
# exactly once each.
law select_covers:
  for labels: List<&2, Nat>
  for +orders: List<&2, List<&2, Nat>>
  for +k: Nat
  for +i: Nat
  {Split.count_rows(Split.folds_from(labels, orders, k, i)) == List.length(&2, Nat, labels) : Nat}

# The rows of the fold f: the ones that test and the ones that train.
def fold_of(+fs: List<&2, Nat>, +f: Nat) -> Fold:
  Fold{Split.test_of(fs, f), Split.train_of(fs, f)}

# The number of rows of the fold f.
def fold_size(fs: List<&2, Nat>, f: Nat) -> Nat:
  List.length(&2, Nat, Split.test_of(fs, f))

# The number of rows of each fold.
def fold_sizes_go(+fs: List<&2, Nat>, k: Nat, +acc: List<&2, Nat>) -> List<&2, Nat>:
  match k:
    case 0n:
      acc
    case 1n++rest:
      fold_sizes_go(fs, rest, Con{fold_size(fs, rest), acc})

# The number of rows of each fold.
def fold_sizes(fs: List<&2, Nat>, k: Nat) -> List<&2, Nat>:
  fold_sizes_go(fs, k, Nil{})

# The fold of every row of the dataset, in the order of the rows.
def folds(labels: List<&2, Nat>, seed: U32, k: Nat) -> List<&2, Nat>:
  Split.folds(labels, seed, k)

# The rows of every fold.
def folds_go(+fs: List<&2, Nat>, k: Nat, +acc: List<&2, Fold>) -> List<&2, Fold>:
  match k:
    case 0n:
      acc
    case 1n++rest:
      folds_go(fs, rest, Con{fold_of(fs, rest), acc})

# The rows of every fold: the test rows of each fold, and the training
# rows of each fold.
def folds_of(labels: List<&2, Nat>, seed: U32, +k: Nat) -> List<&2, Fold>:
  folds_go(Split.folds(labels, seed, k), k, Nil{})

# every row of the dataset is in exactly one fold
def select_covers(labels, orders, k, i):
  match labels:
    case Nil{}:
      {==}
    case Con{h, +t}:
      %select_covers(t, orders, k, Nat.add(i, 1n)) : {1n+Split.count_rows(Split.folds_from(t, orders, k, Nat.add(i, 1n))) == 1n+_ : Nat}
      {==}