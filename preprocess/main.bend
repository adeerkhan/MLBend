# mlbend-preprocess: the transformers of scikit-learn, over bend-ml's matrix.
#
#   import mlbend-preprocess@0.1.0.0/main.bend as P
#
# A transformer takes a matrix and gives a matrix of the same shape. The
# matrix is bend-ml's: `Mat<r, c>` from `bend-ml-tensor`, the one
# bend-ml's autodiff takes, so the estimators that follow can be handed
# to a model without a copy. MLBend writes no matrix of its own.
#
# The standardiser is bend-ml's own arithmetic: a column-wise
# normalisation is a transpose, then `Mat.layernorm` (which centres a
# row and scales it by its own standard deviation), then a transpose
# back. The min-max scaler and the normalisers use `Vec.dot`,
# `Vec.scale` and `Mat.to_rows`.
#
# What is proved: the number of rows of a one-hot matrix, and the shape
# of every result (the type says it, and the checker verifies it in
# every operation). What is tested: the numbers, against NumPy and
# scikit-learn (reference/test_preprocess.py). F32 is not a real
# number, so a value is never proved, only tested.

import Base
import bend-ml-tensor@0.1.0.0/main.bend as T

# The standardiser: every column centred on its mean and scaled by its
# own standard deviation.


# LAW (one_hot_count_length): the one-hot matrix has one row for every
# label.
law one_hot_count_length:
  for ys: List<&2, Nat>
  {one_hot_count_of(ys) == List.length(&2, Nat, ys) : Nat}

# The extremes of every column: the smallest and the largest.
type Bounds is Data:
  Bounds{mins: List<&2, F32>, maxs: List<&2, F32>}

# The largest of two rows, cell by cell.
def bounds_max_go(maxs: List<&2, F32>, row: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match maxs:
    case Nil{}:
      acc
    case Con{a, ta}:
      match row:
        case Con{b, tb}:
          bounds_max_go(ta, tb, List.append(&2, F32, acc, Con{F32.max(a, b), Nil{}}))
        case Nil{}:
          acc

# The largest of two rows, cell by cell.
def bounds_max(maxs: List<&2, F32>, row: List<&2, F32>) -> List<&2, F32>:
  bounds_max_go(maxs, row, Nil{})

# The smallest of two rows, cell by cell.
def bounds_min_go(mins: List<&2, F32>, row: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match mins:
    case Nil{}:
      acc
    case Con{a, ta}:
      match row:
        case Con{b, tb}:
          bounds_min_go(ta, tb, List.append(&2, F32, acc, Con{F32.min(a, b), Nil{}}))
        case Nil{}:
          acc

# The smallest of two rows, cell by cell.
def bounds_min(mins: List<&2, F32>, +row: List<&2, F32>) -> List<&2, F32>:
  bounds_min_go(mins, row, Nil{})

# The extremes of every column, row by row.
def bounds_of_go(+rows: List<&2, List<&2, F32>>, +mins: List<&2, F32>, +maxs: List<&2, F32>) ->
  Bounds:
  match rows:
    case Nil{}:
      Bounds{mins, maxs}
    case Con{+row, t}:
      bounds_of_go(t, bounds_min(mins, row), bounds_max(maxs, row))

# The extremes of every column, row by row.
def bounds_of_rows(+rows: List<&2, List<&2, F32>>) -> Bounds:
  match rows:
    case Nil{}:
      Bounds{Nil{}, Nil{}}
    case Con{+row, t}:
      bounds_of_go(t, row, row)

# The extremes of every column of a matrix.
def bounds_of(+r: Nat, +c: Nat, a: T.Mat<r, c>) -> Bounds:
  bounds_of_rows(T.Mat.to_rows(r, c, a))

# The row, divided, cell by cell.
def l1_finish_go(acc: List<&2, F32>, +k: F32, +out: List<&2, F32>) -> List<&2, F32>:
  match acc:
    case Nil{}:
      out
    case Con{h, t}:
      l1_finish_go(t, k, List.append(&2, F32, out, Con{F32.mul(h, k), Nil{}}))

# The row, divided.
def l1_finish(acc: List<&2, F32>, total: F32) -> List<&2, F32>:
  l1_finish_go(acc, (1.0 / F32.max(total, 0.00000001) : F32), Nil{})

# One row divided by the sum of its absolute values, cell by cell.
def l1_row_go(+row: List<&2, F32>, +acc: List<&2, F32>, +total: F32) -> List<&2, F32>:
  match row:
    case Nil{}:
      l1_finish(acc, total)
    case Con{+h, t}:
      l1_row_go(t, List.append(&2, F32, acc, Con{h, Nil{}}), F32.add(total, F32.abs(h)))

# One row divided by the sum of its absolute values.
def l1_row(+row: List<&2, F32>) -> List<&2, F32>:
  l1_row_go(row, Nil{}, 0.0)

# The rows of a matrix, each divided by the sum of its absolute values.
def l1_normalize_go(+rows: List<&2, List<&2, F32>>, +acc: List<&2, List<&2, F32>>) ->
  List<&2, List<&2, F32>>:
  match rows:
    case Nil{}:
      acc
    case Con{row, t}:
      l1_normalize_go(t, List.append(&2, List<&2, F32>, acc, Con{l1_row(row), Nil{}}))

# A matrix of r rows of c numbers, when there is one.
def mat_of_(+r: Nat, +c: Nat, m: Maybe<&2, T.Mat<r, c>>) -> T.Mat<r, c>:
  match m:
    case Some{v}:
      v
    case None{}:
      T.Mat.fill(r, c, 0.0)

# A matrix of r rows of c numbers. When the rows do not fit, the
# matrix is empty: a shape that does not hold is not a matrix.
def mat_of(+r: Nat, +c: Nat, +rows: List<&2, List<&2, F32>>) -> T.Mat<r, c>:
  mat_of_(r, c, T.Mat.of(r, c, rows))

# The L1 normaliser: every row divided by the sum of its absolute
# values.
def l1_normalize(+r: Nat, +c: Nat, a: T.Mat<r, c>) -> T.Mat<r, c>:
  mat_of(r, c, l1_normalize_go(T.Mat.to_rows(r, c, a), Nil{}))

# The row, divided, cell by cell.
def l2_finish_go(acc: List<&2, F32>, +norm: F32, +out: List<&2, F32>) -> List<&2, F32>:
  match acc:
    case Nil{}:
      out
    case Con{h, t}:
      l2_finish_go(t, norm, List.append(&2, F32, out, Con{(h / F32.max(norm, 0.00000001) : F32), Nil{}}))

# The length of a row, cell by cell.
def l2_norm_go(acc: List<&2, F32>, +acc_sum: F32) -> F32:
  match acc:
    case Nil{}:
      F32.sqrt(acc_sum)
    case Con{+h, t}:
      l2_norm_go(t, F32.add(acc_sum, F32.mul(h, h)))

# The length of a row.
def l2_norm(acc: List<&2, F32>) -> F32:
  l2_norm_go(acc, 0.0)

# The row, divided by its length.
def l2_finish(+acc: List<&2, F32>) -> List<&2, F32>:
  l2_finish_go(acc, l2_norm(acc), Nil{})

# One row divided by the length of its arrow, cell by cell.
def l2_row_go(+row: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match row:
    case Nil{}:
      l2_finish(acc)
    case Con{h, t}:
      l2_row_go(t, List.append(&2, F32, acc, Con{h, Nil{}}))

# One row divided by the length of its arrow.
def l2_row(+row: List<&2, F32>) -> List<&2, F32>:
  l2_row_go(row, Nil{})

# The rows of a matrix, each divided by the length of its arrow.
def l2_normalize_go(+rows: List<&2, List<&2, F32>>, +acc: List<&2, List<&2, F32>>) ->
  List<&2, List<&2, F32>>:
  match rows:
    case Nil{}:
      acc
    case Con{row, t}:
      l2_normalize_go(t, List.append(&2, List<&2, F32>, acc, Con{l2_row(row), Nil{}}))

# The L2 normaliser: every row divided by the length of its arrow.
def l2_normalize(+r: Nat, +c: Nat, +a: T.Mat<r, c>) -> T.Mat<r, c>:
  mat_of(r, c, l2_normalize_go(T.Mat.to_rows(r, c, a), Nil{}))

# One row scaled column by column, cell by cell.
def scale_row_go(+row: List<&2, F32>, mins: List<&2, F32>, maxs: List<&2, F32>,
  +acc: List<&2, F32>) -> List<&2, F32>:
  match row:
    case Nil{}:
      acc
    case Con{h, t}:
      match mins:
        case Nil{}:
          acc
        case Con{+m, ms}:
          match maxs:
            case Nil{}:
              acc
            case Con{mx, mxs}:
              scale_row_go(t, ms, mxs,
                List.append(&2, F32, acc,
                  Con{((h - m : F32) / F32.max((mx - m : F32), 0.00000001) : F32), Nil{}}))

# One row scaled column by column.
def scale_row(+row: List<&2, F32>, mins: List<&2, F32>, maxs: List<&2, F32>) -> List<&2, F32>:
  scale_row_go(row, mins, maxs, Nil{})

# A matrix scaled column by column, row by row.
def scaled_go(+rows: List<&2, List<&2, F32>>, +mins: List<&2, F32>, +maxs: List<&2, F32>,
  +acc: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match rows:
    case Nil{}:
      acc
    case Con{row, t}:
      scaled_go(t, mins, maxs,
        List.append(&2, List<&2, F32>, acc, Con{scale_row(row, mins, maxs), Nil{}}))

# A matrix scaled column by column.
def scaled_of(+r: Nat, +c: Nat, a: T.Mat<r, c>, +b: Bounds) -> List<&2, List<&2, F32>>:
  match b:
    case Bounds{mins, maxs}:
      scaled_go(T.Mat.to_rows(r, c, a), mins, maxs, Nil{})

# A matrix scaled column by column: every cell is (x - min) / (max - min).
def scaled(+r: Nat, +c: Nat, a: T.Mat<r, c>, +b: Bounds) -> List<&2, List<&2, F32>>:
  scaled_of(r, c, a, b)

# The min-max scaler: every column mapped onto [0, 1].
def min_max(+r: Nat, +c: Nat, +a: T.Mat<r, c>) -> T.Mat<r, c>:
  mat_of(r, c, scaled(r, c, a, bounds_of(r, c, a)))

# A list with its nth element replaced, element by element.
def replace_at_go(+xs: List<&2, F32>, +n: Nat, +v: F32, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, +t}:
      match n:
        case 0n:
          List.append(&2, F32, acc, Con{v, t})
        case _:
          replace_at_go(t, Nat.sub(n, 1n), v, List.append(&2, F32, acc, Con{h, Nil{}}))

# A list with its nth element replaced.
def replace_at(+xs: List<&2, F32>, n: Nat, v: F32) -> List<&2, F32>:
  replace_at_go(xs, n, v, Nil{})

# One row of a one-hot matrix: a zero in every column but the label's.
def one_hot_row(+label: Nat, k: Nat) -> List<&2, F32>:
  replace_at(List.replicate(F32, k, 0.0), label, 1.0)

# The rows of a one-hot matrix, label by label.
def one_hot_rows_go(+labels: List<&2, Nat>, +k: Nat, +acc: List<&2, List<&2, F32>>) ->
  List<&2, List<&2, F32>>:
  match labels:
    case Nil{}:
      acc
    case Con{h, t}:
      one_hot_rows_go(t, k, List.append(&2, List<&2, F32>, acc, Con{one_hot_row(h, k), Nil{}}))

# The rows of a one-hot matrix, one for every label.
def one_hot_rows(+labels: List<&2, Nat>, +k: Nat) -> List<&2, List<&2, F32>>:
  one_hot_rows_go(labels, k, Nil{})

# The one-hot encoding: a row of k numbers for every label, with a one
# in the column of the label. It is Nothing when there are not exactly
# n labels.
def one_hot(+k: Nat, +n: Nat, labels: List<&2, Nat>) -> Maybe<&2, T.Mat<n, k>>:
  T.Mat.of(n, k, one_hot_rows(labels, k))

# The number of rows of a one-hot matrix, counted.
def one_hot_count_go(+labels: List<&2, Nat>, +acc: Nat) -> Nat:
  match labels:
    case Nil{}:
      acc
    case Con{h, t}:
      one_hot_count_go(t, Nat.add(acc, 1n))

# The number of rows of a one-hot matrix.
def one_hot_count_of(ys: List<&2, Nat>) -> Nat:
  match ys:
    case Nil{}:
      0n
    case Con{h, t}:
      (1n + one_hot_count_of(t) : Nat)

# the one-hot matrix has one row for every label
def one_hot_count_length(ys):
  match ys:
    case Nil{}:
      {==}
    case Con{h, +t}:
      %one_hot_count_length(t) : {1n+one_hot_count_of(t) == 1n+_ : Nat}
      {==}

def standardize(+r: Nat, +c: Nat, a: T.Mat<r, c>) -> T.Mat<r, c>:
  T.Mat.transpose(c, r,
    T.Mat.layernorm(c, r, 0.00000001, T.Vec.fill(r, 1.0), T.Vec.fill(r, 0.0),
      T.Mat.transpose(r, c, a)))