# mlbend-pipeline: fit on the training rows, transform both sides.
#
#   import mlbend-pipeline@0.1.0.0/main.bend as Pipeline
#
# The leakage an experiment layer exists to prevent: standardising the
# whole dataset before the folds are made, so the scaler has already seen
# the test rows. Here `fit` takes the training rows and nothing else, and
# `transform` applies the fitted means and deviations to any rows.
#
#   +scaler = Pipeline.fit(train_rows)        # train only
#   +train  = Pipeline.transform(scaler, train_rows)
#   +test   = Pipeline.transform(scaler, test_rows)
#
# or, for one fold, `Pipeline.fold(train_rows, test_rows)`, which returns
# both sides already transformed.
#
# What is proved: transforming the rows keeps one output row for every
# input row (`pipeline_preserves_rows`) — no row is dropped and none is
# invented. What is checked: the means, the deviations and the transformed
# values against scikit-learn's `StandardScaler` fitted on the same
# training rows (reference/test_pipeline.py), and the cross-validation
# against scikit-learn's `Pipeline(StandardScaler, LogisticRegression)`
# (reference/test_crossval.py).
#
# The no-leakage property is the API itself: the only rows `fit` can see
# are the ones it is given, and `fold` gives it the training rows. A law
# cannot prove what a caller passes; the tests and the cross-validation
# comparison are what hold that end.

import Base

# A fitted scaler: the mean and the standard deviation of every column.

type Scaler is Data:
  Scaler{means: List<&2, F32>, stds: List<&2, F32>}

# The two sides of one fold, transformed.

type Prepared is Data:
  Prepared{train: List<&2, List<&2, F32>>, test: List<&2, List<&2, F32>>}

# LAW (pipeline_preserves_rows): transforming the rows keeps one output row
# for every input row — no row is dropped and none is invented.
law pipeline_preserves_rows:
  for +scaler: Scaler
  for rows: List<&2, List<&2, F32>>
  {count_rows(transform(scaler, rows)) == count_rows(rows) : Nat}

# How many rows there are.
def count_rows(rows: List<&2, List<&2, F32>>) -> Nat:
  match rows:
    case Nil{}:
      0n
    case Con{h, t}:
      (1n + count_rows(t) : Nat)

# Two rows, added cell by cell.
def zip_add(a: List<&2, F32>, b: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      match b:
        case Con{hb, tb}:
          Con{F32.add(ha, hb), zip_add(ta, tb)}
        case Nil{}:
          Nil{}

# Two rows, cell by cell, the first minus the second.
def zip_sub(+a: List<&2, F32>, b: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      match b:
        case Con{hb, tb}:
          Con{F32.sub(ha, hb), zip_sub(ta, tb)}
        case Nil{}:
          Nil{}

# Two rows, cell by cell, the first divided by the second.
def zip_div(a: List<&2, F32>, b: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      match b:
        case Con{hb, tb}:
          Con{(ha / F32.max(hb, 0.0001) : F32), zip_div(ta, tb)}
        case Nil{}:
          Nil{}

# One row and a number, cell by cell.
def zip_scale(a: List<&2, F32>, +k: F32) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      Con{F32.mul(ha, k), zip_scale(ta, k)}

# The cells of a row, squared.
def squares(+a: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{+ha, ta}:
      Con{F32.mul(ha, ha), squares(ta)}

# The sum of every column, over the rows.
def column_sums_go(rows: List<&2, List<&2, F32>>, +acc: List<&2, F32>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      column_sums_go(t, zip_add(acc, h))

# The sum of every column, over the rows.
def column_sums(rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      column_sums_go(t, h)

# The mean of every column.
def means_of(+rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      zip_scale(column_sums(rows),
        (1.0 / F32.from_nat(count_rows(rows)) : F32))

# The sum of the squared deviations of every column.
def deviation_sums_go(rows: List<&2, List<&2, F32>>, +means: List<&2, F32>,
  +acc: List<&2, F32>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      deviation_sums_go(t, means, zip_add(acc, squares(zip_sub(h, means))))

# The sum of the squared deviations of every column.
def deviation_sums(rows: List<&2, List<&2, F32>>, +means: List<&2, F32>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      deviation_sums_go(t, means, squares(zip_sub(h, means)))

# The square root of every cell of a row.
def roots(a: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      Con{F32.sqrt(ha), roots(ta)}

# The standard deviation of every column: the population one, as
# scikit-learn's StandardScaler uses.
def stds_of(+rows: List<&2, List<&2, F32>>, +means: List<&2, F32>) -> List<&2, F32>:
  roots(zip_scale(deviation_sums(rows, means),
    (1.0 / F32.from_nat(count_rows(rows)) : F32)))

# The scaler of the training rows, and nothing else.
def fit(+rows: List<&2, List<&2, F32>>) -> Scaler:
  +means = means_of(rows)
  +stds = stds_of(rows, means)
  Scaler{means, stds}

# One row, centred and scaled by the fitted values.
def transform_row(scaler: Scaler, row: List<&2, F32>) -> List<&2, F32>:
  match scaler:
    case Scaler{means, stds}:
      zip_div(zip_sub(row, means), stds)

# The rows, centred and scaled by the fitted values.
def transform(+scaler: Scaler, rows: List<&2, List<&2, F32>>) ->
  List<&2, List<&2, F32>>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{transform_row(scaler, h), transform(scaler, t)}

# One fold: fit on the training rows, transform both sides.
def fold(+train: List<&2, List<&2, F32>>, test: List<&2, List<&2, F32>>) -> Prepared:
  +scaler = fit(train)
  Prepared{transform(scaler, train), transform(scaler, test)}

# transforming the rows keeps one output row for every input row
def pipeline_preserves_rows(scaler, rows):
  match rows:
    case Nil{}:
      {==}
    case Con{h, +t}:
      %pipeline_preserves_rows(scaler, t) : {1n+count_rows(transform(scaler, t)) == 1n+_ : Nat}
      {==}
