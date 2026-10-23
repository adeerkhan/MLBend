# Principal component analysis over the wine dataset, fetched from
# HuggingFace at run time.
#
#   bendenv run -- bend examples/pca/main.bend
#
# The first principal component of the standardised features, by power
# iteration on the covariance matrix: start from a fixed vector, multiply
# by the covariance and normalise, a hundred times. The explained
# variance ratio is the Rayleigh quotient of that vector over the trace.
#
# It is a transformer, not a trainer: no gradients, no library, and the
# same rows give the same component. `reference/test_pca.py` compares the
# ratio with scikit-learn's PCA on the same standardised data.
import Base
import mlbend-datasets@0.1.4.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf

# The features and the rounds of the power iteration.

# The cells of a list, from index i on.
def cell_at_go(xs: List<&2, F32>, +k: Nat, +i: Nat) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{+h, t}:
      Bool.pick(F32, Nat.is_eq(i, k), h, cell_at_go(t, k, Nat.add(i, 1n)))

# The cell at an index, or 0 when the row is short.
def cell_at(xs: List<&2, F32>, k: Nat) -> F32:
  cell_at_go(xs, k, 0n)

# One row and a number, cell by cell.
def zip_scale(a: List<&2, F32>, +k: F32) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      Con{F32.mul(ha, k), zip_scale(ta, k)}

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
        (1.0 / F32.from_nat(List.length(&2, List<&2, F32>, rows)) : F32))

# The cells of a row, squared.
def squares(+a: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{+ha, ta}:
      Con{F32.mul(ha, ha), squares(ta)}

# The sum of the squares of the deviations, row by row.
def deviation_sums_go(rows: List<&2, List<&2, F32>>, +means: List<&2, F32>,
  +acc: List<&2, F32>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      deviation_sums_go(t, means, zip_add(acc, squares(zip_sub(h, means))))

# The sum of the squares of the deviations of every column.
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

# The standard deviation of every column.
def stds_of(+rows: List<&2, List<&2, F32>>, means: List<&2, F32>) -> List<&2, F32>:
  roots(zip_scale(deviation_sums(rows, means),
    (1.0 / F32.from_nat(List.length(&2, List<&2, F32>, rows)) : F32)))

# One row, centred and scaled.
def scale_row(row: List<&2, F32>, means: List<&2, F32>, stds: List<&2, F32>) -> List<&2, F32>:
  zip_div(zip_sub(row, means), stds)

# The rows, one at a time.
def standardise_go(+rows: List<&2, List<&2, F32>>, +means: List<&2, F32>, +stds: List<&2, F32>,
  +acc: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      standardise_go(t, means, stds,
        List.append(&2, List<&2, F32>, acc, Con{scale_row(h, means, stds), Nil{}}))

# The rows with every column centred and scaled.
def standardise(+rows: List<&2, List<&2, F32>>, means: List<&2, F32>, stds: List<&2, F32>) ->
  List<&2, List<&2, F32>>:
  standardise_go(rows, means, stds, Nil{})

# One column, as a list.
def column_go(rows: List<&2, List<&2, F32>>, +j: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      column_go(t, j, List.append(&2, F32, acc, Con{cell_at(h, j), Nil{}}))

# One column, as a list.
def column(rows: List<&2, List<&2, F32>>, j: Nat) -> List<&2, F32>:
  column_go(rows, j, Nil{})

# Every column, in the order of the features.
def columns_go(+rows: List<&2, List<&2, F32>>, +n: Nat, +j: Nat,
  +acc: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match n:
    case 0n:
      acc
    case 1n+rest:
      columns_go(rows, rest, Nat.add(j, 1n),
        List.append(&2, List<&2, F32>, acc, Con{column(rows, j), Nil{}}))

# Every column, in the order of the features.
def columns(rows: List<&2, List<&2, F32>>, n: Nat) -> List<&2, List<&2, F32>>:
  columns_go(rows, n, 0n, Nil{})

# The dot product of two columns.
def dot_go(a: List<&2, F32>, b: List<&2, F32>, +acc: F32) -> F32:
  match a:
    case Nil{}:
      acc
    case Con{+ha, ta}:
      match b:
        case Con{+hb, tb}:
          dot_go(ta, tb, F32.add(acc, F32.mul(ha, hb)))
        case Nil{}:
          acc

# The dot product of two columns.
def dot(a: List<&2, F32>, b: List<&2, F32>) -> F32:
  dot_go(a, b, 0.0)

# The norm of a column.
def norm(+xs: List<&2, F32>) -> F32:
  F32.sqrt(dot(xs, xs))

# A column, scaled to unit length.
def normalise(+xs: List<&2, F32>) -> List<&2, F32>:
  zip_scale(xs, (1.0 / F32.max(norm(xs), 0.000000001) : F32))

# One covariance row: the covariance of column a with every column.
def cov_row_go(columns: List<&2, List<&2, F32>>, +a: List<&2, F32>, +n: F32,
  +acc: List<&2, F32>) -> List<&2, F32>:
  match columns:
    case Nil{}:
      acc
    case Con{c, t}:
      cov_row_go(t, a, n,
        List.append(&2, F32, acc, Con{F32.div(dot(a, c), n), Nil{}}))

# One covariance row.
def cov_row(columns: List<&2, List<&2, F32>>, a: List<&2, F32>, n: F32) -> List<&2, F32>:
  cov_row_go(columns, a, n, Nil{})

# The covariance matrix, row by row.
def cov_go(columns: List<&2, List<&2, F32>>, +all: List<&2, List<&2, F32>>, +n: F32,
  +acc: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match columns:
    case Nil{}:
      acc
    case Con{a, t}:
      cov_go(t, all, n, List.append(&2, List<&2, F32>, acc, Con{cov_row(all, a, n), Nil{}}))

# The covariance matrix.
def covariance(+columns: List<&2, List<&2, F32>>, n: F32) -> List<&2, List<&2, F32>>:
  cov_go(columns, columns, n, Nil{})

# The product of a matrix and a column.
def mat_vec_go(matrix: List<&2, List<&2, F32>>, +v: List<&2, F32>,
  +acc: List<&2, F32>) -> List<&2, F32>:
  match matrix:
    case Nil{}:
      acc
    case Con{row, t}:
      mat_vec_go(t, v, List.append(&2, F32, acc, Con{dot(row, v), Nil{}}))

# The product of a matrix and a column.
def mat_vec(matrix: List<&2, List<&2, F32>>, v: List<&2, F32>) -> List<&2, F32>:
  mat_vec_go(matrix, v, Nil{})

# One round of the power iteration: multiply, then normalise.
def iterate_go(fuel: List<&2, Nat>, +matrix: List<&2, List<&2, F32>>,
  +v: List<&2, F32>) -> List<&2, F32>:
  match fuel:
    case Nil{}:
      v
    case Con{u, t}:
      iterate_go(t, matrix, normalise(mat_vec(matrix, v)))

# The first principal component, after n rounds.
def component(matrix: List<&2, List<&2, F32>>, v: List<&2, F32>, n: Nat) ->
  List<&2, F32>:
  iterate_go(List.range(n), matrix, v)

# The diagonal of a matrix, added up.
def trace_go(matrix: List<&2, List<&2, F32>>, +i: Nat, +acc: F32) -> F32:
  match matrix:
    case Nil{}:
      acc
    case Con{row, t}:
      trace_go(t, Nat.add(i, 1n), F32.add(acc, cell_at(row, i)))

# The diagonal of a matrix, added up.
def trace(matrix: List<&2, List<&2, F32>>) -> F32:
  trace_go(matrix, 0n, 0.0)

# The starting vector: 1, 2, ..., n, normalised.
def add_count(acc: List<&2, F32>, n: Nat) -> List<&2, F32>:
  List.append(&2, F32, acc, Con{F32.add(F32.from_nat(n), 1.0), Nil{}})

# The starting vector: 1, 2, ..., n, normalised.
def start(n: Nat) -> List<&2, F32>:
  normalise(List.foldl(&2, Nat, List<&2, F32>, add_count, List.range(n), Nil{}))

# The explained variance ratio of the first component.
def ratio(+cov: List<&2, List<&2, F32>>, +v: List<&2, F32>) -> F32:
  F32.div(dot(v, mat_vec(cov, v)), trace(cov))

# The run, over the rows.
def show_rows(+rows: List<&2, List<&2, F32>>) -> IO(Unit):
  +means = means_of(rows)
  +stds = stds_of(rows, means)
  +prepared = standardise(rows, means, stds)
  +n = List.length(&2, List<&2, F32>, prepared)
  +cols = columns(prepared, 11n)
  +cov = covariance(cols, F32.from_nat(n))
  +v = component(cov, start(11n), 100n)
  do IO<Unit>:
    IO.print(String.append("explained variance ratio: ", F32.show(ratio(cov, v))))

def show_table(t: Datasets.Table) -> IO(Unit):
  match t:
    case Datasets.Table{+rows, +labels}:
      show_rows(rows)

# The ratio, or the reason there is none.
def show(t: Result<&1, &1, Hf.Err, Datasets.Table>) -> IO(Unit):
  match t:
    case Fail{e}:
      IO.print("the dataset failed")
    case Done{table}:
      show_table(table)

def main() -> IO(Unit):
  do IO<Unit>:
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.wine()
    show(t)
