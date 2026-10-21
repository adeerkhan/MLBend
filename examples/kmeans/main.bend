# k-means over the wine dataset, fetched from HuggingFace at run time.
#
#   bendenv run -- bend examples/kmeans/main.bend
#
# The clustering is plain Bend: no gradients and no library. The features
# are standardised first, because the wine dataset mixes scales from 0 to
# 160 and a distance would then be a distance between sulphates.
#
# The arithmetic is done on whole rows: a fold over the rows adds the
# columns up, and a fold over a row adds the cells up. Nothing counts a
# column by its place, so a row may be short.
import Base
import mlbend-datasets@0.1.4.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf

# Two rows, added cell by cell.


# The squares of the differences, from the first column on.
def sum_squares(a: List<&2, F32>, b: List<&2, F32>, +acc: F32) -> F32:
  match a:
    case Nil{}:
      acc
    case Con{+ha, ta}:
      match b:
        case Con{+hb, tb}:
          sum_squares(ta, tb, F32.add(acc, F32.mul(F32.sub(ha, hb), F32.sub(ha, hb))))
        case Nil{}:
          acc

# The squared distance of two rows.
def distance(a: List<&2, F32>, b: List<&2, F32>) -> F32:
  sum_squares(a, b, 0.0)

# The nearest centre, and how far it is.
def nearest_go(centres: List<&2, List<&2, F32>>, +row: List<&2, F32>, +n: Nat,
  +best: Nat, +best_d: F32) -> Nat:
  match centres:
    case Nil{}:
      best
    case Con{h, +t}:
      +d = distance(row, h)
      Bool.pick(Nat, F32.is_lt(d, best_d),
        nearest_go(t, row, Nat.add(n, 1n), n, d),
        nearest_go(t, row, Nat.add(n, 1n), best, best_d))

# The index of the nearest centre.
def nearest(centres: List<&2, List<&2, F32>>, row: List<&2, F32>) -> Nat:
  nearest_go(centres, row, 0n, 0n, 1.0e30)

# The clusters, row by row.
def assign_go(+centres: List<&2, List<&2, F32>>, +rows: List<&2, List<&2, F32>>,
  +acc: List<&2, Nat>) -> List<&2, Nat>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      assign_go(centres, t, List.append(&2, Nat, acc, Con{nearest(centres, h), Nil{}}))

# The cluster of every row: the nearest centre.
def assign(centres: List<&2, List<&2, F32>>, rows: List<&2, List<&2, F32>>) -> List<&2, Nat>:
  assign_go(centres, rows, Nil{})

# A centre, or an empty row when there is none.
def centre_of(m: Maybe<&2, List<&2, F32>>) -> List<&2, F32>:
  match m:
    case Some{c}:
      c
    case None{}:
      Nil{}

# The centre of one cluster, or an empty row when there is none.
def centre_at(centres: List<&2, List<&2, F32>>, n: Nat) -> List<&2, F32>:
  centre_of(List.get(&2, List<&2, F32>, centres, n))
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

# The sum of every column, row by row.
def column_sums_go(rows: List<&2, List<&2, F32>>, +acc: List<&2, F32>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      column_sums_go(t, zip_add(acc, h))

# The sum of every column over every row.
def column_sums(rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      column_sums_go(t, h)

# How many rows are in one cluster, counted.
def count_go(labels: List<&2, Nat>, +k: Nat, +acc: Nat) -> Nat:
  match labels:
    case Nil{}:
      acc
    case Con{l, t}:
      count_go(t, k, Bool.pick(Nat, Nat.is_eq(l, k), Nat.add(acc, 1n), acc))

# How many rows are in one cluster.
def count_of(labels: List<&2, Nat>, k: Nat) -> Nat:
  count_go(labels, k, 0n)

# The cells of a row, squared.
def squares(+a: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{+ha, ta}:
      Con{F32.mul(ha, ha), squares(ta)}

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

# The inertia, row by row.
def inertia_go(+centres: List<&2, List<&2, F32>>, +rows: List<&2, List<&2, F32>>,
  labels: List<&2, Nat>, +acc: F32) -> F32:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      match labels:
        case Nil{}:
          acc
        case Con{l, ls}:
          inertia_go(centres, t, ls, F32.add(acc, distance(h, centre_at(centres, l))))

# The sum of the squared distances of every row to its centre.
def inertia(centres: List<&2, List<&2, F32>>, rows: List<&2, List<&2, F32>>,
  labels: List<&2, Nat>) -> F32:
  inertia_go(centres, rows, labels, 0.0)

# One row and a number, cell by cell.
def zip_scale(+a: List<&2, F32>, +k: F32) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      Con{F32.mul(ha, k), zip_scale(ta, k)}

# The mean of every column.
def means_of(+rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  zip_scale(column_sums(rows), (1.0 / F32.from_nat(List.length(&2, List<&2, F32>, rows)) : F32))

# The mean of the rows of one cluster.
def mean_of(rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      means_of(Con{h, t})

# The rows of one cluster, gathered.
def members_go(+rows: List<&2, List<&2, F32>>, labels: List<&2, Nat>, +k: Nat,
  +acc: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      match labels:
        case Nil{}:
          acc
        case Con{l, ls}:
          members_go(t, ls, k,
            Bool.pick(List<&2, List<&2, F32>>, Nat.is_eq(l, k),
              List.append(&2, List<&2, F32>, acc, Con{h, Nil{}}),
              acc))

# The rows of one cluster.
def members_of(+rows: List<&2, List<&2, F32>>, labels: List<&2, Nat>, k: Nat) ->
  List<&2, List<&2, F32>>:
  members_go(rows, labels, k, Nil{})

# The mean of every cluster, in the order of the cluster indices.
def means_of_clusters(indices: List<&2, Nat>, +rows: List<&2, List<&2, F32>>,
  +labels: List<&2, Nat>) -> List<&2, List<&2, F32>>:
  match indices:
    case Nil{}:
      Nil{}
    case Con{i, t}:
      List.append(&2, List<&2, F32>, Nil{},
        Con{mean_of(members_of(rows, labels, i)), means_of_clusters(t, rows, labels)})

# The centres again, from the clusters.
def recompute(indices: List<&2, Nat>, centres: List<&2, List<&2, F32>>,
  rows: List<&2, List<&2, F32>>, labels: List<&2, Nat>) -> List<&2, List<&2, F32>>:
  means_of_clusters(indices, rows, labels)

# One round: assign, then recompute.
def round(indices: List<&2, Nat>, +centres: List<&2, List<&2, F32>>,
  +rows: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  recompute(indices, centres, rows, assign(centres, rows))

# The rounds, the fuel first.
def rounds(fuel: List<&2, Nat>, +indices: List<&2, Nat>, +centres: List<&2, List<&2, F32>>,
  +rows: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match fuel:
    case Nil{}:
      centres
    case Con{u, t}:
      rounds(t, indices, round(indices, centres, rows), rows)

# The sizes of the clusters, as text.
def sizes_text(indices: List<&2, Nat>, +labels: List<&2, Nat>) -> String:
  match indices:
    case Nil{}:
      ""
    case Con{i, t}:
      String.append(U32.show(U32.from_nat(count_of(labels, i))), String.append(" ", sizes_text(t, labels)))

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

# The clustering, over the standardised rows.
def show_rows(+rows: List<&2, List<&2, F32>>) -> IO(Unit):
  +means = means_of(rows)
  +prepared = standardise(rows, means, stds_of(rows, means))
  +indices = List.range(3n)
  +centres = rounds(List.range(10n), indices,
    List.take(&2, List<&2, F32>, prepared, 3n), prepared)
  +labels = assign(centres, prepared)
  do IO<Unit>:
    IO.print(String.append("inertia: ", F32.show(inertia(centres, prepared, labels))))
    IO.print(String.append("cluster sizes: ", sizes_text(indices, labels)))

# The clustering: three centres, ten rounds, and what it came to.
def show(t: Datasets.Table) -> IO(Unit):
  match t:
    case Datasets.Table{+rows, +labels}:
      show_rows(rows)

# The clustering, or the reason there is none.
def show_result(t: Result<&1, &1, Hf.Err, Datasets.Table>) -> IO(Unit):
  match t:
    case Fail{e}:
      IO.print("the dataset failed")
    case Done{v}:
      show(v)

def main() -> IO(Unit):
  do IO<Unit>:
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.wine()
    show_result(t)