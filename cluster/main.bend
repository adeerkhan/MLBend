# mlbend-cluster: k-means, functional.
#
#   import mlbend-cluster@0.1.0.0/main.bend as Cluster
#
# The clustering is Lloyd's algorithm with a fixed number of rounds: the
# centres are the first k rows, then every round assigns every row to its
# nearest centre and moves each centre to the mean of the rows assigned
# to it. A fixed number of rounds means the result is a function of the
# arguments: there is no loop that waits for the centres to settle, so a
# run is reproducible and the same rows give the same centres.
#
# A row is a list of F32 and a centre is a row: the distance is the sum
# of the squared differences, so the arithmetic is on whole rows and no
# column is found by its place.
#
# What is proved: every row gets exactly one cluster (`cluster_covers`)
# — the assignment holds one label per row, so a row cannot be in two
# clusters and cannot be left out. What is tested: the assignment, the
# cluster sizes and the inertia against NumPy (reference/test_cluster.py),
# and the k-means example that reads the wine dataset through it
# (reference/test_kmeans.py).

import Base

# The centres of the clusters, in the order of the cluster indices.

type Model is Data:
  Model{centres: List<&2, List<&2, F32>>, k: Nat}

# LAW (cluster_covers): every row gets exactly one cluster — the
# assignment holds one label per row, so a row cannot be in two clusters
# and cannot be left out.
law cluster_covers:
  for +centres: List<&2, List<&2, F32>>
  for rows: List<&2, List<&2, F32>>
  {count_labels(assign(centres, rows)) == count_rows(rows) : Nat}

# How many labels an assignment holds.
def count_labels(labels: List<&2, Nat>) -> Nat:
  match labels:
    case Nil{}:
      0n
    case Con{h, t}:
      (1n + count_labels(t) : Nat)

# How many rows there are.
def count_rows(rows: List<&2, List<&2, F32>>) -> Nat:
  match rows:
    case Nil{}:
      0n
    case Con{h, t}:
      (1n + count_rows(t) : Nat)

# The squares of the differences of two rows, added up.
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

# The nearest centre, walking the centres in order.
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

# The index of the nearest centre: the first one when two are equally far.
def nearest(centres: List<&2, List<&2, F32>>, row: List<&2, F32>) -> Nat:
  nearest_go(centres, row, 0n, 0n, 1.0e30)

# The cluster of every row: the index of its nearest centre.
def assign(+centres: List<&2, List<&2, F32>>, rows: List<&2, List<&2, F32>>) ->
  List<&2, Nat>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{nearest(centres, h), assign(centres, t)}

# every row gets exactly one cluster
def cluster_covers(centres, rows):
  match rows:
    case Nil{}:
      {==}
    case Con{h, +t}:
      %cluster_covers(centres, t) : {1n+count_labels(assign(centres, t)) == 1n+_ : Nat}
      {==}

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

# One row and a number, cell by cell.
def zip_scale(a: List<&2, F32>, +k: F32) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      Con{F32.mul(ha, k), zip_scale(ta, k)}

# The mean of every column of the rows.
def means_of(+rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      zip_scale(column_sums(rows),
        (1.0 / F32.from_nat(count_rows(rows)) : F32))

# The rows of one cluster, gathered.
def members_go(rows: List<&2, List<&2, F32>>, labels: List<&2, Nat>, +k: Nat,
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
            Bool.pick(List<&2, List<&2, F32>>, Nat.is_eq(l, k), Con{h, acc}, acc))

# The rows of one cluster.
def members_of(rows: List<&2, List<&2, F32>>, labels: List<&2, Nat>, k: Nat) ->
  List<&2, List<&2, F32>>:
  members_go(rows, labels, k, Nil{})

# The mean of every cluster, in the order of the cluster indices. A
# cluster with no rows keeps an empty centre, so the indices never move.
def means_of_clusters_go(indices: List<&2, Nat>, +rows: List<&2, List<&2, F32>>,
  +labels: List<&2, Nat>) -> List<&2, List<&2, F32>>:
  match indices:
    case Nil{}:
      Nil{}
    case Con{i, t}:
      Con{means_of(members_of(rows, labels, i)),
        means_of_clusters_go(t, rows, labels)}

# The mean of every cluster, in the order of the cluster indices.
def means_of_clusters(indices: List<&2, Nat>, +rows: List<&2, List<&2, F32>>,
  +labels: List<&2, Nat>) -> List<&2, List<&2, F32>>:
  means_of_clusters_go(indices, rows, labels)

# The first k rows: where the centres start.
def starts(k: Nat, +rows: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  List.take(&2, List<&2, F32>, rows, k)

# One round: assign, then move every centre to the mean of its rows.
def round(+indices: List<&2, Nat>, +centres: List<&2, List<&2, F32>>,
  +rows: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  means_of_clusters(indices, rows, assign(centres, rows))

# The rounds, the fuel first.
def rounds_go(fuel: List<&2, Nat>, +indices: List<&2, Nat>,
  +centres: List<&2, List<&2, F32>>, +rows: List<&2, List<&2, F32>>) ->
  List<&2, List<&2, F32>>:
  match fuel:
    case Nil{}:
      centres
    case Con{u, t}:
      rounds_go(t, indices, round(indices, centres, rows), rows)

# The centres after the rounds.
def rounds(n: Nat, indices: List<&2, Nat>, centres: List<&2, List<&2, F32>>,
  rows: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  rounds_go(List.range(n), indices, centres, rows)

# The clustering: the k centres, after the rounds, over the rows.
def fit(+k: Nat, rounds_count: Nat, +rows: List<&2, List<&2, F32>>) -> Model:
  Model{rounds(rounds_count, List.range(k), starts(k, rows), rows), k}

# A centre, or an empty row when there is none.
def centre_of(m: Maybe<&2, List<&2, F32>>) -> List<&2, F32>:
  match m:
    case Some{c}:
      c
    case None{}:
      Nil{}

# The centre at an index, or an empty row when there is none.
def centre_at(centres: List<&2, List<&2, F32>>, n: Nat) -> List<&2, F32>:
  centre_of(List.get(&2, List<&2, F32>, centres, n))

# The squared distances from the rows to their centres, added up.
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

# The sum of the squared distances from every row to its centre.
def inertia(centres: List<&2, List<&2, F32>>, rows: List<&2, List<&2, F32>>,
  labels: List<&2, Nat>) -> F32:
  inertia_go(centres, rows, labels, 0.0)

# The number of rows in one cluster, walking the labels.
def count_go(labels: List<&2, Nat>, +k: Nat, +acc: Nat) -> Nat:
  match labels:
    case Nil{}:
      acc
    case Con{l, t}:
      count_go(t, k, Bool.pick(Nat, Nat.is_eq(l, k), Nat.add(acc, 1n), acc))

# The number of rows in one cluster.
def count_of(labels: List<&2, Nat>, k: Nat) -> Nat:
  count_go(labels, k, 0n)

# The number of rows of every cluster, in the order of the cluster indices.
def sizes_go(indices: List<&2, Nat>, +labels: List<&2, Nat>) -> List<&2, Nat>:
  match indices:
    case Nil{}:
      Nil{}
    case Con{i, t}:
      Con{count_of(labels, i), sizes_go(t, labels)}

# The number of rows of every cluster, in the order of the cluster indices.
def sizes(labels: List<&2, Nat>, k: Nat) -> List<&2, Nat>:
  sizes_go(List.range(k), labels)
