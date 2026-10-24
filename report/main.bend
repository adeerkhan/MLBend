# mlbend-report: the confusion matrix and the per-class report, natively in Bend 2.
#
#   import mlbend-report@0.1.0.0/main.bend as R
#
# What an estimator's predictions look like: the confusion matrix of a
# classification, and the per-class report scikit-learn's
# classification_report prints — the precision, the recall, the F1 and
# the support of every class, with the macro and the weighted averages.
# The labels are Nat: the class 0, the class 1, and so on up to k-1.
#
# The per-class numbers rest on mlbend-metrics' counts and on its
# zero-division convention: a class nobody predicted scores 0.0, the
# way scikit-learn's zero_division=0 does.
#
# What is proved: the shape of the matrix. The matrix of k classes has
# k rows (matrix_rows) and every row k cells (row_length). What is
# tested: the F32 numbers, against scikit-learn's confusion_matrix and
# classification_report (reference/test_report.py). F32 is not a real
# number, so a report's value is never proved, only its shape.

import Base
import mlbend-metrics@0.1.0.0/main.bend as M

# The report of one class: its scores, and how many samples it has.

type Cls is Data:
  Cls{precision: F32, recall: F32, f1: F32, support: Nat}

# The whole report: one Cls per class, the averages over them, and the
# accuracy.

type Report is Data:
  Report{classes: List<&2, Cls>, macro_avg: Cls, weighted_avg: Cls, accuracy: F32}

# The matrix
# ----------
#
# The confusion matrix, cell by cell. The row is the true label, the
# column the predicted one, the way scikit-learn's confusion_matrix
# lays it out.

# The cell of the true label i and the predicted label j: how many
# samples have both.
def cell.go(ys: List<&2, Nat>, ps: List<&2, Nat>, +i: Nat, +j: Nat, n: Nat) -> Nat:
  match ys ps:
    case Con{y, yt} Con{p, pt}:
      cell.go(yt, pt, i, j, (n + M.same(y, i) * M.same(p, j) : Nat))
    case _ _:
      n

def cell(ys: List<&2, Nat>, ps: List<&2, Nat>, +i: Nat, +j: Nat) -> Nat:
  cell.go(ys, ps, i, j, 0n)

# How many rows a matrix has.
def rows_of(m: List<&2, List<&2, Nat>>) -> Nat:
  match m:
    case Nil{}:
      0n
    case Con{h, t}:
      (1n + rows_of(t) : Nat)

# How many cells a row has.
def cells_of(row: List<&2, Nat>) -> Nat:
  match row:
    case Nil{}:
      0n
    case Con{h, t}:
      (1n + cells_of(t) : Nat)

# The row of the class i, from the column j on: n cells, the cell of
# every class from j to j+n-1, ascending. The counted form is the
# public one, because the law walks it: the counter of the cells left
# is the argument that shrinks.
def row_from(+ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat, +i: Nat, n: Nat, +j: Nat) ->
  List<&2, Nat>:
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{cell(ys, ps, i, j), row_from(ys, ps, k, i, rest, (1n + j : Nat))}

# The row of the class i: k cells, ascending.
def row(ys: List<&2, Nat>, ps: List<&2, Nat>, +k: Nat, +i: Nat) -> List<&2, Nat>:
  row_from(ys, ps, k, i, k, 0n)

# The rows of the classes i to i+n-1: n rows, each of k cells, ascending.
def matrix_from(+ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat, n: Nat, +i: Nat) ->
  List<&2, List<&2, Nat>>:
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{row(ys, ps, k, i), matrix_from(ys, ps, k, rest, (1n + i : Nat))}

# The confusion matrix of k classes: k rows of k cells, the row of the
# class 0 first.
def matrix(ys: List<&2, Nat>, ps: List<&2, Nat>, +k: Nat) ->
  List<&2, List<&2, Nat>>:
  matrix_from(ys, ps, k, k, 0n)

# LAW 1 (matrix_rows): n rows of the matrix are n rows.
law matrix_rows:
  for +ys: List<&2, Nat>
  for +ps: List<&2, Nat>
  for +k: Nat
  for +n: Nat
  for +i: Nat
  {rows_of(matrix_from(ys, ps, k, n, i)) == n : Nat}

# n rows of the matrix are n rows
def matrix_rows(ys, ps, k, n, i):
  match n:
    case 0n:
      {==}
    case 1n+rest:
      %matrix_rows(ys, ps, k, rest, (1n + i : Nat)) :
        {1n+rows_of(matrix_from(ys, ps, k, rest, (1n + i : Nat))) == 1n+_ : Nat}
      {==}

# LAW 2 (row_length): n cells of a row are n cells.
law row_length:
  for +ys: List<&2, Nat>
  for +ps: List<&2, Nat>
  for +k: Nat
  for +i: Nat
  for +n: Nat
  for +j: Nat
  {cells_of(row_from(ys, ps, k, i, n, j)) == n : Nat}

# n cells of a row are n cells
def row_length(ys, ps, k, i, n, j):
  match n:
    case 0n:
      {==}
    case 1n+rest:
      %row_length(ys, ps, k, i, rest, (1n + j : Nat)) :
        {1n+cells_of(row_from(ys, ps, k, i, rest, (1n + j : Nat))) == 1n+_ : Nat}
      {==}

# The classes
# -----------
#
# The per-class scores, on the counts of mlbend-metrics. The caller has
# checked that the labels and the predictions agree in length and that
# there is at least one sample; the gated entry is `report`.

# The report of one class: its precision, its recall, its F1 and its
# support.
def cls_of(+ys: List<&2, Nat>, +ps: List<&2, Nat>, +i: Nat) -> Cls:
  +hits = M.count_both(ys, ps, i)
  +support = M.count_true(ys, i)
  +p = M.ratio(hits, M.count_pred(ps, i))
  +r = M.ratio(hits, support)
  Cls{p, r, M.f1_num(p, r), support}

# The report of every class, from the class i on: n of them, ascending.
def classes_of.go(+ys: List<&2, Nat>, +ps: List<&2, Nat>, n: Nat, +i: Nat) ->
  List<&2, Cls>:
  match n:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{cls_of(ys, ps, i), classes_of.go(ys, ps, rest, (1n + i : Nat))}

# The report of every class: k of them, the class 0 first.
def classes_of(ys: List<&2, Nat>, ps: List<&2, Nat>, +k: Nat) -> List<&2, Cls>:
  classes_of.go(ys, ps, k, 0n)

# The averages
# ------------

# The sums the averages rest on: the per-class scores added up, and the
# support beside them.
type Tot is Data:
  Tot{p: F32, r: F32, f: F32, support: Nat}

# One class more, added into the sums.
def tot_add(t: Tot, c: Cls) -> Tot:
  match t:
    case Tot{p, r, f, support}:
      match c:
        case Cls{p2, r2, f2, support2}:
          Tot{(p + p2 : F32), (r + r2 : F32), (f + f2 : F32), (support + support2 : Nat)}

# The sums over the classes.
def tot_of.go(classes: List<&2, Cls>, acc: Tot) -> Tot:
  match classes:
    case Nil{}:
      acc
    case Con{c, t}:
      tot_of.go(t, tot_add(acc, c))

def tot_of(classes: List<&2, Cls>) -> Tot:
  tot_of.go(classes, Tot{0.0, 0.0, 0.0, 0n})

# a divided by b, or 0.0 when b is zero — the zero_division convention
def divide(+a: F32, +b: F32) -> F32:
  Bool.pick(F32, F32.is_eq(b, 0.0), 0.0, (a / b : F32))

def macro_of.go(t: Tot, +k: Nat) -> Cls:
  match t:
    case Tot{p, r, f, support}:
      +n = F32.from_nat(k)
      Cls{divide(p, n), divide(r, n), divide(f, n), support}

# The macro average: the mean of the per-class scores, every class
# weighing the same, the way scikit-learn's macro average does.
def macro_of(classes: List<&2, Cls>, +k: Nat) -> Cls:
  macro_of.go(tot_of(classes), k)

# The weighted sums: every per-class score multiplied by its support.
def weighted_sums(classes: List<&2, Cls>, +p: F32, +r: F32, +f: F32) -> Tot:
  match classes:
    case Nil{}:
      Tot{p, r, f, 0n}
    case Con{c, t}:
      match c:
        case Cls{p2, r2, f2, s2}:
          +w = F32.from_nat(s2)
          weighted_sums(t, (p + p2 * w : F32), (r + r2 * w : F32), (f + f2 * w : F32))

def weighted_of.go(t: Tot, +total: Nat) -> Cls:
  match t:
    case Tot{p, r, f, support}:
      +n = F32.from_nat(total)
      Cls{divide(p, n), divide(r, n), divide(f, n), total}

# The weighted average: the per-class scores, each weighed by its
# support, over the total support, the way scikit-learn's weighted
# average does.
def weighted_of(classes: List<&2, Cls>, +total: Nat) -> Cls:
  weighted_of.go(weighted_sums(classes, 0.0, 0.0, 0.0), total)

# The report
# ----------

# The whole report, or why there is none: the labels and the
# predictions must agree in length, and there must be at least one
# sample.
def report.go(r: Result<&1, &1, M.Bad, Nat>, +ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) ->
  Result<&1, &1, M.Bad, Report>:
  match r:
    case Done{+n}:
      +classes = classes_of(ys, ps, k)
      Done{Report{classes, macro_of(classes, k), weighted_of(classes, n),
        M.ratio(M.count_right(ys, ps), n)}}
    case Fail{M.Mismatch{expected, got}}:
      Fail{M.Mismatch{expected, got}}
    case Fail{M.Empty{}}:
      Fail{M.Empty{}}

def report(+ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) ->
  Result<&1, &1, M.Bad, Report>:
  report.go(M.pairs(&2, Nat, ys, ps), ys, ps, k)
