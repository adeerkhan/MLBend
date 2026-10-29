# k-nearest neighbours over the breast cancer dataset, fetched from
# HuggingFace at run time.
#
#   bendenv run -- bend examples/knn/main.bend
#
# The classifier is plain Bend: no gradients and no library. The features
# are standardised first, because the columns of the dataset run from 0 to
# 4000 and a distance would then be a distance between mitoses.
#
# The rows are split: the first four fifths train, the last fifth test.
# For every test row the three nearest training rows vote.
import Base
import mlbend-datasets@0.1.6.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf
import mlbend-metrics@0.1.0.0/main.bend as M
import mlbend-report@0.1.0.0/main.bend as Report

# A neighbour: how far it is, and what it is.

type Neighbour is Data:
  Neighbour{dist: F32, label: F32}

# The nearer of two neighbours, compared.
def nearer_go(+a: Neighbour, b: Neighbour) -> Bool:
  match a:
    case Neighbour{d, l}:
      match b:
        case Neighbour{d2, l2}:
          F32.is_lt(d, d2)

# The nearer of two neighbours.
def nearer(a: Neighbour, b: Neighbour) -> Bool:
  nearer_go(a, b)

# The squares of the differences, from the first column on.
def sum_squares(+a: List<&2, F32>, b: List<&2, F32>, +acc: F32) -> F32:
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

# The neighbour of a row: how far it is from the query, and what it is.
def neighbour_of(row: List<&2, F32>, label: F32, query: List<&2, F32>) -> Neighbour:
  Neighbour{distance(row, query), label}

# Every training row, one at a time.
def neighbours_go(train: List<&2, List<&2, F32>>, labels: List<&2, F32>,
  +query: List<&2, F32>, +acc: List<&2, Neighbour>) -> List<&2, Neighbour>:
  match train:
    case Nil{}:
      acc
    case Con{h, t}:
      match labels:
        case Nil{}:
          acc
        case Con{l, ls}:
          neighbours_go(t, ls, query, List.append(&2, Neighbour, acc, Con{neighbour_of(h, l, query), Nil{}}))

# Every training row, as a neighbour of the query.
def neighbours_of(train: List<&2, List<&2, F32>>, labels: List<&2, F32>,
  query: List<&2, F32>) -> List<&2, Neighbour>:
  neighbours_go(train, labels, query, Nil{})

# The k nearest neighbours of a query, the nearest first.
def k_nearest(train: List<&2, List<&2, F32>>, labels: List<&2, F32>, query: List<&2, F32>,
  k: Nat) -> List<&2, Neighbour>:
  List.take(&2, Neighbour, List.sort(~Neighbour, ~nearer, neighbours_of(train, labels, query)), k)

# The label of a neighbour.
def label_of(n: Neighbour) -> F32:
  match n:
    case Neighbour{d, l}:
      l

# How many of the neighbours carry a label, counted.
def count_label_go(neighbours: List<&2, Neighbour>, +label: F32, +acc: F32) -> F32:
  match neighbours:
    case Nil{}:
      acc
    case Con{h, t}:
      count_label_go(t, label,
        Bool.pick(F32, F32.is_eq(label_of(h), label), F32.add(acc, 1.0), acc))

# How many of the neighbours carry a label.
def count_label(+neighbours: List<&2, Neighbour>, +label: F32) -> F32:
  count_label_go(neighbours, label, 0.0)

# The label the neighbours vote for: the one they say most.
def vote(+neighbours: List<&2, Neighbour>) -> F32:
  zero = count_label(neighbours, 0.0)
  one = count_label(neighbours, 1.0)
  Bool.pick(F32, F32.is_gt(zero, one), 0.0, 1.0)

# The label the classifier gives a row.
def classify(train: List<&2, List<&2, F32>>, labels: List<&2, F32>, query: List<&2, F32>,
  k: Nat) -> F32:
  vote(k_nearest(train, labels, query, k))

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

# How many of the test rows are classified right, row by row.
def correct_go(+train: List<&2, List<&2, F32>>, +train_labels: List<&2, F32>,
  test: List<&2, List<&2, F32>>, test_labels: List<&2, F32>, +k: Nat, +right: F32,
  total: F32) -> F32:
  match test:
    case Nil{}:
      right
    case Con{h, t}:
      match test_labels:
        case Nil{}:
          right
        case Con{l, ls}:
          hit = F32.is_eq(classify(train, train_labels, h, k), l)
          correct_go(train, train_labels, t, ls, k,
            Bool.pick(F32, hit, F32.add(right, 1.0), right), F32.add(total, 1.0))

# How many of the test rows are classified right.
def correct(train: List<&2, List<&2, F32>>, train_labels: List<&2, F32>,
  +test: List<&2, List<&2, F32>>, test_labels: List<&2, F32>, k: Nat) -> F32:
  correct_go(train, train_labels, test, test_labels, k, 0.0, 0.0)

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

# The distance of a neighbour.
def dist_of(n: Neighbour) -> F32:
  match n:
    case Neighbour{d, l}:
      d

# One row and a number, cell by cell.
def zip_scale(a: List<&2, F32>, +k: F32) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      Con{F32.mul(ha, k), zip_scale(ta, k)}

# The mean of every column.
def means_of(+rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  zip_scale(column_sums(rows), (1.0 / F32.from_nat(List.length(&2, List<&2, F32>, rows)) : F32))

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
def standardise_go(rows: List<&2, List<&2, F32>>, +means: List<&2, F32>, +stds: List<&2, F32>,
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

# The class of a label: the dataset's 0.0 and 1.0 are the classes 0 and 1.
def class_of(l: F32) -> Nat:
  Bool.pick(Nat, F32.is_eq(l, 1.0), 1n, 0n)

# The class of every label, one at a time.
def classes_go(labels: List<&2, F32>, +acc: List<&2, Nat>) -> List<&2, Nat>:
  match labels:
    case Nil{}:
      acc
    case Con{l, ls}:
      classes_go(ls, List.append(&2, Nat, acc, Con{class_of(l), Nil{}}))

# The predicted class of every test row, one at a time.
def predicted_go(+train: List<&2, List<&2, F32>>, +train_labels: List<&2, F32>,
  test: List<&2, List<&2, F32>>, +k: Nat, +acc: List<&2, Nat>) -> List<&2, Nat>:
  match test:
    case Nil{}:
      acc
    case Con{h, t}:
      predicted_go(train, train_labels, t, k,
        List.append(&2, Nat, acc, Con{class_of(classify(train, train_labels, h, k)), Nil{}}))

# One row of the confusion matrix, printed.
def row_text(row: List<&2, Nat>) -> String:
  match row:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(U32.show(U32.from_nat(h)), String.append(" ", row_text(t)))

# The confusion matrix, printed.
def matrix_text.go(m: List<&2, List<&2, Nat>>, acc: String) -> String:
  match m:
    case Nil{}:
      acc
    case Con{h, t}:
      matrix_text.go(t, String.append(String.append(acc, row_text(h)), "\n"))

def matrix_text(m: List<&2, List<&2, Nat>>) -> String:
  matrix_text.go(m, "")

# One class of the report, printed.
def class_text(c: Report.Cls, +i: Nat) -> String:
  match c:
    case Report.Cls{p, r, f, support}:
      String.append("class ", String.append(U32.show(U32.from_nat(i)),
        String.append(" precision ", String.append(F32.show(p),
        String.append(" recall ", String.append(F32.show(r),
        String.append(" f1 ", String.append(F32.show(f),
        String.append(" support ", U32.show(U32.from_nat(support)))))))))))

# Every class of the report, printed.
def classes_text.go(classes: List<&2, Report.Cls>, +i: Nat, acc: String) -> String:
  match classes:
    case Nil{}:
      acc
    case Con{c, t}:
      classes_text.go(t, (1n + i : Nat),
        String.append(String.append(acc, class_text(c, i)), "\n"))

# The averages of the report, printed.
def avg_text(name: String, c: Report.Cls) -> String:
  match c:
    case Report.Cls{p, r, f, support}:
      String.append(name, String.append(" precision ", String.append(F32.show(p),
        String.append(" recall ", String.append(F32.show(r),
        String.append(" f1 ", String.append(F32.show(f), "")))))))

# The report of a report value, printed.
def report_text_of(rep: Report.Report) -> String:
  match rep:
    case Report.Report{classes, macro_avg, weighted_avg, accuracy}:
      String.append(classes_text.go(classes, 0n, ""),
        String.append(avg_text("macro", macro_avg),
        String.append("\n", String.append(avg_text("weighted", weighted_avg), ""))))

# The report, printed, or why there is none.
def report_text(r: Result<&1, &1, M.Bad, Report.Report>) -> String:
  match r:
    case Fail{M.Mismatch{expected, got}}:
      "the report failed: the lengths do not agree"
    case Fail{M.Empty{}}:
      "the report failed: no sample"
    case Done{rep}:
      report_text_of(rep)

# The classifier, over the rows and the labels.
def show_rows(+rows: List<&2, List<&2, F32>>, +labels: List<&2, F32>) -> IO(Unit):
  +means = means_of(rows)
  +prepared = standardise(rows, means, stds_of(rows, means))
  +split = (Nat.mul(List.length(&2, List<&2, F32>, prepared), 4n) / 5n : Nat)
  +train = List.take(&2, List<&2, F32>, prepared, split)
  +train_labels = List.take(&2, F32, labels, split)
  +test = List.drop(&2, List<&2, F32>, prepared, split)
  +test_labels = List.drop(&2, F32, labels, split)
  right = correct(train, train_labels, test, test_labels, 3n)
  total = F32.from_nat(List.length(&2, List<&2, F32>, test))
  +ys = classes_go(test_labels, Nil{})
  +ps = predicted_go(train, train_labels, test, 3n, Nil{})
  do IO<Unit>:
    IO.print(String.append("train rows: ", U32.show(U32.from_nat(List.length(&2, List<&2, F32>, train)))))
    IO.print(String.append("test rows: ", U32.show(U32.from_nat(List.length(&2, List<&2, F32>, test)))))
    IO.print(String.append("accuracy: ", F32.show((right / total : F32))))
    IO.print("matrix:")
    IO.print(matrix_text(Report.matrix(ys, ps, 2n)))
    IO.print(report_text(Report.report(ys, ps, 2n)))

# The classifier, over the online dataset.
def show(t: Datasets.Table) -> IO(Unit):
  match t:
    case Datasets.Table{+rows, +labels}:
      show_rows(rows, labels)

# The classifier, or the reason there is none.
def show_result(t: Result<&1, &1, Hf.Err, Datasets.Table>) -> IO(Unit):
  match t:
    case Fail{e}:
      IO.print("the dataset failed")
    case Done{v}:
      show(v)

def main() -> IO(Unit):
  do IO<Unit>:
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.cancer()
    show_result(t)