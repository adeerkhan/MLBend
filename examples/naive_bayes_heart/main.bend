# Gaussian naive Bayes over the heart failure dataset, fetched from
# HuggingFace at run time.
#
#   bendenv run -- bend examples/naive_bayes/main.bend
#
# The classifier is plain Bend: no gradients and no library. It is the
# baseline the experiment layer keeps beside the learned models: every
# feature is assumed normal within a class, the classes are independent
# given the label, and a row goes to the class whose mean and variance
# make it most likely.
#
# The rows are split: the first four fifths train, the last fifth test, as
# `examples/knn` does. There is no standardising, and none is needed: a
# Gaussian is unchanged by an affine change of one feature, so the means
# and the variances move with the scale and the prediction does not.
import Base
import mlbend-datasets@0.1.5.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf

# One class's model: the mean and the variance of every feature, and the
# log of the class's share of the training rows.

type Params is Data:
  Params{mean: List<&2, F32>, var: List<&2, F32>, log_prior: F32}

# The class of a label.
def class_of(label: F32) -> Nat:
  U32.to_nat(F32.to_u32(label))

# Whether a label names a class.
def is_class(label: F32, +c: Nat) -> Bool:
  F32.is_eq(label, F32.from_nat(c))

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
        (1.0 / F32.from_nat(List.length(&2, List<&2, F32>, rows)) : F32))

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

# The variance of every column: the population one.
def vars_of(+rows: List<&2, List<&2, F32>>, +means: List<&2, F32>) -> List<&2, F32>:
  zip_scale(deviation_sums(rows, means),
    (1.0 / F32.from_nat(List.length(&2, List<&2, F32>, rows)) : F32))

# The rows of one class, one at a time.
def rows_of_class_go(rows: List<&2, List<&2, F32>>, labels: List<&2, F32>, +c: Nat,
  +acc: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      match labels:
        case Nil{}:
          acc
        case Con{l, ls}:
          rows_of_class_go(t, ls, c,
            Bool.pick(List<&2, List<&2, F32>>, is_class(l, c),
              List.append(&2, List<&2, F32>, acc, Con{h, Nil{}}), acc))

# The rows of one class.
def rows_of_class(rows: List<&2, List<&2, F32>>, labels: List<&2, F32>, c: Nat) ->
  List<&2, List<&2, F32>>:
  rows_of_class_go(rows, labels, c, Nil{})

# How many labels name a class.
def count_go(labels: List<&2, F32>, +c: Nat, +acc: Nat) -> Nat:
  match labels:
    case Nil{}:
      acc
    case Con{l, t}:
      count_go(t, c, Bool.pick(Nat, is_class(l, c), Nat.add(acc, 1n), acc))

# How many labels name a class.
def count_of(labels: List<&2, F32>, c: Nat) -> Nat:
  count_go(labels, c, 0n)

# One cell's contribution to a class's log likelihood: the log of the
# normal density at that cell.
def gauss_cell(+mean: F32, +var: F32, +x: F32) -> F32:
  +v = F32.max(var, 0.000000001)
  +d = F32.sub(x, mean)
  F32.mul((0.0 - 0.5 : F32),
    F32.add(F32.log(F32.mul(6.2831855, v)), F32.div(F32.mul(d, d), v)))

# A row's contribution to a class's log likelihood.
def ll_go(mean: List<&2, F32>, var: List<&2, F32>, row: List<&2, F32>, +acc: F32) -> F32:
  match mean:
    case Nil{}:
      acc
    case Con{+hm, tm}:
      match var:
        case Nil{}:
          acc
        case Con{+hv, tv}:
          match row:
            case Nil{}:
              acc
            case Con{+hx, tx}:
              ll_go(tm, tv, tx, F32.add(acc, gauss_cell(hm, hv, hx)))

# The log likelihood of a row, given a class's parameters.
def score(p: Params, row: List<&2, F32>) -> F32:
  match p:
    case Params{mean, var, log_prior}:
      F32.add(log_prior, ll_go(mean, var, row, 0.0))

# The class a row is more likely under: 0 when it is 0, 1 when it is 1.
def predict(+zero: Params, +one: Params, +row: List<&2, F32>) -> Nat:
  Bool.pick(Nat, F32.is_gt(score(one, row), score(zero, row)), 1n, 0n)

# The parameters of one class.
def params_of(+rows: List<&2, List<&2, F32>>, +labels: List<&2, F32>, +c: Nat,
  total: Nat) -> Params:
  +selected = rows_of_class(rows, labels, c)
  +mean = means_of(selected)
  +var = vars_of(selected, mean)
  +share = F32.div(F32.from_nat(count_of(labels, c)), F32.from_nat(total))
  +lp = F32.log(F32.max(share, 0.000000000001))
  Params{mean, var, lp}

# How many test rows the classifier gets right.
def correct_go(rows: List<&2, List<&2, F32>>, labels: List<&2, F32>, +zero: Params,
  +one: Params, +acc: Nat) -> Nat:
  match rows:
    case Nil{}:
      acc
    case Con{r, rs}:
      match labels:
        case Nil{}:
          acc
        case Con{l, ls}:
          correct_go(rs, ls, zero, one,
            Bool.pick(Nat, Nat.is_eq(predict(zero, one, r), class_of(l)), Nat.add(acc, 1n), acc))

# A list of rows, as text.
def rows_text(rows: List<&2, List<&2, F32>>) -> String:
  U32.show(U32.from_nat(List.length(&2, List<&2, F32>, rows)))

# The split and the score, over the rows.
def show_rows(+rows: List<&2, List<&2, F32>>, +labels: List<&2, F32>) -> IO(Unit):
  +n = List.length(&2, List<&2, F32>, rows)
  +train_n = Nat.div(Nat.mul(n, 4n), 5n)
  +train_rows = List.take(&2, List<&2, F32>, rows, train_n)
  +test_rows = List.drop(&2, List<&2, F32>, rows, train_n)
  +train_labels = List.take(&2, F32, labels, train_n)
  +test_labels = List.drop(&2, F32, labels, train_n)
  +zero = params_of(train_rows, train_labels, 0n, train_n)
  +one = params_of(train_rows, train_labels, 1n, train_n)
  +total = List.length(&2, List<&2, F32>, test_rows)
  +right = correct_go(test_rows, test_labels, zero, one, 0n)
  do IO<Unit>:
    IO.print(String.append("train rows: ", rows_text(train_rows)))
    IO.print(String.append("test rows: ", rows_text(test_rows)))
    IO.print(String.append("accuracy: ", F32.show((F32.div(F32.from_nat(right), F32.from_nat(total)) : F32))))

def show_table(t: Datasets.Table) -> IO(Unit):
  match t:
    case Datasets.Table{+rows, +labels}:
      show_rows(rows, labels)

# The score, or the reason there is none.
def show(t: Result<&1, &1, Hf.Err, Datasets.Table>) -> IO(Unit):
  match t:
    case Fail{e}:
      IO.print("the dataset failed")
    case Done{table}:
      show_table(table)

def main() -> IO(Unit):
  do IO<Unit>:
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.heart()
    show(t)
