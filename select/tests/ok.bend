# The rows of every fold: the numbers are checked against scikit-learn by
# reference/test_select.py.
#
#| fold 0: 1 3 5 6 9 10
#| train of fold 0: 0 2 4 7 8 11
#| sizes: 6 3 3
import Base
import mlbend-select@0.1.0.0/main.bend as Select

# The cells of a list, as text.

def cells_go(+xs: List<&2, Nat>, +acc: String) -> String:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      cells_go(t, String.append(acc, String.append(U32.show(U32.from_nat(h)), " ")))

# The cells of a list, as text.
def cells(+xs: List<&2, Nat>) -> String:
  cells_go(xs, "")

# The rows that test in the fold f, out of a fold.
def fold_test(f: Select.Fold) -> List<&2, Nat>:
  match f:
    case Select.Fold{test, train}:
      test

# The rows that train in the fold f, out of a fold.
def fold_train(f: Select.Fold) -> List<&2, Nat>:
  match f:
    case Select.Fold{test, train}:
      train

# Four rows of one class.
def four(+c: Nat) -> List<&2, Nat>:
  Con{c, Con{c, Con{c, Con{c, Nil{}}}}}

# Twelve rows of three classes, four of each.
def labels() -> List<&2, Nat>:
  List.append(&2, Nat, List.append(&2, Nat, four(0n), four(1n)), four(2n))

# The folds of the rows, for the seed 42.
def folds_of() -> List<&2, Nat>:
  Select.folds(labels(), 42, 3n)

# The number of rows of each fold.
def sizes() -> List<&2, Nat>:
  Select.fold_sizes(folds_of(), 3n)

# The rows that test in the fold 0.
def test_of() -> List<&2, Nat>:
  fold_test(Select.fold_of(folds_of(), 0n))

# The rows that train in the fold 0.
def train_of() -> List<&2, Nat>:
  fold_train(Select.fold_of(folds_of(), 0n))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(String.append("fold 0: ", cells(test_of())))
    IO.print(String.append("train of fold 0: ", cells(train_of())))
    IO.print(String.append("sizes: ", cells(sizes())))