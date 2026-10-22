# A seeded, stratified split of the rows: the numbers are checked against
# scikit-learn by reference/test_split.py.
#
#| folds: 1 0 2 0 1 0 0 2 1 0 0 2
#| test of fold 0: 1 3 5 6 9 10
#| train of fold 0: 0 2 4 7 8 11
#| the same seed twice: 1 0 2 0 1 0 0 2 1 0 0 2
import Base
import mlbend-split@0.1.0.0/main.bend as Split

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

# Four rows of one class.
def four(+c: Nat) -> List<&2, Nat>:
  Con{c, Con{c, Con{c, Con{c, Nil{}}}}}

# Twelve rows of three classes, four of each.
def labels() -> List<&2, Nat>:
  List.append(&2, Nat, List.append(&2, Nat, four(0n), four(1n)), four(2n))

# The folds of the rows, for the seed 42 again.
def folds_again() -> List<&2, Nat>:
  Split.folds(labels(), 42, 3n)

# The folds of the rows, for the seed 42.
def folds_of() -> List<&2, Nat>:
  Split.folds(labels(), 42, 3n)

# The rows that test, out of a split.
def split_test(s: Split.Split) -> List<&2, Nat>:
  match s:
    case Split.Split{train, test}:
      test

# The rows that test in the fold 0.
def test_of() -> List<&2, Nat>:
  split_test(Split.split(labels(), 42, 3n, 0n))

# The rows that train, out of a split.
def split_train(s: Split.Split) -> List<&2, Nat>:
  match s:
    case Split.Split{train, test}:
      train

# The rows that train in the fold 0.
def train_of() -> List<&2, Nat>:
  split_train(Split.split(labels(), 42, 3n, 0n))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(String.append("folds: ", cells(folds_of())))
    IO.print(String.append("test of fold 0: ", cells(test_of())))
    IO.print(String.append("train of fold 0: ", cells(train_of())))
    IO.print(String.append("the same seed twice: ", cells(folds_again())))