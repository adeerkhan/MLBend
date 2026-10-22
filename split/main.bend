# mlbend-split: a seeded, deterministic split of the rows, stratified by label.
#
#   import mlbend-split@0.1.0.0/main.bend as Split
#
# The rows of a class are shuffled and then dealt out over the folds, so
# every fold holds the same proportion of every class — the rule
# scikit-learn's StratifiedKFold uses. The shuffle is bend-kit-random's
# xoshiro128**, seeded: the same seed gives the same split, and another
# seed gives another one. Nothing here generates numbers of its own.
#
# The labels are class indices, 0 to the number of classes.
#
# What is proved: every row of the dataset is placed in exactly one fold
# (`split_covers`) — the fold list holds one entry per row, so the folds
# are a partition of the rows: the test rows of a fold and its training
# rows are disjoint, and together they hold every row. What is tested:
# that the sizes and the classes of the folds agree with scikit-learn's
# (reference/test_split.py).

import Base
import bend-kit-random@0.1.0.1/random.bend as Rand

# The split of the rows: the training rows and the test rows, as their
# indices in the dataset.

type Split is Data:
  Split{train: List<&2, Nat>, test: List<&2, Nat>}

# LAW (split_covers): every row of the dataset is placed in a fold — the
# fold list holds one entry per row, so the folds are a partition of the
# rows: the test rows of a fold and its training rows are disjoint, and
# together they hold every row.
law split_covers:
  for labels: List<&2, Nat>
  for +orders: List<&2, List<&2, Nat>>
  for +k: Nat
  for +i: Nat
  {count_rows(folds_from(labels, orders, k, i)) == List.length(&2, Nat, labels) : Nat}

# The rows of one class, with their indices.
def rows_of_class(labels: List<&2, Nat>, +c: Nat, +i: Nat, +acc: List<&2, Nat>) -> List<&2, Nat>:
  match labels:
    case Nil{}:
      acc
    case Con{h, t}:
      rows_of_class(t, c, Nat.add(i, 1n),
        Bool.pick(List<&2, Nat>, Nat.is_eq(h, c), List.append(&2, Nat, acc, Con{i, Nil{}}), acc))

# The list of a pair: the list, and the generator that comes after it.
def shuffled_list(pair: List<&2, Nat> & Rand.Rng) -> List<&2, Nat>:
  (xs, r) = pair
  xs

# The rows of one class, shuffled by the generator.
def shuffled_class(labels: List<&2, Nat>, +c: Nat, +seed: U32) -> List<&2, Nat>:
  shuffled_list(Rand.shuffle(~Nat, rows_of_class(labels, c, 0n, Nil{}),
    Rand.seed(U32.add(seed, U32.from_nat(c)))))

# The shuffled rows of every class, one class after the other.
def class_orders_go(+labels: List<&2, Nat>, +seed: U32, remaining: Nat, +next: Nat,
  +acc: List<&2, List<&2, Nat>>) -> List<&2, List<&2, Nat>>:
  match remaining:
    case 0n:
      acc
    case 1n+rest:
      class_orders_go(labels, seed, rest, Nat.add(next, 1n),
        List.append(&2, List<&2, Nat>, acc, Con{shuffled_class(labels, next, seed), Nil{}}))

# The shuffled rows of every class, one class after the other.
def class_orders(labels: List<&2, Nat>, seed: U32, c: Nat) -> List<&2, List<&2, Nat>>:
  class_orders_go(labels, seed, c, 0n, Nil{})

# The largest label of the labels.
def largest_of(labels: List<&2, Nat>, +acc: Nat) -> Nat:
  match labels:
    case Nil{}:
      acc
    case Con{+h, t}:
      largest_of(t, Bool.pick(Nat, Nat.is_gt(h, acc), h, acc))

# The number of classes of the labels: one more than the largest label.
def classes_of(labels: List<&2, Nat>) -> Nat:
  Nat.add(1n, largest_of(labels, 0n))

# The number of rows a list of folds holds.
def count_rows(fs: List<&2, Nat>) -> Nat:
  match fs:
    case Nil{}:
      0n
    case Con{h, t}:
      (1n + count_rows(t) : Nat)

# The place of a row among the shuffled rows of its class.
def position_go(order: List<&2, Nat>, +i: Nat, +at: Nat) -> Nat:
  match order:
    case Nil{}:
      at
    case Con{h, t}:
      Bool.pick(Nat, Nat.is_eq(h, i), at, position_go(t, i, Nat.add(at, 1n)))

# The place of a row among the shuffled rows of its class.
def position_of(order: List<&2, Nat>, i: Nat) -> Nat:
  position_go(order, i, 0n)

# The fold of a row, given the shuffled rows of its class.
def fold_in_of(order: List<&2, Nat>, k: Nat, i: Nat) -> Nat:
  Nat.mod(position_of(order, i), k)

# The first of the shuffled rows of every class.
def order_head(orders: List<&2, List<&2, Nat>>) -> List<&2, Nat>:
  match orders:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      h

# The orders of every class but the first.
def order_tail(+orders: List<&2, List<&2, Nat>>) -> List<&2, List<&2, Nat>>:
  match orders:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      t

# The shuffled rows of the class of a label: the n-th of the orders.
def order_at(n: Nat, +orders: List<&2, List<&2, Nat>>) -> List<&2, Nat>:
  match n:
    case 0n:
      order_head(orders)
    case 1n+rest:
      order_at(rest, order_tail(orders))

# The fold of a row: its place among the shuffled rows of its class,
# dealt out over the folds in turn.
def fold_in(orders: List<&2, List<&2, Nat>>, k: Nat, label: Nat, i: Nat) -> Nat:
  fold_in_of(order_at(label, orders), k, i)

# The fold of every row from the row i on, given the shuffled rows of
# every class.
def folds_from(labels: List<&2, Nat>, +orders: List<&2, List<&2, Nat>>, +k: Nat, +i: Nat) ->
  List<&2, Nat>:
  match labels:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{fold_in(orders, k, h, i), folds_from(t, orders, k, Nat.add(i, 1n))}

# The fold of every row of the dataset, in the order of the rows.
def folds(+labels: List<&2, Nat>, seed: U32, k: Nat) -> List<&2, Nat>:
  folds_from(labels, class_orders(labels, seed, classes_of(labels)), k, 0n)

# The row, in the fold.
def test_row(fold: Nat, f: Nat, +test: List<&2, Nat>, i: Nat) -> List<&2, Nat>:
  Bool.pick(List<&2, Nat>, Nat.is_eq(fold, f), List.append(&2, Nat, test, Con{i, Nil{}}), test)

# The row, out of the fold.
def train_row(fold: Nat, f: Nat, +train: List<&2, Nat>, i: Nat) -> List<&2, Nat>:
  Bool.pick(List<&2, Nat>, Nat.is_eq(fold, f), train, List.append(&2, Nat, train, Con{i, Nil{}}))

# The rows of one fold: the ones that test and the ones that train.
def split_of(fs: List<&2, Nat>, +f: Nat, +i: Nat, +test: List<&2, Nat>, +train: List<&2, Nat>) -> Split:
  match fs:
    case Nil{}:
      Split{train, test}
    case Con{+h, t}:
      split_of(t, f, Nat.add(i, 1n), test_row(h, f, test, i), train_row(h, f, train, i))

# The rows of one fold: the ones that test and the ones that train.
def split(labels: List<&2, Nat>, seed: U32, k: Nat, f: Nat) -> Split:
  split_of(folds(labels, seed, k), f, 0n, Nil{}, Nil{})

# every row of the dataset is placed in a fold
def split_covers(labels, orders, k, i):
  match labels:
    case Nil{}:
      {==}
    case Con{h, +t}:
      %split_covers(t, orders, k, Nat.add(i, 1n)) : {1n+count_rows(folds_from(t, orders, k, Nat.add(i, 1n))) == 1n+_ : Nat}
      {==}

# The rows of the fold f: the ones that test.
def test_of_go(+fs: List<&2, Nat>, +f: Nat, +i: Nat, +acc: List<&2, Nat>) -> List<&2, Nat>:
  match fs:
    case Nil{}:
      acc
    case Con{+h, t}:
      test_of_go(t, f, Nat.add(i, 1n), test_row(h, f, acc, i))

# The rows of the fold f: the ones that test.
def test_of(fs: List<&2, Nat>, f: Nat) -> List<&2, Nat>:
  test_of_go(fs, f, 0n, Nil{})

# The rows of the fold f: the ones that train.
def train_of_go(+fs: List<&2, Nat>, +f: Nat, +i: Nat, +acc: List<&2, Nat>) -> List<&2, Nat>:
  match fs:
    case Nil{}:
      acc
    case Con{+h, t}:
      train_of_go(t, f, Nat.add(i, 1n), train_row(h, f, acc, i))

# The rows of the fold f: the ones that train.
def train_of(fs: List<&2, Nat>, f: Nat) -> List<&2, Nat>:
  train_of_go(fs, f, 0n, Nil{})