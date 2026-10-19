# mlbend-metrics: the evaluation surface of machine learning, natively in Bend 2.
#
#   import mlbend-metrics@0.1.0.0/main.bend as M
#
# A metric compares two sequences of the same length. The length is not
# trusted: every metric answers a Result, and a mismatch of lengths (or
# an empty input) is a value, never a crash.
#
# What is proved: the counts. What is tested: the F32 numbers, against
# NumPy and scikit-learn (reference/test_metrics.py). F32 is not a real
# number, so a metric's value is never proved, only its shape.

import Base
import bend-ml-nat-lemmas@0.1.1.0/main.bend as NL

# Bad
# ---
#
# Why a metric could not answer.

type Bad is Data:
  Mismatch{expected: Nat, got: Nat}
  Empty{}

# Counts
# ------
#
# The counting core, over Nat, so that it can be proved. A count walks
# the labels; the walks over two sequences stop at the shorter one (the
# metrics check the lengths first).

# 1 when a and b are the same label, 0 otherwise
def same_of(s: Bool) -> Nat:
  match s:
    case True{}:
      1n
    case False{}:
      0n

def same(a: Nat, b: Nat) -> Nat:
  same_of(Nat.is_eq(a, b))

# the number of samples
def count(ys: List<&2, Nat>) -> Nat:
  match ys:
    case Nil{}:
      0n
    case Con{h, t}:
      (1n + count(t) : Nat)

# the number of samples whose true label is k
def count_true.go(ys: List<&2, Nat>, +k: Nat, n: Nat) -> Nat:
  match ys:
    case Nil{}:
      n
    case Con{y, yt}:
      count_true.go(yt, k, (n + same(y, k) : Nat))

def count_true(ys: List<&2, Nat>, k: Nat) -> Nat:
  count_true.go(ys, k, 0n)

# the number of samples whose predicted label is k
def count_pred.go(ps: List<&2, Nat>, +k: Nat, n: Nat) -> Nat:
  match ps:
    case Nil{}:
      n
    case Con{p, pt}:
      count_pred.go(pt, k, (n + same(p, k) : Nat))

def count_pred(ps: List<&2, Nat>, k: Nat) -> Nat:
  count_pred.go(ps, k, 0n)

# the number of samples whose true label is k and whose predicted label
# is k too: the true positives of the class k
def count_both.go(ys: List<&2, Nat>, ps: List<&2, Nat>, +k: Nat, n: Nat) -> Nat:
  match ys ps:
    case Con{y, yt} Con{p, pt}:
      count_both.go(yt, pt, k, (n + same(y, k) * same(p, k) : Nat))
    case _ _:
      n

def count_both(ys: List<&2, Nat>, ps: List<&2, Nat>, k: Nat) -> Nat:
  count_both.go(ys, ps, k, 0n)

# the number of samples predicted right
def count_right.go(ys: List<&2, Nat>, ps: List<&2, Nat>, n: Nat) -> Nat:
  match ys ps:
    case Con{y, yt} Con{p, pt}:
      count_right.go(yt, pt, (n + same(y, p) : Nat))
    case _ _:
      n

def count_right(ys: List<&2, Nat>, ps: List<&2, Nat>) -> Nat:
  count_right.go(ys, ps, 0n)

# LAW 1 (count_length): the number of samples is the length of the list
# of labels. (The count is this package's own walk, so the law pins it
# to the length Base counts.)
law count_length:
  for ys: List<&2, Nat>
  {count(ys) == List.length(&2, Nat, ys) : Nat}

# the number of samples is the length of the list of labels
def count_length(ys):
  match ys:
    case Nil{}:
      {==}
    case Con{h, t}:
      %count_length(t) : {1n+count(t) == 1n+_ : Nat}
      {==}

# Sums
# -----
#
# The sums the error metrics need: the walk over the two sequences, with
# the number of pairs walked beside the sum.

type Sum is Data:
  Sum{n: Nat, s: F32}

def sum_add(a: Sum, x: F32) -> Sum:
  match a:
    case Sum{n, s}:
      Sum{(1n + n : Nat), (s + x : F32)}

# the sum of the squared differences of the pairs walked
def sq_sum.go(ys: List<&2, F32>, ps: List<&2, F32>, acc: Sum) -> Sum:
  match ys ps:
    case Con{y, yt} Con{p, pt}:
      +d = (y - p : F32);
      sum_add(sq_sum.go(yt, pt, acc), (d * d : F32))
    case _ _:
      acc

# the sum of the absolute differences of the pairs walked
def abs_sum.go(ys: List<&2, F32>, ps: List<&2, F32>, acc: Sum) -> Sum:
  match ys ps:
    case Con{y, yt} Con{p, pt}:
      d = F32.abs((y - p : F32));
      sum_add(abs_sum.go(yt, pt, acc), d)
    case _ _:
      acc

# Ratios
# ------
#
# A metric that divides: the ratio of two counts, and the mean of a sum
# over a number of pairs. A zero denominator answers 0.0, the
# convention scikit-learn uses (zero_division=0).

# the ratio a over b, or 0.0 when b is zero
def ratio_of(z: Bool, +a: Nat, +b: Nat) -> F32:
  match z:
    case True{}:
      0.0
    case False{}:
      (F32.from_nat(a) / F32.from_nat(b) : F32)

def ratio(+a: Nat, +b: Nat) -> F32:
  ratio_of(Nat.is_eq(b, 0n), a, b)

# the mean of a sum over n pairs
def mean(s: Sum, n: Nat) -> F32:
  match s:
    case Sum{n0, s0}:
      (s0 / F32.from_nat(n) : F32)

# Metrics
# -------
#
# Every metric answers a Result: the value, or why it could not answer.

# the number of pairs to walk, or why the two sequences do not agree
def same_len_of(s: Bool, +n: Nat, +m: Nat) -> Result<&1, &1, Bad, Nat>:
  match s:
    case True{}:
      Done{n}
    case False{}:
      Fail{Mismatch{n, m}}

def same_len(+n: Nat, +m: Nat) -> Result<&1, &1, Bad, Nat>:
  same_len_of(Nat.is_eq(n, m), n, m)

# the number of pairs to walk, or why there are none
def not_empty_of(z: Bool, n: Nat) -> Result<&1, &1, Bad, Nat>:
  match z:
    case True{}:
      Fail{Empty{}}
    case False{}:
      Done{n}

# the number of pairs, or why there is no answer
def not_empty(r: Result<&1, &1, Bad, Nat>) -> Result<&1, &1, Bad, Nat>:
  match r:
    case Done{+n}:
      not_empty_of(Nat.is_eq(n, 0n), n)
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

# the number of pairs the two sequences hold, or why there is no answer
def pairs(a, -A: Kind(a), ys: List<a, A>, ps: List<a, A>) -> Result<&1, &1, Bad, Nat>:
  not_empty(same_len(List.length(a, A, ys), List.length(a, A, ps)))

# the mean squared error between the targets and the predictions
def mse.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, F32>, +ps: List<&2, F32>) -> Result<&1, &1, Bad, F32>:
  match r:
    case Done{n}:
      Done{mean(sq_sum.go(ys, ps, Sum{0n, 0.0}), n)}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def mse(+ys: List<&2, F32>, +ps: List<&2, F32>) -> Result<&1, &1, Bad, F32>:
  mse.go(pairs(&2, F32, ys, ps), ys, ps)

# the mean absolute error between the targets and the predictions
def mae.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, F32>, +ps: List<&2, F32>) -> Result<&1, &1, Bad, F32>:
  match r:
    case Done{n}:
      Done{mean(abs_sum.go(ys, ps, Sum{0n, 0.0}), n)}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def mae(+ys: List<&2, F32>, +ps: List<&2, F32>) -> Result<&1, &1, Bad, F32>:
  mae.go(pairs(&2, F32, ys, ps), ys, ps)

# the root mean squared error between the targets and the predictions
def rmse.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, F32>, +ps: List<&2, F32>) -> Result<&1, &1, Bad, F32>:
  match r:
    case Done{n}:
      Done{F32.sqrt(mean(sq_sum.go(ys, ps, Sum{0n, 0.0}), n))}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def rmse(+ys: List<&2, F32>, +ps: List<&2, F32>) -> Result<&1, &1, Bad, F32>:
  rmse.go(pairs(&2, F32, ys, ps), ys, ps)

# the fraction of the samples predicted right
def accuracy.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, Nat>, +ps: List<&2, Nat>) -> Result<&1, &1, Bad, F32>:
  match r:
    case Done{n}:
      Done{ratio(count_right(ys, ps), n)}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def accuracy(+ys: List<&2, Nat>, +ps: List<&2, Nat>) -> Result<&1, &1, Bad, F32>:
  accuracy.go(pairs(&2, Nat, ys, ps), ys, ps)

# the fraction of the samples predicted as k that are really k: the
# precision of the class k
def precision.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  match r:
    case Done{n}:
      Done{ratio(count_both(ys, ps, k), count_pred(ps, k))}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def precision(+ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  precision.go(pairs(&2, Nat, ys, ps), ys, ps, k)

# the fraction of the samples really k that are predicted as k: the
# recall of the class k
def recall.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  match r:
    case Done{n}:
      Done{ratio(count_both(ys, ps, k), count_true(ys, k))}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def recall(+ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  recall.go(pairs(&2, Nat, ys, ps), ys, ps, k)

# the harmonic mean of two scores
def f1_num_of(z: Bool, +p: F32, +r: F32) -> F32:
  match z:
    case True{}:
      0.0
    case False{}:
      ((2.0 * p * r / (p + r) : F32))

def f1_num(+p: F32, +r: F32) -> F32:
  f1_num_of(F32.is_eq((p + r : F32), 0.0), p, r)

# the harmonic mean of a precision and a recall
def f1_of(p: Result<&1, &1, Bad, F32>, r: Result<&1, &1, Bad, F32>) -> Result<&1, &1, Bad, F32>:
  match p:
    case Done{pv}:
      match r:
        case Done{rv}:
          Done{f1_num(pv, rv)}
        case Fail{Mismatch{expected, got}}:
          Fail{Mismatch{expected, got}}
        case Fail{Empty{}}:
          Fail{Empty{}}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

# the harmonic mean of the precision and the recall of the class k
def f1.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  match r:
    case Done{+n}:
      f1_of(precision.go(Done{n}, ys, ps, k), recall.go(Done{n}, ys, ps, k))
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def f1(+ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  f1.go(pairs(&2, Nat, ys, ps), ys, ps, k)

