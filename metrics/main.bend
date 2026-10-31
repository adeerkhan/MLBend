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

# The accounting walks
# --------------------
#
# The samples predicted right, the samples predicted wrong, and the pairs
# walked: three plain recursions over the two lists. Each stops on whichever
# list ends first, so the law needs no length hypothesis; count_length pins the
# walked count to the list's length when the two sequences are the same length,
# which is the case the metrics check before they count.

# the number of samples predicted wrong, of one label comparison
def diff_of(s: Bool) -> Nat:
  match s:
    case True{}:
      0n
    case False{}:
      1n

def diff(a: Nat, b: Nat) -> Nat:
  diff_of(Nat.is_eq(a, b))

# the number of samples predicted right
def right(ys: List<&2, Nat>, ps: List<&2, Nat>) -> Nat:
  match ys ps:
    case Con{y, yt} Con{p, pt}:
      Nat.add(same(y, p), right(yt, pt))
    case _ _:
      0n

# the number of samples predicted wrong
def wrong(ys: List<&2, Nat>, ps: List<&2, Nat>) -> Nat:
  match ys ps:
    case Con{y, yt} Con{p, pt}:
      Nat.add(diff(y, p), wrong(yt, pt))
    case _ _:
      0n

# the number of pairs walked
def walked(ys: List<&2, Nat>, ps: List<&2, Nat>) -> Nat:
  match ys ps:
    case Con{y, yt} Con{p, pt}:
      Nat.add(1n, walked(yt, pt))
    case _ _:
      0n

# (a + b) + (c + d) == (a + c) + (b + d)
def add4(+a: Nat, +b: Nat, +c: Nat, +d: Nat) -> {Nat.add(Nat.add(a, b), Nat.add(c, d)) == Nat.add(Nat.add(a, c), Nat.add(b, d)) : Nat}:
  +u1 = Equal.sym(Nat, Nat.add(a, Nat.add(b, Nat.add(c, d))), Nat.add(Nat.add(a, b), Nat.add(c, d)), NL.add_assoc(a, b, Nat.add(c, d)))
  +u2 = Equal.cong(Nat, Nat, z => Nat.add(a, z), Nat.add(b, Nat.add(c, d)), Nat.add(Nat.add(b, c), d), NL.add_assoc(b, c, d))
  +u3 = Equal.cong(Nat, Nat, z => Nat.add(a, Nat.add(z, d)), Nat.add(b, c), Nat.add(c, b), NL.add_comm(b, c))
  +u4 = Equal.cong(Nat, Nat, z => Nat.add(a, z), Nat.add(Nat.add(c, b), d), Nat.add(c, Nat.add(b, d)), Equal.sym(Nat, Nat.add(c, Nat.add(b, d)), Nat.add(Nat.add(c, b), d), NL.add_assoc(c, b, d)))
  Equal.trans(Nat, Nat.add(Nat.add(a, b), Nat.add(c, d)), Nat.add(a, Nat.add(b, Nat.add(c, d))), Nat.add(Nat.add(a, c), Nat.add(b, d)), u1,
    Equal.trans(Nat, Nat.add(a, Nat.add(b, Nat.add(c, d))), Nat.add(a, Nat.add(Nat.add(b, c), d)), Nat.add(Nat.add(a, c), Nat.add(b, d)), u2,
      Equal.trans(Nat, Nat.add(a, Nat.add(Nat.add(b, c), d)), Nat.add(a, Nat.add(Nat.add(c, b), d)), Nat.add(Nat.add(a, c), Nat.add(b, d)), u3,
        Equal.trans(Nat, Nat.add(a, Nat.add(Nat.add(c, b), d)), Nat.add(a, Nat.add(c, Nat.add(b, d))), Nat.add(Nat.add(a, c), Nat.add(b, d)), u4,
          NL.add_assoc(a, c, Nat.add(b, d))))))

# one pair: s + d == 1, so (s + cr) + (d + cm) == 1 + cw when cr + cm == cw
def split_step(+s: Nat, +d: Nat, +cr: Nat, +cm: Nat, +cw: Nat, hs: {Nat.add(s, d) == 1n : Nat}, h: {Nat.add(cr, cm) == cw : Nat}) -> {Nat.add(Nat.add(s, cr), Nat.add(d, cm)) == Nat.add(1n, cw) : Nat}:
  +u1 = add4(s, cr, d, cm)
  +u2 = Equal.cong(Nat, Nat, z => Nat.add(z, Nat.add(cr, cm)), Nat.add(s, d), 1n, hs)
  +u3 = Equal.cong(Nat, Nat, z => Nat.add(1n, z), Nat.add(cr, cm), cw, h)
  Equal.trans(Nat, Nat.add(Nat.add(s, cr), Nat.add(d, cm)), Nat.add(Nat.add(s, d), Nat.add(cr, cm)), Nat.add(1n, cw), u1,
    Equal.trans(Nat, Nat.add(Nat.add(s, d), Nat.add(cr, cm)), Nat.add(1n, Nat.add(cr, cm)), Nat.add(1n, cw), u2, u3))

# The parent step, stated over the two constructors so the branch goal reduces
# the way ASet.node_t's does: the subtree's fact arrives as an argument, never
# as the result of a recursive call made inside the match.
def split_pair(+y: Nat, +p: Nat, +yt: List<&2, Nat>, +pt: List<&2, Nat>, b: Bool, hf: {Nat.add(right(yt, pt), wrong(yt, pt)) == walked(yt, pt) : Nat}) -> {Nat.add(Nat.add(same_of(b), right(yt, pt)), Nat.add(diff_of(b), wrong(yt, pt))) == Nat.add(1n, walked(yt, pt)) : Nat}:
  match b:
    case True{}:
      split_step(1n, 0n, right(yt, pt), wrong(yt, pt), walked(yt, pt), NL.add_zero(1n), hf)
    case False{}:
      split_step(0n, 1n, right(yt, pt), wrong(yt, pt), walked(yt, pt),
        Equal.trans(Nat, Nat.add(0n, 1n), Nat.add(1n, 0n), 1n, NL.add_comm(0n, 1n), NL.add_zero(1n)), hf)

def split_ok(+ys: List<&2, Nat>, +ps: List<&2, Nat>) -> {Nat.add(right(ys, ps), wrong(ys, ps)) == walked(ys, ps) : Nat}:
  match ys ps:
    case Con{y, yt} Con{p, pt}:
      split_pair(y, p, yt, pt, Nat.is_eq(y, p), split_ok(yt, pt))
    case Nil{} _:
      {==}
    case Con{y, yt} Nil{}:
      {==}

# LAW 2 (accounting): the samples predicted right and the samples predicted
# wrong add up to the samples walked.
law accounting:
  for ys: List<&2, Nat>
  for ps: List<&2, Nat>
  {Nat.add(right(ys, ps), wrong(ys, ps)) == walked(ys, ps) : Nat}

def accounting(ys, ps):
  split_ok(ys, ps)

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

# the total of a sum
def sum_total(s: Sum) -> F32:
  match s:
    case Sum{n, s0}:
      s0

# the sum of the squared residuals, as a number
def ss_res(ys: List<&2, F32>, ps: List<&2, F32>) -> F32:
  sum_total(sq_sum.go(ys, ps, Sum{0n, 0.0}))

# ---- r2 ----
def ysum.go(ys: List<&2, F32>, acc: F32) -> F32:
  match ys:
    case Nil{}:
      acc
    case Con{h, t}:
      ysum.go(t, (acc + h : F32))

def ysum(ys: List<&2, F32>) -> F32:
  ysum.go(ys, 0.0)

def dev.go(ys: List<&2, F32>, +m: F32, acc: F32) -> F32:
  match ys:
    case Nil{}:
      acc
    case Con{h, t}:
      +d = (h - m : F32)
      dev.go(t, m, (acc + d * d : F32))

def dev(ys: List<&2, F32>, m: F32) -> F32:
  dev.go(ys, m, 0.0)

def r2_of(z: Bool, +sst: F32, +sse: F32) -> F32:
  match z:
    case True{}:
      Bool.pick(F32, F32.is_eq(sse, 0.0), 1.0, 0.0)
    case False{}:
      (1.0 - (sse / sst : F32) : F32)

def r2.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, F32>, +ps: List<&2, F32>) -> Result<&1, &1, Bad, F32>:
  match r:
    case Done{+n}:
      +m = (ysum(ys) / F32.from_nat(n) : F32)
      +sst = dev(ys, m)
      Done{r2_of(F32.is_eq(sst, 0.0), sst, ss_res(ys, ps))}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def r2(+ys: List<&2, F32>, +ps: List<&2, F32>) -> Result<&1, &1, Bad, F32>:
  r2.go(pairs(&2, F32, ys, ps), ys, ps)

# ---- log loss ----
def prob_at(xs: List<&2, F32>, +i: Nat) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{h, t}:
      match i:
        case 0n:
          h
        case 1n+r:
          prob_at(t, r)

def log_sum.go(ys: List<&2, Nat>, ps: List<&2, List<&2, F32>>, acc: F32) -> F32:
  match ys ps:
    case Con{y, yt} Con{p, pt}:
      +q = F32.max(prob_at(p, y), 0.0000001)
      log_sum.go(yt, pt, (acc - F32.log(q) : F32))
    case _ _:
      acc

def log_loss.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, Nat>, +ps: List<&2, List<&2, F32>>) -> Result<&1, &1, Bad, F32>:
  match r:
    case Done{+n}:
      Done{(log_sum.go(ys, ps, 0.0) / F32.from_nat(n) : F32)}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def log_loss(+ys: List<&2, Nat>, +ps: List<&2, List<&2, F32>>) -> Result<&1, &1, Bad, F32>:
  log_loss.go(not_empty(same_len(List.length(&2, Nat, ys), List.length(&2, List<&2, F32>, ps))), ys, ps)

# ---- the confusion matrix ----
def cell.go(ys: List<&2, Nat>, ps: List<&2, Nat>, +i: Nat, +j: Nat, acc: Nat) -> Nat:
  match ys ps:
    case Con{y, yt} Con{p, pt}:
      cell.go(yt, pt, i, j, Nat.add(acc, Nat.mul(same(y, i), same(p, j))))
    case _ _:
      acc

def cell(ys: List<&2, Nat>, ps: List<&2, Nat>, i: Nat, j: Nat) -> Nat:
  cell.go(ys, ps, i, j, 0n)

def row_at(+n: Nat, +ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat, +i: Nat) -> List<&2, Nat>:
  match n:
    case 0n:
      Nil{}
    case 1n+r:
      Con{cell(ys, ps, i, Nat.sub(k, n)), row_at(r, ys, ps, k, i)}

def row(+ys: List<&2, Nat>, +ps: List<&2, Nat>, i: Nat, +k: Nat) -> List<&2, Nat>:
  row_at(k, ys, ps, k, i)

def rows_at(+n: Nat, +ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> List<&2, List<&2, Nat>>:
  match n:
    case 0n:
      Nil{}
    case 1n+r:
      Con{row(ys, ps, Nat.sub(k, n), k), rows_at(r, ys, ps, k)}

def confusion.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, List<&2, List<&2, Nat>>>:
  match r:
    case Done{+n}:
      Done{rows_at(k, ys, ps, k)}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def confusion(+ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, List<&2, List<&2, Nat>>>:
  confusion.go(pairs(&2, Nat, ys, ps), ys, ps, k)

# ---- the averages ----
type Scores is Data:
  Scores{p: F32, r: F32, f: F32}

def p_of(s: Scores) -> F32:
  match s:
    case Scores{p, r, f}:
      p

def r_of(s: Scores) -> F32:
  match s:
    case Scores{p, r, f}:
      r

def f_of(s: Scores) -> F32:
  match s:
    case Scores{p, r, f}:
      f

def val_of(r: Result<&1, &1, Bad, F32>) -> F32:
  match r:
    case Done{v}:
      v
    case Fail{e}:
      0.0

def div_or(z: Bool, +a: F32, +b: F32) -> F32:
  match z:
    case True{}:
      0.0
    case False{}:
      (a / b : F32)

def sums_go(+left: Nat, +ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat, +total: Nat, +w: Bool, +acc: Scores) -> Scores:
  match left:
    case 0n:
      acc
    case 1n+r:
      +i = Nat.sub(k, left)
      +pv = val_of(precision.go(Done{total}, ys, ps, i))
      +rv = val_of(recall.go(Done{total}, ys, ps, i))
      +fv = val_of(f1.go(Done{total}, ys, ps, i))
      +wt = Bool.pick(F32, w, F32.from_nat(count_true(ys, i)), 1.0)
      sums_go(r, ys, ps, k, total, w, Scores{(p_of(acc) + pv * wt : F32), (r_of(acc) + rv * wt : F32), (f_of(acc) + fv * wt : F32)})

def sums(+ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat, +total: Nat, w: Bool) -> Scores:
  sums_go(k, ys, ps, k, total, w, Scores{0.0, 0.0, 0.0})

def macro_precision.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  match r:
    case Done{+n}:
      Done{div_or(Nat.is_eq(k, 0n), p_of(sums(ys, ps, k, n, False{})), F32.from_nat(k))}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def macro_precision(+ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  macro_precision.go(pairs(&2, Nat, ys, ps), ys, ps, k)

def weighted_precision.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  match r:
    case Done{+n}:
      Done{(p_of(sums(ys, ps, k, n, True{})) / F32.from_nat(n) : F32)}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def weighted_precision(+ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  weighted_precision.go(pairs(&2, Nat, ys, ps), ys, ps, k)

def macro_recall.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  match r:
    case Done{+n}:
      Done{div_or(Nat.is_eq(k, 0n), r_of(sums(ys, ps, k, n, False{})), F32.from_nat(k))}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def macro_recall(+ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  macro_recall.go(pairs(&2, Nat, ys, ps), ys, ps, k)

def macro_f1.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  match r:
    case Done{+n}:
      Done{div_or(Nat.is_eq(k, 0n), f_of(sums(ys, ps, k, n, False{})), F32.from_nat(k))}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def macro_f1(+ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  macro_f1.go(pairs(&2, Nat, ys, ps), ys, ps, k)

def weighted_recall.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  match r:
    case Done{+n}:
      Done{(r_of(sums(ys, ps, k, n, True{})) / F32.from_nat(n) : F32)}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def weighted_recall(+ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  weighted_recall.go(pairs(&2, Nat, ys, ps), ys, ps, k)

def weighted_f1.go(r: Result<&1, &1, Bad, Nat>, +ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  match r:
    case Done{+n}:
      Done{(f_of(sums(ys, ps, k, n, True{})) / F32.from_nat(n) : F32)}
    case Fail{Mismatch{expected, got}}:
      Fail{Mismatch{expected, got}}
    case Fail{Empty{}}:
      Fail{Empty{}}

def weighted_f1(+ys: List<&2, Nat>, +ps: List<&2, Nat>, +k: Nat) -> Result<&1, &1, Bad, F32>:
  weighted_f1.go(pairs(&2, Nat, ys, ps), ys, ps, k)

def show(r: Result<&1, &1, Bad, F32>) -> String:
  match r:
    case Done{v}:
      F32.show(v)
    case Fail{e}:
      "fail"

def row_show(rs: List<&2, Nat>) -> String:
  match rs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(U32.show(U32.from_nat(h)), String.append(" ", row_show(t)))

def mat_show(rs: List<&2, List<&2, Nat>>) -> String:
  match rs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(row_show(h), mat_show(t))

def cm(r: Result<&1, &1, Bad, List<&2, List<&2, Nat>>>) -> String:
  match r:
    case Done{v}:
      mat_show(v)
    case Fail{e}:
      "fail"
