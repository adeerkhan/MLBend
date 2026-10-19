# mlbend-metrics

The evaluation surface of machine learning, natively in Bend 2: the error metrics
of a regression, the scores of a classification, and the counts they rest on.

- Bend: **2.0.37**
- License: MIT
- No `@unsafe`, no `?TODO`: `bend metrics/main.bend` prints `ALL PROOFS CHECK`,
  and `bend metrics/main.bend --verdict` re-checks it with the Lean-proved kernel.

## Usage

```python
import Base
import mlbend-metrics@0.1.0.0/main.bend as M

# a metric answers a Result: the value, or why it could not answer
def show(r: Result<&1, &1, M.Bad, F32>) -> String:
  match r:
    case Done{v}:
      F32.show(v)
    case Fail{M.Mismatch{expected, got}}:
      String.append("mismatch ", U32.show(U32.from_nat(expected)))
    case Fail{M.Empty{}}:
      "empty"

def main():
  do IO<Unit>:
    Unit <- IO.print(show(M.mse([3.0, 0.5, 2.5], [2.5, 1.0, 2.0])))
    return Unit{}
```

## The metrics

| Function | What it answers | What scikit-learn calls it |
|---|---|---|
| `mse(ys, ps)` | the mean squared error | `mean_squared_error` |
| `mae(ys, ps)` | the mean absolute error | `mean_absolute_error` |
| `rmse(ys, ps)` | the root mean squared error | `root_mean_squared_error` |
| `accuracy(ys, ps)` | the fraction of the samples predicted right | `accuracy_score` |
| `precision(ys, ps, k)` | of the samples predicted `k`, the fraction really `k` | `precision_score` |
| `recall(ys, ps, k)` | of the samples really `k`, the fraction predicted `k` | `recall_score` |
| `f1(ys, ps, k)` | the harmonic mean of the precision and the recall | `f1_score` |

`ys` and `ps` are the targets and the predictions: `List<&2, F32>` for a
regression, `List<&2, Nat>` (the labels) for a classification. The two
sequences must have the same length.

## The counts

The counting core is over `Nat`, so it can be proved, and the metrics rest on it.

| Function | What it answers |
|---|---|
| `count(ys)` | the number of samples |
| `count_true(ys, k)` | the number of samples whose true label is `k` |
| `count_pred(ps, k)` | the number of samples whose predicted label is `k` |
| `count_both(ys, ps, k)` | the number of samples whose true and predicted labels are both `k` |
| `count_right(ys, ps)` | the number of samples predicted right |

## When a metric cannot answer

A metric never crashes: it answers a `Result`.

| Answer | When |
|---|---|
| `Done{value}` | the metric |
| `Fail{Mismatch{expected, got}}` | the two sequences do not have the same length |
| `Fail{Empty{}}` | there is no sample |

A metric whose denominator is zero (the precision of a class nobody predicted)
answers `0.0`, the convention scikit-learn uses with `zero_division=0`.

## The laws

| Law | What it says |
|---|---|
| `count_length` | the number of samples is the length of the list of labels |

## What is proved and what is tested

The counts are proved; the `F32` numbers are not, because `F32` is not a real
number. `reference/test_metrics.py` compares the ten metrics of
`metrics/tests/ok.bend` with NumPy and scikit-learn, within a relative `1e-5`
(the two round differently). The trust base is in [`docs/AUDIT.md`](../docs/AUDIT.md).
