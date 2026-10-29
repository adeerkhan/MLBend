# mlbend-report

The confusion matrix and the per-class report of a classification, natively in
Bend 2: what an estimator's predictions *look like*, beside the numbers
[`mlbend-metrics`](../metrics/README.md) answers.

- Bend: **2.0.38**
- License: MIT
- No `@unsafe`, no `?TODO`: `bend report/main.bend` prints `ALL PROOFS CHECK`,
  and `bend report/main.bend --verdict` re-checks it with the Lean-proved kernel.

## Usage

```python
import Base
import mlbend-report@0.1.0.0/main.bend as R

def main():
  do IO<Unit>:
    r = R.report([0n, 0n, 1n, 1n, 2n, 2n, 2n], [0n, 1n, 1n, 0n, 2n, 2n, 0n], 4n)
    # r is a Result: the report, or why there is none
    return Unit{}
```

The labels are `Nat`: the class 0, the class 1, and so on up to `k-1`. A class
nobody predicted scores `0.0`, the convention scikit-learn's
`zero_division=0` uses.

## The matrix

| Function | What it answers |
|---|---|
| `cell(ys, ps, i, j)` | the number of samples whose true label is `i` and whose predicted label is `j` |
| `row_from(ys, ps, k, i, n, j)` | `n` cells of the class-`i` row, from the column `j` on, ascending |
| `row(ys, ps, k, i)` | the class-`i` row: `k` cells, ascending |
| `matrix_from(ys, ps, k, n, i)` | `n` rows, from the class `i` on, each of `k` cells |
| `matrix(ys, ps, k)` | the confusion matrix of `k` classes: `k` rows of `k` cells |

The row is the true label and the column the predicted one, the way
scikit-learn's `confusion_matrix` lays it out.

## The report

| Function | What it answers |
|---|---|
| `cls_of(ys, ps, i)` | the class `i`'s `Cls`: precision, recall, F1, support |
| `classes_of(ys, ps, k)` | one `Cls` per class, the class 0 first |
| `macro_of(classes, k)` | the mean of the per-class scores, every class weighing the same |
| `weighted_of(classes, total)` | the per-class scores, each weighed by its support |
| `report(ys, ps, k)` | the whole `Report`, or why there is none |

A `Report` carries one `Cls` per class, the macro and the weighted averages,
and the accuracy. The per-class numbers rest on `mlbend-metrics`' counts
(`count_true`, `count_pred`, `count_both`) and on its zero-division
convention. The macro and the weighted averages match scikit-learn's
`classification_report` with `zero_division=0`.

## When a report cannot answer

`report(ys, ps, k)` answers a `Result`:

| Answer | When |
|---|---|
| `Done{report}` | the report |
| `Fail{Mismatch{expected, got}}` | the two sequences do not have the same length |
| `Fail{Empty{}}` | there is no sample |

## The laws

| Law | What it says |
|---|---|
| `matrix_rows` | `n` rows of the matrix are `n` rows |
| `row_length` | `n` cells of a row are `n` cells |

The laws are stated over the counted forms (`matrix_from`, `row_from`),
because those are the functions whose recursion the checker can follow: the
counter of the rows (or of the cells) left is the argument that shrinks. That
the matrix of `k` classes has `k` rows of `k` cells is the laws at `n := k`.

**The debt.** "The cells add up to the number of samples" — every sample lands
in exactly one cell — is a count that depends on an element's *value*, the same
wall `mlbend-metrics`' accounting law and `mlbend-split`'s sum law record
([issue #1](https://github.com/adeerkhan/MLBend/issues/1),
[issue #8](https://github.com/adeerkhan/MLBend/issues/8),
[issue #9](https://github.com/adeerkhan/MLBend/issues/9)). It is a debt of
this package, not a law.

## What is proved and what is tested

The shape of the matrix is proved; the `F32` numbers are not, because `F32` is
not a real number. `reference/test_report.py` compares the matrix, the
per-class scores, the averages and the accuracy of `report/tests/ok.bend`
with scikit-learn's `confusion_matrix` and `classification_report`, within a
relative `1e-5`. `examples/classification/knn` prints the report beside its accuracy, and
`reference/test_knn.py` checks both against scikit-learn's kNN on the same
predictions. The trust base is in [`docs/AUDIT.md`](../docs/AUDIT.md).
