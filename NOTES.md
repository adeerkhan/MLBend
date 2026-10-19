# Notes

Every finding, every measurement and every discarded hypothesis, by date. Read the
relevant entry before changing performance-sensitive code or a law.

## 2026-10-20

- **v0.1: `mlbend-metrics`.** The first package: the evaluation surface (MSE,
  MAE, RMSE, accuracy, precision, recall, F1) and the counts they rest on. The
  ten metrics of `metrics/tests/ok.bend` match NumPy and scikit-learn within a
  relative 1e-5 (`reference/test_metrics.py`). One law proved so far:
  `count_length`, the number of samples is the length of the label list.
- **The plan** is in `docs/PLAN.md`: the packages, the order and the stop
  criteria. The order is metrics -> linalg -> preprocess -> split -> linear ->
  cluster -> pipeline -> datasets -> select.
- **Bend traps this package paid for** (they are in `AGENTS.md`): `match` only
  inspects a parameter, so every comparison is computed by the caller and passed
  in (`same_len_of(Nat.is_eq(n, m), n, m)`); there is no `let` keyword (`x = v`,
  and `+x = v` to reuse); a def cannot be used before it is defined, so the
  helpers come first; `Result` is a `Type` and cannot be reused with `+`; the
  comparison operators are not terms (`Nat.is_eq`, `F32.is_eq`); a negative
  literal is not a term (`(0.0 - 0.5 : F32)`).
- **Debt, not in the package.**
  - The accounting law ("the samples predicted right and the samples predicted
    wrong add up to the number of samples") is not in v0.1. It needs the
    hypothesis that the two label lists have the same length, and the proof
    needs the counts to come out of one walk, so the statement can name both.
    It is v0.2's first law, with the `mlbend-linalg` inner-product lemmas.
  - The classification metrics are stated per class (`precision(ys, ps, k)`);
    scikit-learn also has a macro and a weighted average over the classes.
    Those come with the confusion matrix, in v0.2.
  - `r2` and the log loss are not here: both need the variance of the targets
    (a second sum) and a zero-denominator case of their own. v0.2.
