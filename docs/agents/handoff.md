# Handoff

One file, overwritten at the end of every session. It is the literal memory
context: a fresh agent reads this first, then the issue it names, then
`AGENTS.md`.

**Write it before you stop.** An agent that ends a session without writing this
leaves the next one blind.

## 2026-10-20 — v0.1, `mlbend-metrics`

**Where we are.** The first package is written, checked and pushed: the
evaluation surface (MSE, MAE, RMSE, accuracy, precision, recall, F1) and the
counts they rest on. One law is proved (`count_length`). The ten metrics of
`metrics/tests/ok.bend` match NumPy and scikit-learn within a relative 1e-5.

**What was run.**
- `bend metrics/main.bend --check-only` → `ALL PROOFS CHECK`.
- `bend metrics/main.bend --verdict` → **the Lean kernel proves the law**
  (CI, run 39372667256: `ok metrics/main.bend --verdict (the kernel, as Lean)`).
  So `count_length` is proved, not merely checked.
- `bend metrics/tests/ok.bend`, `metrics/tests/error.bend` → the ten metrics and
  the three error answers, and the output matches the `#|` block pinned in the
  file.
- `bend metrics/tests/bad_affine.bend`, `bad_type.bend` → refused, as they must
  be.
- `reference/test_metrics.py` → 10/10 against NumPy and scikit-learn
  (numpy 2.5.4, scikit-learn 1.9.1, Python 3.14).
- `make check` → 11/11 locally (the kernel check is skipped without Lean, and
  says so) and 12/12 in CI.

**What was NOT run.** Nothing of the plan is unverified except what is listed
below as unverified; the only thing this machine cannot do is the kernel check
(it has no Lean 4.34.0), and CI does it.

**What is next.** `docs/PLAN.md` has the order; the two open issues have the
detail:
- issue #1: the accounting law (the samples right and the samples wrong add up
  to the number of samples), `r2`, the log loss, the macro averages.
- issue #2: the containers — lists or bend-ml's `Mat` over a flat `Array<F32>`.
  v0.2 measures both before choosing; the estimators cannot be written until
  this is settled.

**What is unverified.** Everything in `NOTES.md` that is marked as measured on
one machine. The numbers of the metrics are stable (F32, a fixed order of
sums); the *performance* of the packages is not measured at all yet.

**Open questions for the maintainer.**
- Should the packages be published to BendHub from this repository, and under
  which account? The first version is `mlbend-metrics@0.1.0.0`; publishing is
  public and permanent, and it needs `bend login`.
- The `mlbend-linalg` container choice (issue #2) is worth a word from whoever
  wrote bend-ml's tensor packages.
