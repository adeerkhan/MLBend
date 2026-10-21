# Handoff

One file, overwritten at the end of every session. It is the literal memory
context: a fresh agent reads this first, then the issue it names, then
`AGENTS.md`.

**Write it before you stop.** An agent that ends a session without writing this
leaves the next one blind.

## 2026-10-20 — the environment, and v0.1 `mlbend-metrics`

**Where we are.** Two things. The first package is written, checked and
pushed: the evaluation surface (MSE, MAE, RMSE, accuracy, precision,
recall, F1) and the counts they rest on, with one law proved. And the
repository is a workspace now, with one lockfile and one environment for
every package here, built by `bendenv` (https://github.com/adeerkhan/bend-env).

**What was run (2026-10-21).**
- `bendenv sync` → 22 packages, 233 files linked out of the global cache.
- `bendenv run -- bend datasets/tests/iris.bend` → `rows: 150, classes: 0,0,`:
  the iris dataset fetched from HuggingFace at run time, parsed in Bend,
  and put into a `Mat<150n, 4n>` with its labels.
- `bendenv run -- bend hf/tests/hello.bend` → `dataset: mstz/iris` and
  `first row label: 5`: the dataset page and the rows server, both read
  over HTTP with `bend-kit-http` and `bend-kit-json`.
- `bendenv check` → the environment matches the lock.
- `bendenv lock` → the collision report names `Text` between
  `emerging-ezhttp`, `bend2/base.bend` and `ezx`. That is the clash that
  stops `ezx@1.5.0.0` from compiling against bend 2.0.37.

**What was run.**
- `bend metrics/main.bend --check-only` → `ALL PROOFS CHECK`.
- `bend metrics/main.bend --verdict` → **the Lean kernel proves the law**
  (CI, run 39372667256). So `count_length` is proved, not merely checked.
- `bend metrics/tests/ok.bend`, `metrics/tests/error.bend` → the ten
  metrics and the three error answers, matching the `#|` block pinned in
  the file.
- `bend metrics/tests/bad_affine.bend`, `bad_type.bend` → refused, as they
  must be.
- `reference/test_metrics.py` → 10/10 against NumPy and scikit-learn
  (numpy 2.5.4, scikit-learn 1.9.1, Python 3.14).
- `make check` → 11/11 locally (the kernel check is skipped without Lean,
  and says so) and 12/12 in CI.
- `make env` → the lock and the environment: 27 packages.
- `make env-check` → the environment matches the lock, and
  `examples/stacks/main.bend` runs over it:
  `bend-ml-tensor-array: [12, 12, 12, 12] | bendtorch-core: built a
  tensor of 4 words`.
- `bendenv run -- bend examples/logistic/main.bend` → a logistic
  regression over the iris dataset, fetched from HuggingFace at run
  time, trained with bend-ml's autodiff: `loss before: 1.0986128` then
  `1.086162`, `1.0759476`, `1.0671557`.
- `bendenv run -- bend examples/tape/tape.bend` → the same model over
  bendtorch's tape: it compiles and runs, and the `linear` node refuses
  the operands (error 106).
- `bendenv hash bendtorch.bend` on `aaaxn/bendtorch`'s checkout →
  `0x1689e5deb278c16abd3bd45eac101ef3` over 196 files, the hash the hub
  holds for `bendtorch-core@0.1.0.0`.

**What was NOT run.** `bendenv publish` was never run: publishing is
public and permanent, and it needs `bend login`. Only the hash was
computed, and it matches the hub's.

**What is next.** `docs/PLAN.md` has the order; the two open issues have
the detail. The next piece of work is an estimator over the online
data: the iris rows are here, the models are bend-ml's and bendtorch's,
and the examples show both driving a training step. The open question is
the shape convention of bendtorch's weights, which its `linear` node
refuses (error 106); its own MNIST demo is the reference.
- issue #1: the accounting law (the samples right and the samples wrong
  add up to the number of samples), `r2`, the log loss, the macro averages.
- issue #2: the containers — lists or bend-ml's `Mat` over a flat
  `Array<F32>`. The environment answers half of it: the flat one is
  `bend-ml-tensor-array`, the list one is `bend-ml-tensor`, and they are
  two different `Mat` types that do not mix. The measurements still go
  in v0.2.

**What is unverified.** Everything in `NOTES.md` that is marked as
measured on one machine. The numbers of the metrics are stable (F32, a
fixed order of sums); the *performance* of the packages is not measured
at all yet. The environment was built and checked on one machine, with
the hub about 4,000 km away.

**Open questions for the maintainer.**
- Should the packages be published to BendHub from this repository, and
  under which account? The first version is `mlbend-metrics@0.1.0.0`;
  publishing is public and permanent, and it needs `bend login`.
- The flat tensor library has no autodiff behind it: `bend-ml-autograd`
  is written against the list-based `bend-ml-tensor`. Which way should
  the deep-learning half go — an autodiff over `bend-ml-tensor-array`, or
  the list-based tensors everywhere? The answer decides what v0.3 and
  v0.6 are written against.
