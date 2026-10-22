# Handoff

One file, overwritten at the end of every session. It is the literal memory
context: a fresh agent reads this first, then the issue it names, then
`AGENTS.md`.

**Write it before you stop.** An agent that ends a session without writing this
leaves the next one blind.

## 2026-10-22 — the preprocess package, and the containers answered

**Where we are.** `mlbend-preprocess` (v0.3, issue #5) is written, checked,
published to the hub (`0xa8182854f375cd61135ccd32993993cc`) and on GitHub
under the tag `v0.1.0.0`. The transformers are standardize, min-max, L1
and L2 normalisation and one-hot, and they run over **bend-ml's** matrix
(`bend-ml-tensor`'s `Mat<r, c>`) — not over a container of ours. The
logistic example now standardises its features, so the learning rate of
0.5 is sane.

**What was run.**
- `bend preprocess/main.bend --check-only` → `ALL PROOFS CHECK`.
- `bend preprocess/tests/ok.bend` → the standardised, min-maxed, L1 and
  L2 normalised and one-hot matrices, and the number of rows of the
  one-hot, matching the `#|` block pinned in the file.
- `reference/.venv/bin/python reference/test_preprocess.py` → the numbers
  match scikit-learn's `StandardScaler`, `MinMaxScaler`, `Normalizer`
  (l1 and l2) and `np.eye`, to the precision of an F32.
- `bend -o /tmp/logistic examples/logistic/main.bend && /tmp/logistic` →
  `loss before: 1.0986128`, then `0.7287295`, `0.6035278`, `0.5403903`
  at a learning rate of 0.5. On the raw features that rate makes the
  loss rise.

**What was not run.**
- The kernel check (`--verdict`) of the new package: Lean 4.34.0 is not
  on this machine. CI proves it.
- `make check` end to end after the last commits; it was green before
  them (run 39641765719, 31/31).

**What I learned.**
- A transformer over `Mat<r, c>` must declare `+r: Nat, +c: Nat`
  (linear): bend-ml's `transpose` and `layernorm` take the width as a
  runtime argument, and the dimensions are erased type parameters.
- `List.replicate` and a "replace the nth element" fold are the cheap
  way to build a one-hot row; a counter is never a decreasing argument.

**What is next.**
1. `mlbend-split` (v0.4): a seeded, stratified split, with the partition
   law — the first law about the data rather than about arithmetic, and
   the first one the kernel can prove.
2. The bendtorch tape path (`examples/tape/tape.bend`): the shapes are
   settled (the weight is [out, in], the bias is [out] — see NOTES.md),
   and the example now builds them so, but it still refuses with 106.
   The next step is to read the shapes back out of the tensors and
   compare them with what the tape recorded.
3. `mlbend-cluster` (v0.7): the k-means of `examples/kmeans/main.bend`
   promoted to a package, with the law that every point is in exactly
   one cluster.

## 2026-10-21 — the datasets, and k-means over one of them

**Where we are.** The datasets package grew a generic CSV loader, and four
datasets come from HuggingFace over the network at run time: iris (150
rows), wine (6497), breast cancer (683) and heart failure (299). The first
model that is *not* a neural network is next to them: k-means over the wine
dataset, in plain Bend, with no gradients and no library.

**What was published.**
- `mlbend-datasets@0.1.3.0` → `0xa77cd723e17e54a2544513eecb372be6`, and then
  `mlbend-datasets@0.1.4.0` → `0x805aa9d73de5385dd9f66c3249152265`: the
  generic CSV loader and then the loader that leaves out the columns at
  the end of a row, so the label of the wine dataset is the quality and
  not the `is_red` column next to it. Both are on the hub and on
  GitHub, under the tag of the same version.

**What was run.**
- `bendenv run -- bend datasets/tests/datasets.bend` (built with `bend -o`,
  see below) → `iris: 150 rows, 150`, `wine: 6497 rows, 6497`,
  `cancer: 683 rows, 683`, `heart: 299 rows, 299`, and the same numbers are
  pinned in the file.
- `bend -o /tmp/kmeans examples/kmeans/main.bend && /tmp/kmeans` →
  `inertia: 57343.812` and `cluster sizes: 1071 717 4709`.
- `reference/.venv/bin/python reference/test_kmeans.py` → the cluster sizes
  are the same as NumPy's (1071, 717, 4709) and the inertia agrees
  (57343.812 against 57344.11955916257).
- `reference/.venv/bin/python reference/kmeans_wine.py` → the NumPy side on
  its own: rows 6497, inertia 57344.12, sizes 1071 717 4709.
- `bend -o /tmp/knn examples/knn/main.bend && /tmp/knn` → `train rows: 546`,
  `test rows: 137`, `accuracy: 0.9781022`.
- `reference/.venv/bin/python reference/test_knn.py` → scikit-learn's
  `KNeighborsClassifier(n_neighbors=3)` over the same split gives
  0.9781021897810219.

**What was not run.**
- The kernel check (`--verdict`): the Lean toolchain (4.34.0) was still
  downloading on this machine, so the laws are *checked*, not *proved*, here.
  CI proves them.
- `make check` → **28/28** on this machine, with the kernel check skipped
  (no Lean here).
- CI (run 39641765719) → **31/31**, with the kernel check: the law of
  `mlbend-metrics` is *proved* by Lean 4.34.0.

**What was broken, and is fixed now.**
- **The CI was red before this session.** Every run since 09:55 failed. The
  reason: `bend hf/main.bend` ends in "SOME PROOFS FAIL: 312 defs rely on
  unsafe or foreign code". A file with no `main` is checked by the verdict
  path, and the verdict refuses a book that holds any foreign code — the
  socket bindings of the http package are. A package that does IO can
  never have a clean verdict, so the check now reads it as a pass when the
  only complaint is the foreign code.
- **clang was gone.** The wrapper at `~/.local/bin/clang` pointed at
  libraries symlinked into `/tmp/opencode`, and `/tmp` is cleaned: the
  builds failed with "bend needs clang 14 or newer (found no clang)". The
  wrapper now names the tree's own libraries, and `scripts/setup.sh` asks
  for clang.
- The examples are not part of `make check`: the checks live in
  `reference/test_*.py` and in the packages' own `tests/`.

**What I learned (the cost of the day).**
- **A counter is never a decreasing argument.** The checker reads the
  arguments of a self-call left to right and wants the first changed one to
  shrink structurally (`term_descend` in `bend2/bend.ts`), so
  `f(Nat.sub(k, 1n), ...)` is refused. A counted loop is
  `List.range(n)` and a loop over a list is the tail; the index rides along
  as an accumulator. This is the rule I lost the most time to.
- **A def body has one result.** Two statements in a row are a parse error;
  effects go in a `do IO<Unit>:` block, and a `match` heads a def body, so
  it needs a def of its own.
- **The interpreter's stack is a few hundred frames.** A fold over 300 rows
  overflows `bend file.bend`; the built program runs. Every test in
  `reference/check_all.py` is now built and then run.
- **`scripts/reuse.py`** marks the parameters, the pattern binders and the
  binds that a body reads twice, so the `+` that Bend wants is not found by
  hand. **`scripts/sort_defs.py`** now keeps a comment with the def it
  describes.

**What is next.**
1. `mlbend-preprocess`: standardise the features as a package, so the
   logistic example can use a learning rate that is not 0.01.
2. The estimators of `docs/PLAN.md` v0.3–v0.6 over the online data: the
   nearest-centroid classifier, k-NN, and the linear model.
3. The bendtorch tape path (`examples/tape/tape.bend`): the weight
   orientation and the rank of the bias are still unsettled, and the
   reference is `bendtorch/examples/mnist/train.bend`.
4. A regression model over the wine quality: the dataset now carries the
   quality as its label (eleven features), so the least squares is the
   next estimator, and `mlbend-preprocess` is what makes it fit.

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
