# Handoff

One file, overwritten at the end of every session. It is the literal memory
context: a fresh agent reads this first, then the issue it names, then
`AGENTS.md`.

**Write it before you stop.** An agent that ends a session without writing this
leaves the next one blind.

## 2026-10-24 — the cache, a warm corpus, and a clean check

**Where we are.** The foundation is settled: **Bend 2.0.38**, the maintained
**`bendtorch@1.0.1.0`** (`nuxyel/BendTorch`), `mlbend-datasets@0.1.5.0`, nine
tasks, and a full check that passes.

- **The dataset cache is real.** `mlbend-datasets@0.1.5.0` reads
  `MLBEND_DATA` when it is set, laid out as `<repo>/<path>`
  (`mstz/iris/iris.data`); a missing file falls back to the hub. The runner
  fills `reference/.cache/datasets` with the exact bytes it hashes and sets the
  variable before the Bend run. The cache is gitignored; the SHA-256 stays in
  every manifest. The release is on the hub at
  `0x508fc33b44130576819939e645eba7da` and tagged
  `mlbend-datasets-v0.1.5.0`.
- **The corpus is warm.** The last corpus run has `fetch` 0–1 ms on every task,
  so `compute` is the run itself: iris-crossval **0.098 s**, cancer-knn
  **0.225 s**, transformer-periodic **0.032 s**, wine-pca **5.108 s**. The
  first run on a cold cache pays the fetch; `make experiments` twice (or one
  warm-up) is the honest sequence. `results/RESULTS.md` says this in its intro.
- **`make check` → 66/66** on 2.0.38, including the new
  `reference/test_datasets_cache.py` (a five-row iris cache must produce five
  rows, not the 150 the hub would give). **CI on `0ef436f` passed 74/74 in
  22m27s.**
- **The NumPy adapter covers all nine tasks** under `results/latest-numpy/`;
  the same task and manifest, NumPy as the trainer.
- **The transformer stays the trained model**: loss **2.3591 → 0.0021**,
  accuracy **1.0**; the NumPy twin starts at 2.3591 (6.8e-7 apart) and ends at
  0.1351 (0.9375). `reference/test_transformer.py` rewrites
  `results/transformer/run.json` and the curve on every check.
- **The structural graph** at `.ua/knowledge-graph.json` was built at `05eacbd`:
  102 files, 958 nodes, 905 edges, 6 layers, 7 tour steps, no validation
  issues. It is structural, not per-file LLM enrichment; use it to orient, not
  as evidence.

**What was run.**
- `make experiments` on the warm cache, then `make results`; everything passed.
- `reference/experiment.py --all --trainer numpy`; reports under
  `results/latest-numpy/`.
- `make check` with `BEND_BIN` pointing at the pinned 2.0.38; **66/66**.

**What is next.**
1. Move the kNN classification report into a Bend package with a law; the
   Python report is the oracle.
2. A second transformer task with a classification head or a copy task, with
   accuracy beside the loss.
3. Run the W&B sweep (`reference/wandb_sweep.yaml`), which is configured but
   has not been started (it needs a login and a sweep id).

**What is unverified.**
- The timings are one machine and a warm cache; the fetch column is the
  cache-fill/hash time, not a network measurement.
- The cache reads a file at its size and then `File.read` once; a very large
  file with a short read would still be a silent truncation. The datasets here
  are small, and the cached bytes are the hashed bytes.
- `/understand`'s graph is structural; no per-file LLM enrichment.

## 2026-10-23 — the first transformer block


**Where we are.** The deep-learning track has started. `examples/attention` is
a single-head self-attention block — Q, K, V, the scaled dot-product scores, a
softmax over every row, the weighted sum of V — written in plain Bend over four
tokens of eight features. The weights are a fixed formula, generated the same
way in the PyTorch reference; every output number agrees to **1.5e-8**
(`reference/test_attention.py`). It is a task in the experiment layer,
`attention-parity` (checksum 1.76295, delta 3e-9).

**The finding: training a transformer is blocked on BendTorch.** Its tape
records add, sub, mul, matmul, relu, slice, gather, cross-entropy and MSE. It
has no recorded `exp`, no broadcast `div`, and so no softmax backward. The
forward pass is pure Bend today; a trained attention task needs those
operations (and their VJPs) on the tape first.

**What was run.**
- `reference/test_attention.py` → max absolute difference **1.49e-8**, checksum
  1.762954 against 1.762954.
- `reference/experiment.py --task attention-parity` → PASS.
- The full `make check` is running; the test is new and self-contained.

**What is next.**
1. An issue on `nuxyel/BendTorch` for `exp`, broadcast `div` and `sum_dims`
   gradients (softmax), with the attention forward as the use case. If accepted,
   a recorded `softmax` is the smaller surface.
2. Then a trained attention task: same block, a sequence task, PyTorch as the
   oracle.
3. A BendTorch deep MLP task (the MNIST example is the prototype) to have a
   trained deep model in the corpus before the transformer trains.

## 2026-10-23 — the pipeline, and no leakage


**Where we are.** `mlbend-pipeline@0.1.0.0` is written, published
(`0xea73a86e2118529cadd597fb3727979b`, tag `mlbend-pipeline-v0.1.0.0`) and
used by `examples/crossval`. The scaler is fitted on the training rows of a
fold and applied to both sides; the leakage that standardised the whole matrix
before the split is gone. The law `pipeline_preserves_rows` (one output row per
input row) is checked by Bend's checker; `reference/test_pipeline.py` compares
the means, the deviations and the transformed rows — unseen test rows included
— with scikit-learn's `StandardScaler`, and `reference/test_crossval.py` now
compares against `Pipeline(StandardScaler, LogisticRegression)`, the reference
with no leakage of its own.

**What was run.**
- `make check` → **64/64** locally (the pipeline package, its test, and
  `test_pipeline.py` added to the previous 58).
- `reference/test_crossval.py` → Bend mean **0.9333**, scikit-learn **0.9533**;
  the mean is unchanged, the gap is the two shuffles, not the scaling.
- `reference/experiment.py --task iris-crossval` → PASS, 0.9333 against 0.9533.
- A resolution check with `BEND_LIB` unset: the published package runs from the
  hub, so CI does not need a local workspace.

**What CI did.** The run on `42ad0f4` was cancelled at **30m16s**: 58 checks,
each building a program, do not fit in 30 minutes on a 4-vCPU runner.
`.github/workflows/ci.yml` now allows **60** minutes and pushes only on `main`
(the old `devel` filter is gone). The next run validates the pipeline.

**What was not run.** The kernel `--verdict` here (no Lean 4.34.0); CI runs it.
The pipeline imports only Base, so the kernel has no foreign code to refuse.

**What is next.**
1. The results methodology: cache the datasets, split `fetch_ms` from
   `compute_ms`, and put the dataset's SHA-256 in the manifest. The current
   task timings include the HuggingFace fetch on both sides.
2. The adapter protocol: BendTorch and bend-ml's autograd behind one task
   interface, so a trainer changes without a task changing.
3. Model selection and the report (grid/nested CV, confusion matrix, R²).

## 2026-10-23 — the experiment layer, on main, with results kept


**Where we are.** `devel` was merged into `main` (fast-forward, 55 commits) and
`main` is the work branch now. MLBend is no longer "the classic-ML half"; it is
the **experiment layer**: bend-ml owns the tensors and the laws, BendTorch owns
the training, and this repository owns the data, splits, transformations,
selection, metrics, reports and manifests. The boundary rule is in `AGENTS.md`,
`README.md` and `docs/PLAN.md`: a model enters through an adapter; no tensor, no
autodiff, no optimizer, no second trainer.

**What was built.**
- `reference/experiment.py`: seven tasks, each with a parser, a NumPy or
  scikit-learn reference, a tolerance table and a manifest (git commit and
  dirtiness, Bend version, every `mlbend-*`/`bend-ml-*`/`bendtorch-*` hub hash,
  machine, seed, build time, run time, reference time).
- **Results are committed**: `results/RESULTS.md` (the table),
  `results/BENCH.md` (the kernel benchmark), `results/latest/<task>.json`,
  `results/raw/`. `make experiments`, `make bench`, `make results`.
- **Weights & Biases**: `--wandb` logs the manifest, offline without
  `WANDB_API_KEY`; verified with `--offline` (wandb 0.31.0, offline-run file
  written). `make experiment TASK=…` runs one task.
- Two new examples and tasks: `examples/naive_bayes` (0.98540145, exactly
  scikit-learn's `GaussianNB`) and `examples/pca` (first component 0.275443,
  scikit-learn 0.2754426).
- `bench/matmul` + `reference/bench_matmul.py`: a compute-only kernel
  benchmark, 50 sequential 128x128 products, Bend 20.9 ms / 10.0 GFLOP/s
  against NumPy 1.0 ms / 201.1 GFLOP/s (NumPy 19x; one Bend thread against
  BLAS).

**What was run.**
- `make check` → **58/58** locally (the two new reference tests included).
- `make experiments` → every task PASS; the committed table has the numbers.
  `iris-crossval` reports per-fold scores but gates only the mean: the two
  libraries shuffle with different generators, so a single fold is not
  comparable.
- `make bench` → 20.9 ms Bend, 1.0 ms NumPy.
- The task timings include the HuggingFace fetch on both sides; the kernel
  benchmark is the fetch-free comparison. This is stated in `RESULTS.md`.

**What was not run.** The kernel `--verdict` here (no Lean 4.34.0); CI runs it.
W&B online (no API key); offline was verified. `make experiments` was not added
to CI: the task timings would be noisy and the fetch needs the network.

**Laws: the rule we settled.** A law is for a pure structural claim — a count,
a partition, a cover, a composition — and the kernel proves it. `F32` values,
wall times, dataset fetches and W&B runs are tested or measured, not proved. So
the runner and the two new examples add no laws (they are examples, not
packages); the substrate keeps the seven it has; and the **next package owes
one**: the no-leakage pipeline, whose law is that fitting sees only the
training rows of the fold.

**What is next.**
1. `mlbend-pipeline`: fit every transformer on the training rows of the fold
   only, and prove it. `examples/crossval` currently standardises before the
   split; that is the leakage the package removes.
2. The adapter protocol, written down: BendTorch and bend-ml's autograd behind
   one task interface, so a trainer changes without a task changing.
3. Model selection: grid/random search, nested CV, per-fold tables.
4. Reports: confusion matrix, per-class precision/recall/F1, R², intervals.
5. The corpus: more datasets and seeds, and a committed table per release.

## 2026-10-23 — the cluster package, and two more examples


**Where we are.** The seven packages now include `mlbend-cluster` (k-means),
and two examples sit beside the existing ones: a two-layer network on
bend-ml and a linear regression on bendtorch's tape. `examples/kmeans` no
longer holds the algorithm; it holds the data and the standardising, and the
clustering is the package's.

**What was published.**
- `mlbend-cluster@0.1.0.0` → `0x0c4fbc7d44d2e3e47247a6c1f12c05e0`, on the hub
  and tagged `mlbend-cluster-v0.1.0.0` in this repository. Its law,
  `cluster_covers` (every row gets exactly one cluster), is checked by Bend's
  checker. The package imports only Base, so the kernel has no foreign code to
  refuse — unlike the split and the select, whose verdict carries
  bend-kit-random's two operating-system defs.

**What was run.**
- `make check` → **56/56 checks passed**: the six older packages and the
  cluster, the refused programs, the pinned outputs, and every NumPy and
  scikit-learn comparison, including the new `test_cluster.py`,
  `test_mlp.py` and `test_regression.py`.
- `examples/mlp`: loss 1.1172239 → 0.11065235, accuracy **0.9733**, against
  scikit-learn's `MLPClassifier(hidden_layer_sizes=(8,))` on the same data,
  also 0.9733.
- `examples/regression`: MSE 1.0000114 → **0.7078516**, against
  scikit-learn's `LinearRegression` at **0.7078631** (0.002 % apart).
- `examples/kmeans` through the package: inertia **50645.64**, sizes
  **2384 990 3123** — the numbers it had before the move.
- A resolution check with `BEND_LIB` unset: `cluster/tests/ok.bend` builds and
  runs against the hub's copy, so CI does not depend on a local workspace.

**What CI ran.** The kernel `--verdict` is not installed here (no Lean 4.34.0),
so CI ran it: run 39839677275 on `478fd1f` finished green in 18m37s with
**63/63 checks passed**, and `cluster/main.bend --verdict` says **ALL PROOFS
CHECK** — the kernel certifies `cluster_covers`, as expected for a package
that holds no foreign code. The other seven verdicts are the same as before.

**What I learned.**
- bendtorch's backward accumulates into the slots, and `Sgd.step` keeps the
  slot, so a training loop needs `Backward.zero_grad` between steps. Without
  it, 1000 steps ended at 0.9293 where 200 ended at 0.9000: more steps,
  worse loss. `NOTES.md` has the detail.
- A rectified network cannot start from zero weights: the gradient of the
  rectifier at zero is zero, so the hidden units never move. The example
  starts from a fixed small pattern and is reproducible.

**What is next.**
1. `mlbend-linear` (v0.6): ordinary least squares, ridge and logistic
   regression as a package. `examples/regression` is the prototype and
   `examples/logistic` the second.
2. `mlbend-pipeline` (v0.8): the typed chain of the transformers and an
   estimator.
3. The standardising is written three times now (`kmeans`, `regression`, and
   `crossval`), in list form. Either it moves into a package of its own or the
   two examples move onto `mlbend-preprocess` over bend-ml's matrix; decide
   when the pipeline lands, because the pipeline is the thing that would
   absorb it.

## 2026-10-22 — one repository for the six packages

**Where we are.** The six hub packages — `mlbend-metrics`,
`mlbend-preprocess`, `mlbend-split`, `mlbend-select`, `mlbend-datasets`,
`mlbend-huggingface` — are directories of this repository, in the workspace
`bend.toml`, and nothing else. The per-package GitHub repositories
(`adeerkhan/mlbend-*`) are redundant and are being deleted; `bend-env` stays
in a repository of its own, because it is the package manager and not a
package.

**What changed.**
- `scripts/release.sh` no longer copies a package into a repository of its
  own. It publishes to the hub, commits the released tree here (after
  refusing to start on a dirty tree), tags it `<name>-v<version>` and pushes
  the branch and the tag to `origin`. Tags already given per package
  (`v0.1.0.0`) would collide here; the name is in the tag now.
- The released versions of the six packages are tagged in this repository at
  the commit whose package directory holds the published bytes:
  `mlbend-metrics-v0.1.0.0`, `mlbend-preprocess-v0.1.0.0`,
  `mlbend-split-v0.1.1.0`, `mlbend-select-v0.1.0.0`,
  `mlbend-datasets-v0.1.4.0`, `mlbend-huggingface-v0.1.1.0`.
- The datasets README was the one file newer in the standalone repository
  (the four datasets, the `Table` type, the `drop` argument); it is here now,
  so both trees hold the same bytes.
- `README.md`, `AGENTS.md` and `docs/agents/domain.md` list all six packages
  and say there is one repository; the stale `mlbend-linalg` row is gone.
- `mlbend-datasets/data/` — the 132 KB of CSVs from the old design — was
  removed; the loader fetches the files at run time.

**What was run.** `bash -n scripts/release.sh` and
`python3 scripts/fix_versions.py` → every sibling import holds its current
version. `make check` → **48/48 checks passed** on this tree (the proofs, the
rejected programs, the pinned outputs and the scikit-learn comparisons). Lean
4.34.0 was not installed here, so the kernel `--verdict` was skipped and CI
runs it, as usual; no Bend source moved, only the datasets README, the release
script and the maps.

**What is next.** Delete the six `adeerkhan/mlbend-*` GitHub repositories;
their history is in this repository and their published bytes are on the hub.
`make release PKG=<dir>` is the only release path from here.

## 2026-10-22 — the split and the select, and the plan's v0.9 reached

**Where we are.** Three of the plan's items are now done. The
cross-validation over the iris dataset reaches the plan's stop criterion
for v0.9, and the plan's v0.4 (`mlbend-split`, the seeded stratified
split) and v0.5 (`mlbend-select`, the folds of the cross-validation on
top of it) are written, published and proved.

**What was run.**
- `bendenv run -- bend examples/crossval/main.bend` → five stratified
  folds: 0.9667, 0.8333, 0.9667, 0.9333, 0.9667; **mean accuracy
  0.9333**. The example sits on `mlbend-select` now, so the split is
  written in one place; the per-fold numbers moved with the assignment,
  the mean did not.
- `reference/.venv/bin/python reference/test_crossval.py` →
  scikit-learn's LogisticRegression over its own stratified folds gives a
  mean of 0.9533; the Bend mean is within 0.02 of it. The folds are not
  the same folds (numpy shuffles with its own generator, we shuffle with
  bend-kit-random's), so what agrees is the procedure.
- `reference/.venv/bin/python reference/test_split.py` and
  `reference/.venv/bin/python reference/test_select.py` → the fold sizes
  and the class counts match scikit-learn's `StratifiedKFold` exactly:
  30 rows a fold and ten of each class a fold, on both sides.
- `make check` → see the log below.

**What was published.**
- `mlbend-select@0.1.0.0` -> `0xf6594aa2392363c8ab774bc81fb20b9d`, on the
  hub and on GitHub under the tag `v0.1.0.0`.
- `mlbend-split@0.1.1.0` -> `0xe8b7953f71ec5b441a90a08e0961269b` (it gained
  `test_of` and `train_of`), on the hub and on GitHub under the tag
  `v0.1.1.0`; `mlbend-split@0.1.0.0` -> `0x97faf074f5fec7518a7b67151e3be4f7`
  is the first one.
- `mlbend-preprocess@0.1.0.0` went out earlier today.

**What was not run.**
- The kernel check (`--verdict`) of the two new packages: Lean 4.34.0 was
  still downloading at the end of the session, so I did not see the
  verdict here; CI runs it. What the checker already says is that the
  verdict will carry bend-kit-random's `entropy` and `from_os` — foreign
  code the split never calls — so the verdict is a refusal to certify,
  the same shape as the HTTP packages. Their laws are therefore *checked*
  by Bend's checker, not certified by the kernel; both READMEs and
  `docs/AUDIT.md` say so. The metrics package's law, with no such
  dependency, is certified.

**What I learned (the cost of the day).**
- A `def` that proves a `law` takes the law's variables, and a `+` on one
  of them is a syntax error. `scripts/reuse.py` now leaves those defs
  alone.
- A `def` with no return type *fills a law*: the checker looks for a law
  of the same name, so a helper needs a return type.
- A hypothesis `%f(args) : {shape}` must have the *goal's* shape, with
  `_` marking the part the call fills in — not the lemma's own shape.
- `Nat.add(a, b)` and `Nat.add(b, a)` are different terms to the checker.
- A self-call has to pass its arguments unchanged until one shrinks, and
  the shrinking one comes first: a walk that counts *up* has to be written
  with the `1n+rest` pattern and the recursion on `rest`.
- A `case` binder read twice needs `++`: `case 1n++rest:`.
- A record has no field access: `split(...).test` is not a term.
- `Bool.pick` evaluates both branches.

**What is next.**
1. The sum law — "the sizes add up and no index is in both" — is issue #8,
   and it is the same wall issue #1 records for the metrics' accounting
   law: a count that depends on an element's *value* cannot be lifted to
   the shape the checker wants.
2. The datasets package needs a law (issue #6).
3. `mlbend-linear` (v0.6) and `mlbend-cluster` (v0.7).

## 2026-10-22 — the tape trains, and the plan's v0.9 is reached

**Where we are.** Two things that were open are now done. The bendtorch
tape trains (the 106 was the weight shape), and the iris dataset goes
through standardize → logistic regression → cross-validation with the
accuracy matching scikit-learn's — the plan's stop criterion for v0.9.

**What was run.**
- `bendenv run -- bend examples/tape/tape.bend` → `loss before:
  1.0986128`, `loss after:  0.7287296`. bendtorch's tape and bend-ml's
  autodiff, two implementations of the same gradient step.
- `reference/.venv/bin/python reference/test_tape.py` → the two agree:
  1.0986128 before, 0.7287295 (bend-ml) and 0.7287296 (bendtorch) after.
- `bendenv run -- bend examples/crossval/main.bend` → five stratified
  folds: 0.9667, 0.9, 0.9333, 0.9, 0.9667; **mean accuracy 0.9333**.
- `reference/.venv/bin/python reference/test_crossval.py` →
  scikit-learn's LogisticRegression over its own stratified folds gives a
  mean of 0.9533; the Bend mean is within 0.02 of it. The folds are not
  the same folds (numpy shuffles with its own generator, we shuffle with
  bend-kit-random's xoshiro128**), so what agrees is the procedure.
- `reference/.venv/bin/python reference/test_preprocess.py` → the five
  transformers match scikit-learn's `StandardScaler`, `MinMaxScaler`,
  `Normalizer` and `np.eye` to F32 precision.
- `make check` → see below.

**What was published.**
- `mlbend-preprocess@0.1.0.0` → `0xa8182854f375cd61135ccd32993993cc`,
  on the hub and on GitHub under the tag `v0.1.0.0`.

**What was not run.**
- The kernel check (`--verdict`) of the new package: Lean 4.34.0 is not
  on this machine, so `mlbend-preprocess`'s law is *checked*, not
  *proved*, here. CI proves it.

**What I learned (the cost of the day).**
- bendtorch's `linear` records `x @ W^T`, so the weights are **[out,
  in]** and the bias a vector of **[out]** — this is in bendtorch's own
  source (`src/tensor/matmul.bend`, `src/autograd/tensor_backward.bend`),
  and it is what the 106 was about.
- A node's value is fixed when it is recorded, so the loss *after* a
  step is a second forward pass on the stepped tape.
- A match cannot scrutinize a computed value — not even `Nat.is_eq(a,
  b)`; a comparison needs its own def or the value has to arrive as a
  parameter.
- A record has no field access: `l.before` is not a term.
- `Bool.pick` evaluates both branches, so it cannot choose between two
  lists without building both.
- A `Nat` counter does not shrink by arithmetic the checker can see
  (`Nat.sub(k, 1n)` is refused as a self-call); it shrinks by being
  unpacked (`case 1n+rest`, pass `rest`), the way
  `bend-ml-nat-lemmas` proves `add_comm`. A list shrinks by its tail.
- `List.range(n)` counts *down*, and `List.append` takes the element
  type as its second argument.

**What is next.**
1. `mlbend-split` (v0.4): the fold logic of `examples/crossval` as a
   package, with the partition law — the sizes add up and no index is in
   both. `bend-kit-random@0.1.0.1` is the generator, so nothing of ours
   generates numbers.
2. The datasets package needs a law (issue #4): "the loader reads the
   rows the file has" over the bytes, or the honest version of it.
3. `mlbend-select` (v0.5): k-fold cross-validation on top of the split,
   as a package rather than as the loop in the example.
4. `mlbend-cluster` (v0.7): the k-means of `examples/kmeans` promoted to
   a package, with the law that every point is in exactly one cluster.

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
