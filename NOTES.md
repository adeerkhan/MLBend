# Notes

Every finding, every measurement and every discarded hypothesis, by date. Read the
relevant entry before changing performance-sensitive code or a law.

## 2026-10-24

- **The evidence stage (2026-10-24).** Three things changed under the
  corpus. The datasets are pinned to a HuggingFace revision (`resolve/<sha>`)
  and each task carries the expected SHA-256; the runner verifies the bytes it
  reads. `MLBEND_DATA` is now authoritative: a miss is an error, not a quiet
  fetch, so a manifest can never hash one file while the program reads
  another. `mlbend-datasets@0.1.6.0` carries both changes.
- **The same folds, and five seeds (2026-10-24).** `examples/crossval` prints
  the fold of every row and takes the seed as its first argument; the runner
  builds once, runs seeds 42–46, and the NumPy reference reads that assignment
  and runs the same 60 SGD steps. The means agree to **5.9e-8** and the folds to
  **2e-8**, instead of the old 0.03 tolerance for two different partitions.
  `RESULTS.md` now reports mean ± sample deviation; every seed must pass.
- **The naive Bayes variance (2026-10-24).** The exact comparison found the
  heart baseline at **0.9333** against scikit-learn's **0.95**. The cause was
  not the split or the data: scikit-learn adds `var_smoothing = 1e-9 ×
  max(variance)` to every variance, and the Bend examples did not. With the
  same smoothing both baselines agree exactly: heart **0.95**, breast
  **0.98540145**. The heart test now gates at 1e-6.
- **Upstream (2026-10-24).** `bendlang/bend#1845` makes `File.read` fill its
  buffer instead of answering one `read(2)`; the repro is a pipe that writes
  three bytes, sleeps, then three more, and a six-byte read that used to
  answer three. The PR has a FIFO regression on both lanes. The bend-ml
  finding — `bend-ml-tensor` is the only `Mat` autograd trains, while
  `bend-ml-tensor-array` is the faster one — is architectural and already
  documented as frozen/superseded; it is not a PR-shaped change.
- **The dataset cache (2026-10-24).** `mlbend-datasets@0.1.5.0` reads a
  local file when `MLBEND_DATA` is set: the directory is laid out as
  `<repo>/<path>`, the same shape as the hub (`mstz/iris/iris.data`), and a
  missing file falls back to the network. The experiment runner fills
  `reference/.cache/datasets` with the exact bytes it hashes and sets the
  variable before the Bend run, so a warm task records `fetch: 0.0` and its
  run time is compute. A cold first run still pays the fetch; run the corpus
  twice, or warm the cache once, before reading the timing column. The
  iris-crossval run fell from **0.85 s** to **0.098 s** of compute. The
  loader reads the file at its real size (`File.size`, then `File.read`)
  rather than a fixed cap, so a larger dataset is not silently truncated;
  the cache is gitignored and the SHA-256 stays in the manifest.
- **The structural graph (2026-10-24).** `/understand` built a fresh graph
  at `05eacbd`: 102 files, 958 nodes, 905 edges, 6 layers, 7 tour steps and
  no validation issues. It is structural, not LLM-enriched per file, and it
  is at `.ua/knowledge-graph.json`; the point is orientation, not evidence.
- **The foundation correction (2026-10-24).** The “BendTorch has no softmax”
  finding was true only of `bendtorch-core@0.1.0.0` (`aaaxn/bendtorch`, the old
  line). The maintained library is `bendtorch@1.0.1.0` (`nuxyel/BendTorch`),
  which has `Attention`, `RMSNorm`, `Embedding`, `CrossEntropy`, `AdamW` and
  `Train`, and targets Bend 2.0.38. MLBend now pins **Bend 2.0.38** and depends
  on the real `bendtorch`. The two artifacts opened against the wrong tree were
  closed; no upstream issue is open for a gap that does not exist.
- **The first trained transformer (2026-10-24).** `examples/transformer` is a
  one-block causal transformer on the real library: embedding, RMSNorm,
  single-head causal attention, output projection, RMSNorm, linear head,
  AdamW, 200 steps on next-token prediction over the periodic sequence 0..7.
  The loss falls from **2.3591** to **0.0021**; a NumPy twin with the same
  deterministic weights starts at **2.3591** (the two agree to 6.8e-7) and ends
  at **0.1351**. The Bend update moves faster after the first step; the curves
  and the table are in `results/transformer/` (`loss.svg`, `loss.tsv`,
  `run.json`), and the task is `transformer-periodic` in the experiment layer.
  The plain-Bend `attention` example remains the library-free forward parity
  check (1.5e-8).
- **Bend 2.0.38 (2026-10-24).** `scripts/setup.sh`, the check pin, the Makefile
  and the docs moved from 2.0.37; the SHA256s are the release tarballs'. The three old examples (`tape`, `regression`, `stacks`) were retired on
  2026-10-24, and `bendtorch-core` left `bend.toml` with them.

## 2026-10-23 — the experiment layer, and what a law is for

- **The merge (2026-10-23).** `devel` went into `main` as a fast-forward (55
  commits) and the work continues on `main`. The old branch stays for history;
  the rules in `AGENTS.md` and the issue tracker now say `main`.
- **The identity (2026-10-23).** MLBend is the experiment layer, not another
  model library. bend-ml owns the tensors and the laws, BendTorch owns the
  training, and this repository owns the data, the splits, the transformations,
  the selection, the metrics, the reports and the manifests. The boundary rule
  is now in `AGENTS.md`, `README.md` and `docs/PLAN.md`: a model enters through
  an adapter; no tensor, no autodiff, no optimizer, no second trainer.
- **The runner (2026-10-23).** `reference/experiment.py` turns a task into a
  report card: it builds and runs the program, parses the number, computes the
  reference with NumPy or scikit-learn, compares them within a declared
  tolerance, and writes the manifest — git commit and dirtiness, Bend version,
  every `mlbend-*`, `bend-ml-*` and `bendtorch-*` package with its hub hash,
  machine, seed, build time, run time and reference time. The reports are
  committed under `results/`; `--wandb` sends the same manifest to Weights &
  Biases, offline without an API key.
- **The timings are two different questions (2026-10-23).** A task's Bend run
  and its Python reference both fetch the dataset from HuggingFace, so their
  ratio compares the whole program, not the kernel. The compute-only benchmark
  is the second answer: fifty sequential 128 x 128 float32 products take
  **18.4 ms** in Bend over `bend-ml-tensor-array` and **1.0 ms** in NumPy —
  11.4 against 218.4 GFLOP/s, NumPy 19x faster, one Bend thread against BLAS.
  The two numbers belong in different tables, and `results/` keeps them apart.
- **The two new examples (2026-10-23).** Gaussian naive Bayes over breast
  cancer scores **0.98540145**, the same as scikit-learn's `GaussianNB` on the
  same split; PCA's first component explains **0.275443** of the variance,
  against scikit-learn's **0.2754426**. Neither needs standardising (a Gaussian
  is affine-invariant) or a trainer.
- **The pipeline, and the leakage it removes (2026-10-23).** `mlbend-pipeline`
  fits the scaler on the training rows of a fold and transforms both sides;
  `examples/crossval` had standardised the whole matrix first, so the test rows
  were in the means and the deviations. The law `pipeline_preserves_rows`
  proves one output row per input row; the numbers are compared with
  scikit-learn's `StandardScaler` and with `Pipeline(StandardScaler,
  LogisticRegression)` on the same folds. The iris mean is unchanged (0.9333)
  because standardising everywhere is nearly harmless there — the API is what
  changed, and the leakage is now impossible by construction.
- **CI hit its 30-minute ceiling (2026-10-23).** The run on `42ad0f4` was
  cancelled at 30m16s: 58 checks, each building a program, do not fit in 30
  minutes on a 4-vCPU runner. `.github/workflows/ci.yml` now allows 60, and
  the retired `devel` branch is out of the push filter.
- **The two spellings of an effect (2026-10-23).** A return type written
  `IO<Unit>` rather than `IO(Unit)` reports `a declared datatype (unknown:
  IO)`, and the reverse (`do IO(Unit)`) reports `expected '<', observed '('`.
  Neither names the pair or the position; the minimal repro and the suggested
  message are for `bendlang/bend` (the `#1632` diagnostic series).
- **The first transformer block, and the op it needs (2026-10-23).** A
  single-head self-attention block over four tokens of eight features — Q, K,
  V, scaled dot-product scores, row softmax, the weighted sum of V — is written
  in plain Bend (`examples/attention`) and agrees with PyTorch to **1.5e-8**
  on every output number (`reference/test_attention.py`). The weights are a
  fixed formula, so the task is `attention-parity` and the checksum agrees to
  3e-9.
  **The gap is training.** BendTorch's tape records add, sub, mul, matmul,
  relu, slice, gather, cross-entropy and MSE; it has no recorded `exp`, no
  broadcast `div`, and therefore no softmax backward. A transformer cannot be
  trained on it until those operations (and their VJPs) exist. The forward is
  pure Bend over lists for now; the next upstream step is a BendTorch issue for
  `softmax` (or `exp`/`div`/`sum_dims` with gradients), then a trained attention
  task here.
- **A function is a Type, so a builder cannot be passed and reused (2026-10-23).**
  Writing one `row(builder, ...)` and passing `x_cell`, `wq_cell`, ... needs
  `+f`, which the checker refuses: `+x` requires a `Data` type, and a function
  is a `Type`. The example builds by a `kind` number instead
  (`cell(kind, i, j)`), which is also what made the termination check simple:
  the decreasing `n` must sit before the increasing column index in the
  recursive call.
- **Laws: what they are for here (2026-10-23).** A law belongs to a pure
  structural claim — a count, a partition, a cover, a composition — and the
  kernel proves it. A number (`F32`), a wall time, a dataset fetch and a W&B
  log are not laws: they are tested against a reference and recorded in a
  manifest. So the experiment runner adds no laws, the two examples add none
  (they are examples, not packages), and the next Bend package — the
  no-leakage pipeline — owes one: *fitting sees only the training rows of the
  fold*. Until that proof exists, the claim is an issue, not a package.

## 2026-10-23

- **The tape accumulates gradients until `zero_grad` (2026-10-23).** A
  backward pass in bendtorch adds its contribution to each working slot, and
  `Sgd.step` keeps the slot after it steps (`Backward.zero_grad` is what
  empties it). A training loop that calls `backward` and `step` without
  `zero_grad` therefore applies the sum of every gradient so far: the first
  regression run printed 0.9000 after 200 steps at 0.1 and 0.9293 after 1000
  steps at 0.3 — more steps, worse loss, because the step was growing. With
  `Backward.zero_grad(tape)` after each step the same run reaches **0.7078516**
  where scikit-learn's `LinearRegression` gives **0.7078631**
  (`reference/test_regression.py`). The one-step `examples/tape` had hidden
  this: one backward, one step, nothing to accumulate.
- **The two-layer network is the rectifier's chain (2026-10-23).** A hidden
  layer of eight rectified units over iris, trained with `AG.linear`,
  `AG.mask.rows`, `AG.linear`, `AG.ce_loss`, `AG.ce_grad`,
  `AG.linear_bwd`, `AG.relu_bwd` and `AG.linear_bwd` again, reaches accuracy
  **0.9733** after 400 steps at 0.5, the same accuracy scikit-learn's
  `MLPClassifier(hidden_layer_sizes=(8,))` reaches on the same data. The
  starting weights are a fixed small pattern: all-zero weights leave the
  hidden units with a zero gradient, because the gradient of the rectifier at
  zero is zero.
- **k-means is a package now (2026-10-23).** `mlbend-cluster@0.1.0.0` carries
  the Lloyd rounds, the assignment and the inertia from `examples/kmeans`,
  with the law `cluster_covers` (one label per row). The example that read the
  wine dataset keeps only the data and the standardising, and the numbers did
  not move: inertia 50645.64, sizes 2384 990 3123.

## 2026-10-22

- **48/48 on the consolidated tree (2026-10-22).** `make check` after the six
  packages became one repository: every proof checks, the two programs that
  must be refused are refused, the pinned outputs match, and the NumPy and
  scikit-learn numbers match. The kernel `--verdict` was skipped for want of
  Lean 4.34.0 here; CI runs it. Verification is the same command as before —
  consolidation changed where the packages live, not what they compute.
- **One repository for the six packages (2026-10-22).** The per-package
  `mlbend-*` repositories were a mistake in shape, not in content: the packages
  were always developed side by side in the workspace `bend.toml`, and the
  release script existed only to copy each directory into a repository of its
  own. A package's identity is its name and version on BendHub, not a
  repository; the commit and the tag `<name>-v<version>` here record where the
  published bytes came from. `bend-env` keeps its repository because it is a
  tool, not a package, and is consumed by checkout rather than by the hub.
- **The one file that had drifted.** The datasets README was newer in the
  standalone clone than in `datasets/`: the four-dataset table, the `Table`
  type and the `drop` argument were only there. The package's `main.bend` was
  identical in both places, so the hub hash was never in question; the README
  is merged now.

- **bendtorch's shapes are settled (2026-10-22).** The question the
  handoff carried — the weight orientation and the rank of the bias — is
  answered by bendtorch's own source, not by guessing:
  - `src/tensor/matmul.bend`: "x @ W^T for x of [rows, inner] and W of
    [columns, inner]: cell (row, column) adds x[row, k] * W[column, k]".
    So the weight is **[out, in]** (PyTorch's layout) and the input is
    [rows, in].
  - `src/autograd/tensor_backward.bend`: "Linear: records x @ W^T with W
    of [out, in], PyTorch's layout, then adds the bias b of [out] to
    every row." So the bias is a **vector of [out]**.
  - The MNIST demo agrees: `weight1 [128, 784]`, `bias1 [128]`,
    `weight2 [10, 128]`, `bias2 [10]`.
  - The tape example now builds the weights as `[3, 4]` (three classes,
    four features) and the bias as `[3]`, and standardises the features
    through `mlbend-preprocess` so it trains the same model as
    `examples/logistic`.
  - **The 106 was the shape, after all.** bendtorch's `linear` records
    `x @ W^T`, so the inner dimensions are x's columns and W's columns:
    with the data of [150, 4] the weights must be [3, 4] and not [4, 3].
    The bias is [3], a vector of the outputs. With the shapes right the
    refusal is gone.
  - **A node's value is fixed when it is recorded.** The loss *after* the
    step is therefore a second forward pass on the stepped tape: the step
    writes the new weights into the leaves, and a recording after it
    reads them ("Recording an operation after a step reads the new
    values", as bendtorch's sgd.bend says). bendtorch's own MNIST demo
    rebuilds the tape each step for the same reason.
  - **The two stacks agree.** `examples/tape` (bendtorch) and
    `examples/logistic` (bend-ml) train the same model on the same
    standardised iris data with a learning rate of 0.5, and the losses
    agree to F32 precision: 1.0986128 before, 0.7287295 and 0.7287296
    after (`reference/test_tape.py`). Two independent libraries, one
    gradient step.

- **`mlbend-preprocess` (v0.3, issue #5).** The transformers of
  scikit-learn over **bend-ml's** matrix: `Mat<r, c>` from
  `bend-ml-tensor`, the one bend-ml's autodiff takes. MLBend writes no
  matrix of its own.
  - The standardiser is bend-ml's own arithmetic: a column-wise
    normalisation is `Mat.transpose`, then `Mat.layernorm` (which
    centres a row and scales it by its own standard deviation), then
    `Mat.transpose` back. The layernorm's variance is the population
    one (divided by the count), the same as scikit-learn's.
  - The numbers match scikit-learn's `StandardScaler`,
    `MinMaxScaler`, `Normalizer` (l1 and l2) and `np.eye` on the same
    matrix, to the precision of an F32 (`reference/test_preprocess.py`).
  - What is proved: the one-hot matrix has one row for every label
    (`one_hot_count_length`, an induction over the labels). What is
    only tested: the values, because F32 is opaque to the checker.
    "The standardised column has mean 0 and variance 1" is the debt of
    the issue.
  - `Mat.transpose` and `Mat.layernorm` take the *width* as a runtime
    argument, so a generic transformer over `Mat<r, c>` has to declare
    `+r: Nat, +c: Nat` (linear) — the dimensions are erased type
    parameters, and a transformer that needs them at run time cannot
    have them erased.
- **The logistic example standardises its features (2026-10-22).** With
  the standardiser the learning rate of 0.5 is sane: the loss goes
  1.0986128, 0.7287295, 0.6035278, 0.5403903. On the raw features (which
  run from 0.1 to 7) that same rate overshoots and the loss *rises* —
  the step overshooting, not a wrong gradient.
- **Issue #2 (the containers) is answered by the work.** The matrix is
  bend-ml's `bend-ml-tensor` `Mat`: it is a list of rows with the
  dimensions in the type, it is the one `bend-ml-autograd` takes (so
  the estimators can feed a model without a copy), and it has
  `col_sums`, `to_rows`, `transpose` and `layernorm` — everything the
  transformers need. The flat `bend-ml-tensor-array` is a *different*
  `Mat` type: using it would mean a second matrix in the ecosystem,
  which `docs/PLAN.md` forbids.

- **The bendtorch tape trains (2026-10-22).** The 106 was the shape after
  all: bendtorch's `linear` records `x @ W^T`, so the weights are
  `[out, in]` — three rows of four for the iris data, not four rows of
  three — and the bias is a vector of `[out]`. The second half of the
  puzzle was that a node's value is fixed when it is recorded: the loss
  *after* the step is a second forward pass on the stepped tape, because
  the step writes the new weights into the leaves. `examples/tape` now
  trains, and `reference/test_tape.py` checks it against
  `examples/logistic`: bendtorch and bend-ml are two implementations of
  the same gradient step, and they agree to F32 precision (1.0986128
  before, 0.7287295 and 0.7287296 after).
- **The cross-validation example now sits on the packages (2026-10-22).** It
  used to shuffle and deal the rows out itself; the folds come from
  `mlbend-select` now, so the split is written in one place. The
  assignment changed with it (the package deals the rows of a class out
  in turn, the example dealt them out in blocks), so the per-fold
  accuracies moved: 0.9667, 0.8333, 0.9667, 0.9333, 0.9667. The **mean is
  the same, 0.9333**, and scikit-learn's is 0.9533, so
  `reference/test_crossval.py` still agrees within 0.02.
- **The plan's v0.9 is reached (2026-10-22).** The iris dataset goes
  through standardize -> logistic regression -> cross-validation, and
  the accuracy matches scikit-learn's:
  - the Bend program (5 stratified folds, seed 42) scores 0.9667, 0.9,
    0.9333, 0.9, 0.9667 — a mean of 0.9333;
  - scikit-learn's LogisticRegression over its own stratified folds
    scores a mean of 0.9533.
  - The folds are *not* the same folds: scikit-learn shuffles each class
    with numpy's generator, the Bend program with bend-kit-random's
    xoshiro128**. What agrees is the procedure, to within the spread of
    two random partitions of 150 rows.
- **`mlbend-preprocess` (v0.3, issue #5) is published** as
  `mlbend-preprocess@0.1.0.0` (`0xa8182854f375cd61135ccd32993993cc`).
  The transformers run over bend-ml's `Mat<r, c>` — bend-ml's matrix,
  not one of ours — and the standardiser is bend-ml's own arithmetic:
  `Mat.transpose`, then `Mat.layernorm` (which centres a row and scales
  it by its own standard deviation), then `Mat.transpose` back. The
  numbers match scikit-learn's `StandardScaler`, `MinMaxScaler` and
  `Normalizer` to the precision of an F32.
- **The PRNG question is answered.** `bend-kit-random@0.1.0.1` is on the
  hub: xoshiro128**, pure (a draw returns the value beside the next
  state), with unbiased ranges and Fisher-Yates shuffles. It is someone
  else's package, so the split uses it rather than a generator of our
  own. The plan's open question about the PRNG is closed.
- **`mlbend-split` (v0.4) is written and published** as
  `mlbend-split@0.1.0.0` (`0x97faf074f5fec7518a7b67151e3be4f7`), on the
  hub and on GitHub under the tag `v0.1.0.0`. The package deals the rows
  of a class out over the folds in turn, so every fold holds the same
  proportion of every class. The shuffle is `bend-kit-random`'s
  xoshiro128**, seeded: the same seed gives the same split, and another
  seed gives another one. The law `fold_covers` says every row of the
  dataset is placed in a fold, so the folds partition the rows.
- **The fold sizes and the class counts match scikit-learn's exactly**
  (150 rows, three classes of fifty, five folds: 30 rows a fold, ten of
  each class a fold, on both sides). The two programs do not agree on
  *which* rows go where — scikit-learn shuffles with numpy's generator,
  the package with bend-kit-random's — so the comparison is of the shape
  of the split, not of its contents.
- **The sum law is the same wall as the accounting law (2026-10-22).** I
  tried to prove "the rows of a fold and the rows outside it add up to all
  the rows" (`Nat.add(count_in(xs, f), count_out(xs, f)) ==
  List.length(xs)`) and the checker named the shape it wants:
  `Nat.add(count_in_go(t, f, same_of(Nat.is_eq(h, f))),
  count_out_go(t, f, same_of(Nat.is_ne(h, f)))) == 1n+_`. The head's
  contribution is *data-dependent* — `same_of` is a `Bool` the checker
  cannot reduce — so the tail's two counts do not line up with the
  hypothesis's `Nat.add(count_in_go(t, f, 0n), count_out_go(t, f, 0n))`.
  This is the wall issue #1 records for the metrics' accounting law: a
  count that depends on the *value* of an element cannot be lifted to a
  `1n + ...` shape. The way out is a count that walks with the test
  decided by the caller, or arithmetic over `Nat` that the checker can
  reassociate — `bend-ml-nat-lemmas` has `add_comm` and `add_assoc`, but
  the hypothesis shape has to be the goal's shape, and neither lemma
  matches it.
- **`mlbend-select` (v0.5) is written and published** as
  `mlbend-select@0.1.0.0` (`0xf6594aa2392363c8ab774bc81fb20b9d`), on the
  hub and on GitHub under the tag `v0.1.0.0`. It hands out the folds
  `mlbend-split` makes, so the two agree by construction, and adds the
  walk over the folds: the test rows of each fold, the training rows,
  and the sizes. The law `select_covers` says every row of the dataset is
  in exactly one fold. The fold sizes and the class counts match
  scikit-learn's `StratifiedKFold` exactly (150 rows, three classes of
  fifty, five folds: 30 rows a fold, ten of each class a fold, on both
  sides).
- **`mlbend-split` is at 0.1.1.0** (`0xe8b7953f71ec5b441a90a08e0961269b`):
  it gained `test_of` and `train_of`, which take the fold list instead of
  the labels, so the folds are computed once and asked of as many times
  as needed.
- **The two new packages' laws are checked, not certified (2026-10-22).**
  `mlbend-split` and `mlbend-select` depend on `bend-kit-random`, which
  carries `entropy` and `from_os` — two defs that call the operating
  system. Neither package calls them (both seed the generator), but
  Bend's verdict refuses a book that holds any foreign code, and it
  cannot tell that nothing calls them. So the two laws are *checked* by
  Bend's checker, and the verdict is a refusal to certify — the same
  shape as the HTTP packages. The metrics package's law has no such
  dependency and is certified. The price of not writing a generator of
  our own is a weaker guarantee, and the READMEs and `docs/AUDIT.md` say
  so.
- **Bend facts paid for today.**
  - A `def` with no return type *fills a law*: the checker looks for a
    law of the same name, so a helper that is not a proof needs a
    return type.
  - A self-call has to pass its arguments unchanged until one of them
    shrinks, and the shrinking one comes first. A walk that counts
    *up* therefore has to be written with the `1n+rest` pattern and the
    recursion on `rest`, and the fold it handles is `rest`, not `f`.
  - A `case` binder read twice needs `++`: `case 1n++rest:`. The `+`
    on a *def* parameter is a different marker.
  - A hypothesis `%f(args) : {shape}` must have the *goal's* shape, with
    `_` marking the part the call fills in — not the lemma's own
    shape.
  - A match cannot scrutinize a computed value — not even
    `Nat.is_eq(fold, f)`; the comparison needs its own def, or the
    value has to arrive as a parameter.
  - A record has no field access: `l.before` is not a term. The record
    is destructured by a match, and the fields come out as binders.
  - `Bool.pick` evaluates both branches, so it cannot be used to choose
    between two lists without building both.
  - The decreasing argument of a self-call has to be *structurally*
    smaller. A `Nat` counter does not shrink by arithmetic that the
    checker can see (`Nat.sub(k, 1n)` is refused); it shrinks by being
    unpacked (`case 1n+rest` and passing `rest`), the way
    `bend-ml-nat-lemmas` proves `add_comm`. A list shrinks by its tail.
  - `List.range(n)` takes the count alone, and it counts *down*:
    `List.range(3n)` is `[2n, 1n, 0n]`.
  - `List.append(a, -A, xs, ys)` takes the element type as its second
    argument, and its two list arguments are the accumulator and the
    element.
  - The PRNG's shuffle returns a pair: the list and the generator that
    comes after it, which has to be threaded by hand.

## 2026-10-21

- **The CI was red before this session, and the reason was the verdict.**
  `bend hf/main.bend` and `bend datasets/main.bend` end in "SOME PROOFS
  FAIL: 312 defs rely on unsafe or foreign code". A file with no `main`
  is checked by the *verdict* path (`cli_verdict` in `bend2/main.ts`),
  and the verdict refuses a book that holds any foreign code — the
  socket bindings of the http package are foreign code. So a package
  that does IO can never have a clean verdict, and the check now reads
  it as a pass when the *only* complaint is the foreign code (a type
  error has "expected" and "observed" in it).
  - The laws of `mlbend-metrics` are still proved by the kernel — that
    package is pure and its verdict is clean.
- **clang lives in the tree, not in /tmp (2026-10-21).** The wrapper at
  `~/.local/bin/clang` pointed its `LD_LIBRARY_PATH` at a directory whose
  libraries were symlinks into `/tmp/opencode`. `/tmp` is cleaned, and
  one morning the builds failed with "bend needs clang 14 or newer
  (found no clang)". The wrapper now names the tree's own libraries, and
  `scripts/setup.sh` asks for clang — the checks build every program
  before it runs it.
- **k-nearest neighbours over the breast cancer dataset
  (`examples/knn/main.bend`).** Plain Bend again: the neighbours are
  sorted by distance with `List.sort`, and the three nearest vote. The
  accuracy over the last fifth of the rows is 0.9781022, and
  scikit-learn's `KNeighborsClassifier(n_neighbors=3)` over the same
  split gives 0.9781021897810219 (`reference/test_knn.py`).
  - Two Bend details cost the most here. `List.sort` takes its
    comparator as a *template*: `List.sort(~Neighbour, ~nearer, xs)`,
    with the `~` at the call site.
  - A vote that counts the labels must count them over the whole list,
    not over the tail of the recursion: the count is computed once and
    the recursion only walks to the end.
- **The datasets are the UCI sets, not scikit-learn's subsets.** The `mstz`
  mirrors on HuggingFace carry the whole file: wine is 6497 rows (not 178),
  breast cancer 683 (not 569), heart failure 299. The loader is now one
  generic CSV reader instead of a function per dataset: `mlbend-datasets`
  fetches the file over the hub client, drops the header, reads every cell
  as a number (`True` is 1, `False` is 0) and takes the last column as the
  label. The labels are `F32`, not `Nat`: a class label is the index of the
  class and a regression label is the value, and the model decides which.
- **k-means over the wine dataset (`examples/kmeans/main.bend`).** Plain
  Bend: no gradients, no library, no tensors. The features are standardised
  first (the wine columns run from 0 to 160), the first three rows are the
  centres, and ten rounds of Lloyd's algorithm follow.
  - The result matches NumPy: the cluster sizes are the same (1071, 717,
    4709) and the inertia agrees to the fifth digit (57343.812 against
    57344.120). `reference/test_kmeans.py` runs the Bend program and
    `reference/kmeans_wine.py` is the NumPy side.
  - The arithmetic is done on whole rows, never by the place of a column:
    a fold over the rows adds the columns up (`zip_add`), a fold over a row
    adds the cells up. A row may then be short, which the wine file's last
    row is.
- **A counter is never a decreasing argument (2026-10-21).** The checker
  reads the arguments of a self-call left to right and wants the first
  changed argument to *shrink*: the parameter has to be a constructor and
  the argument a subterm of it (`term_descend` in `bend2/bend.ts`). So
  `f(Nat.sub(k, 1n), ...)` is refused: `Nat.sub(k, 1n)` is a call, not a
  smaller term. A loop over a number is written with `List.range(n)` and a
  loop over a list with the tail, and the index rides along as an
  accumulator. This is the same rule as "no forward references" and it
  costs the same number of round trips.
- **A def body has one result.** `def f(): e1` and then `e2` is a parse
  error: the second statement wants a new `def`. Several effects go in a
  `do IO<Unit>:` block, and a `match` heads a def body, so it cannot be a
  statement inside a `do` block: it goes in a def of its own.
- **The interpreter's stack is a few hundred frames.** `bend file.bend`
  overflows on a fold over 300 rows (`memory fault (machine stack
  overflow?)`); the same program built with `bend -o` runs. Every test in
  `reference/check_all.py` is therefore built and then run.
- **`scripts/reuse.py` and `scripts/sort_defs.py` (2026-10-21).** Bend
  consumes a value once unless it is bound with `+`, and the error for a
  value read twice ("consumed more than once") came up once per def.
  `scripts/reuse.py` marks the parameters, the pattern binders and the
  binds that the body reads twice; it is a script over the source, not a
  change to the language. `scripts/sort_defs.py` now keeps a comment with
  the def it describes: the comments at the end of a block belong to the
  next one.

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
- **The process, and where the memory lives.** An agent's durable context is
  this repository, not a transcript: `AGENTS.md` (the rules, the read order, the
  traps), `docs/agents/handoff.md` (where we are, what was run, what was not),
  `docs/agents/domain.md` (the vocabulary and the design rules),
  `docs/agents/issue-tracker.md` (how the issues carry the work), `NOTES.md`
  (what was measured and what failed), `docs/AUDIT.md` (the trust base) and the
  issues themselves (the debt). The laws are the real memory: a claim that
  survives the Lean kernel needs no log to stay true.
- **The bend repository stays where it is.** It is not vendored here: it is a
  fork with two remotes, it moves daily, and the copy would be wrong tomorrow.
  The pinned install (`~/.bend`) is the versioned reference (the guide and
  `base.bend` come with it), the hub cache (`~/.bend/lib`) is an offline mirror
  of the published packages, and `make sync-bend` fetches the pinned tag into
  `../bend-ref` when a read-only copy of the compiler is wanted.
- **Housekeeping on the bend fork**: the `/understand` knowledge graph (`.ua/`)
  is generated and must never be committed — it is now in that repository's
  `.git/info/exclude`, because it would otherwise fail bend's own repo gate (the
  allow list) the first time an agent ran `git add -A`.
- **The base layer exists (2026-10-20).** The plan was wrong about
  `mlbend-linalg`: the ecosystem already has the containers, the CSV reader,
  the bytes, the integers, the soft float, the proved lemmas and the parser
  combinators (`bend-kit-csv`, `-bytes`, `-int`, `-f64`, `-lemmas`, `-parse`),
  and bend-ml has the tensors and the reverse-mode autodiff. MLBend writes the
  *machine-learning* surface and imports the rest. A second `Mat` and a second
  set of lemmas would be the third copy of the same thing in the ecosystem, so
  the `mlbend-linalg` package is off the plan.
- **The numbers gap, and the ask (2026-10-20).** Base can print a number but
  not read one: no `U32.of_string`, no `F32.of_string`, and `bend-kit-f64`
  says so outright ("Decimal text is not here"). The data layer needs it, so
  the ask is upstream: bendlang/bend#1789, with the shape (Base's own
  `Result<&1, &1, U32 & String, A>`) and the law (`U32.of_string
  (U32.show x) == x`, provable). Until it lands, v0.2 carries a small decimal
  parser and drops it when Base has one.
- **The environment (2026-10-20).** The repository is a workspace now: a
  root `bend.toml` names the members (`metrics`), one `bend.lock` holds
  every package in the closure, and one `.bend/env` holds them all.
  `bendenv` is at <https://github.com/adeerkhan/bend-env>; it is a Rust
  tool that takes uv's shape (a global content-addressed cache, hard
  links into the environment, parallel downloads, one lock per
  workspace). `make env` builds it, `make env-check` verifies it and
  runs `examples/stacks/main.bend` over it.
  - **The example is the proof.** One program, one environment, two
    stacks: `bend-ml-tensor-array@0.1.14.0` computes a 2x2 product
    (`[12, 12, 12, 12]`) and `bendtorch-core@0.1.0.0` builds a tensor
    of four words. Both come from the lock.
  - **The ecosystem's tensors do not mix, and the tool says so.** A
    Bend type belongs to the package that declares it, so
    `bend-ml-tensor` and `bend-ml-tensor-array` are two different
    `Mat`, and so are the `G` of `bend-ml-autograd` and of
    `bend-ml-tensor-array`. `bendenv` reports every type two packages
    of the closure declare. The consequence: **the flat tensor library
    has no autodiff behind it** — `bend-ml-autograd@0.1.1.0` is written
    against the list-based `bend-ml-tensor@0.1.0.0`. That is the gap
    the deep-learning half has to close.
  - **The versions skew too.** `bend-ml-autograd@0.1.1.0` imports
    `bend-ml-nat-lemmas@0.1.0.0` while this repository asks for
    `0.1.1.0`; both are in the lock, and the tool says so. Not a bug:
    a Bend source names its dependencies exactly, so two versions can
    live side by side.
  - **The publish path is verified.** `bendenv hash <file.bend>`
    computes the hash a package would publish under, and on
    `aaaxn/bendtorch`'s checkout it gives
    `0x1689e5deb278c16abd3bd45eac101ef3` — the hash the hub holds for
    `bendtorch-core@0.1.0.0`, over the same 196 files. So the file
    set, the paths, the digests and the hash are the compiler's, byte
    for byte.
- **The datasets come from HuggingFace (2026-10-21).** Two new packages:
  `mlbend-hf` (a HuggingFace client: the dataset page, the file tree, a
  file's bytes, and the rows server's JSON rows) and `mlbend-datasets`
  (the datasets of scikit-learn, fetched at run time). The iris dataset
  comes from the file the dataset's own repository holds, which holds the
  same numbers scikit-learn ships: 150 rows of four features and the class
  of each, as a `Mat<150n, 4n>` and a list of the labels. Nothing but
  Bend: `bend-kit-http` for the HTTP, `bend-kit-json` for the JSON, and a
  parser of my own for the rows.
  - **The shape of the code is Bend's, not mine.** The checker will not
    let a match scrutinize a computed value, so every helper that
    unwraps a `Maybe` takes it as a parameter and matches on that, and a
    caller gets one call each. A def must also come before the defs that
    use it, which is a rule I broke often enough to write a sorter for.
  - **The toolchain is up.** `clang` 22 (the Fedora RPMs, unpacked into
    `~/opt/llvm-tree`, no root) and a native `bend` 2.0.37 at
    `~/.bend/bin/bend` (a compiled bun binary; a build is 0.8 s). Without
    clang the bend compiler cannot build anything.
- **The ecosystem's other package manager does not compile (2026-10-21).**
  `ezx` is on the hub: "ez, the package manager for Bend 2", by
  noah-emp, v1.5.0.0, 74 files. Its manifest is `ez.toml`, a ledger of
  every dependency with its hash and its origin (a hub name or a git
  commit with a NAR hash), it vendors into `.ez/lib`, and it pins tools.
  It does not compile against bend 2.0.37: `emerging-ezhttp` declares
  `Body.Text{text: String}` and Base's `Event` declares
  `Event.Text{text: String}`, and a program that imports both has two
  `Text` constructors in scope. `bendenv lock` now says so before you
  try:
  `Text: 0xf1c957a470368870a6d1d62a8c0cbe32, bend2/base.bend, ezx`.
  That is the second time the collision report has earned its place: Base
  is not on the hub and it moves, so a constructor it gains can break a
  package published before it.
  - **ezx pins its dependencies by hash, not by name.** All five of them.
    The resolver follows those now.
- **The tensors and the autodiff are bend-ml's and bendtorch's (2026-10-21).**
  Two packages of our own, `mlbend-nn` and `mlbend-torch`, re-implemented
  what bend-ml and bendtorch already provide, and they are gone. The
  examples use the libraries directly: the forward pass is `AG.linear`,
  the loss is `AG.ce_loss`, the gradient of the loss is `AG.ce_grad`, the
  backward of the layer is `AG.linear_bwd`, the step is `AG.sgd`, and on
  the other side the tape is `Backward.leaf`, `Backward.linear`,
  `Backward.cross_entropy`, `Backward.backward` and `Sgd.step`.
  - What this repository adds is the classic-ML surface around them:
    the metrics, the datasets, and the estimators that come next.
  - The logistic example over the online iris dataset loses 1.0986 to
    1.0672 in three steps. The learning rate is 0.01 because the
    features are not standardised: at 0.5 the loss *rises* (1.099 to
    1.537 to 6.88), which is the step overshooting and not a wrong
    gradient. `mlbend-preprocess`, which standardises, is what makes a
    larger rate sane.
  - The bendtorch path compiles and runs -- the data, the weights and
    the bias are built, the leaves go on the tape, the loss reads back --
    and the `linear` node still refuses the operands (error 106, an inner
    dimension mismatch). The shape convention of the weights (PyTorch's
    `[out, in]` or the mathematical `[in, out]`) and the rank of the
    bias are not settled yet; bendtorch's own MNIST demo is the
    reference.
- **The data (2026-10-20).** iris (150 rows, 4 features) and breast cancer
  (569 rows, 30 features) are in `mlbend-datasets/data/`, taken from
  scikit-learn's copies (which come from UCI). A hub package is one file, so
  the data lives with the program that uses it, not in the package: the loaders
  take a path.
