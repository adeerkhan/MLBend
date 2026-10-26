# Handoff

One file, overwritten at the end of every session. It is the literal memory
context: a fresh agent reads this first, then the issue it names, then
`AGENTS.md`.

**Write it before you stop.** An agent that ends a session without writing this
leaves the next one blind.

## 2026-10-26 — the generative package, the GPU lane, the digits done

**Where we are.** Three things moved; one is now finished rather than
parked.

- **`mlbend-generative@0.1.0.0`** (issue #10): the counter RNG (murmur3's
  fmix32 **plus its closed-form inverse**), the cosine and the linear
  schedules, and the time embedding — four length laws proved, the twin
  bit-exact, `mix32`'s bijection tested-not-proved (the issue's debt).
  `diffusion-2d` now runs on it: T=64 linear (few-step cosine *diverges* —
  a reverse step with β near 1 multiplies prediction error by
  1/sqrt(1-β)), σ_t fixed (the old sampler added raw z, a random walk),
  4000 steps; first loss **152.954, delta −1.5e-5**; the cloud's
  statistics are report-mode with the measured reason in NOTES (a 1-ulp
  weight perturbation changes nothing, but optimizer rounding over 4000
  steps makes different denoisers).
- **The GPU lane is real and measured on this machine.** `bend -o`
  builds the device program when `CUDA_HOME` has `nvrtc.h` (NVRTC at
  build; three RPMs unpacked into `~/.local/cuda-nvrtc` — the recipe and
  the failure it fixes are in NOTES). Programs reach the GPU only through
  bangs = `bend-ml-tensor-gpu`; bendtorch's `torch/gpu.bend` is the
  drop-in. On the RTX 5060: `demo/gpu_test.bend` → all eight linear
  comparisons **bit-identical on the GPU** (0.8 s, 20 % util, 1.7 GB);
  `gpu_layers_test` → SiLU 4 ulps / attention in CUDA-exp ulps — exactly
  their documented policy. Ours stay CPU because they use the proved
  `main.bend` layers. **The `libnvrtc.so.13` failure was environmental
  and is fixed in bendtorch (`6ee6721`, local commit):** the default
  `CUDA_HOME` named a nonexistent `~/.local/cuda12`, the libs are in
  `lib64`, and `LD_LIBRARY_PATH` was unset — `make gpu` and
  `scripts/gpu_expcr.sh` now pass (all zeros) with
  `PY=/home/adeerkhan/github/fatima/.venv/bin/python` (torch 2.14.1+cu130;
  the local venvs have no torch). Do not re-edit the Makefile's CUDA lines
  13–15; they are correct as committed. **Not yet filed**: the bendlang
  doc issue (the CUDA_HOME/bangs/`--gpu-build` recipe is only in the
  compiler source).
- **`diffusion-digits` is done and in the corpus** (was parked): the last
  layout bug — `row_of` skipping one pixel per row, and rows newest-first —
  is fixed in `2f2631b` (`drop64_go`, `rows_rev`); the test is un-parked as
  `reference/test_diffusion_digits.py` (the 90th check, 440 s) and the task
  is wired into `experiment.py`: first run **PASS**, first loss delta
  **4.69e-05**, data mean delta **4.6e-9**, report card in
  `results/latest/diffusion-digits.json`, five rows in RESULTS.md, README
  with `grid.svg`. The sampler still explodes (`sample_mean` 50256.9 vs
  96.7) — report-mode, and the reason: an underfit denoiser, so the next
  model step is a residual head (`T.Residual`), then the GPU import switch.
  NOTES has the print-alignment finding (the printed `loss:` lines carry
  `curve[i-1]`; `final_loss:` carries `curve[-1]`).

**What was run.** `experiment.py --task diffusion-digits` → PASS;
`--summary` rewrote RESULTS (12 tasks' rows). `make check` → **90/90**
(log `/tmp/opencode/check11.log`). `make gpu` and `scripts/gpu_expcr.sh`
in bendtorch → all zeros. `make_example_readmes.py` → READMEs for both
diffusion examples (they had none) and refreshed the crossval/heart text
that predated the shared folds and `var_smoothing`.

**What is next.**
1. The residual head (`T.Residual`) for the digits sampler, then the GPU
   import switch (`torch/gpu.bend`, one import line) with a CPU-vs-GPU
   step measurement.
2. A CIFAR-10 task (3072 outputs) with the Bend-vs-NumPy/PyTorch
   comparison and a sample grid under `results/`.
3. `make release PKG=report` and `PKG=generative` still wait on the
   maintainer (CI red on the kNN check until report ships).
4. File the bendlang doc issue for the GPU recipe.
5. The W&B sweep over the selected parameter.
6. **CI on `db495cf` is 87/90, and the three failures are all package
   resolution**: `test_diffusion` and `test_diffusion_digits` fail because
   `mlbend-generative@0.1.0.0` is not on BendHub, `test_knn` because
   `mlbend-report@0.1.0.0` is not. CI deliberately resolves from the hub
   (`scripts/setup.sh` creates no workspace, so `BEND_LIB` is unset there);
   both packages wait on the maintainer's go for `make release`. The other
   87 checks — every proof, kernel verdict and number — pass. Await
   bend-ml#19 and bendlang/bend#1845 as well.

## 2026-10-25 (later) — the first diffusion model

**Where we are.** `diffusion-2d` is the corpus's tenth task: a DDPM
denoiser (three `Linear` layers with `SiLU`) trained by `T.Train.step`
with AdamW to predict the noise of a 32-step linear-β forward process
over two moons, then sampled by the ancestral DDPM loop — all on
`bendtorch@1.0.1.0`, in `examples/diffusion/main.bend`. The gate is the
first training loss, **6.542548, against a float32 NumPy twin to 1.8e-7**;
final loss, sample statistics and the scatter (`results/diffusion/`) are
report-mode. The run is 3.3 s; the reference 3.8 s.

- **Every "random" number is a pure function of a counter**: murmur3's
  fmix32 over (step-salt·φ, index), Box-Muller over `(m24+1)/2^24`
  uniforms — never zero, exactly representable. Base's `F32.sin/cos/log/
  exp` (builtin bodies behind laws) made real Gaussians and a sinusoidal
  time embedding possible.
- **The finding for upstream**: model parameters hold arrays, so they are
  affine — a sampling loop cannot reuse them. `T.Layer.fwd` returns them
  in `Fwd{p, y, s}`; the sampler threads them through a `PS{p, x}` record
  the way nuxyel's own eval loop does (`smol.bend`'s `ev_loop`). `Mat.dup`
  is the one-shot duplication; a loop wants the threading shape. NOTES.md
  has the full list of checker rules the file cost.
- **The bug the exact comparison caught**: the moons list is interleaved
  (u0, l0, u1, l1, …) and the twin first wrote the moons sequentially.
  Row 0 matched, the loss sat 2 % off, and every scalar stage matched to
  an ulp — order-blind statistics (mean/std) hid it; only the gated loss
  caught it. Fixed in the twin.

**What was run.**
- `reference/experiment.py --task diffusion-2d` → PASS (first_loss delta
  −1.8e-7); `--summary` reg rewrote `results/RESULTS.md`.
- `reference/test_diffusion.py` → first loss agrees, data statistics
  agree, writes `results/diffusion/{loss.tsv,loss.svg,scatter.svg,run.json}`.
- `make check` → see the log below (the diffusion test joined).

**What is next.**
1. `make release PKG=report` still waits on the maintainer (CI red on kNN
   until then).
2. The W&B sweep over the selected parameter (handoff item 3).
3. The sampler's quality: the sampled cloud is broader than the data
   (std 3.0 vs 0.87) — more steps, a cosine schedule, or an ε-learned
   variance would tighten it. The task's job was the pipeline, not the
   picture, but the picture is now measurable.
4. Watch bend-ml#20 and bend#1848/#1835 for answers.

## 2026-10-25 — mlbend-report, and two upstream issues

**Where we are.** The report half of the evaluation surface is a package:
`mlbend-report@0.1.0.0` (issue #9) carries the confusion matrix and the
per-class report — precision, recall, F1, support, the macro and the weighted
averages — over `Nat` labels, on `mlbend-metrics`' counts. Both laws
(`matrix_rows`, `row_length`) are **proved by the Lean kernel on this machine**
(Lean 4.34.0 was installed during the session, so `--verdict` no longer waits
for CI). The laws are stated over the counted forms (`matrix_from`,
`row_from`): the checker refused the law over the `k`-classes wrapper, because
the walker keeps the total `k` fixed while only the rows-remaining counter
shrinks — see NOTES.md.

- **The numbers match scikit-learn.** `reference/test_report.py` compares the
  matrix, every per-class score, the averages and the accuracy of
  `report/tests/ok.bend` (seven samples, three classes present, one absent —
  the zero-support class exercises `zero_division=0`) with
  `confusion_matrix` and `classification_report`: every check within F32
  precision. `cancer-knn` prints the report beside its accuracy, and
  `reference/test_knn.py` now checks the matrix `[[100, 2], [1, 34]]` and the
  per-class scores against scikit-learn's kNN on the same predictions.
- **Two upstream issues filed.** `bendlang/bend#1848`: the two spellings of an
  effect (`IO<Unit>` in a type position reports "a declared datatype (unknown:
  IO)"; `do IO(Unit)` reports `expected '<', observed '('`), both repros
  verified on 2.0.38. `nuxyel/bend-ml#20`: the design question — re-base
  autodiff on the maintained `tensor-array`, or declare BendTorch the answer
  and standardize the conversion seam; the leaning stated there is the latter,
  with the conversion surface as the cheaper artifact.
- **The toolchain caught up.** `~/.bend/bin/bend` was still 2.0.37 and the
  Python venv predated the repository's move (its interpreter pointed at the
  old path); `scripts/setup.sh` rebuilt both. `make check` is now **81/81**
  locally, with the kernel verdicts running here instead of only in CI.

**What was run.**
- `make check` → **81/81**, including `report/main.bend --verdict` (the kernel
  proved both laws) and the new `reference/test_report.py`.
- `reference/test_knn.py` → the accuracy, the matrix and the per-class report
  all agree with scikit-learn.
- `bendenv lock`/`bendenv sync` → `mlbend-report@0.1.0.0` is in the lock and
  the environment as a local member.

**What is not done, and what is next.**
1. **`mlbend-report@0.1.0.0` is not published.** `examples/knn` imports it by
   hub path, and CI resolves packages from the hub — so CI stays red for the
   kNN check until `make release PKG=report` runs. Publishing is public and
   permanent and needs `bend login`; it is waiting on the maintainer's go.
2. The second transformer task with a classification head (the handoff's item
   2), which can print a real report from day one.
3. The W&B sweep over the selected parameter (item 3).
4. The answers on bend-ml#20 and bend#1848, when they come.

**What is unverified.** The kNN report's agreement is one split of one
dataset; the report's own test is the broader check. Timings unchanged (no
hot path moved).

## 2026-10-24 — pinned bytes, shared folds, and a real oracle

**Where we are.** The experiment layer now has the two things it was missing:
the bytes cannot move, and the numbers have an uncertainty and an exact
reference.

- **Pinned data.** Every dataset URL names a HuggingFace revision, not `main`,
  and each task carries the expected SHA-256. The runner verifies the bytes it
  reads and fails on a mismatch. `MLBEND_DATA` is authoritative:
  `mlbend-datasets@0.1.6.0` treats a cache miss as an error, so the hash in the
  manifest is always the file the program read.
- **Shared folds, five seeds.** `examples/crossval` prints the fold of every
  row and takes its seed as the first argument. The runner builds once, runs
  seeds **42–46**, and the NumPy reference reads the printed assignment and
  runs the same 60 SGD steps. The mean agrees to **5.9e-8**, each fold to
  **2e-8**; the old 0.03 tolerance for two different partitions is gone.
- **A real naive-Bayes bug.** The exact comparison showed heart at **0.9333**
  against scikit-learn's **0.95**. scikit-learn adds `var_smoothing = 1e-9 ×
  max(variance)`; the Bend examples did not. Both baselines now match exactly:
  heart **0.95**, breast **0.98540145**, gated at 1e-6.
- **Upstream PR #1845** (`bendlang/bend`): `File.read` now fills its buffer
  instead of answering one `read(2)`. The repro is a pipe that writes three
  bytes, sleeps, then three more; a six-byte read used to answer three. The
  PR carries a FIFO regression on the C and JS lanes.
- **The bend-ml gap is not a PR.** `bend-ml-tensor` is the only `Mat` autograd
  trains; `bend-ml-tensor-array` is the faster one with no autodiff. Their
  README freezes autograd and moves training to BendTorch, and their
  CONTRIBUTING asks for an issue before an architectural change. A PR without
  a design would be noise; the useful next artifact is an `autograd-array`
  proposal, not a patch.

**What was run.**
- `make experiments` with the pinned bytes and five seeds; every task passes.
- `reference/experiment.py --all --trainer numpy`; all nine adapter reports.
- `reference/test_crossval.py` compares the shared folds and the same model.
- `reference/test_datasets_cache.py` checks both the cache hit and the strict
  miss.
- `make check` → **66/66**; CI on `ef247ae` → **74/74** in 22m18s.

**What is next.**
1. `mlbend-report`: confusion matrix and per-class precision/recall/F1 with a
   law, then wire `cancer-knn` to print it.
2. A second transformer task with a classification head, accuracy beside loss.
3. The W&B sweep over the selected parameter, now that the folds are shared.
4. If the maintainers want it, an `autograd-array` design issue for bend-ml;
   otherwise use BendTorch for training and keep tensor-array for inference.

**What is unverified.**
- Timings are one machine and a warm cache.
- The pinned revision is the current `main` at the time of pinning; moving the
  pin is a deliberate commit, and the expected hash catches a bad move.
- `/understand`'s graph is structural.

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
