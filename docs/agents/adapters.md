# The adapter protocol

A task is data plus a protocol. An adapter runs one trainer against it. The
runner is `reference/experiment.py`; this file says what a task and an adapter
must provide, so a new model or a new trainer does not edit the runner.

## A task

A task is an entry in `TASKS` with these fields:

| field | what it means |
|---|---|
| `description` | one line, printed in `--list` and in the report card |
| `program` | the Bend program, relative to the repository root |
| `dataset` | `name`, `rows`, `features`, `source`, and `files` (the exact URLs hashed and fetched) |
| `seed` | the seed the program uses, or `0` when it is deterministic by construction |
| `parse` | `stdout -> {metric: value}`; the adapter's output grammar |
| `reference` | a function returning the same metric names from NumPy or scikit-learn |
| `metrics` | per metric: `tolerance` and `mode` (`absolute`, `relative`, `exact`, `report`) |

A task whose metric has `mode: report` is printed but not gated: it is a
number the reference does not reproduce exactly (a different fold partition,
or a different convergence speed).

## An adapter

There are two adapters today. `--trainer bend` builds the program with
`bend -o`, runs it, and hands its stdout to the task's `parse`. `--trainer
numpy` runs the reference function as the trainer and records it under
`results/latest-numpy/`, so the same task can be compared A/B without changing
the task. The reference remains the second number in the report card.

A new trainer (BendTorch, bend-ml, a future library) enters by adding a
`parse` for its output and a `program`, or by adding a branch next to
`--trainer`. The task's `dataset`, `metrics` and tolerances do not change.

## The manifest

Every run writes:

- the git commit and whether the tree was dirty;
- the Bend version;
- every `mlbend-*`, `bend-ml-*` and `bendtorch-*` package at its hub hash;
- the dataset's `files`, the pinned `expected_sha256`, the bytes and SHA-256
  read, and the fetch time;
- the machine, the seed list, the build time, the run time (minimum of the
  repeats), the compute estimate, and the reference time;
- the metric table: the Bend number, the reference number, the delta, the
  tolerance, and pass or fail.

The runner fills `reference/.cache/datasets/<repo>/<path>` with the exact
bytes it hashes and sets `MLBEND_DATA`, so a warm run reads the file and does
not fetch: `fetch` is 0 and the run time is compute. On a cold cache the first
run pays the fetch and records it; the next run is warm. When the variable is
set, a miss is an error: the manifest hashes the cached file, so the program
must not quietly read another one. The compute-only kernel benchmark is
`results/BENCH.md`.

## Seeds and shared folds

A task may carry `seeds` (the cross-validation task carries five). The runner
builds once, runs the binary once per seed with the seed as its first argument,
and compares every seed against the reference. `--seeds` overrides the list.
The report says mean ± sample deviation and passes only if every seed passes.

The reference must use the same partition as the program. `iris-crossval`
prints the fold of every row (`folds: ...`) and its seed; the NumPy reference
reads that assignment, standardises inside each fold and runs the same 60 SGD
steps, so the two agree to F32 precision instead of sharing a 0.03 tolerance.

## Weights & Biases

`--wandb` logs the manifest to the project (`mlbend` by default), offline when
there is no `WANDB_API_KEY`. `reference/wandb_sweep.yaml` is the sweep
definition; `make sweep` starts it and `make sweep-agent` runs an agent.
