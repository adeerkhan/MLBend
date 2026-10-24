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

The only adapter today is the Bend program: build it with `bend -o`, run it,
and hand its stdout to the task's `parse`. The reference is the second
adapter: it computes the same names in Python.

A new trainer (BendTorch, bend-ml, a future library) enters by adding a
`parse` for its output and a `program`; if it needs a different invocation,
add a small function next to `run_bend` and call it from the task's `trainer`
field. The task's `dataset`, `metrics` and tolerances do not change.

## The manifest

Every run writes:

- the git commit and whether the tree was dirty;
- the Bend version;
- every `mlbend-*`, `bend-ml-*` and `bendtorch-*` package at its hub hash;
- the dataset's `files`, byte count and SHA-256, and the fetch time;
- the machine, the seed, the build time, the run time (minimum of the
  repeats), the compute estimate, and the reference time;
- the metric table: the Bend number, the reference number, the delta, the
  tolerance, and pass or fail.

The fetch time is a Python probe of the same files; Bend fetches them inside
its own run, so `compute_seconds = run_seconds - fetch_seconds` is an estimate,
not a kernel measurement. The compute-only kernel benchmark is
`results/BENCH.md`.

## Weights & Biases

`--wandb` logs the manifest to the project (`mlbend` by default), offline when
there is no `WANDB_API_KEY`. `reference/wandb_sweep.yaml` is the sweep
definition; `make sweep` starts it and `make sweep-agent` runs an agent.
