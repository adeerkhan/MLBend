# MLBend

Machine learning for [Bend 2](https://bend-lang.com), in the shape scikit-learn has:
estimators, transformers, pipelines, metrics, model selection. What Bend changes
is the ground under all of it: an estimator is a **value** and not an object with
mutated state, the **shape** of the data is in the type, and the structural
guarantees are **laws** that Bend's kernel checks.

This is the classic-ML half of the ecosystem. The tensor, autodiff and model
half lives in [bend-ml](https://github.com/nuxyel/bend-ml); MLBend builds the
estimator surface on top of it and does not repeat it.

## The design, in five rules

1. **Values, not objects.** `scaler = Scaler.fit(x)` and then `z = scaler.transform(y)`:
   a straight line of data, no `self`, nothing mutated. Bend's affine rule makes
   this the only shape a program can have, and a fitted model is a value you can
   pass, store and thread.
2. **Shapes in the types.** `Mat<r, c>` for the data, a fitted model over `c`
   features is `Model<c, k>`, and `predict : Mat<n, c> -> Mat<n, k>`. A model
   applied to the wrong number of features does not compile.
3. **Structure proved, numbers tested.** Sizes, partitions and compositions are
   laws the kernel checks; `F32` results are compared with NumPy and
   scikit-learn. `F32` is not a real number, so no value is ever proved.
4. **Effects at the edges only.** Fetching a dataset, reading a file, seeding a
   generator. The estimators themselves are pure, so a run is reproducible.
5. **Errors are values.** A length that does not match is a `Result`, not a
   crash: a metric answers `Done{value}` or `Fail{Mismatch{...}}`.

## Packages

| Package | What it is | Status |
|---|---|---|
| [`mlbend-metrics`](metrics) | the evaluation surface: MSE, MAE, RMSE, accuracy, precision, recall, F1, and the counts they rest on | v0.1: written, checked, one law proved |

More are planned; the order and the stop criteria are in [`docs/PLAN.md`](docs/PLAN.md).

## What is proved, and what is trusted

[`docs/AUDIT.md`](docs/AUDIT.md) says what each package proves, what it only
tests, and what it trusts. The short version: the counts are proved, the `F32`
numbers are tested against NumPy and scikit-learn.

## For an agent

Start at [`AGENTS.md`](AGENTS.md): the rules, the read order, the commands, and
where the language lives (the guide and the Base source come with the pinned
install; the language's own source is `bendlang/bend`, in its own repository, and
is not vendored here). [`docs/agents/handoff.md`](docs/agents/handoff.md) says
where the work is right now; [`docs/agents/domain.md`](docs/agents/domain.md) is
the map of the packages; [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md)
is how the issues carry the work.

## Checks

```bash
make setup        # the pinned Bend 2.0.37, Lean 4.34.0 and the Python venv
make check        # the proofs, the kernel (--verdict), the rejected programs, the NumPy comparison
make laws         # every law statement of every package
```

`make check` is the gate before a commit; CI runs it on every push.

## License

MIT. Everything in this repository is MIT, and a package published to BendHub
carries its own `LICENSE` next to the entry file.
