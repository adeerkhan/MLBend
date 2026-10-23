# mlbend-pipeline

Fit on the training rows only, transform both sides.

```python
import mlbend-pipeline@0.1.0.0/main.bend as Pipeline
```

The leakage an experiment layer exists to prevent: standardising the whole
dataset before the folds are made, so the scaler has already seen the test
rows. Here the scaler is fitted on the training rows and nothing else:

```python
+scaler = Pipeline.fit(train_rows)              # train only
+train  = Pipeline.transform(scaler, train_rows)
+test   = Pipeline.transform(scaler, test_rows)
```

or, for one fold:

```python
+p = Pipeline.fold(train_rows, test_rows)
# p : Pipeline.Prepared, with p.train and p.test already transformed
```

| function | what it does |
| --- | --- |
| `fit(rows)` | the mean and the population standard deviation of every column of the training rows |
| `transform(scaler, rows)` | every row centred and scaled by those fitted values |
| `fold(train, test)` | fit on `train`, return both sides transformed |

The fitted state is a value:

```python
type Scaler is Data:
  Scaler{means: List<&2, F32>, stds: List<&2, F32>}
```

## What is proved

`pipeline_preserves_rows`: transforming the rows keeps one output row for
every input row — no row is dropped and none is invented. The proof is a
recursion over the rows, the same shape as `mlbend-cluster`'s
`cluster_covers`.

The no-leakage property itself is the API: `fit` can only see the rows it is
given, and `fold` gives it the training rows. A law cannot prove what a caller
passes; the tests hold that end.

## What is tested

The means, the deviations and every transformed number — the unseen test rows
included — against scikit-learn's `StandardScaler` fitted on the same training
rows (`reference/test_pipeline.py`). `examples/crossval` uses the package, and
its mean accuracy is compared with scikit-learn's
`Pipeline(StandardScaler, LogisticRegression)` on the same folds
(`reference/test_crossval.py`).

## The limits, stated

- The columns of a row and the fitted means must have the same length; a short
  row is transformed against the columns it has.
- A column with no spread is divided by `0.0001`, not by zero.
- This is the scaler. Min-max, one-hot and the normalisers are still in
  `mlbend-preprocess`; the next version either grows a chain here or gives that
  package the same `fit`/`transform` split.
