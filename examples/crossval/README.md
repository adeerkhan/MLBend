# Cross-validation, without leakage

The whole chain: iris from HuggingFace, the scaler fitted by `mlbend-pipeline` on the training rows of each fold, logistic regression with bend-ml's autodiff, the folds from `mlbend-select`, the score from `mlbend-metrics`.

```console
$ bendenv run -- bend examples/crossval/main.bend
```

Mean accuracy **0.9333**, against scikit-learn's `Pipeline(StandardScaler, LogisticRegression)` at **0.9533** on the same folds. The two libraries shuffle with different generators, so the folds differ; the procedure agrees.

![plot](plot.svg)
