# The examples

Programs that use the libraries and the packages, rather than packages of
their own. Bend-ml and bendtorch are the libraries: nothing here
re-implements them.

| Example | What it shows |
|---|---|
| [`logistic/main.bend`](logistic/main.bend) | a logistic regression over the iris dataset, fetched from HuggingFace at run time, with bend-ml's tensors and bend-ml's autodiff |
| [`mlp/main.bend`](mlp/main.bend) | a two-layer network with a rectifier, over the same data and the same library: the chain of gradients through both layers |
| [`tape/tape.bend`](tape/tape.bend) | the logistic model over bendtorch's tape: the forward records the nodes, the backward walks them, the step reads the gradients off the leaves |
| [`regression/main.bend`](regression/main.bend) | ridge regression over the wine dataset, over bendtorch's tape with `mse_loss` and a weight decay in the step |
| [`kmeans/main.bend`](kmeans/main.bend) | k-means over the wine dataset, through `mlbend-cluster`, after standardising the columns |
| [`pca/main.bend`](pca/main.bend) | the first principal component of wine, by power iteration: a transformer, not a trainer |
| [`attention/main.bend`](attention/main.bend) | a single-head self-attention block over four tokens: the core of a transformer, checked against PyTorch |
| [`transformer/main.bend`](transformer/main.bend) | a one-block causal transformer trained on a periodic sequence with the real BendTorch, against a NumPy twin, with the loss curve saved |
| [`knn/main.bend`](knn/main.bend) | k-nearest neighbours over the breast cancer dataset: no gradients, no library, a distance and a vote |
| [`naive_bayes/main.bend`](naive_bayes/main.bend) | Gaussian naive Bayes over the same data: the gradient-free baseline the experiment layer keeps beside the learned models |
| [`crossval/main.bend`](crossval/main.bend) | the whole chain end to end: dataset → scaler fitted on each training fold → logistic regression → stratified folds → accuracy, with no leakage |
| [`stacks/main.bend`](stacks/main.bend) | the two stacks side by side, over one environment |

## The logistic example

```console
$ bendenv run -- bend examples/logistic/main.bend
loss before: 1.0986128
loss after:  1.086162
loss after:  1.0759476
loss after:  1.0671557
```

The forward pass is `AG.linear`, the loss is `AG.ce_loss`, the gradient of
the loss with respect to the logits is `AG.ce_grad`, the backward of the
layer is `AG.linear_bwd`, and the step is `AG.sgd`. The program adds the
model and the loop over it, and nothing else.

The learning rate is 0.01 because the features are not standardised:
iris runs to 7.0, and at 0.5 the loss *rises* (1.099 → 1.537 → 6.88),
which is the step overshooting and not a wrong gradient.
`mlbend-preprocess`, which standardises, is what makes a larger rate
sane.

## The MLP example

```console
$ bendenv run -- bend examples/mlp/main.bend
loss before: 1.1172239
loss after:  0.11065235
accuracy: 0.97333336
```

Eight rectified hidden units between the features and the three classes,
400 full-batch steps at 0.5. The forward pass is `AG.linear`,
`AG.mask.rows` (the rectifier) and `AG.linear` again; the loss is
`AG.ce_loss`; the backward pass is `AG.ce_grad`, `AG.linear_bwd`,
`AG.relu_bwd` and `AG.linear_bwd` again; the step is `AG.sgd` and
`AG.sgd_vec`. The starting weights are a fixed small pattern, so the run
is reproducible: the weights are varied by place, because zero weights
would leave the hidden units with no gradient at all.

scikit-learn's `MLPClassifier` with eight hidden units reaches the same
0.9733 on the same data (`reference/test_mlp.py`); it trains with Adam
and another initialisation, so what agrees is the result, not the path.

## The tape example

The same logistic model, over bendtorch's tape. It compiles and runs: the
data, the weights and the bias are built, the leaves go on the tape, and
the loss reads back from it. The weights are `[out, in]` (PyTorch's
layout) and the bias is `[out]`, so the tape's `linear` accepts them; the
losses of the step agree with `examples/logistic`'s to F32 precision.

## The regression example

Ridge regression over the wine dataset, on bendtorch's tape. The features
and the target are standardised, the forward is `linear`, the loss is
`mse_loss`, and the step is `sgd` over the weights and the bias, with a
weight decay applied to the gradient. The MSE is compared with
scikit-learn's `Ridge` (`reference/test_regression.py`).

## The k-means example

The wine dataset, standardised, clustered by `mlbend-cluster` into three
clusters for ten rounds. The example only holds the data and the
standardising; the algorithm lives in the package, where every row is
proved to get exactly one cluster.

## The naive Bayes example

A Gaussian per feature per class, trained on the first four fifths of the
breast cancer dataset: a gradient-free baseline that needs no library and
no standardising. It scores 0.9854, the same as scikit-learn's
`GaussianNB` on the same split. It is also a task in the experiment layer
(`cancer-naive-bayes`).

## The transformer example

A one-block causal transformer on `bendtorch@1.0.1.0`: embedding, RMSNorm,
single-head causal attention, output projection, RMSNorm, linear head, trained
with AdamW for 200 steps on next-token prediction over the periodic sequence
0..7. The loss falls from **2.3591** to **0.0021**; a NumPy twin with the same
deterministic weights starts at the same 2.3591 and ends at 0.1351. The two
curves, the step-by-step losses and the run's manifest are written to
`results/transformer/` (`loss.tsv`, `loss.svg`, `run.json`).

It is a task in the experiment layer (`transformer-periodic`). The
architecture uses the maintained `bendtorch` package, not the old
`bendtorch-core` tape; the plain-Bend `attention` example above remains the
library-free parity check.

## The attention example

The core of a transformer in plain Bend: Q, K and V projections, the scaled
dot-product scores, a softmax over every row, and the weighted sum of V. Four
tokens of eight features with fixed weights, the same formulas in the PyTorch
reference; every output agrees within 1.5e-8 (`reference/test_attention.py`).
This is the forward pass. Training the block needs a softmax/exp gradient on
BendTorch's tape, which does not exist yet — that is the next upstream step.

## The PCA example

The covariance matrix of the standardised wine features, and a hundred
power iterations for its first eigenvector. The explained variance ratio
is 0.275443, against scikit-learn's 0.275443 on the same data. It is a
transformer, not a trainer, and it is a task in the experiment layer
(`wine-pca`).

## The cross-validation example

The whole chain: the iris dataset comes from HuggingFace, the scaler is fitted
by `mlbend-pipeline` on the training rows of each fold — never on the whole
dataset — the model is trained with bend-ml's
autodiff, the folds come from `mlbend-select` over `mlbend-split`, and the
score is `mlbend-metrics`. The program adds the loop over the folds and
nothing else.

## The stacks example

bend-ml's tensors and bendtorch's tensors in one program, resolved by one
lockfile and materialized into one environment. A broken environment shows
up as a missing import rather than as a silent difference.
