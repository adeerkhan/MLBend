# The examples

Programs that use the libraries, rather than packages of our own.
Bend-ml and bendtorch are the libraries: nothing here re-implements
them.

| Example | What it shows |
|---|---|
| [`logistic/main.bend`](logistic/main.bend) | a logistic regression over the iris dataset, fetched from HuggingFace at run time, with bend-ml's tensors and bend-ml's autodiff |
| [`tape/tape.bend`](tape/tape.bend) | the same model over bendtorch's tape: the forward records the nodes, the backward walks them, the step reads the gradients off the leaves |
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

## The tape example

The same model, over bendtorch's tape. It compiles and runs: the data,
the weights and the bias are built, the leaves go on the tape, and the
loss reads back from it. The `linear` node still refuses the operands —
error 106, an inner dimension mismatch — so the shape convention of the
weights (PyTorch's `[out, in]` or the mathematical `[in, out]`) and the
rank of the bias are not settled yet. `bendtorch`'s own MNIST demo is the
reference for both.
