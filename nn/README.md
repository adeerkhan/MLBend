# mlbend-nn

The neural-network surface: a linear layer with gradients, over
[bend-ml](https://github.com/nuxyel/bend-ml)'s tensors and its autodiff.

```python
import mlbend-nn@0.1.0.0/main.bend as NN

m0 = NN.start()                                  # every weight zero
l  = NN.loss(m0, x, labels)                      # the mean cross-entropy
m1 = NN.step(m0, x, labels, 0.01)                # one step of SGD
```

The shapes are in the types: the layer is a `Mat<4n, 3n>` of weights and a
`Vec<3n>` of bias, so a layer of the wrong width does not compile, and the
gradient of a layer has the shape of the layer.

The gradients are bend-ml's: `linear_bwd` is the backward of a linear
layer, `ce_grad` is the gradient of the loss with respect to the logits,
and `sgd` is the step.

The tensors here are bend-ml's list-based ones, the ones its autodiff is
written against. The flat tensors are fifty times faster and have no
autodiff behind them: that is the trade, and the tool that resolves them
says so.
