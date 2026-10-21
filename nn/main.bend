# mlbend-nn: the neural-network surface, over bend-ml's tensors and its
# autodiff.
#
#   import mlbend-nn@0.1.0.0/main.bend as NN
#
# A linear layer with gradients: the weights and the bias, a forward
# pass, the mean cross-entropy, and a step of SGD. The gradients are
# bend-ml's: `linear_bwd` is the backward of a linear layer, `ce_grad` is
# the gradient of the loss with respect to the logits, and `sgd` is the
# step.
#
# The shapes are in the types: the layer is a `Mat<4n, 3n>` of weights
# and a `Vec<3n>` of bias, so a layer of the wrong width does not
# compile, and the gradient of a layer has the shape of the layer.
#
# The tensors here are bend-ml's list-based ones, the ones its autodiff
# is written against. The flat tensors are fifty times faster and have
# no autodiff behind them: that is the trade, and the tool that resolves
# them says so.

import Base
import bend-ml-tensor@0.1.0.0/main.bend as T
import bend-ml-autograd@0.1.1.0/main.bend as AG

# The rows of the data, the features of each.
type Rows is Data:
  Rows{rows: List<&2, List<&2, F32>>, labels: List<&2, Nat>}

# A fitted model: the weights and the bias of a linear layer.
type Linear is Data:
  Linear{+w: T.Mat<4n, 3n>, +b: T.Vec<3n>}

# The starting model: every weight zero, so every class scores the same
# and the loss is the log of the number of classes.
def logits(m: Linear, x: T.Mat<150n, 4n>) -> T.Mat<150n, 3n>:
  match m:
    case Linear{+w, +b}:
      AG.linear(150n, 4n, 3n, x, w, b)

# The loss: the mean cross-entropy of the logits against the labels.
def gradient(m: Linear, x: T.Mat<150n, 4n>, labels: List<&2, Nat>) -> T.Mat<150n, 3n>:
  AG.ce_grad(150n, 3n, logits(m, x), labels)

# One step of SGD: the gradient of the loss, and the step.
def loss(m: Linear, x: T.Mat<150n, 4n>, labels: List<&2, Nat>) -> F32:
  AG.ce_loss(150n, 3n, logits(m, x), labels)

# The gradient of the loss with respect to the logits.
def mat_of_maybe(m: Maybe<&2, T.Mat<150n, 4n>>) -> Result<&1, &1, String, T.Mat<150n, 4n>>:
  match m:
    case Some{matrix}:
      Done{matrix}
    case None{}:
      Fail{"the rows are not 150 of four"}

# The forward pass: the logits of every row.
def mat_of(rows: List<&2, List<&2, F32>>) -> Result<&1, &1, String, T.Mat<150n, 4n>>:
  mat_of_maybe(T.Mat.of(150n, 4n, rows))

# The matrix of the rows, when there is one.
def start() -> Linear:
  Linear{T.Mat.fill(4n, 3n, 0.0), T.Vec.fill(3n, 0.0)}

# The data as bend-ml's tensor, and the labels beside it.
def with_gradient(w: T.Mat<4n, 3n>, b: T.Vec<3n>, +lr: F32,
  g: T.Mat<150n, 4n> & (T.Mat<4n, 3n> & T.Vec<3n>)) -> Linear:
  (dx, (dw, db)) = g
  Linear{AG.sgd(4n, 3n, lr, w, dw), AG.sgd_vec(3n, lr, b, db)}

def step(m: Linear, +x: T.Mat<150n, 4n>, labels: List<&2, Nat>, +lr: F32) -> Linear:
  match m:
    case Linear{+w, +b}:
      with_gradient(w, b, lr, AG.linear_bwd(150n, 4n, 3n, x, w, gradient(m, x, labels)))

# The step, from the gradient of the layer.
def tensor_of(r: Rows) -> Result<&1, &1, String, T.Mat<150n, 4n>>:
  match r:
    case Rows{rows, labels}:
      mat_of(rows)

# The rows as a matrix, when they are 150 of four.