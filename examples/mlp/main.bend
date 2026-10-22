# A two-layer neural network over the iris dataset, with bend-ml's
# tensors and bend-ml's autodiff.
#
#   bendenv run -- bend examples/mlp/main.bend
#
# The forward pass is `AG.linear` twice, with a rectifier between them:
#
#   hidden  = relu(x · W1 + b1)        x: 150 x 4,  W1: 4 x 8
#   logits  = hidden · W2 + b2         W2: 8 x 3
#   loss    = cross-entropy of the logits against the labels
#
# The backward pass is the chain of the same functions: the gradient of
# the loss with respect to the logits is `AG.ce_grad`, `AG.linear_bwd`
# gives the gradient for the second layer and the gradient that arrives
# at the hidden layer, `AG.relu_bwd` passes it through the rectifier, and
# `AG.linear_bwd` again gives the gradient for the first layer. The step
# is `AG.sgd` and `AG.sgd_vec`.
#
# Nothing here re-implements bend-ml: the layer, the rectifier, the loss,
# the gradients and the step are the library's. What this program adds is
# the model, the initial weights and the loop.
#
# The features are standardised by `mlbend-preprocess`, over bend-ml's
# matrix, as `examples/logistic` does, so the learning rate is sane.
import Base
import bend-ml-tensor@0.1.0.0/main.bend as T
import bend-ml-autograd@0.1.1.0/main.bend as AG
import mlbend-datasets@0.1.4.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf
import mlbend-preprocess@0.1.0.0/main.bend as P

# The model: two layers and their biases.

type Model is Data:
  Model{w1: T.Mat<4n, 8n>, b1: T.Vec<8n>, w2: T.Mat<8n, 3n>, b2: T.Vec<3n>}

# The class of a label.
def class_of(label: F32) -> Nat:
  U32.to_nat(F32.to_u32(label))

# The labels of the dataset, as the class indices they are, one by one.
def class_labels_go(+labels: List<&2, F32>, +acc: List<&2, Nat>) -> List<&2, Nat>:
  match labels:
    case Nil{}:
      acc
    case Con{h, t}:
      class_labels_go(t, List.append(&2, Nat, acc, Con{class_of(h), Nil{}}))

# The labels of the dataset, as the class indices they are.
def class_labels(+labels: List<&2, F32>) -> List<&2, Nat>:
  class_labels_go(labels, Nil{})

# One starting weight, from its place: small, and different for every
# place, so the hidden units do not start as copies of each other.
def cell(+i: Nat, +j: Nat) -> F32:
  F32.sub(F32.mul(F32.from_nat(Nat.mod(Nat.add(Nat.mul(i, 3n), Nat.mul(j, 5n)), 7n)), 0.1), 0.3)

# One row of starting weights.
def init_row_go(+i: Nat, +c: Nat, +j: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match c:
    case 0n:
      acc
    case 1n+rest:
      init_row_go(i, rest, Nat.add(j, 1n),
        List.append(&2, F32, acc, Con{cell(i, j), Nil{}}))

# One row of starting weights.
def init_row(i: Nat, c: Nat) -> List<&2, F32>:
  init_row_go(i, c, 0n, Nil{})

# The rows of starting weights.
def init_rows_go(+n: Nat, +c: Nat, +i: Nat, +acc: List<&2, List<&2, F32>>) ->
  List<&2, List<&2, F32>>:
  match n:
    case 0n:
      acc
    case 1n+rest:
      init_rows_go(rest, c, Nat.add(i, 1n),
        List.append(&2, List<&2, F32>, acc, Con{init_row(i, c), Nil{}}))

# The rows of starting weights.
def init_rows(n: Nat, c: Nat) -> List<&2, List<&2, F32>>:
  init_rows_go(n, c, 0n, Nil{})

# The first layer's weights.
def weights1() -> T.Mat<4n, 8n>:
  T.Mat{init_rows(4n, 8n)}

# The second layer's weights.
def weights2() -> T.Mat<8n, 3n>:
  T.Mat{init_rows(8n, 3n)}

# The model the run starts from: varied weights and zero biases.
def start() -> Model:
  Model{weights1(), T.Vec.fill(8n, 0.0), weights2(), T.Vec.fill(3n, 0.0)}

# The first layer before the rectifier.
def pre(w1: T.Mat<4n, 8n>, b1: T.Vec<8n>, x: T.Mat<150n, 4n>) -> T.Mat<150n, 8n>:
  AG.linear(150n, 4n, 8n, x, w1, b1)

# The rectifier, over the rows of a matrix.
def rectified(m: T.Mat<150n, 8n>) -> T.Mat<150n, 8n>:
  match m:
    case T.Mat{rows}:
      T.Mat{AG.mask.rows(rows)}

# The hidden layer: the first layer through the rectifier.
def hidden(+w1: T.Mat<4n, 8n>, +b1: T.Vec<8n>, x: T.Mat<150n, 4n>) ->
  T.Mat<150n, 8n>:
  rectified(pre(w1, b1, x))

# The logits of every row.
def logits(+m: Model, x: T.Mat<150n, 4n>) -> T.Mat<150n, 3n>:
  match m:
    case Model{+w1, +b1, +w2, +b2}:
      AG.linear(150n, 8n, 3n, hidden(w1, b1, x), w2, b2)

# The loss: the mean cross-entropy of the logits against the labels.
def loss(+m: Model, +x: T.Mat<150n, 4n>, labels: List<&2, Nat>) -> F32:
  AG.ce_loss(150n, 3n, logits(m, x), labels)

# The model after both layers have been stepped, from the gradient of
# the first layer. A tuple is destructured by a def that takes it as a
# parameter: a computed value cannot be scrutinised in place.
def finished(w1: T.Mat<4n, 8n>, b1: T.Vec<8n>, w2: T.Mat<8n, 3n>, b2: T.Vec<3n>, +lr: F32,
  g1: T.Mat<150n, 4n> & (T.Mat<4n, 8n> & T.Vec<8n>)) -> Model:
  (d_x, (dw1, db1)) = g1
  Model{AG.sgd(4n, 8n, lr, w1, dw1), AG.sgd_vec(8n, lr, b1, db1), w2, b2}

# The second layer stepped, the rectifier passed, and the first layer's
# gradient taken: the model follows.
def after_hidden(+w1: T.Mat<4n, 8n>, +b1: T.Vec<8n>, +w2: T.Mat<8n, 3n>,
  +b2: T.Vec<3n>, +lr: F32, +x: T.Mat<150n, 4n>, z1: T.Mat<150n, 8n>,
  g2: T.Mat<150n, 8n> & (T.Mat<8n, 3n> & T.Vec<3n>)) -> Model:
  (d_h, (dw2, db2)) = g2
  +d_z1 = AG.relu_bwd(150n, 8n, z1, d_h)
  finished(w1, b1, AG.sgd(8n, 3n, lr, w2, dw2), AG.sgd_vec(3n, lr, b2, db2), lr,
    AG.linear_bwd(150n, 4n, 8n, x, w1, d_z1))

# One step of SGD, through both layers.
def step(+m: Model, +x: T.Mat<150n, 4n>, labels: List<&2, Nat>, +lr: F32) -> Model:
  match m:
    case Model{+w1, +b1, +w2, +b2}:
      +z1 = pre(w1, b1, x)
      +h = rectified(z1)
      +out = AG.linear(150n, 8n, 3n, h, w2, b2)
      +d_out = AG.ce_grad(150n, 3n, out, labels)
      after_hidden(w1, b1, w2, b2, lr, x, z1, AG.linear_bwd(150n, 8n, 3n, h, w2, d_out))

# The steps, the fuel first.
def train_go(fuel: List<&2, Nat>, +m: Model, +x: T.Mat<150n, 4n>,
  +labels: List<&2, Nat>) -> Model:
  match fuel:
    case Nil{}:
      m
    case Con{u, t}:
      train_go(t, step(m, x, labels, 0.5), x, labels)

# The model after n steps.
def train(n: Nat, m: Model, x: T.Mat<150n, 4n>, labels: List<&2, Nat>) -> Model:
  train_go(List.range(n), m, x, labels)

# The place of the largest number of a row.
def argmax_go(+row: List<&2, F32>, +i: Nat, +best: Nat, +best_v: F32) -> Nat:
  match row:
    case Nil{}:
      best
    case Con{+h, t}:
      Bool.pick(Nat, F32.is_gt(h, best_v),
        argmax_go(t, Nat.add(i, 1n), i, h),
        argmax_go(t, Nat.add(i, 1n), best, best_v))

# The place of the largest number of a row.
def argmax(row: List<&2, F32>) -> Nat:
  argmax_go(row, 0n, 0n, (0.0 - 1.0e30 : F32))

# How many rows the model gets right, row by row.
def correct_go(labels: List<&2, Nat>, +rows: List<&2, List<&2, F32>>, +acc: Nat) -> Nat:
  match labels:
    case Nil{}:
      acc
    case Con{l, ls}:
      match rows:
        case Nil{}:
          acc
        case Con{r, rs}:
          correct_go(ls, rs, Bool.pick(Nat, Nat.is_eq(argmax(r), l), Nat.add(acc, 1n), acc))

# How many rows the model gets right.
def correct(labels: List<&2, Nat>, +rows: List<&2, List<&2, F32>>) -> Nat:
  correct_go(labels, rows, 0n)

# The rows of a matrix, out of it.
def rows_of(m: T.Mat<150n, 3n>) -> List<&2, List<&2, F32>>:
  match m:
    case T.Mat{rows}:
      rows

# The accuracy, as a number between 0 and 1.
def accuracy(labels: List<&2, Nat>, rows: List<&2, List<&2, F32>>) -> F32:
  (F32.from_nat(correct(labels, rows)) / 150.0 : F32)

# The two losses and the accuracy, as one line each.
def report(before: F32, after: F32, acc: F32) -> String:
  String.append(
    String.append(
      String.append(
        String.append("loss before: ", F32.show(before)),
        "\nloss after:  "),
      F32.show(after)),
    String.append("\naccuracy: ", F32.show(acc)))

# The rows as bend-ml's matrix, standardised.
def standard_of(m: Maybe<&2, T.Mat<150n, 4n>>) -> Result<&1, &1, String, T.Mat<150n, 4n>>:
  match m:
    case Some{matrix}:
      Done{P.standardize(150n, 4n, matrix)}
    case None{}:
      Fail{"the rows are not 150 of four"}

# The rows as bend-ml's matrix, when they have the right shape.
def tensor_of(+rows: List<&2, List<&2, F32>>) -> Result<&1, &1, String, T.Mat<150n, 4n>>:
  standard_of(T.Mat.of(150n, 4n, rows))

# The run: the loss before, the loss after, and the accuracy.
def run(+x: T.Mat<150n, 4n>, +labels: List<&2, Nat>) -> String:
  +m0 = start()
  l0 = loss(m0, x, labels)
  +m1 = train(400n, m0, x, labels)
  l1 = loss(m1, x, labels)
  report(l0, l1, accuracy(labels, rows_of(logits(m1, x))))

def show_tensor(t: Result<&1, &1, String, T.Mat<150n, 4n>>, labels: List<&2, Nat>) -> IO(Unit):
  match t:
    case Fail{why}:
      IO.print("the rows are wrong")
    case Done{x}:
      IO.print(run(x, labels))

# The matrix of the rows, when there is one.
def show_rows(+rows: List<&2, List<&2, F32>>, +labels: List<&2, F32>) -> IO(Unit):
  show_tensor(tensor_of(rows), class_labels(labels))

def show_table(t: Datasets.Table) -> IO(Unit):
  match t:
    case Datasets.Table{+rows, +labels}:
      show_rows(rows, labels)

def show(d: Result<&1, &1, Hf.Err, Datasets.Table>) -> IO(Unit):
  match d:
    case Fail{e}:
      IO.print("the dataset failed")
    case Done{table}:
      show_table(table)

def main() -> IO(Unit):
  do IO<Unit>:
    d : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.iris()
    show(d)
