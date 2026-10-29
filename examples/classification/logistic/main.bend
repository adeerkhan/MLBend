# A logistic regression over the iris dataset, fetched from HuggingFace
# at run time, with bend-ml's tensors and bend-ml's autodiff.
#
#   bendenv run -- bend examples/logistic/main.bend
#
# Nothing here re-implements bend-ml: the forward pass is `AG.linear`,
# the loss is `AG.ce_loss`, the gradient of the loss with respect to the
# logits is `AG.ce_grad`, the backward of the layer is `AG.linear_bwd`
# and the step is `AG.sgd`. The features are standardised by
# `mlbend-preprocess`, which is bend-ml's arithmetic over bend-ml's
# matrix. What this program adds is the model and the loop over it.
#
# The standardising is what makes the learning rate of 0.5 sane: on the
# raw features (which run from 0.1 to 7) that rate overshoots and the
# loss rises.
import Base
import bend-ml-tensor@0.1.0.0/main.bend as T
import bend-ml-autograd@0.1.1.0/main.bend as AG
import mlbend-datasets@0.1.6.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf
import mlbend-preprocess@0.1.0.0/main.bend as P

# The model: the weights of the layer and its bias.

type Model is Data:
  Model{+w: T.Mat<4n, 3n>, +b: T.Vec<3n>}

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

# The forward pass: the logits of every row.
def logits(+m: Model, x: T.Mat<150n, 4n>) -> T.Mat<150n, 3n>:
  match m:
    case Model{+w, +b}:
      AG.linear(150n, 4n, 3n, x, w, b)

# The loss: the mean cross-entropy of the logits against the labels.
def gradient(+m: Model, x: T.Mat<150n, 4n>, labels: List<&2, Nat>) -> T.Mat<150n, 3n>:
  AG.ce_grad(150n, 3n, logits(m, x), labels)

# One step of SGD.
def join4(a: String, b: String, c: String, d: String) -> String:
  String.append(String.append(String.append(a, "\n"), String.append(b, "\n")),
    String.append(String.append(c, "\n"), d))

def loss(+m: Model, x: T.Mat<150n, 4n>, labels: List<&2, Nat>) -> F32:
  AG.ce_loss(150n, 3n, logits(m, x), labels)

# The gradient of the loss with respect to the logits.
def start() -> Model:
  Model{T.Mat.fill(4n, 3n, 0.0), T.Vec.fill(3n, 0.0)}

# The data as bend-ml's tensor, and the labels beside it.
def stepped(w: T.Mat<4n, 3n>, b: T.Vec<3n>, +lr: F32,
  g: T.Mat<150n, 4n> & (T.Mat<4n, 3n> & T.Vec<3n>)) -> Model:
  (dx, (dw, db)) = g
  Model{AG.sgd(4n, 3n, lr, w, dw), AG.sgd_vec(3n, lr, b, db)}

# The starting model: every weight zero, so every class scores the same.
def step(+m: Model, +x: T.Mat<150n, 4n>, labels: List<&2, Nat>, lr: F32) -> Model:
  match m:
    case Model{+w, +b}:
      stepped(w, b, lr, AG.linear_bwd(150n, 4n, 3n, x, w, gradient(m, x, labels)))

# The step, from the gradient of the layer.
def training(+x: T.Mat<150n, 4n>, +labels: List<&2, Nat>) -> String:
  +m0 = start()
  l0 = loss(m0, x, labels)
  +m1 = step(m0, x, labels, 0.5)
  l1 = loss(m1, x, labels)
  +m2 = step(m1, x, labels, 0.5)
  l2 = loss(m2, x, labels)
  +m3 = step(m2, x, labels, 0.5)
  l3 = loss(m3, x, labels)
  join4(
    String.append("loss before: ", F32.show(l0)),
    String.append("loss after:  ", F32.show(l1)),
    String.append("loss after:  ", F32.show(l2)),
    String.append("loss after:  ", F32.show(l3)))

def show_tensor(t: Result<&1, &1, String, T.Mat<150n, 4n>>, labels: List<&2, Nat>) -> IO(Unit):
  match t:
    case Fail{why}:
      IO.print("the rows are wrong")
    case Done{x}:
      IO.print(training(x, labels))

# The rows as bend-ml's matrix, standardised.
def standard_of(m: Maybe<&2, T.Mat<150n, 4n>>) -> Result<&1, &1, String, T.Mat<150n, 4n>>:
  match m:
    case Some{matrix}:
      Done{P.standardize(150n, 4n, matrix)}
    case None{}:
      Fail{"the rows are not 150 of four"}

def tensor_of(+rows: List<&2, List<&2, F32>>) -> Result<&1, &1, String, T.Mat<150n, 4n>>:
  standard_of(T.Mat.of(150n, 4n, rows))

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

# The losses of four steps of SGD.
def mat_of(m: Maybe<&2, T.Mat<150n, 4n>>) -> Result<&1, &1, String, T.Mat<150n, 4n>>:
  match m:
    case Some{matrix}:
      Done{matrix}
    case None{}:
      Fail{"the rows are not 150 of four"}