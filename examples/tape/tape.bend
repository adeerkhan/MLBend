# The same logistic regression, over bendtorch's tape.
#
#   bendenv run -- bend examples/tape/tape.bend
#
# bend-ml's autodiff is functional: the backward pass is a function of
# the forward. bendtorch's is a tape: the forward records the nodes, the
# backward walks them, and the step reads the gradients off the leaves.
#
# Nothing here re-implements bendtorch: the tape is
# `Backward.leaf`, `Backward.linear`, `Backward.cross_entropy` and
# `Backward.backward`, and the step is `Sgd.step`. What this program
# adds is the model and the loop over it.
#
# The nodes of the tape: the weights are leaf 0, the bias leaf 1 and the
# data node 2. The data is a constant, as PyTorch's input tensor has
# requires_grad=False, so backward never computes its gradient.
#
# bendtorch's `linear` records `x @ W^T` with the weights of [out, in]
# (PyTorch's layout) and the bias of [out]: so the weights here are three
# rows of four (three classes, four features) and the bias is three
# numbers.
#
# The features are standardised by `mlbend-preprocess`, over bend-ml's
# matrix, so that this program and `examples/logistic` train the same
# model on the same data and can be compared.
#
# A node's value is fixed when it is recorded, so the loss *after* the
# step is a second forward pass on the stepped tape: the step writes the
# new weights into the leaves, and a recording after it reads them.
import Base
import bendtorch-core@0.1.0.0/src/tensor/core.bend as Core
import bendtorch-core@0.1.0.0/src/autograd/tensor_backward.bend as Backward
import bendtorch-core@0.1.0.0/src/autograd/sgd.bend as Sgd
import bend-ml-tensor@0.1.0.0/main.bend as T
import mlbend-datasets@0.1.4.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf
import mlbend-preprocess@0.1.0.0/main.bend as P

# Why a step did not come: the code of the first failure.

type Err is Data:
  Err{code: U32}

# The losses of one step: the loss before it and the loss after it.
type Losses is Data:
  Losses{before: F32, after: F32}

# One row's numbers, appended to the numbers so far.
def add_row(acc: List<&2, F32>, row: List<&2, F32>) -> List<&2, F32>:
  List.append(&2, F32, acc, row)

# One label, as the class index the tape wants.
def target_of(label: F32) -> U32:
  U32.from_nat(U32.to_nat(F32.to_u32(label)))

# One label, appended to the labels so far.
def add_target(acc: List<&2, U32>, n: F32) -> List<&2, U32>:
  List.append(&2, U32, acc, target_of(n) <> Nil{})

# One zero, appended to the numbers so far.
def add_zero(acc: List<&2, F32>, n: Nat) -> List<&2, F32>:
  List.append(&2, F32, acc, 0.0 <> Nil{})

# The number a tensor holds, when it holds one.
def read_of(pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> Result<&1, &1, Err, F32>:
  (dynamic, result) = pair
  match result:
    case Fail{error}:
      Fail{Err{Core.error_code(error)}}
    case Done{value}:
      Done{value}

# The value of a node, as a number.
def number_of(pair: Backward.Tape & Result<&2, &1, Backward.Error, Core.Dynamic>) ->
  Result<&1, &1, Err, F32>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{Err{Backward.error_code(error)}}
    case Done{dynamic}:
      read_of(Core.read_dynamic(dynamic, 0))

# The loss of the tape as it is now.
def after_go(pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>) ->
  Result<&1, &1, Err, F32>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{Err{Backward.error_code(error)}}
    case Done{id}:
      number_of(Backward.value(tape, id))

# The logits of every row: the linear of the data with the weights and
# the bias, and the id of the node.
def logits(tape: Backward.Tape) -> Backward.Tape & Result<&2, &2, Backward.Error, Nat>:
  Backward.linear(tape, 2n, 0n,  1n)

# The loss of the logits, and the id of the loss node.
def loss(pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>, targets: List<&2, U32>) ->
  Backward.Tape & Result<&2, &2, Backward.Error, Nat>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      (tape, Fail{error})
    case Done{id}:
      Backward.cross_entropy(tape, id, targets)

# The loss of the tape as it is now, and the tape.
def after_of(tape: Backward.Tape, targets: List<&2, U32>) -> Result<&1, &1, Err, F32>:
  after_go(loss(logits(tape), targets))

# count numbers, all zero.
def zeros(count: Nat) -> List<&2, F32>:
  List.foldl(&2, Nat, List<&2, F32>, add_zero, List.range(count), Nil{})

# The bias as bendtorch's own tensor: three zeros.
def bias_of() -> Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.construct([3], zeros(3n))

# The rows of the data as one flat list of numbers.
def flat_of(rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  List.foldl(&2, List<&2, F32>, List<&2, F32>, add_row, rows, Nil{})

# The rows, standardised, when there is a matrix of them.
def prepared_of(+rows: List<&2, List<&2, F32>>, m: Maybe<&2, T.Mat<150n, 4n>>) ->
  List<&2, List<&2, F32>>:
  match m:
    case Some{matrix}:
      T.Mat.to_rows(150n, 4n, P.standardize(150n, 4n, matrix))
    case None{}:
      rows

# The rows, standardised, over bend-ml's matrix.
def prepared(+rows: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  prepared_of(rows, T.Mat.of(150n, 4n, rows))

# The data as bendtorch's own tensor: 150 rows of four, standardised.
def data_of(+rows: List<&2, List<&2, F32>>) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.construct([150, 4], flat_of(prepared(rows)))

# The tape of a pair: the tape and the id of the node.
def tape_of(pair: Backward.Tape & Nat) -> Backward.Tape:
  (tape, id) = pair
  tape

# The leaves of the tape: the weights (0), the bias (1), the data (2).
def leaves(w: Core.Dynamic, b: Core.Dynamic, x: Core.Dynamic) -> Backward.Tape:
  tape_of(Backward.constant(tape_of(Backward.leaf(tape_of(Backward.leaf(Backward.empty(), w)), b)), x))

# Why the step did not come.
def show_error(error: Err) -> IO(Unit):
  match error:
    case Err{code}:
      IO.print(String.append("the step failed: ", U32.show(code)))

# The two losses, as one line each.
def two(l: Losses) -> String:
  match l:
    case Losses{before, after}:
      String.append(String.append("loss before: ", F32.show(before)),
        String.append("\nloss after:  ", F32.show(after)))

# The step, and the two losses.
def show_losses(r: Result<&1, &1, Err, Losses>) -> IO(Unit):
  match r:
    case Fail{error}:
      show_error(error)
    case Done{losses}:
      IO.print(two(losses))

# The labels as the numbers the tape wants.
def targets_of(+labels: List<&2, F32>) -> List<&2, U32>:
  List.foldl(&2, F32, List<&2, U32>, add_target, labels, Nil{})

# The weights as bendtorch's own tensor: three rows of four, all zero.
def weights_of() -> Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.construct([3, 4], zeros(12n))

# The step, and the loss before it.
def stepped_go4(pair: Backward.Tape & Result<&2, &2, Sgd.Error, Unit>, value: F32) ->
  Result<&1, &1, Err, (F32 & Backward.Tape)>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{Err{Sgd.error_code(error)}}
    case Done{unit}:
      Done{(value, tape)}

# The backward pass, and the step on the leaves.
def stepped_go3(pair: Backward.Tape & Result<&2, &2, Backward.Error, Unit>, value: F32,
  lr: F32) -> Result<&1, &1, Err, (F32 & Backward.Tape)>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{Err{Backward.error_code(error)}}
    case Done{unit}:
      stepped_go4(Sgd.step(tape, lr, [0n, 1n]), value)

# The loss before the step, and the backward pass.
def stepped_go2(r: Result<&1, &1, Err, F32>,
  pair: Backward.Tape & Result<&2, &2, Backward.Error, Unit>, lr: F32) ->
  Result<&1, &1, Err, (F32 & Backward.Tape)>:
  match r:
    case Fail{error}:
      Fail{error}
    case Done{value}:
      stepped_go3(pair, value, lr)

# The value of the loss, and the backward pass.
def stepped_go(pair: Backward.Tape & Result<&2, &1, Backward.Error, Core.Dynamic>, id: Nat,
  lr: F32) -> Result<&1, &1, Err, (F32 & Backward.Tape)>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{Err{Backward.error_code(error)}}
    case Done{dynamic}:
      stepped_go2(read_of(Core.read_dynamic(dynamic, 0)), Backward.backward(tape, id), lr)

# The forward pass, the loss, its value, the backward pass and the step:
# the loss before the step and the tape that holds the new weights.
def stepped(pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>, lr: F32) ->
  Result<&1, &1, Err, (F32 & Backward.Tape)>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{Err{Backward.error_code(error)}}
    case Done{+id}:
      stepped_go(Backward.value(tape, id), id, lr)

# The loss after the step, and the loss before it.
def training_go2(r: Result<&1, &1, Err, F32>, before: F32) -> Result<&1, &1, Err, Losses>:
  match r:
    case Fail{error}:
      Fail{error}
    case Done{after}:
      Done{Losses{before, after}}

# The loss before the step, and the tape to step.
def training_go(r: Result<&1, &1, Err, (F32 & Backward.Tape)>, lr: F32,
  targets: List<&2, U32>) -> Result<&1, &1, Err, Losses>:
  match r:
    case Fail{error}:
      Fail{error}
    case Done{pair}:
      (before, tape) = pair
      training_go2(after_of(tape, targets), before)

# The losses of one step: the loss before it and the loss after it.
def training(tape: Backward.Tape, +targets: List<&2, U32>, +lr: F32) ->
  Result<&1, &1, Err, Losses>:
  training_go(stepped(loss(logits(tape), targets), lr), lr, targets)

# The tape, with the bias to go.
def with_bias(b: Result<&2, &1, Core.Error, Core.Dynamic>, w: Core.Dynamic,
  x: Core.Dynamic, targets: List<&2, U32>) -> Result<&1, &1, Err, Losses>:
  match b:
    case Fail{error}:
      Fail{Err{Core.error_code(error)}}
    case Done{bias}:
      training(leaves(w, bias, x), targets, 0.5)

# The tape, with the data to go.
def with_data(d: Result<&2, &1, Core.Error, Core.Dynamic>, targets: List<&2, U32>,
  w: Core.Dynamic) -> Result<&1, &1, Err, Losses>:
  match d:
    case Fail{error}:
      Fail{Err{Core.error_code(error)}}
    case Done{x}:
      with_bias(bias_of(), w, x, targets)

# The tape, with the weights and the data to go.
def with_weights(w: Result<&2, &1, Core.Error, Core.Dynamic>,
  d: Result<&2, &1, Core.Error, Core.Dynamic>, targets: List<&2, U32>) ->
  Result<&1, &1, Err, Losses>:
  match w:
    case Fail{error}:
      Fail{Err{Core.error_code(error)}}
    case Done{weights}:
      with_data(d, targets, weights)

# The tape, from the tensors the caller made.
def tape_of_rows(rows: List<&2, List<&2, F32>>, labels: List<&2, F32>) ->
  Result<&1, &1, Err, Losses>:
  with_weights(weights_of(), data_of(rows), targets_of(labels))

# The step, and the two losses.
def show_rows(+rows: List<&2, List<&2, F32>>, +labels: List<&2, F32>) -> IO(Unit):
  show_losses(tape_of_rows(rows, labels))

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