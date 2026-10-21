# The same logistic regression, over bendtorch's tape.
#
#   bendenv run -- bend examples/tape/main.bend
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
# The nodes of the tape: the weights are leaf 0, the bias leaf 1, the
# data leaf 2, the logits node 3 and the loss node 4.
import Base
import bendtorch-core@0.1.0.0/src/tensor/core.bend as Core
import bendtorch-core@0.1.0.0/src/autograd/tensor_backward.bend as Backward
import bendtorch-core@0.1.0.0/src/autograd/sgd.bend as Sgd
import mlbend-datasets@0.1.4.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf

# The model: the weights of the layer and its bias.
type Model is Type:
  Model{weights: Core.Dynamic, bias: Core.Dynamic}

# The leaves of the tape and their ids.
type Leaves is Type:
  Leaves{tape: Backward.Tape, w: Nat, b: Nat, x: Nat}

# The tape: the leaves, the ids of the parameters, and the loss node.
type Tape is Type:
  Tape{tape: Backward.Tape, w: Nat, b: Nat, x: Nat, loss: Nat}

# Why an answer did not come.
type Err is Data:
  ErrBackward{error: Backward.Error}
  ErrSgd{error: Sgd.Error}
  ErrCore{code: U32}

# The rows of the data as one flat list of numbers.
def add_row(acc: List<&2, F32>, row: List<&2, F32>) -> List<&2, F32>:
  List.append(&2, F32, acc, row)

# count numbers, all zero.
def add_target(acc: List<&2, U32>, n: Nat) -> List<&2, U32>:
  List.append(&2, U32, acc, U32.from_nat(n) <> Nil{})

def add_zero(acc: List<&2, F32>, n: Nat) -> List<&2, F32>:
  List.append(&2, F32, acc, 0.0 <> Nil{})

# The data as bendtorch's own tensor: 150 rows of four.
def two(before: F32, mid: F32) -> String:
  String.append(String.append("loss before: ", F32.show(before)),
    String.append("\nloss after:  ", F32.show(mid)))

def again(r: Result<&1, &1, Err, (F32 & Tape)>, before: F32) -> String:
  match r:
    case Fail{e}:
      "the second step failed"
    case Done{pair}:
      (mid, t2) = pair
      two(before, mid)

# The two losses, as one line each.
def zeros(count: Nat) -> List<&2, F32>:
  List.foldl(&2, Nat, List<&2, F32>, add_zero, List.range(count), Nil{})

# One zero, appended to the numbers so far.
def bias_of() -> Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.construct([3], zeros(3n))

# The tape, from the tensors the caller made.
def leaf_x(pair: Backward.Tape & Nat, w1: Nat, b1: Nat) ->
  Result<&1, &1, Err, Leaves>:
  (t3, x1) = pair
  Done{Leaves{t3, w1, b1, x1}}

# The forward pass: the logits, and then the loss.
def leaf_b(pair: Backward.Tape & Nat, w1: Nat, x: Core.Dynamic) ->
  Result<&1, &1, Err, Leaves>:
  (t2, b1) = pair
  leaf_x(Backward.leaf(t2, x), w1, b1)

# The leaves, all in.
def leaf_w(pair: Backward.Tape & Nat, x: Core.Dynamic, b: Core.Dynamic) ->
  Result<&1, &1, Err, Leaves>:
  (t1, w1) = pair
  leaf_b(Backward.leaf(t1, b), w1, x)

# The leaves, with the data to go.
def leaves(x: Core.Dynamic, w: Core.Dynamic, b: Core.Dynamic) ->
  Result<&1, &1, Err, Leaves>:
  leaf_w(Backward.leaf(Backward.empty(), w), x, b)

# The leaves, with the bias and the data to go.
def loss_of(pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>) ->
  Result<&1, &1, Err, Tape>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{ErrBackward{error}}
    case Done{loss}:
      Done{Tape{tape, 0n, 1n, 2n, loss}}

# The step: the loss before it, and the tape after it.
def cross_entropy(tape: Backward.Tape, logits: Nat, targets: List<&2, U32>) ->
  Result<&1, &1, Err, Tape>:
  loss_of(Backward.cross_entropy(tape, logits, targets))

# The tape with the loss on it.
def logits_of(pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>, targets: List<&2, U32>) ->
  Result<&1, &1, Err, Tape>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{ErrBackward{error}}
    case Done{logits}:
      cross_entropy(tape, logits, targets)

# The loss node, and the tape that holds it.
def linear(tape: Backward.Tape, x: Nat, w: Nat, b: Nat, targets: List<&2, U32>) ->
  Result<&1, &1, Err, Tape>:
  logits_of(Backward.linear(tape, x, w, b), targets)

# The loss of the logits.
def forward(lv: Leaves, targets: List<&2, U32>) -> Result<&1, &1, Err, Tape>:
  match lv:
    case Leaves{tape, w, b, x}:
      linear(tape, x, w, b, targets)

# The logits node, and the loss on it.
def with_leaves(lv: Result<&1, &1, Err, Leaves>, targets: List<&2, U32>) ->
  Result<&1, &1, Err, Tape>:
  match lv:
    case Fail{error}:
      Fail{error}
    case Done{leaves}:
      forward(leaves, targets)

# The leaves of the tape: the weights, the bias and the data.
def built(x: Core.Dynamic, w: Core.Dynamic, b: Core.Dynamic, targets: List<&2, U32>) ->
  Result<&1, &1, Err, Tape>:
  with_leaves(leaves(x, w, b), targets)

# The tape, with the leaves to go.
def flat_of(rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  List.foldl(&2, List<&2, F32>, List<&2, F32>, add_row, rows, Nil{})

# One row's numbers, appended to the numbers so far.
def data_of(rows: List<&2, List<&2, F32>>) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.construct([150, 4], flat_of(rows))

# The weights as bendtorch's own tensor: four rows of three, all zero.
def stepped_sgd(pair: Backward.Tape & Result<&2, &2, Sgd.Error, Unit>, +w: Nat, +b: Nat,
  +x: Nat, +loss: Nat, before: F32) -> Result<&1, &1, Err, (F32 & Tape)>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{ErrSgd{error}}
    case Done{value}:
      Done{(before, Tape{tape, w, b, x, loss})}

# The tape, from the rows of the data.
def sgd(tape: Backward.Tape, +w: Nat, +b: Nat, +x: Nat, +loss: Nat, lr: F32, before: F32) ->
  Result<&1, &1, Err, (F32 & Tape)>:
  stepped_sgd(Sgd.step(tape, lr, [w, b]), w, b, x, loss, before)

# The step, and the tape that holds it.
def stepped(pair: Backward.Tape & Result<&2, &2, Backward.Error, Unit>, +w: Nat, +b: Nat, +x: Nat,
  +loss: Nat, lr: F32, before: F32) -> Result<&1, &1, Err, (F32 & Tape)>:
  (after, result) = pair
  match result:
    case Fail{error}:
      Fail{ErrBackward{error}}
    case Done{value}:
      sgd(after, w, b, x, loss, lr, before)

# The step, from the tape that holds the gradients.
def scalar_pair(pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>, tape: Backward.Tape,
  +w: Nat, +b: Nat, +x: Nat, +loss: Nat, lr: F32) -> Result<&1, &1, Err, (F32 & Tape)>:
  (dynamic, result) = pair
  match result:
    case Fail{error}:
      Fail{ErrCore{Core.error_code(error)}}
    case Done{v}:
      stepped(Backward.backward(tape, loss), w, b, x, loss, lr, v)

# The backward pass, and the step on the leaves.
def loss_pair(pair: Backward.Tape & Result<&2, &1, Backward.Error, Core.Dynamic>,
  +w: Nat, +b: Nat, +x: Nat, +loss: Nat, lr: F32) -> Result<&1, &1, Err, (F32 & Tape)>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{ErrBackward{error}}
    case Done{dynamic}:
      scalar_pair(Core.read_dynamic(dynamic, 0), tape, w, b, x, loss, lr)

# The loss as a number, and the tape to step.
def step_of(t: Tape, lr: F32) -> Result<&1, &1, Err, (F32 & Tape)>:
  match t:
    case Tape{tape, +w, +b, +x, +loss}:
      loss_pair(Backward.value(tape, loss), w, b, x, loss, lr)

# The loss before the step, and the tape to step.
def second(t: Tape, before: F32) -> String:
  again(step_of(t, 0.01), before)

# The second step, and the two losses.
def first(r: Result<&1, &1, Err, (F32 & Tape)>) -> String:
  match r:
    case Fail{e}:
      "the step failed"
    case Done{pair}:
      (before, t1) = pair
      second(t1, before)

# The step again, and the loss before it.
def training(t: Tape) -> String:
  first(step_of(t, 0.01))

# The first step, and the loss before it.
def show_tape(t: Result<&1, &1, Err, Tape>) -> IO(Unit):
  match t:
    case Fail{e}:
      IO.print("the tape failed")
    case Done{tape}:
      IO.print(training(tape))

# The loss before the step and the loss after it.
def weights_of() -> Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.construct([4, 3], zeros(12n))

# The bias as bendtorch's own tensor: three zeros.
def tape_of(x: Core.Dynamic, w: Core.Dynamic, b: Core.Dynamic, targets: List<&2, U32>) ->
  Result<&1, &1, Err, Tape>:
  built(x, w, b, targets)

# The tape, once the leaves are in it.
def with_bias(b: Result<&2, &1, Core.Error, Core.Dynamic>, w: Core.Dynamic, x: Core.Dynamic,
  targets: List<&2, U32>) -> Result<&1, &1, Err, Tape>:
  match b:
    case Fail{error}:
      Fail{ErrCore{Core.error_code(error)}}
    case Done{bias}:
      tape_of(x, w, bias, targets)

# The labels as the numbers the tape wants.
def with_weights(w: Result<&2, &1, Core.Error, Core.Dynamic>, x: Core.Dynamic,
  targets: List<&2, U32>) -> Result<&1, &1, Err, Tape>:
  match w:
    case Fail{error}:
      Fail{ErrCore{Core.error_code(error)}}
    case Done{weights}:
      with_bias(bias_of(), weights, x, targets)

# The tape, with the bias to go.
def with_data(d: Result<&2, &1, Core.Error, Core.Dynamic>, targets: List<&2, U32>) ->
  Result<&1, &1, Err, Tape>:
  match d:
    case Fail{error}:
      Fail{ErrCore{Core.error_code(error)}}
    case Done{x}:
      with_weights(weights_of(), x, targets)

# The tape, with the weights to go.
def tape_of_rows(rows: List<&2, List<&2, F32>>, targets: List<&2, U32>) ->
  Result<&1, &1, Err, Tape>:
  with_data(data_of(rows), targets)

# The tape, with the data to go.
def targets_of(labels: List<&2, Nat>) -> List<&2, U32>:
  List.foldl(&2, Nat, List<&2, U32>, add_target, labels, Nil{})

# One label, appended to the labels so far.
def show_rows(+rows: List<&2, List<&2, F32>>, +labels: List<&2, Nat>) -> IO(Unit):
  show_tape(tape_of_rows(rows, targets_of(labels)))

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
