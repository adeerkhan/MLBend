# The same linear model, over bendtorch's tape.
#
# bend-ml's autodiff is functional: the backward pass is a function of
# the forward. bendtorch's is a tape: the forward records the nodes, the
# backward walks them, and the step reads the gradients off the leaves.
#
# The two hold different tensors -- bend-ml's are lists, bendtorch's are
# its own storage with a shape in the type -- so the model is built
# twice, from the same numbers, and the two losses are compared.
#
# The nodes of the tape: the weights are leaf 0, the bias leaf 1, the
# data leaf 2, the logits node 3 and the loss node 4.

import Base
import bendtorch-core@0.1.0.0/src/tensor/core.bend as Core
import bendtorch-core@0.1.0.0/src/autograd/tensor_backward.bend as Backward
import bendtorch-core@0.1.0.0/src/autograd/sgd.bend as Sgd

# Why an answer did not come: the tape's error, or the step's.
type Err is Data:
  ErrBackward{error: Backward.Error}
  ErrSgd{error: Sgd.Error}
  ErrCore{code: U32}

# The code of the error the tape failed with.
type Leaves is Type:
  Leaves{tape: Backward.Tape, w: Nat, b: Nat, x: Nat}

# A tape with the loss on it.
type Tape is Type:
  Tape{tape: Backward.Tape, w: Nat, b: Nat, x: Nat, loss: Nat}

# The rows of the data as one flat list of numbers.
def add_row(acc: List<&2, F32>, row: List<&2, F32>) -> List<&2, F32>:
  List.append(&2, F32, acc, row)

# count numbers, all zero.
def add_zero(acc: List<&2, F32>, n: Nat) -> List<&2, F32>:
  List.append(&2, F32, acc, 0.0 <> Nil{})

# The data as bendtorch's own tensor: 150 rows of four.
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

# The loss of a tape, and the tape again: the tape is a Type, so it is
# used once and handed back.
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
def code_of(e: Err) -> U32:
  match e:
    case ErrBackward{error}:
      Backward.error_code(error)
    case ErrSgd{error}:
      U32.from_nat(0n)
    case ErrCore{code}:
      code

# The leaves of the tape and their ids.
def flat_of(rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  List.foldl(&2, List<&2, F32>, List<&2, F32>, add_row, rows, Nil{})

# One row's numbers, appended to the numbers so far.
def data_of(rows: List<&2, List<&2, F32>>) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.construct([150, 4], flat_of(rows))

# The weights as bendtorch's own tensor: four rows of three, all zero.
def read_scalar(pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>, tape: Backward.Tape,
  +w: Nat, +b: Nat, +x: Nat, +loss: Nat) -> Result<&1, &1, Err, (F32 & Tape)>:
  (dynamic, result) = pair
  match result:
    case Fail{error}:
      Fail{ErrCore{Core.error_code(error)}}
    case Done{v}:
      Done{(v, Tape{tape, w, b, x, loss})}

# The step: the loss before it, and the tape after it.
def read_loss(pair: Backward.Tape & Result<&2, &1, Backward.Error, Core.Dynamic>,
  +w: Nat, +b: Nat, +x: Nat, +loss: Nat) -> Result<&1, &1, Err, (F32 & Tape)>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{ErrBackward{error}}
    case Done{dynamic}:
      read_scalar(Core.read_dynamic(dynamic, 0), tape, w, b, x, loss)

# One number of a tensor, as a number.
def loss_of_tape(t: Tape) -> Result<&1, &1, Err, (F32 & Tape)>:
  match t:
    case Tape{tape, +w, +b, +x, +loss}:
      read_loss(Backward.value(tape, loss), w, b, x, loss)

# The loss, as a number, and the tape again.
def stepped_sgd(pair: Backward.Tape & Result<&2, &2, Sgd.Error, Unit>, +w: Nat, +b: Nat,
  +x: Nat, +loss: Nat, before: F32) -> Result<&1, &1, Err, (F32 & Tape)>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{ErrSgd{error}}
    case Done{value}:
      Done{(before, Tape{tape, w, b, x, loss})}

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
def tape_of(x: Core.Dynamic, w: Core.Dynamic, b: Core.Dynamic, targets: List<&2, U32>) ->
  Result<&1, &1, Err, Tape>:
  built(x, w, b, targets)

# The tape, once the leaves are in it.
def weights_of() -> Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.construct([4, 3], zeros(12n))

# The bias as bendtorch's own tensor: three zeros.