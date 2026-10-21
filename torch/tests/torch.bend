# The same logistic regression, over bendtorch's tape.
import Base
import bendtorch-core@0.1.0.0/src/tensor/core.bend as Core
import bendtorch-core@0.1.0.0/src/autograd/tensor_backward.bend as Backward
import mlbend-torch@0.1.0.0/main.bend as Torch
import mlbend-datasets@0.1.2.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf

def add_target(acc: List<&2, U32>, n: Nat) -> List<&2, U32>:
  List.append(&2, U32, acc, U32.from_nat(n) <> Nil{})

# The tape, from the data and the weights.
def two(before: F32, mid: F32) -> String:
  String.append(String.append("loss before: ", F32.show(before)),
    String.append("\nloss after:  ", F32.show(mid)))

def again(r: Result<&1, &1, Torch.Err, (F32 & Torch.Tape)>, before: F32) -> String:
  match r:
    case Fail{e}:
      "the second step failed"
    case Done{pair}:
      (mid, t2) = pair
      two(before, mid)

# The two losses, as one line each.
def torch_error(error: Core.Error) -> Torch.Err:
  Torch.ErrBackward{Backward.TensorError{error}}

# The labels as the numbers the tape wants.
def with_bias(b: Result<&2, &1, Core.Error, Core.Dynamic>, w: Core.Dynamic, x: Core.Dynamic,
  targets: List<&2, U32>) -> Result<&1, &1, Torch.Err, Torch.Tape>:
  match b:
    case Fail{error}:
      Fail{torch_error(error)}
    case Done{bias}:
      Torch.tape_of(x, w, bias, targets)

def with_weights(w: Result<&2, &1, Core.Error, Core.Dynamic>, x: Core.Dynamic,
  targets: List<&2, U32>) -> Result<&1, &1, Torch.Err, Torch.Tape>:
  match w:
    case Fail{error}:
      Fail{torch_error(error)}
    case Done{weights}:
      with_bias(Core.construct([3], Torch.zeros(3n)), weights, x, targets)

# The tape, with everything in.
def with_data(d: Result<&2, &1, Core.Error, Core.Dynamic>, targets: List<&2, U32>) ->
  Result<&1, &1, Torch.Err, Torch.Tape>:
  match d:
    case Fail{error}:
      Fail{torch_error(error)}
    case Done{x}:
      with_weights(Core.construct([4, 3], Torch.zeros(12n)), x, targets)

# The tape, with the bias to go.
def build(rows: List<&2, List<&2, F32>>, targets: List<&2, U32>) ->
  Result<&1, &1, Torch.Err, Torch.Tape>:
  with_data(Core.construct([150, 4], Torch.flat_of(rows)), targets)

# The tape, with the weights to go.
def second(t: Torch.Tape, before: F32) -> String:
  again(Torch.step_of(t, 0.01), before)

# The second step, and the two losses.
def first(r: Result<&1, &1, Torch.Err, (F32 & Torch.Tape)>) -> String:
  match r:
    case Fail{e}:
      "the step failed"
    case Done{pair}:
      (before, t1) = pair
      second(t1, before)

# The step again, and the loss before it.
def losses(t: Torch.Tape) -> String:
  first(Torch.step_of(t, 0.01))

# The first step, and the loss before it.
def show_tape(t: Result<&1, &1, Torch.Err, Torch.Tape>) -> IO(Unit):
  match t:
    case Fail{e}:
      IO.print(String.append("the tape failed: ", U32.show(Torch.code_of(e))))
    case Done{tape}:
      IO.print(losses(tape))

# The loss before the step, and the tape to step again.
def targets_of(labels: List<&2, Nat>) -> List<&2, U32>:
  List.foldl(&2, Nat, List<&2, U32>, add_target, labels, Nil{})

# One label, appended to the labels so far.
def show_rows(+rows: List<&2, List<&2, F32>>, +labels: List<&2, Nat>) -> IO(Unit):
  show_tape(build(rows, targets_of(labels)))

# A Core error, as the package's error.
def show_table(t: Datasets.Table) -> IO(Unit):
  match t:
    case Datasets.Table{rows, labels}:
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
