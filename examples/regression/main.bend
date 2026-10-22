# Linear regression over the wine dataset, on bendtorch's tape.
#
#   bendenv run -- bend examples/regression/main.bend
#
# The wine dataset's label is the quality, a number from 3 to 8, so this
# is regression rather than classification. The features and the target
# are standardised, the model is one linear layer, and the loss is the
# mean squared error:
#
#   prediction = x · Wᵀ + b        x: 6497 x 11,  W: 1 x 11
#   loss       = mse(prediction, y)
#
# Nothing here re-implements bendtorch: the layer is `Backward.linear`,
# the loss is `Backward.mse_loss`, the backward pass is
# `Backward.backward` and the step is `Sgd.step`. What this program adds
# is the model, the data and the loop.
#
# The weights start at zero; the gradient of the mean squared error at
# zero is not zero, so every step moves them.
import Base
import bendtorch-core@0.1.0.0/src/tensor/core.bend as Core
import bendtorch-core@0.1.0.0/src/autograd/tensor_backward.bend as Backward
import bendtorch-core@0.1.0.0/src/autograd/sgd.bend as Sgd
import mlbend-datasets@0.1.4.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf

# Why a step did not come: the code of the first failure.

type Err is Data:
  Err{code: U32}

# The two losses of the run: before the training and after it.
type Losses is Data:
  Losses{before: F32, after: F32}

# One row's numbers, appended to the numbers so far.
def add_row(acc: List<&2, F32>, row: List<&2, F32>) -> List<&2, F32>:
  List.append(&2, F32, acc, row)

# One number, appended to the numbers so far.
def add_value(acc: List<&2, F32>, n: F32) -> List<&2, F32>:
  List.append(&2, F32, acc, Con{n, Nil{}})

# One zero, appended to the numbers so far.
def add_zero(acc: List<&2, F32>, n: Nat) -> List<&2, F32>:
  List.append(&2, F32, acc, Con{0.0, Nil{}})

# count numbers, all zero.
def zeros(count: Nat) -> List<&2, F32>:
  List.foldl(&2, Nat, List<&2, F32>, add_zero, List.range(count), Nil{})

# The rows of the data as one flat list of numbers.
def flat_of(rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  List.foldl(&2, List<&2, F32>, List<&2, F32>, add_row, rows, Nil{})

# Two rows, cell by cell, the first minus the second.
def zip_sub(+a: List<&2, F32>, b: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      match b:
        case Con{hb, tb}:
          Con{F32.sub(ha, hb), zip_sub(ta, tb)}
        case Nil{}:
          Nil{}

# Two rows, cell by cell, the first divided by the second.
def zip_div(a: List<&2, F32>, b: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      match b:
        case Con{hb, tb}:
          Con{(ha / F32.max(hb, 0.0001) : F32), zip_div(ta, tb)}
        case Nil{}:
          Nil{}

# Two rows, added cell by cell.
def zip_add(a: List<&2, F32>, b: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      match b:
        case Con{hb, tb}:
          Con{F32.add(ha, hb), zip_add(ta, tb)}
        case Nil{}:
          Nil{}

# The sum of every column, over the rows.
def column_sums_go(rows: List<&2, List<&2, F32>>, +acc: List<&2, F32>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      column_sums_go(t, zip_add(acc, h))

# The sum of every column, over the rows.
def column_sums(rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      column_sums_go(t, h)

# One row and a number, cell by cell.
def zip_scale(+a: List<&2, F32>, +k: F32) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      Con{F32.mul(ha, k), zip_scale(ta, k)}

# The cells of a row, squared.
def squares(+a: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{+ha, ta}:
      Con{F32.mul(ha, ha), squares(ta)}

# The sum of the squares of the deviations, row by row.
def deviation_sums_go(rows: List<&2, List<&2, F32>>, +means: List<&2, F32>,
  +acc: List<&2, F32>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      deviation_sums_go(t, means, zip_add(acc, squares(zip_sub(h, means))))

# The sum of the squares of the deviations of every column.
def deviation_sums(rows: List<&2, List<&2, F32>>, +means: List<&2, F32>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      deviation_sums_go(t, means, squares(zip_sub(h, means)))

# The square root of every cell of a row.
def roots(a: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      Con{F32.sqrt(ha), roots(ta)}

# The mean of every column.
def means_of(+rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      zip_scale(column_sums(rows),
        (1.0 / F32.from_nat(List.length(&2, List<&2, F32>, rows)) : F32))

# The standard deviation of every column.
def stds_of(+rows: List<&2, List<&2, F32>>, means: List<&2, F32>) -> List<&2, F32>:
  roots(zip_scale(deviation_sums(rows, means),
    (1.0 / F32.from_nat(List.length(&2, List<&2, F32>, rows)) : F32)))

# One row, centred and scaled.
def scale_row(row: List<&2, F32>, means: List<&2, F32>, stds: List<&2, F32>) -> List<&2, F32>:
  zip_div(zip_sub(row, means), stds)

# The rows, one at a time.
def standardise_go(+rows: List<&2, List<&2, F32>>, +means: List<&2, F32>, +stds: List<&2, F32>,
  +acc: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      standardise_go(t, means, stds,
        List.append(&2, List<&2, F32>, acc, Con{scale_row(h, means, stds), Nil{}}))

# The rows with every column centred and scaled.
def standardise(+rows: List<&2, List<&2, F32>>, means: List<&2, F32>, stds: List<&2, F32>) ->
  List<&2, List<&2, F32>>:
  standardise_go(rows, means, stds, Nil{})

# The sum of a list of numbers.
def sum_of(+xs: List<&2, F32>, +acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      sum_of(t, F32.add(acc, h))

# The mean of a list of numbers.
def mean_of(+xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{h, t}:
      F32.div(sum_of(xs, 0.0), F32.from_nat(List.length(&2, F32, xs)))

# The sum of the squared deviations of a list of numbers.
def deviation_of(+xs: List<&2, F32>, +mean: F32, +acc: F32) -> F32:
  match xs:
    case Nil{}:
      acc
    case Con{+h, t}:
      deviation_of(t, mean, F32.add(acc, F32.mul(F32.sub(h, mean), F32.sub(h, mean))))

# The standard deviation of a list of numbers.
def std_of(+xs: List<&2, F32>, mean: F32) -> F32:
  F32.sqrt(F32.div(deviation_of(xs, mean, 0.0),
    F32.from_nat(List.length(&2, F32, xs))))

# One target, centred and scaled.
def scale_label(label: F32, mean: F32, std: F32) -> F32:
  F32.div(F32.sub(label, mean), F32.max(std, 0.0001))

# The targets, one at a time.
def scale_labels_go(+labels: List<&2, F32>, +mean: F32, +std: F32,
  +acc: List<&2, F32>) -> List<&2, F32>:
  match labels:
    case Nil{}:
      acc
    case Con{h, t}:
      scale_labels_go(t, mean, std, List.append(&2, F32, acc, Con{scale_label(h, mean, std), Nil{}}))

# The targets, centred and scaled.
def scale_labels(+labels: List<&2, F32>, mean: F32, std: F32) -> List<&2, F32>:
  scale_labels_go(labels, mean, std, Nil{})

# The number a tensor holds, when it holds one.
def read_of(pair: Core.Dynamic & Result<&2, &2, Core.Error, F32>) -> Result<&1, &1, Err, F32>:
  (dynamic, result) = pair
  match result:
    case Fail{error}:
      Fail{Err{Core.error_code(error)}}
    case Done{value}:
      Done{value}

# The prediction of every row, and the id of the node.
def prediction(tape: Backward.Tape) -> Backward.Tape & Result<&2, &2, Backward.Error, Nat>:
  Backward.linear(tape, 2n, 0n, 1n)

# The loss of the prediction against the target, and the id of the node.
def loss_node(pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>) ->
  Backward.Tape & Result<&2, &2, Backward.Error, Nat>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      (tape, Fail{error})
    case Done{id}:
      Backward.mse_loss(tape, id, 3n)

# The step, and the tape that holds the new weights.
def stepped_sgd(pair: Backward.Tape & Result<&2, &2, Sgd.Error, Unit>) ->
  Result<&1, &1, Err, Backward.Tape>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{Err{Sgd.error_code(error)}}
    case Done{unit}:
      Done{Backward.zero_grad(tape)}

# The backward pass, and the step on the weights and the bias.
def stepped_backward(pair: Backward.Tape & Result<&2, &2, Backward.Error, Unit>,
  lr: F32) -> Result<&1, &1, Err, Backward.Tape>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{Err{Backward.error_code(error)}}
    case Done{unit}:
      stepped_sgd(Sgd.step(tape, lr, [0n, 1n]))

# The value of the loss, and the backward pass and the step.
def stepped_read(r: Result<&1, &1, Err, F32>, tape: Backward.Tape, id: Nat,
  lr: F32) -> Result<&1, &1, Err, Backward.Tape>:
  match r:
    case Fail{error}:
      Fail{error}
    case Done{value}:
      stepped_backward(Backward.backward(tape, id), lr)

# The loss's value, the backward pass and the step: the tape follows.
def stepped_value(pair: Backward.Tape & Result<&2, &1, Backward.Error, Core.Dynamic>, id: Nat,
  lr: F32) -> Result<&1, &1, Err, Backward.Tape>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{Err{Backward.error_code(error)}}
    case Done{dynamic}:
      stepped_read(read_of(Core.read_dynamic(dynamic, 0)), tape, id, lr)

# The loss node, the backward pass and the step: the tape follows.
def stepped(pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>) ->
  Result<&1, &1, Err, Backward.Tape>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{Err{Backward.error_code(error)}}
    case Done{+id}:
      stepped_value(Backward.value(tape, id), id, 0.3)

# One round: the prediction, the loss, the backward pass and the step.
def step(tape: Backward.Tape) -> Result<&1, &1, Err, Backward.Tape>:
  stepped(loss_node(prediction(tape)))

# One round, from the result of the round before.
def advance(pair: Result<&1, &1, Err, Backward.Tape>, u: Nat) ->
  Result<&1, &1, Err, Backward.Tape>:
  match pair:
    case Fail{error}:
      Fail{error}
    case Done{tape}:
      step(tape)

# The training: 1000 rounds at a learning rate of 0.3.
def training(tape: Backward.Tape) -> Result<&1, &1, Err, Backward.Tape>:
  List.foldl(&2, Nat, Result<&1, &1, Err, Backward.Tape>, advance, List.range(1000n),
    Done{tape})

# The value of the loss, from the tape's read: the tape comes back too.
def loss_read(pair: Backward.Tape & Result<&2, &1, Backward.Error, Core.Dynamic>) ->
  Backward.Tape & Result<&1, &1, Err, F32>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      (tape, Fail{Err{Backward.error_code(error)}})
    case Done{dynamic}:
      (tape, read_of(Core.read_dynamic(dynamic, 0)))

# The loss of the tape as it is now, and the tape with its nodes.
def forward_loss(pair: Backward.Tape & Result<&2, &2, Backward.Error, Nat>) ->
  Backward.Tape & Result<&1, &1, Err, F32>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      (tape, Fail{Err{Backward.error_code(error)}})
    case Done{id}:
      loss_read(Backward.value(tape, id))

# The two losses, from the loss after the training.
def from_after(before: F32, pair: Backward.Tape & Result<&1, &1, Err, F32>) ->
  Result<&1, &1, Err, Losses>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{error}
    case Done{after}:
      Done{Losses{before, after}}

# The loss after the training, from the loss before it.
def run_after(before: F32, pair: Result<&1, &1, Err, Backward.Tape>) ->
  Result<&1, &1, Err, Losses>:
  match pair:
    case Fail{error}:
      Fail{error}
    case Done{tape}:
      from_after(before, forward_loss(loss_node(prediction(tape))))

# The training, from the tape and its loss before.
def run_before(pair: Backward.Tape & Result<&1, &1, Err, F32>) ->
  Result<&1, &1, Err, Losses>:
  (tape, result) = pair
  match result:
    case Fail{error}:
      Fail{error}
    case Done{before}:
      run_after(before, training(tape))

# The two losses of one tape: the loss before the training and after it.
def run_on(tape: Backward.Tape) -> Result<&1, &1, Err, Losses>:
  run_before(forward_loss(loss_node(prediction(tape))))

# The tape of a pair: the tape and the id of the node.
def tape_of(pair: Backward.Tape & Nat) -> Backward.Tape:
  (tape, id) = pair
  tape

# The leaves of the tape: the weights (0), the bias (1), the data (2)
# and the target (3).
def leaves(w: Core.Dynamic, b: Core.Dynamic, x: Core.Dynamic, y: Core.Dynamic) -> Backward.Tape:
  tape_of(Backward.constant(tape_of(Backward.constant(
    tape_of(Backward.leaf(tape_of(Backward.leaf(Backward.empty(), w)), b)), x)), y))

# The tape, from the four tensors.
def built(w: Core.Dynamic, b: Core.Dynamic, x: Core.Dynamic, y: Core.Dynamic) ->
  Result<&1, &1, Err, Losses>:
  run_on(leaves(w, b, x, y))

# The tape, with the target to go.
def with_target(y: Result<&2, &1, Core.Error, Core.Dynamic>, x: Core.Dynamic,
  b: Core.Dynamic, w: Core.Dynamic) -> Result<&1, &1, Err, Losses>:
  match y:
    case Fail{error}:
      Fail{Err{Core.error_code(error)}}
    case Done{target}:
      built(w, b, x, target)

# The tape, with the data to go.
def with_data(x: Result<&2, &1, Core.Error, Core.Dynamic>, b: Core.Dynamic,
  w: Core.Dynamic, y: Result<&2, &1, Core.Error, Core.Dynamic>) -> Result<&1, &1, Err, Losses>:
  match x:
    case Fail{error}:
      Fail{Err{Core.error_code(error)}}
    case Done{data}:
      with_target(y, data, b, w)

# The tape, with the weights to go.
def with_weight(w: Result<&2, &1, Core.Error, Core.Dynamic>, bias: Core.Dynamic,
  x: Result<&2, &1, Core.Error, Core.Dynamic>, y: Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Result<&1, &1, Err, Losses>:
  match w:
    case Fail{error}:
      Fail{Err{Core.error_code(error)}}
    case Done{weights}:
      with_data(x, bias, weights, y)

# The tape, with the bias to go.
def with_bias(b: Result<&2, &1, Core.Error, Core.Dynamic>, w: Result<&2, &1, Core.Error, Core.Dynamic>,
  x: Result<&2, &1, Core.Error, Core.Dynamic>, y: Result<&2, &1, Core.Error, Core.Dynamic>) ->
  Result<&1, &1, Err, Losses>:
  match b:
    case Fail{error}:
      Fail{Err{Core.error_code(error)}}
    case Done{bias}:
      with_weight(w, bias, x, y)

# The weights as bendtorch's own tensor: one row of eleven, all zero.
def weight_tensor() -> Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.construct([1, 11], zeros(11n))

# The bias as bendtorch's own tensor: one number, zero.
def bias_tensor() -> Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.construct([1], zeros(1n))

# The data as bendtorch's own tensor: the standardised rows.
def data_tensor(+rows: List<&2, List<&2, F32>>) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.construct([6497, 11], flat_of(rows))

# The target as bendtorch's own tensor: one number a row.
def target_tensor(+labels: List<&2, F32>) -> Result<&2, &1, Core.Error, Core.Dynamic>:
  Core.construct([6497, 1], List.foldl(&2, F32, List<&2, F32>, add_value, labels, Nil{}))

# The two losses, as one line each.
def two(l: Losses) -> String:
  match l:
    case Losses{before, after}:
      String.append(String.append("mse before: ", F32.show(before)),
        String.append("\nmse after:  ", F32.show(after)))

# Why the training did not come.
def show_error(error: Err) -> IO(Unit):
  match error:
    case Err{code}:
      IO.print(String.append("the training failed: ", U32.show(code)))

# The two losses, or why there are none.
def show_losses(r: Result<&1, &1, Err, Losses>) -> IO(Unit):
  match r:
    case Fail{error}:
      show_error(error)
    case Done{losses}:
      IO.print(two(losses))

# The run: the rows and the targets, standardised, through the tape.
def show_rows(+rows: List<&2, List<&2, F32>>, +labels: List<&2, F32>) -> IO(Unit):
  +means = means_of(rows)
  +stds = stds_of(rows, means)
  +prepared = standardise(rows, means, stds)
  +label_mean = mean_of(labels)
  +label_std = std_of(labels, label_mean)
  +targets = scale_labels(labels, label_mean, label_std)
  show_losses(with_bias(bias_tensor(), weight_tensor(), data_tensor(prepared), target_tensor(targets)))

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
    d : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.wine()
    show(d)
