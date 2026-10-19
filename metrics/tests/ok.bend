import Base
import ../main.bend as M

# The metrics on a small dataset: the targets and the predictions of a
# regression, and the labels and the predicted labels of a
# classification. The numbers are checked against NumPy and
# scikit-learn by reference/test_metrics.py.

def targets() -> List<&2, F32>:
  [3.0, 0.5, 2.5, 7.0]

def predictions() -> List<&2, F32>:
  [2.5, 1.0, 2.0, 8.0]

def labels() -> List<&2, Nat>:
  [0n, 1n, 1n, 0n]

def predicted() -> List<&2, Nat>:
  [0n, 0n, 1n, 1n]

# a metric, or why it could not answer
def show(r: Result<&1, &1, M.Bad, F32>) -> String:
  match r:
    case Done{v}:
      F32.show(v)
    case Fail{M.Mismatch{expected, got}}:
      String.append("mismatch ", U32.show(U32.from_nat(expected)))
    case Fail{M.Empty{}}:
      "empty"

law main:
  IO(Unit)

def main():
  do IO<Unit>:
    Unit <- IO.print(String.append("mse ", show(M.mse(targets(), predictions()))))
    Unit <- IO.print(String.append("mae ", show(M.mae(targets(), predictions()))))
    Unit <- IO.print(String.append("rmse ", show(M.rmse(targets(), predictions()))))
    Unit <- IO.print(String.append("accuracy ", show(M.accuracy(labels(), predicted()))))
    Unit <- IO.print(String.append("precision_0 ", show(M.precision(labels(), predicted(), 0n))))
    Unit <- IO.print(String.append("recall_0 ", show(M.recall(labels(), predicted(), 0n))))
    Unit <- IO.print(String.append("f1_0 ", show(M.f1(labels(), predicted(), 0n))))
    Unit <- IO.print(String.append("precision_1 ", show(M.precision(labels(), predicted(), 1n))))
    Unit <- IO.print(String.append("recall_1 ", show(M.recall(labels(), predicted(), 1n))))
    Unit <- IO.print(String.append("f1_1 ", show(M.f1(labels(), predicted(), 1n))))
    return Unit{}
