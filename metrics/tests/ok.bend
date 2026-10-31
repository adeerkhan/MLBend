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

# the confusion matrix, one line per cell so the golden names each count
def cell_lines(xs: List<&2, Nat>, +i: Nat, +j: Nat) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append("confusion_", String.append(U32.show(U32.from_nat(i)), String.append("_", String.append(U32.show(U32.from_nat(j)), String.append(" ", String.append(U32.show(U32.from_nat(h)), String.append("\n", cell_lines(t, i, Nat.add(j, 1n)))))))))

def matrix_text(xs: List<&2, List<&2, Nat>>, +i: Nat) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(cell_lines(h, i, 0n), matrix_text(t, Nat.add(i, 1n)))

def confusion_text(r: Result<&1, &1, M.Bad, List<&2, List<&2, Nat>>>) -> String:
  match r:
    case Done{v}:
      matrix_text(v, 0n)
    case Fail{e}:
      "confusion_fail 0\n"

def probabilities() -> List<&2, List<&2, F32>>:
  [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.6, 0.3, 0.1]]

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

    Unit <- IO.print(String.append("r2 ", show(M.r2(targets(), predictions()))))
    Unit <- IO.print(String.append("log_loss ", show(M.log_loss(labels(), probabilities()))))
    Unit <- IO.print(String.append("macro_precision ", show(M.macro_precision(labels(), predicted(), 2n))))
    Unit <- IO.print(String.append("macro_recall ", show(M.macro_recall(labels(), predicted(), 2n))))
    Unit <- IO.print(String.append("macro_f1 ", show(M.macro_f1(labels(), predicted(), 2n))))
    Unit <- IO.print(String.append("weighted_precision ", show(M.weighted_precision(labels(), predicted(), 2n))))
    Unit <- IO.print(String.append("weighted_recall ", show(M.weighted_recall(labels(), predicted(), 2n))))
    Unit <- IO.print(String.append("weighted_f1 ", show(M.weighted_f1(labels(), predicted(), 2n))))
    Unit <- IO.print(confusion_text(M.confusion(labels(), predicted(), 2n)))
    return Unit{}

#|mse 0.4375
#|mae 0.625
#|rmse 0.6614378
#|accuracy 0.5
#|precision_0 0.5
#|recall_0 0.5
#|f1_0 0.5
#|precision_1 0.5
#|recall_1 0.5
#|f1_1 0.5
#|r2 0.92134833
#|log_loss 0.6750205
#|macro_precision 0.5
#|macro_recall 0.5
#|macro_f1 0.5
#|weighted_precision 0.5
#|weighted_recall 0.5
#|weighted_f1 0.5
#|confusion_0_0 1
#|confusion_0_1 1
#|confusion_1_0 1
#|confusion_1_1 1
