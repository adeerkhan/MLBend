import Base
import ../main.bend as R
import mlbend-metrics@0.1.0.0/main.bend as M

#|matrix:
#|1 1 0 0
#|1 1 0 0
#|1 0 2 0
#|0 0 0 0
#|class 0 precision 0.33333334 recall 0.5 f1 0.4 support 2
#|class 1 precision 0.5 recall 0.5 f1 0.5 support 2
#|class 2 precision 1 recall 0.6666667 f1 0.8 support 3
#|class 3 precision 0 recall 0 f1 0 support 0
#|macro precision 0.45833334 recall 0.4166667 f1 0.425
#|weighted precision 0.6666667 recall 0.5714286 f1 0.59999996
#|accuracy 0.5714286

# The report of a small classification: seven samples, three classes
# present and one absent, so the zero-support class and the
# zero-division convention are exercised. The numbers are checked
# against scikit-learn's confusion_matrix and classification_report by
# reference/test_report.py.

def labels() -> List<&2, Nat>:
  [0n, 0n, 1n, 1n, 2n, 2n, 2n]

def predicted() -> List<&2, Nat>:
  [0n, 1n, 1n, 0n, 2n, 2n, 0n]

# A row of the matrix, printed.
def show_row(row: List<&2, Nat>) -> String:
  match row:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(U32.show(U32.from_nat(h)), String.append(" ", show_row(t)))

# The matrix, printed.
def show_matrix.go(m: List<&2, List<&2, Nat>>, acc: String) -> String:
  match m:
    case Nil{}:
      acc
    case Con{h, t}:
      show_matrix.go(t, String.append(String.append(acc, show_row(h)), "\n"))

def show_matrix(m: List<&2, List<&2, Nat>>) -> String:
  show_matrix.go(m, "")

# One class of the report, printed.
def class_text(c: R.Cls, +i: Nat) -> String:
  match c:
    case R.Cls{p, r, f, support}:
      String.append("class ", String.append(U32.show(U32.from_nat(i)),
        String.append(" precision ", String.append(F32.show(p),
        String.append(" recall ", String.append(F32.show(r),
        String.append(" f1 ", String.append(F32.show(f),
        String.append(" support ", U32.show(U32.from_nat(support)))))))))))

# Every class of the report, printed.
def classes_text.go(classes: List<&2, R.Cls>, +i: Nat, acc: String) -> String:
  match classes:
    case Nil{}:
      acc
    case Con{c, t}:
      classes_text.go(t, (1n + i : Nat),
        String.append(String.append(acc, class_text(c, i)), "\n"))

# One average, printed.
def avg_text(name: String, c: R.Cls) -> String:
  match c:
    case R.Cls{p, r, f, support}:
      String.append(name, String.append(" precision ", String.append(F32.show(p),
        String.append(" recall ", String.append(F32.show(r),
        String.append(" f1 ", String.append(F32.show(f), "")))))))

# The whole report, printed, or why there is none.
def show(r: Result<&1, &1, M.Bad, R.Report>) -> String:
  match r:
    case Fail{M.Mismatch{expected, got}}:
      "the report failed: the lengths do not agree"
    case Fail{M.Empty{}}:
      "the report failed: no sample"
    case Done{rep}:
      match rep:
        case R.Report{classes, macro_avg, weighted_avg, accuracy}:
          String.append("matrix:\n",
            String.append(show_matrix(R.matrix(labels(), predicted(), 4n)),
            String.append(classes_text.go(classes, 0n, ""),
            String.append(avg_text("macro", macro_avg),
            String.append("\n", String.append(avg_text("weighted", weighted_avg),
            String.append("\n", String.append("accuracy ", F32.show(accuracy)))))))))

law main:
  IO(Unit)

def main():
  do IO<Unit>:
    Unit <- IO.print(show(R.report(labels(), predicted(), 4n)))
    return Unit{}
