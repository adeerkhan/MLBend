# The transformers of scikit-learn, over bend-ml's matrix: the numbers
# are checked against NumPy and scikit-learn by reference/test_preprocess.py.
#
#| standardize: -1.2247448 -1.1355501 0 -0.1622215 1.2247448 1.2977713
#| min_max: 0 0 0.5 0.4 1 1
#| l1: 0.33333334 0.6666667 0.42857146 0.5714286 0.4166667 0.5833334
#| l2: 0.4472136 0.8944272 0.6 0.8 0.5812382 0.81373346
#| one_hot: 1 0 0 0 0 1 0 1 0
#| one_hot rows: 3
import Base
import bend-ml-tensor@0.1.0.0/main.bend as T
import mlbend-preprocess@0.1.0.0/main.bend as P

# A row of two numbers.


# The cells of a row, as text.
def row_text_go(+xs: List<&2, F32>, +acc: String) -> String:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      row_text_go(t, String.append(acc, String.append(F32.show(h), " ")))

# The cells of a row, as text.
def row_text(+xs: List<&2, F32>) -> String:
  row_text_go(xs, "")

# The cells of a matrix, one after the other, as text.
def cells_go(+rows: List<&2, List<&2, F32>>, +acc: String) -> String:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      cells_go(t, String.append(acc, row_text(h)))

# The cells of a matrix, one after the other, as text.
def cells(+rows: List<&2, List<&2, F32>>) -> String:
  cells_go(rows, "")

# The number of rows of the one-hot matrix of three labels.
def count_rows() -> Nat:
  P.one_hot_count_of(Con{0n, Con{2n, Con{1n, Nil{}}}})

# The one-hot matrix of three labels, when there is one.
def one_hot_of_(m: Maybe<&2, T.Mat<3n, 3n>>) -> T.Mat<3n, 3n>:
  match m:
    case Some{v}:
      v
    case None{}:
      T.Mat.fill(3n, 3n, 0.0)

# The one-hot matrix of three labels.
def one_hot_of() -> T.Mat<3n, 3n>:
  one_hot_of_(P.one_hot(3n, 3n, Con{0n, Con{2n, Con{1n, Nil{}}}}))
def row(a: F32, b: F32) -> List<&2, F32>:
  Con{a, Con{b, Nil{}}}

# The matrix of the test: three rows of two numbers.
def sample() -> T.Mat<3n, 2n>:
  T.Mat{Con{row(1.0, 2.0), Con{row(3.0, 4.0), Con{row(5.0, 7.0), Nil{}}}}}

# The rows of a matrix, as text.
def show(r: Nat, c: Nat, m: T.Mat<r, c>) -> String:
  cells(T.Mat.to_rows(r, c, m))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(String.append("standardize: ", show(3n, 2n, P.standardize(3n, 2n, sample()))))
    IO.print(String.append("min_max: ", show(3n, 2n, P.min_max(3n, 2n, sample()))))
    IO.print(String.append("l1: ", show(3n, 2n, P.l1_normalize(3n, 2n, sample()))))
    IO.print(String.append("l2: ", show(3n, 2n, P.l2_normalize(3n, 2n, sample()))))
    IO.print(String.append("one_hot: ", show(3n, 3n, one_hot_of())))
    IO.print(String.append("one_hot rows: ", U32.show(U32.from_nat(count_rows()))))