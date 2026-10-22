# The k-means over a fixed dataset: the labels, the sizes and the inertia.
# The numbers are checked against NumPy by reference/test_cluster.py.
#
#| labels: 0 0 0 1 1 1
#| sizes: 3 3
#| inertia: 2.6666665
#| centre 0: 0.33333334 0.33333334
#| centre 1: 10.333334 10.333334
import Base
import mlbend-cluster@0.1.0.0/main.bend as Cluster

# A row of two numbers.
def row(a: F32, b: F32) -> List<&2, F32>:
  Con{a, Con{b, Nil{}}}

# The six rows: two tight groups, the first two rows are the start.
def sample() -> List<&2, List<&2, F32>>:
  Con{row(0.0, 0.0),
    Con{row(0.0, 1.0),
      Con{row(1.0, 0.0),
        Con{row(10.0, 10.0),
          Con{row(10.0, 11.0),
            Con{row(11.0, 10.0), Nil{}}}}}}}

# The model of the test: two clusters, two rounds.
def model() -> Cluster.Model:
  Cluster.fit(2n, 2n, sample())

# The centres of a model, out of it.
def centres_of(m: Cluster.Model) -> List<&2, List<&2, F32>>:
  match m:
    case Cluster.Model{centres, k}:
      centres

# A list of numbers, as text.
def cells_go(+xs: List<&2, Nat>, +acc: String) -> String:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      cells_go(t, String.append(acc, String.append(U32.show(U32.from_nat(h)), " ")))

# A list of numbers, as text.
def cells(+xs: List<&2, Nat>) -> String:
  cells_go(xs, "")

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

# The labels of the rows.
def labels() -> List<&2, Nat>:
  Cluster.assign(centres_of(model()), sample())

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(String.append("labels: ", cells(labels())))
    IO.print(String.append("sizes: ", cells(Cluster.sizes(labels(), 2n))))
    IO.print(String.append("inertia: ", F32.show(Cluster.inertia(centres_of(model()), sample(), labels()))))
    IO.print(String.append("centre 0: ", row_text(Cluster.centre_at(centres_of(model()), 0n))))
    IO.print(String.append("centre 1: ", row_text(Cluster.centre_at(centres_of(model()), 1n))))
