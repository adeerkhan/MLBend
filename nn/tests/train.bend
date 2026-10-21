# A logistic regression over the iris dataset, fetched from HuggingFace at
# run time and trained with bend-ml's autodiff.
import Base
import bend-ml-tensor@0.1.0.0/main.bend as T
import mlbend-nn@0.1.0.0/main.bend as NN
import mlbend-datasets@0.1.2.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf

def join4(a: String, b: String, c: String, d: String) -> String:
  String.append(String.append(String.append(a, "\n"), String.append(b, "\n")),
    String.append(String.append(c, "\n"), d))

def training(+x: T.Mat<150n, 4n>, +labels: List<&2, Nat>) -> String:
  +m0 = NN.start()
  l0 = NN.loss(m0, x, labels)
  +m1 = NN.step(m0, x, labels, 0.01)
  l1 = NN.loss(m1, x, labels)
  +m2 = NN.step(m1, x, labels, 0.01)
  l2 = NN.loss(m2, x, labels)
  +m3 = NN.step(m2, x, labels, 0.01)
  l3 = NN.loss(m3, x, labels)
  join4(
    String.append("loss before: ", F32.show(l0)),
    String.append("loss after:  ", F32.show(l1)),
    String.append("loss after:  ", F32.show(l2)),
    String.append("loss after:  ", F32.show(l3)))

def show_tensor(t: Result<&1, &1, String, T.Mat<150n, 4n>>, labels: List<&2, Nat>) -> IO(Unit):
  match t:
    case Fail{why}:
      IO.print("the rows are wrong")
    case Done{x}:
      IO.print(training(x, labels))

# The losses of four steps of SGD.
def show_rows(+rows: List<&2, List<&2, F32>>, +labels: List<&2, Nat>) -> IO(Unit):
  show_tensor(NN.tensor_of(NN.Rows{rows, labels}), labels)

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
