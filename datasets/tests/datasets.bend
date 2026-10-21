# The datasets, fetched from the hub at run time.
#
#| iris: 150 rows, 150
#| wine: 6497 rows, 6497
#| cancer: 683 rows, 683
#| heart: 299 rows, 299
import Base
import mlbend-datasets@0.1.3.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf

def count_of(t: Result<&1, &1, Hf.Err, Datasets.Table>) -> String:
  match t:
    case Fail{e}:
      "failed"
    case Done{v}:
      match v:
        case Datasets.Table{rows, labels}:
          String.append(U32.show(U32.from_nat(List.length(&2, List<&2, F32>, rows))),
            String.append(" rows, ", U32.show(U32.from_nat(List.length(&2, F32, labels)))))

def show_cancer() -> IO(Unit):
  do IO<Unit>:
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.cancer()
    IO.print(String.append("cancer: ", count_of(t)))

def show_heart() -> IO(Unit):
  do IO<Unit>:
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.heart()
    IO.print(String.append("heart: ", count_of(t)))

def show_iris() -> IO(Unit):
  do IO<Unit>:
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.iris()
    IO.print(String.append("iris: ", count_of(t)))

def show_wine() -> IO(Unit):
  do IO<Unit>:
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.wine()
    IO.print(String.append("wine: ", count_of(t)))

def main() -> IO(Unit):
  do IO<Unit>:
    show_iris()
    show_wine()
    show_cancer()
    show_heart()
