# The iris dataset, fetched from the hub: 150 rows, four features, three
# classes.
import Base
import mlbend-datasets@0.1.5.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf


def at_of(m: Maybe<&2, F32>) -> F32:
  match m:
    case Some{v}:
      v
    case None{}:
      0.0

def at(+xs: List<&2, F32>, n: Nat) -> F32:
  at_of(List.get(&2, F32, xs, n))

def classes_of(+xs: List<&2, F32>) -> String:
  String.append(String.append(F32.show(at(xs, 0n)), ","),
    String.append(F32.show(at(xs, 1n)), ","))

def show(d: Result<&1, &1, Hf.Err, Datasets.Table>) -> String:
  match d:
    case Fail{e}:
      "the dataset failed"
    case Done{v}:
      match v:
        case Datasets.Table{features, +labels}:
          head = String.append("rows: ", U32.show(U32.from_nat(List.length(&2, F32, labels))))
          tail = String.append(", classes: ", classes_of(labels))
          String.append(head, tail)

def main() -> IO(Unit):
  do IO<Unit>:
    d : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.iris()
    IO.print(show(d))