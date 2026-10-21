# The iris dataset, fetched from the hub: 150 rows, four features, three
# classes.
import Base
import mlbend-datasets@0.1.2.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf

def at_of(m: Maybe<&2, Nat>) -> Nat:
  match m:
    case Some{v}:
      v
    case None{}:
      0n

def at(+xs: List<&2, Nat>, n: Nat) -> Nat:
  at_of(List.get(&2, Nat, xs, n))

def classes_of(+xs: List<&2, Nat>) -> String:
  String.append(String.append(U32.show(U32.from_nat(at(xs, 0n))), ","),
    String.append(U32.show(U32.from_nat(at(xs, 1n))), ","))

def show(d: Result<&1, &1, Hf.Err, Datasets.Table>) -> String:
  match d:
    case Fail{e}:
      "the dataset failed"
    case Done{v}:
      match v:
        case Datasets.Table{features, +labels}:
          head = String.append("rows: ", U32.show(U32.from_nat(List.length(&2, Nat, labels))))
          tail = String.append(", classes: ", classes_of(labels))
          String.append(head, tail)

def main() -> IO(Unit):
  do IO<Unit>:
    d : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.iris()
    IO.print(show(d))
