import Base
import ../main.bend as M

# The two ways a metric cannot answer: the two sequences do not have the
# same length, and there is no sample at all. Neither is a crash: the
# answer is a value.

def targets() -> List<&2, F32>:
  [3.0, 0.5, 2.5]

def predictions() -> List<&2, F32>:
  [2.5, 1.0]

def none() -> List<&2, F32>:
  []

def labels() -> List<&2, Nat>:
  [0n, 1n]

def predicted() -> List<&2, Nat>:
  [0n]

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
    Unit <- IO.print(String.append("mse_empty ", show(M.mse(none(), none()))))
    Unit <- IO.print(String.append("accuracy ", show(M.accuracy(labels(), predicted()))))
    return Unit{}

#|mse mismatch 3
#|mse_empty empty
#|accuracy mismatch 2
