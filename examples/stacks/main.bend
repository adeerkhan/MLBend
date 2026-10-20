# The two ML stacks, in one environment.
#
#   bendenv sync
#   bendenv run -- bend examples/stacks/main.bend
#
# bend-ml's tensors over a flat buffer, and bendtorch's tensors with
# their own storage, both from BendHub, both resolved by one lockfile
# and materialized into one environment. The program reads a shape out
# of one and builds a tensor in the other, so a broken environment
# shows up as a missing import rather than as a silent difference.

import Base
import bend-ml-nat-lemmas@0.1.1.0/main.bend as NL
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import bendtorch-core@0.1.0.0/src/tensor/core.bend as Core

# ---- bend-ml: a 2x2 of 2.0 times a 2x2 of 3.0 ----

def rows_of(-r: Nat, -c: Nat, x: TA.ML<r, c>) -> List<&2, F32>:
  match x:
    case TA.ML{m, l}:
      l

def after(x: TA.MMul<2n, 2n, 2n>) -> List<&2, F32>:
  match x:
    case TA.MMul{a, b, c}:
      rows_of(2n, 2n, TA.Mat.to_list(2n, 2n, c))

def bendml(a: TA.Mat<2n, 2n>, b: TA.Mat<2n, 2n>) -> String:
  String.append("bend-ml-tensor-array: ",
    List.show(&2, F32, F32.show, after(TA.Mat.matmul(2n, 2n, 2n, 0n, a, b))))

# ---- bendtorch: a 2x2 of four words ----

def count_pair(pair: Core.Dynamic & U32) -> U32:
  (t2, +count) = pair
  count

def count_of(t: Core.Dynamic) -> U32:
  count_pair(Core.count(t))

def bendtorch(result: Result<&2, &1, Core.Error, Core.Dynamic>) -> String:
  match result:
    case Fail{error}:
      String.append("bendtorch-core: refused with code ", U32.show(Core.error_code(error)))
    case Done{tensor}:
      String.append(String.append("bendtorch-core: built a tensor of ", U32.show(count_of(tensor))), " words")

def main() -> IO(Unit):
  a = TA.Mat.fill(2n, 2n, 2.0)
  b = TA.Mat.fill(2n, 2n, 3.0)
  IO.print(String.append(bendml(a, b),
    String.append(" | ", bendtorch(Core.construct([2, 2], [1.0, 2.0, 3.0, 4.0])))))
