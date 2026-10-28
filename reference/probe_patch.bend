# The token staging of the ViT denoiser (issue #16, milestone M1): an 8x8
# digit becomes 16 tokens of 2x2 pixels, in token order. The probe prints
# the staged list for a fixed 64-value input, so the NumPy side can check
# the permutation exactly (`reference/digits_vit_probe.py` uses the same
# one); the round trip (unpatch after patch) is checked here by printing
# the recovered first element and length.
import Base
import mlbend-generative@0.1.1.0/main.bend as G

def at_go(xs: List<&2, F32>, +k: Nat, +i: Nat) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{h, t}:
      Bool.pick(F32, Nat.is_eq(i, k), h, at_go(t, k, (1n + i : Nat)))

def at(xs: List<&2, F32>, k: Nat) -> F32:
  at_go(xs, k, 0n)

def show_go(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_go(t)))

# token k covers rows 2r, 2r+1 and columns 2c, 2c+1 (r = k / 4, c = k mod 4)
def tok_go(+x: List<&2, F32>, left: Nat, +k: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      +i = (Nat.add(Nat.mul(16n, Nat.div(k, 4n)), Nat.mul(2n, Nat.mod(k, 4n))) : Nat)
      tok_go(x, rest, (1n + k : Nat),
        List.append(&2, F32, acc,
          Con{at(x, i), Con{at(x, (1n + i : Nat)),
            Con{at(x, (8n + i : Nat)), Con{at(x, (9n + i : Nat)), Nil{}}}}}))
    case _:
      acc

def patch(x: List<&2, F32>) -> List<&2, F32>:
  tok_go(x, 16n, 0n, Nil{})

# the inverse: pixel i sits in token k = 4*(i/16) + (i mod 8)/2, at
# offset 2*((i/8) mod 2) + (i mod 2); walking pixels keeps pixel order
def unpatch_go(+t: List<&2, F32>, left: Nat, +i: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      +k = (Nat.add(Nat.mul(4n, Nat.div(Nat.div(i, 8n), 2n)),
        Nat.div(Nat.mod(i, 8n), 2n)) : Nat)
      +d = (Nat.add(Nat.mul(2n, Nat.mod(Nat.div(i, 8n), 2n)), Nat.mod(i, 2n)) : Nat)
      unpatch_go(t, rest, (1n + i : Nat),
        List.append(&2, F32, acc, Con{at(t, Nat.add(Nat.mul(4n, k), d)), Nil{}}))
    case _:
      acc

def unpatch(t: List<&2, F32>) -> List<&2, F32>:
  unpatch_go(t, 64n, 0n, Nil{})

def main() -> IO(Unit):
  +x = G.noise_of(7n, 64n)
  +p = patch(x)
  +u = unpatch(p)
  do IO<Unit>:
    IO.print(String.append("patch: ", show_go(p)))
    IO.print(String.append("len: ", Nat.show(List.length(&2, F32, p))))
    IO.print(String.append("u0 ", F32.show(at(u, 0n))))
    IO.print(String.append("u63 ", F32.show(at(u, 63n))))
    IO.print(String.append("x0 ", F32.show(at(x, 0n))))
    IO.print(String.append("x63 ", F32.show(at(x, 63n))))
    return Unit{}
