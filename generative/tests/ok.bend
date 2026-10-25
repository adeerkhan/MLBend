import Base
import ../main.bend as G

# The generative skeleton on a few counters: the uniforms and the
# Gaussians, the cosine schedule and its betas, the time embedding, and
# the round trip of mix32's inverse. The numbers are checked against the
# NumPy twin by reference/test_generative.py.

#|u0 0.79354215
#|u1 0.047255337
#|g0 0.6503142
#|g1 0.19896899
#|g100 0.7640456
#|alphas 0.95780456 0.8470122 0.6842266 0.49384364 0.3043949 0.14427207 0.037471935 1.9109826e-15
#|beta1 0.04219544
#|beta8 0.99
#|temb 0.000375 0.99999994 0.003749991 0.99999297 0.037491214 0.99929696 0.36627254 0.9305076
#|roundtrip 64

def show_f(tag: String, +v: F32) -> String:
  String.append(tag, String.append(" ", F32.show(v)))

def show_n(tag: String, +v: U32) -> String:
  String.append(tag, String.append(" ", U32.show(v)))

def show_list_go(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_list_go(t)))

def show_list(tag: String, xs: List<&2, F32>) -> String:
  String.append(tag, String.append(" ", String.append(show_list_go(xs), "")))

def roundtrip_go(+left: Nat, +acc: U32) -> U32:
  match left:
    case 0n:
      acc
    case 1n+rest:
      roundtrip_go(rest,
        U32.add(acc, U32.from_nat(Bool.pick(Nat,
          U32.is_eq(G.mix32_inv(G.mix32(U32.from_nat(rest))), U32.from_nat(rest)), 1n, 0n))))

law main:
  IO(Unit)

def main():
  do IO<Unit>:
    IO.print(show_f("u0", G.uniform(7n, 0n)))
    IO.print(show_f("u1", G.uniform(7n, 1n)))
    IO.print(show_f("g0", G.gauss_at(7n, 0n)))
    IO.print(show_f("g1", G.gauss_at(7n, 1n)))
    IO.print(show_f("g100", G.gauss_at(42n, 100n)))
    IO.print(show_list("alphas", G.alphas(8n)))
    IO.print(show_f("beta1", G.beta_at(1n, 8n)))
    IO.print(show_f("beta8", G.beta_at(8n, 8n)))
    IO.print(show_list("temb", G.temb_of(3n, 8n, 4n)))
    IO.print(show_n("roundtrip", roundtrip_go(64n, (0 : U32))))
    return Unit{}
