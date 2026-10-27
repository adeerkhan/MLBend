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
#|strided3 0.6842266 0.14427207
#|strided0 0.95780456 0.49384364 0.037471935
#|stridedn 2 5 8
#|groups_a 4
#|groups_b 3
#|groups_c 16
#|groups_d 1
#|groups_e 0
#|groups_f 1
#|groups_g 7
#|groups_h 1
#|sigma1 0.57735026
#|sigma0 0
#|sigmac 0
#|step 2 0.8660254 0.70710677 0.4082483 0.57735026
#|ddim 1.3325639 3.5690765 -1.5294058
#|ddimz 1.3325639 2.9917264 -0.95205563
#|eta0a 1.3624496 2.9319549 -0.8623981
#|eta0b 1.3624496 2.9319549 -0.8623981

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

# ---- the DDIM pieces: the strided walk, the step and one update ----

def show_nlist_go(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(U32.show(U32.from_nat(h)), String.append(" ", show_nlist_go(t)))

def show_nlist(tag: String, xs: List<&2, Nat>) -> String:
  String.append(tag, String.append(" ", show_nlist_go(xs)))

# the five coefficients of a Step, as a list to print
def step_list(co: G.Step) -> List<&2, F32>:
  match co:
    case G.Step{c0, c1, c2, c3, sig}:
      Con{c0, Con{c1, Con{c2, Con{c3, Con{sig, Nil{}}}}}}

# a three-element sample, its predicted noise, a draw, and a draw big
# enough that reading it by mistake could not hide
def x_demo() -> List<&2, F32>:
  Con{1.0, Con{2.0, Con{(0.0 - 0.5 : F32), Nil{}}}}

def e_demo() -> List<&2, F32>:
  Con{0.1, Con{(0.0 - 0.2 : F32), Con{0.3, Nil{}}}}

def z_demo() -> List<&2, F32>:
  Con{0.0, Con{1.0, Con{(0.0 - 1.0 : F32), Nil{}}}}

def z_big() -> List<&2, F32>:
  Con{1000.0, Con{(0.0 - 1000.0 : F32), Con{500.0, Nil{}}}}

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
    IO.print(show_list("strided3", G.strided(F32, G.alphas(8n), 3n)))
    IO.print(show_list("strided0", G.strided_from(F32, G.alphas(8n), 3n, 0n)))
    IO.print(show_nlist("stridedn", G.strided(Nat, List.range(10n), 3n)))
    IO.print(show_n("groups_a", U32.from_nat(G.groups_of(10n, 3n, 0n))))
    IO.print(show_n("groups_b", U32.from_nat(G.groups_of(10n, 3n, 2n))))
    IO.print(show_n("groups_c", U32.from_nat(G.groups_of(64n, 4n, 3n))))
    IO.print(show_n("groups_d", U32.from_nat(G.groups_of(5n, 4n, 3n))))
    IO.print(show_n("groups_e", U32.from_nat(G.groups_of(0n, 4n, 0n))))
    IO.print(show_n("groups_f", U32.from_nat(G.groups_of(8n, 8n, 7n))))
    IO.print(show_n("groups_g", U32.from_nat(G.groups_of(7n, 1n, 0n))))
    IO.print(show_n("groups_h", U32.from_nat(G.groups_of(3n, 5n, 0n))))
    IO.print(show_f("sigma1", G.sigma_of(0.25, 0.5, 1.0)))
    IO.print(show_f("sigma0", G.sigma_of(0.25, 0.5, 0.0)))
    IO.print(show_f("sigmac", G.sigma_of(0.6, 0.5, 1.0)))
    IO.print(show_list("step", step_list(G.step_of(0.25, 0.5, 1.0))))
    IO.print(show_list("ddim", G.ddim(G.step_of(0.25, 0.5, 1.0), x_demo(), e_demo(), z_demo())))
    IO.print(show_list("ddimz", G.ddim(G.step_of(0.25, 0.5, 1.0), x_demo(), e_demo(), Nil{})))
    IO.print(show_list("eta0a", G.ddim(G.step_of(0.25, 0.5, 0.0), x_demo(), e_demo(), z_demo())))
    IO.print(show_list("eta0b", G.ddim(G.step_of(0.25, 0.5, 0.0), x_demo(), e_demo(), z_big())))
    return Unit{}
