# A diffusion model trained and sampled in Bend, with the real BendTorch
# (`bendtorch@1.0.1.0`).
#
#   bendenv run -- bend examples/diffusion/main.bend
#
# The model is a small denoiser — three linear layers with SiLU — trained
# by `T.Train.step` with AdamW to predict the noise ε of a DDPM forward
# process over a two-moons dataset of 128 points, then sampled by the
# ancestral DDPM loop. Everything that looks random is a pure function of
# counters: the Gaussian noise is Box-Muller over `mix32`-hashed uniforms
# (murmur3's finaliser, over U32 arithmetic that wraps), so a NumPy twin
# can draw the same numbers bit for bit (reference/test_diffusion.py).
#
# The forward process is x_t = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·ε over 32 steps
# of a linear β schedule; the model input concatenates x_t (2 features)
# with an 8-number sinusoidal time embedding. F32.sin, F32.cos, F32.log
# and F32.exp are Base's own. The data, the noise, the embeddings and the
# samples move as List<&2, F32> (duplicable); only the model's matrices
# are bend-ml-tensor-array's, at the edges.
import Base
import bendtorch@1.0.1.0/main.bend as T
import bend-ml-tensor-array@0.1.14.0/main.bend as TA

# The shape: 128 points of 2 features, a time embedding of 8, a hidden
# width of 32, 32 diffusion steps, 400 training steps.

def n() -> Nat:
  128n

def d() -> Nat:
  2n

def inwidth() -> Nat:
  10n

def hidden() -> Nat:
  32n

def steps() -> Nat:
  400n

def X() -> Type:
  TA.Mat<n(), inwidth()>

def Y() -> Type:
  TA.Mat<n(), d()>

def M2(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

# A pair of numbers, as a record: the affine rule has no field access, so
# a pair is matched through a def, never read.

type Pt is Data:
  Pt{x: F32, y: F32}

def pt_x(p: Pt) -> F32:
  match p:
    case Pt{a, b}:
      a

def pt_y(p: Pt) -> F32:
  match p:
    case Pt{a, b}:
      b

# The number at an index of a list (linear walk, the attention example's
# `at`).

def at_go(xs: List<&2, F32>, +k: Nat, +i: Nat) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{h, t}:
      Bool.pick(F32, Nat.is_eq(i, k), h, at_go(t, k, (1n + i : Nat)))

def at(xs: List<&2, F32>, k: Nat) -> F32:
  at_go(xs, k, 0n)

# ---- the denoiser: linear, SiLU, linear, SiLU, linear ----

def B1P() -> Type:
  T.Two<T.Lin<inwidth(), hidden()>, Unit>

def B1S() -> Type:
  T.Two<M2(n(), inwidth()), M2(n(), hidden())>

def B2P() -> Type:
  T.Two<B1P, T.Lin<hidden(), hidden()>>

def B2S() -> Type:
  T.Two<B1S, M2(n(), hidden())>

def B3P() -> Type:
  T.Two<B2P, Unit>

def B3S() -> Type:
  T.Two<B2S, M2(n(), hidden())>

def B4P() -> Type:
  T.Two<B3P, T.Lin<hidden(), d()>>

def B4S() -> Type:
  T.Two<B3S, M2(n(), hidden())>

def B1() -> T.Layer<B1P, X(), M2(n(), hidden()), B1S>:
  T.Seq(~T.Lin<inwidth(), hidden()>, ~Unit, ~X(), ~M2(n(), hidden()),
    ~M2(n(), hidden()), ~M2(n(), inwidth()), ~M2(n(), hidden()),
    ~T.LinearFirst(n(), inwidth(), hidden(), 0n), ~T.SiLU(n(), hidden()))

def B2() -> T.Layer<B2P, X(), M2(n(), hidden()), B2S>:
  T.Seq(~B1P, ~T.Lin<hidden(), hidden()>, ~X(), ~M2(n(), hidden()),
    ~M2(n(), hidden()), ~B1S, ~M2(n(), hidden()), ~B1(), ~T.Linear(n(), hidden(), hidden(), 0n))

def B3() -> T.Layer<B3P, X(), M2(n(), hidden()), B3S>:
  T.Seq(~B2P, ~Unit, ~X(), ~M2(n(), hidden()), ~M2(n(), hidden()),
    ~B2S, ~M2(n(), hidden()), ~B2(), ~T.SiLU(n(), hidden()))

def Model() -> T.Layer<B4P, X(), Y(), B4S>:
  T.Seq(~B3P, ~T.Lin<hidden(), d()>, ~X(), ~M2(n(), hidden()), ~Y(),
    ~B3S, ~M2(n(), hidden()), ~B3(), ~T.Linear(n(), hidden(), d(), 0n))

# ---- the counters' hash and the Gaussian it answers ----

# murmur3's fmix32 finaliser, over U32 arithmetic that wraps
def mix32(+z: U32) -> U32:
  +a = U32.xor(z, U32.shrn(z, 16n))
  +b = (a * 2246822519 : U32)
  +c = U32.xor(b, U32.shrn(b, 13n))
  +e = (c * 3266489917 : U32)
  U32.xor(e, U32.shrn(e, 16n))

# the hash of one counter: the step salt times the golden ratio, the
# index added beside it
def hash_of(+s: Nat, +i: Nat) -> U32:
  +k = (U32.from_nat(s) * 2654435769 : U32)
  (k + (U32.from_nat(i) + 305419898 : U32) : U32)

# a uniform in (0, 1]: the low 24 bits of the hash plus one, exactly
# representable, so log(u) is never log(0)
def uniform(+s: Nat, +i: Nat) -> F32:
  h = mix32(hash_of(s, i))
  m = ((U32.and(h, U32.from_nat(16777215n)) + 1 : U32) : U32)
  (F32.from_nat(U32.to_nat(m)) / 16777216.0 : F32)

# one Gaussian of a counter, by parity: the even index of the pair brings
# the radius, the odd one the angle
def gauss_body(is_even: Bool, +r: F32, +ang: F32) -> F32:
  match is_even:
    case True{}:
      (r * F32.cos(ang) : F32)
    case False{}:
      (r * F32.sin(ang) : F32)

def gauss_at(+s: Nat, +j: Nat) -> F32:
  +pair = Nat.sub(j, Nat.mod(j, 2n))
  +ua = uniform(s, pair)
  +ub = uniform(s, (1n + pair : Nat))
  +r = F32.sqrt((0.0 - (2.0 * F32.log(ua) : F32) : F32))
  gauss_body(Nat.is_eq(Nat.mod(j, 2n), 0n), r, (6.2831855 * ub : F32))

# ---- the data: two moons, half a circle each, on a fixed grid ----

# one moon point: the upper half of a circle, or the lower one flipped
def moon_point(is_upper: Bool, +k: Nat) -> Pt:
  match is_upper:
    case True{}:
      +th = ((3.1415927 * (F32.from_nat(k) + 0.5 : F32)) / 64.0 : F32)
      Pt{F32.cos(th), F32.sin(th)}
    case False{}:
      +th = ((3.1415927 * (F32.from_nat(k) + 0.5 : F32)) / 64.0 : F32)
      Pt{(1.0 - F32.cos(th) : F32), (0.5 - F32.sin(th) : F32)}

def moons_go(+left: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      +p1 = moon_point(True{}, Nat.sub(64n, left))
      +p2 = moon_point(False{}, Nat.sub(64n, left))
      a1 = List.append(&2, F32, acc, Con{pt_x(p1), Nil{}})
      a2 = List.append(&2, F32, a1, Con{pt_y(p1), Nil{}})
      a3 = List.append(&2, F32, a2, Con{pt_x(p2), Nil{}})
      moons_go(rest, List.append(&2, F32, a3, Con{pt_y(p2), Nil{}}))

# the 256 numbers of the two moons, the upper moon first
def moons() -> List<&2, F32>:
  moons_go(64n, Nil{})

# ---- the schedule: a linear β, ᾱ by repeated multiplication ----

# β_t, linear from 0.0001 to 0.3 over the 32 steps (t = 1 .. 32)
def beta_of(+t: Nat) -> F32:
  (0.0001 + ((0.2999 * (F32.from_nat(t) / 32.0 : F32)) : F32) : F32)

def alphas_go(+left: Nat, +t: Nat, +acc: F32, +lst: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      lst
    case 1n+rest:
      +acc2 = (acc * (1.0 - beta_of(t) : F32) : F32)
      alphas_go(rest, (1n + t : Nat), acc2,
        List.append(&2, F32, lst, Con{acc2, Nil{}}))

# ᾱ_1 .. ᾱ_32, in order
def alphas() -> List<&2, F32>:
  alphas_go(32n, 1n, 1.0, Nil{})

def abar(+t: Nat) -> F32:
  at(alphas(), Nat.sub(t, 1n))

# ---- the time embedding: four frequency rungs, a sine and a cosine each ----

def freq_of(+p: Nat) -> F32:
  match p:
    case 0n:
      0.001
    case 1n:
      0.01
    case 2n:
      0.1
    case _:
      1.0

def temb_pair(+tau: F32, +p: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  +f = freq_of(p)
  a1 = List.append(&2, F32, acc, Con{F32.sin((tau * f : F32)), Nil{}})
  List.append(&2, F32, a1, Con{F32.cos((tau * f : F32)), Nil{}})

def temb_go(+t: Nat, +left: Nat, +p: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      temb_go(t, rest, (1n + p : Nat),
        temb_pair((F32.from_nat(t) / 32.0 : F32), p, acc))

def temb_of(+t: Nat) -> List<&2, F32>:
  temb_go(t, 4n, 0n, Nil{})

# ---- the model input: x_t beside its time embedding ----

def input_row(+s: Nat, +r: Nat, +a: F32, +b: F32, +x0: List<&2, F32>,
  +te: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  +i0 = Nat.mul(r, 2n)
  ax = (a * at(x0, i0) : F32)
  bx = (b * gauss_at(s, i0) : F32)
  ay = (a * at(x0, (1n + i0 : Nat)) : F32)
  by = (b * gauss_at(s, (1n + i0 : Nat)) : F32)
  a1 = List.append(&2, F32, acc, Con{(ax + bx : F32), Nil{}})
  a2 = List.append(&2, F32, a1, Con{(ay + by : F32), Nil{}})
  a3 = List.append(&2, F32, a2, Con{at(te, 0n), Nil{}})
  a4 = List.append(&2, F32, a3, Con{at(te, 1n), Nil{}})
  a5 = List.append(&2, F32, a4, Con{at(te, 2n), Nil{}})
  a6 = List.append(&2, F32, a5, Con{at(te, 3n), Nil{}})
  a7 = List.append(&2, F32, a6, Con{at(te, 4n), Nil{}})
  a8 = List.append(&2, F32, a7, Con{at(te, 5n), Nil{}})
  a9 = List.append(&2, F32, a8, Con{at(te, 6n), Nil{}})
  List.append(&2, F32, a9, Con{at(te, 7n), Nil{}})

def input_go(+s: Nat, +a: F32, +b: F32, +x0: List<&2, F32>, +te: List<&2, F32>,
  +left: Nat, +r: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      input_go(s, a, b, x0, te, rest, (1n + r : Nat),
        input_row(s, r, a, b, x0, te, acc))

# the input of training step s at noise level t: 128 rows of 10 numbers,
# x_t then its embedding, flat in row-major order
def input_of(+s: Nat, +t: Nat, +a: F32, +b: F32, +x0: List<&2, F32>) -> List<&2, F32>:
  input_go(s, a, b, x0, temb_of(t), 128n, 0n, Nil{})

def noise_go(+s: Nat, +left: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      j = Nat.sub(256n, left)
      noise_go(s, rest, List.append(&2, F32, acc, Con{gauss_at(s, j), Nil{}}))

# the noise of counter s: 256 numbers
def noise_of(+s: Nat) -> List<&2, F32>:
  noise_go(s, 256n, Nil{})

# ---- the training loop ----

def cfg() -> T.OptCfg:
  T.CAdamW{0.01, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

def mse(y: Y(), tgt: Y()) -> T.LossOut<Y()>:
  T.MSE(n(), d(), y, tgt)

# the starting weights: a fixed small pattern, not zeros
def value(+i: Nat, +j: Nat, +seed: Nat) -> F32:
  F32.sub(F32.mul(F32.from_nat(Nat.mod(Nat.add(Nat.add(Nat.mul(i, 3n), Nat.mul(j, 5n)), seed), 7n)),
    0.1), 0.2)

def row_vals(+c: Nat, +i: Nat, +j: Nat, +seed: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match c:
    case 0n:
      acc
    case 1n+rest:
      row_vals(rest, i, (1n + j : Nat), seed,
        List.append(&2, F32, acc, Con{value(i, j, seed), Nil{}}))

def rows_vals(+r: Nat, +c: Nat, +i: Nat, +seed: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match r:
    case 0n:
      acc
    case 1n+rest:
      rows_vals(rest, c, (1n + i : Nat), seed,
        List.append(&2, F32, acc, row_vals(c, i, 0n, seed, Nil{})))

def mat_of(+r: Nat, +c: Nat, +seed: Nat) -> TA.Mat<r, c>:
  TA.Mat.from_list(r, c, rows_vals(r, c, 0n, seed, Nil{}))

def params() -> B4P():
  T.Two{
    T.Two{
      T.Two{T.Two{T.Lin{mat_of(32n, 10n, 1n), TA.Mat.fill(1n, 32n, 0.0)}, Unit{}},
      T.Lin{mat_of(32n, 32n, 2n), TA.Mat.fill(1n, 32n, 0.0)}},
    Unit{}},
  T.Lin{mat_of(2n, 32n, 3n), TA.Mat.fill(1n, 2n, 0.0)}}

# sqrt(ᾱ_t) and sqrt(1-ᾱ_t), the forward process's two coefficients
def coef_a(+t: Nat) -> F32:
  F32.sqrt(abar(t))

def coef_b(+t: Nat) -> F32:
  F32.sqrt((1.0 - abar(t) : F32))

def do_step(+s: Nat, tr: T.Train<B4P()>) -> T.Train<B4P()>:
  +t = Nat.add(Nat.mod(s, 32n), 1n)
  T.Train.step(~B4P(), ~X(), ~Y(), ~B4S(), ~Y(), ~Model(), ~mse, cfg(), tr,
    TA.Mat.from_list(n(), inwidth(), input_of(s, t, coef_a(t), coef_b(t), moons())),
    TA.Mat.from_list(n(), d(), noise_of(s)))

def should_print(+i: Nat) -> Bool:
  Bool.or(Bool.or(Nat.is_eq(i, 1n), Nat.is_eq(Nat.mod((1n + i : Nat), 40n), 0n)),
    Nat.is_eq((1n + i : Nat), steps()))

def print_or(sp: Bool, +l: F32) -> IO(Unit):
  match sp:
    case True{}:
      do IO<Unit>:
        IO.print(String.append("loss: ", F32.show(l)))
        return Unit{}
    case False{}:
      do IO<Unit>:
        return Unit{}

def loop(left: Nat, +i: Nat, tr: T.Train<B4P()>) -> IO(T.Train<B4P()>):
  match left tr:
    case 0n T.Train{p, g, st, q1, q2, +l}:
      do IO<T.Train<B4P()>>:
        IO.print(String.append("final_loss: ", F32.show(l)))
        return T.Train{p, g, st, q1, q2, l}
    case 1n+rest T.Train{p, g, st, q1, q2, +l}:
      do IO<T.Train<B4P()>>:
        Unit <- print_or(should_print(i), l)
        loop(rest, (1n + i : Nat), do_step(i, T.Train{p, g, st, q1, q2, l}))

# ---- the sample: the ancestral DDPM loop, from pure noise to a moon ----
#
# The parameters are affine (they hold arrays), so they are not copied but
# returned: `Layer.fwd` answers `Fwd{p, y, s}` with the parameters inside,
# and the loop threads them through a record, the way bendtorch's own
# eval loop does.

# the parameters and the current sample, threaded through the loop

type PS is Type:
  PS{p: B4P(), x: List<&2, F32>}

# one denoise: the parameters come back beside the predicted noise

type PE is Type:
  PE{p: B4P(), eps: List<&2, F32>}

# the flat list of a matrix (Mat.to_list answers ML{mat, list})
def flat_of(m: TA.ML<n(), d()>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

def eps_list(m: Y()) -> List<&2, F32>:
  flat_of(TA.Mat.to_list(n(), d(), m))

# the denoiser's forward pass over one input list
def fwd_eps_of(r: T.Fwd<B4P(), Y(), B4S()>) -> PE:
  match r:
    case T.Fwd{p2, y, s}:
      PE{p2, eps_list(y)}

# the denoiser's answer for one sample (a = 1, b = 0: the sample is not
# noised again, the embedding rides beside it)
def predict(p: B4P(), +t: Nat, +x: List<&2, F32>) -> PE:
  fwd_eps_of(T.Layer.fwd(B4P(), X(), Y(), B4S(), Model(), p,
    TA.Mat.from_list(n(), inwidth(), input_of((1000000n + t : Nat), t, 1.0, 0.0, x))))

def rev_elem(+coef: F32, +scale: F32, +xv: F32, +eps: F32, +z: F32, last: Bool) -> F32:
  match last:
    case True{}:
      (scale * (xv - (coef * eps : F32) : F32) : F32)
    case False{}:
      (scale * ((xv - (coef * eps : F32) : F32) + z : F32) : F32)

def rev_go(+coef: F32, +scale: F32, +x: List<&2, F32>, +eps: List<&2, F32>,
  +z: List<&2, F32>, +last: Bool, +left: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      +j = Nat.sub(256n, left)
      rev_go(coef, scale, x, eps, z, last, rest,
        List.append(&2, F32, acc,
          Con{rev_elem(coef, scale, at(x, j), at(eps, j), at(z, j), last), Nil{}}))

# one reverse step, over the denoise's answer: the new parameters beside
# the new sample
def rev_pe(pe: PE, +coef: F32, +scale: F32, +z: List<&2, F32>, +last: Bool,
  +x: List<&2, F32>) -> PS:
  match pe:
    case PE{p2, eps}:
      PS{p2, rev_go(coef, scale, x, eps, z, last, 256n, Nil{})}

def reverse_step(+t: Nat, last: Bool, st: PS) -> PS:
  match st:
    case PS{p, +x}:
      +beta = beta_of(t)
      +coef = (beta / F32.sqrt((1.0 - abar(t) : F32)) : F32)
      +scale = (1.0 / F32.sqrt((1.0 - beta : F32)) : F32)
      rev_pe(predict(p, t, x), coef, scale, noise_of((1000000n + t : Nat)), last, x)

def sample_go(t: Nat, st: PS) -> PS:
  match t:
    case 0n:
      st
    case 1n++rest:
      last = Nat.is_eq(t, 1n)
      sample_go(rest, reverse_step(t, last, st))

# x_32 from pure noise (a counter past the training ones), then down to x_0
def sample(p: B4P()) -> PS:
  sample_go(32n, PS{p, noise_of(2000000n)})

def sample_x(st: PS) -> List<&2, F32>:
  match st:
    case PS{p, x}:
      x

# ---- the statistics, printed ----

def sum2_go(+arr: List<&2, F32>, +left: Nat, +sx: F32, +sy: F32) -> Pt:
  match left:
    case 0n:
      Pt{sx, sy}
    case 1n+rest:
      +i = Nat.mul(Nat.sub(128n, left), 2n)
      sum2_go(arr, rest,
        (sx + at(arr, i) : F32),
        (sy + at(arr, (1n + i : Nat)) : F32))

def div2_of(p: Pt) -> Pt:
  match p:
    case Pt{sx, sy}:
      Pt{(sx / 128.0 : F32), (sy / 128.0 : F32)}

def means(arr: List<&2, F32>) -> Pt:
  div2_of(sum2_go(arr, 128n, 0.0, 0.0))

def var2_go(+arr: List<&2, F32>, +left: Nat, +mx: F32, +my: F32, +vx: F32, +vy: F32) -> Pt:
  match left:
    case 0n:
      Pt{vx, vy}
    case 1n+rest:
      +i = Nat.mul(Nat.sub(128n, left), 2n)
      +dx = (at(arr, i) - mx : F32)
      +dy = (at(arr, (1n + i : Nat)) - my : F32)
      var2_go(arr, rest, mx, my, (vx + (dx * dx : F32) : F32), (vy + (dy * dy : F32) : F32))

def vars_of(arr: List<&2, F32>, m: Pt) -> Pt:
  match m:
    case Pt{mx, my}:
      var2_go(arr, 128n, mx, my, 0.0, 0.0)

def sqrt2_of(p: Pt) -> Pt:
  match p:
    case Pt{vx, vy}:
      Pt{F32.sqrt((vx / 128.0 : F32)), F32.sqrt((vy / 128.0 : F32))}

def stds(+arr: List<&2, F32>) -> Pt:
  sqrt2_of(vars_of(arr, means(arr)))

def show_pair(tag: String, p: Pt) -> String:
  match p:
    case Pt{a, b}:
      String.append(tag, String.append(" ", String.append(F32.show(a),
        String.append(" ", F32.show(b)))))

def params_of(tr: T.Train<B4P()>) -> B4P():
  match tr:
    case T.Train{p, g, st, q1, q2, l}:
      p

def pair_text(p: Pt) -> String:
  match p:
    case Pt{a, b}:
      String.append(F32.show(a), String.append(" ", F32.show(b)))

def print_points_go(+x: List<&2, F32>, +k: Nat, +acc: String) -> String:
  match k:
    case 0n:
      acc
    case 1n+rest:
      +i = Nat.mul(Nat.sub(128n, k), 2n)
      rest_txt = print_points_go(x, rest, acc)
      String.append(rest_txt, String.append(" ",
        String.append(F32.show(at(x, i)), String.append(" ", F32.show(at(x, (1n + i : Nat)))))))

def print_sample_x(+x: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    IO.print(show_pair("sample_mean:", means(x)))
    IO.print(show_pair("sample_std:", stds(x)))
    IO.print(String.append("samples:", print_points_go(x, 128n, "")))
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(cfg())
    tr : T.Train<B4P()> <- loop(steps(), 0n,
      T.Train.new(~B4P(), ~X(), ~Y(), ~B4S(), ~Model(), cfg(), params()))
    IO.print(show_pair("data_mean:", means(moons())))
    IO.print(show_pair("data_std:", stds(moons())))
    Unit <- print_sample_x(sample_x(sample(params_of(tr))))
    return Unit{}
