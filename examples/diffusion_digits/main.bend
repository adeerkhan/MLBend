# A DDPM over real data: handwritten digits (8x8, the optdigits set),
# trained and sampled in Bend with the real BendTorch.
#
#   bendenv run -- bend examples/diffusion_digits/main.bend
#
# The dataset is mstz/optdigits at a pinned revision: 3823 rows of 64
# pixels (0..16) and the label, read through mlbend-datasets' generic
# csv_of; the first 1024 rows are the training set, the pixels scaled to
# [0, 1]. The denoiser is a three-linear-layer MLP with SiLU over the
# noised 64 pixels beside an 8-number time embedding, trained full-batch
# by AdamW to predict the noise of the linear schedule of
# mlbend-generative, then 64 fresh digits are sampled by the ancestral
# loop and printed as a grid (reference/test_diffusion_digits.py draws
# them beside the data).
#
# The same findings as examples/diffusion govern the shape: the parameters
# thread through the sampling loop (Layer.fwd returns them), and every
# "random" number is a counter.
import Base
import bendtorch@1.0.1.0/main.bend as T
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import mlbend-generative@0.1.0.0/main.bend as G
import mlbend-datasets@0.1.6.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf

def n() -> Nat:
  1024n

def d() -> Nat:
  64n

def inwidth() -> Nat:
  72n

def hidden() -> Nat:
  64n

def steps() -> Nat:
  3000n

def samples() -> Nat:
  64n

def X() -> Type:
  TA.Mat<n(), inwidth()>

def Y() -> Type:
  TA.Mat<n(), d()>

def SY() -> Type:
  TA.Mat<samples(), d()>

def SX() -> Type:
  TA.Mat<samples(), inwidth()>

def M2(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

# ---- the denoiser: linear, SiLU, linear, SiLU, linear ----

def B1P() -> Type:
  T.Two<T.Lin<inwidth(), hidden()>, Unit>

def B1S(r: Nat) -> Type:
  T.Two<M2(r, inwidth()), M2(r, hidden())>

def B2P() -> Type:
  T.Two<B1P, T.Lin<hidden(), hidden()>>

def B2S(r: Nat) -> Type:
  T.Two<B1S(r), M2(r, hidden())>

def B3P() -> Type:
  T.Two<B2P, Unit>

def B3S(r: Nat) -> Type:
  T.Two<B2S(r), M2(r, hidden())>

def B4P() -> Type:
  T.Two<B3P, T.Lin<hidden(), d()>>

def B4S(r: Nat) -> Type:
  T.Two<B3S(r), M2(r, hidden())>

def B1(~r: Nat) -> T.Layer<B1P, M2(r, inwidth()), M2(r, hidden()), B1S(r)>:
  T.Seq(~T.Lin<inwidth(), hidden()>, ~Unit, ~M2(r, inwidth()), ~M2(r, hidden()),
    ~M2(r, hidden()), ~M2(r, inwidth()), ~M2(r, hidden()),
    ~T.LinearFirst(r, inwidth(), hidden(), 0n), ~T.SiLU(r, hidden()))

def B2(~r: Nat) -> T.Layer<B2P, M2(r, inwidth()), M2(r, hidden()), B2S(r)>:
  T.Seq(~B1P, ~T.Lin<hidden(), hidden()>, ~M2(r, inwidth()), ~M2(r, hidden()),
    ~M2(r, hidden()), ~B1S(r), ~M2(r, hidden()), ~B1(~r), ~T.Linear(r, hidden(), hidden(), 0n))

def B3(~r: Nat) -> T.Layer<B3P, M2(r, inwidth()), M2(r, hidden()), B3S(r)>:
  T.Seq(~B2P, ~Unit, ~M2(r, inwidth()), ~M2(r, hidden()), ~M2(r, hidden()),
    ~B2S(r), ~M2(r, hidden()), ~B2(~r), ~T.SiLU(r, hidden()))

def Model(~r: Nat) -> T.Layer<B4P, M2(r, inwidth()), M2(r, d()), B4S(r)>:
  T.Seq(~B3P, ~T.Lin<hidden(), d()>, ~M2(r, inwidth()), ~M2(r, hidden()), ~M2(r, d()),
    ~B3S(r), ~M2(r, hidden()), ~B3(~r), ~T.Linear(r, hidden(), d(), 0n))

# ---- the data: 1024 digits, the pixels scaled to [0, 1] ----

# the loader runs once, so the rows are built the plain way: appended,
# ascending. (The training input is built per step; that one needs the
# cons-and-reverse walk below.)
def scale_go(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      scale_go(t, List.append(&2, F32, acc, Con{(h / 16.0 : F32), Nil{}}))

# one reverse of a whole consed list
def list_rev(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      list_rev(t, Con{h, acc})

def take_rows(left: Nat, xs: List<&2, List<&2, F32>>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left xs:
    case 0n Nil{}:
      acc
    case 0n Con{row, rows}:
      acc
    case 1n+rest Con{row, rows}:
      take_rows(rest, rows, List.append(&2, F32, acc, scale_go(row, Nil{})))
    case 1n+rest Nil{}:
      acc

def digits_rows(t: Datasets.Table) -> List<&2, F32>:
  match t:
    case Datasets.Table{rows, labels}:
      take_rows(1024n, rows, Nil{})

# ---- the schedule and the coefficients ----

def at_go(xs: List<&2, F32>, +k: Nat, +i: Nat) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{h, t}:
      Bool.pick(F32, Nat.is_eq(i, k), h, at_go(t, k, (1n + i : Nat)))

def at(xs: List<&2, F32>, k: Nat) -> F32:
  at_go(xs, k, 0n)

def abar(+t: Nat) -> F32:
  at(G.lin_alphas(64n), Nat.sub(t, 1n))

def coef_a(+t: Nat) -> F32:
  F32.sqrt(abar(t))

def coef_b(+t: Nat) -> F32:
  F32.sqrt((1.0 - abar(t) : F32))

# ---- the model input: the noised pixels beside the embedding ----
#
# Built in three plainly-correct stages: each row's cells are appended in
# order (row-local cost only); the rows are consed as finished blocks; one
# flatten-cons walk and one reverse turn the blocks into the flat
# row-major list. The noise index j rides across rows: pixel p of row r
# pairs with counter index r*64+p, the same index the target noise list
# holds.

def row_temb_go(+te: List<&2, F32>, +k: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match k:
    case 0n:
      acc
    case 1n+rest:
      row_temb_go(te, rest, List.append(&2, F32, acc, Con{at(te, Nat.sub(8n, k)), Nil{}}))

# the pixels of one row, noised and appended in order; j is the row's
# base noise index, riding +1 a pixel
def row_pixels_go(+s: Nat, +a: F32, +b: F32, +te: List<&2, F32>,
  x0: List<&2, F32>, +left: Nat, +j: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match x0 left:
    case Con{h, tail} 1n+rest:
      row_pixels_go(s, a, b, te, tail, rest, (1n + j : Nat),
        List.append(&2, F32, acc,
          Con{(a * h + (b * G.gauss_at(s, j) : F32) : F32), Nil{}}))
    case Con{h, tail} 0n:
      row_temb_go(te, 8n, acc)
    case Nil{} 0n:
      row_temb_go(te, 8n, acc)
    case Nil{} 1n+rest:
      row_temb_go(te, 8n, acc)

# one row consumed: the finished block and the rest of the list

type RR is Data:
  RR{frag: List<&2, F32>, rest: List<&2, F32>}

def row_of(+s: Nat, +a: F32, +b: F32, +te: List<&2, F32>, x0: List<&2, F32>, +j: Nat) -> RR:
  match x0:
    case Nil{}:
      RR{Nil{}, Nil{}}
    case Con{h, +tail}:
      RR{row_pixels_go(s, a, b, te, Con{h, tail}, 64n, j, Nil{}), tail}

def rr_frag(rr: RR) -> List<&2, F32>:
  match rr:
    case RR{frag, rest}:
      frag

def rr_rest(rr: RR) -> List<&2, F32>:
  match rr:
    case RR{frag, rest}:
      rest

# the rows, consed as blocks: the first row ends up deepest
def rows_go(+s: Nat, +a: F32, +b: F32, +t: Nat, +te: List<&2, F32>,
  +rows_left: Nat, +j: Nat, x0: List<&2, F32>, +acc: List<&2, List<&2, F32>>) ->
  List<&2, List<&2, F32>>:
  match rows_left:
    case 0n:
      acc
    case 1n+rest:
      +rr = row_of(s, a, b, te, x0, j)
      rows_go(s, a, b, t, te, rest, (64n + j : Nat), rr_rest(rr), Con{rr_frag(rr), acc})

# flatten the blocks by consing (each block lands reversed), then one
# reverse of the whole: the flat row-major list
def cons_all(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      cons_all(t, Con{h, acc})

def flat_go(rows: List<&2, List<&2, F32>>, +acc: List<&2, F32>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      acc
    case Con{row, rest}:
      flat_go(rest, cons_all(row, acc))

def input_of(+s: Nat, +t: Nat, +a: F32, +b: F32, +x0: List<&2, F32>) -> List<&2, F32>:
  list_rev(flat_go(rows_go(s, a, b, t, G.temb_of(t, 64n, 4n), 1024n, 0n, x0, Nil{}), Nil{}), Nil{})

# the sampler's input: 64 rows, the sample not noised again
def sinput_of(+t: Nat, x: List<&2, F32>) -> List<&2, F32>:
  list_rev(flat_go(rows_go((1000000n + t : Nat), 1.0, 0.0, t, G.temb_of(t, 64n, 4n), 64n, 0n, x, Nil{}), Nil{}), Nil{})

# ---- the training loop ----

def cfg() -> T.OptCfg:
  T.CAdamW{0.01, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

def mse(y: Y(), tgt: Y()) -> T.LossOut<Y()>:
  T.MSE(n(), d(), y, tgt)

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
      T.Two{T.Two{T.Lin{mat_of(hidden(), inwidth(), 1n), TA.Mat.fill(1n, hidden(), 0.0)}, Unit{}},
      T.Lin{mat_of(hidden(), hidden(), 2n), TA.Mat.fill(1n, hidden(), 0.0)}},
    Unit{}},
  T.Lin{mat_of(d(), hidden(), 3n), TA.Mat.fill(1n, d(), 0.0)}}

def noise_of(+s: Nat) -> List<&2, F32>:
  G.noise_of(s, 65536n)

def do_step(+s: Nat, tr: T.Train<B4P()>, +x0: List<&2, F32>) -> T.Train<B4P()>:
  +t = Nat.add(Nat.mod(s, 64n), 1n)
  T.Train.step(~B4P(), ~X(), ~Y(), ~B4S(n()), ~Y(), ~Model(n()), ~mse, cfg(), tr,
    TA.Mat.from_list(n(), inwidth(), input_of(s, t, coef_a(t), coef_b(t), x0)),
    TA.Mat.from_list(n(), d(), noise_of(s)))

def should_print(+i: Nat) -> Bool:
  Bool.or(Bool.or(Nat.is_eq(i, 1n), Nat.is_eq(Nat.mod((1n + i : Nat), 100n), 0n)),
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

def loop(rows_left: Nat, +i: Nat, +x0: List<&2, F32>, tr: T.Train<B4P()>) -> IO(T.Train<B4P()>):
  match rows_left tr:
    case 0n T.Train{p, g, st, q1, q2, +l}:
      do IO<T.Train<B4P()>>:
        IO.print(String.append("final_loss: ", F32.show(l)))
        return T.Train{p, g, st, q1, q2, l}
    case 1n+rest T.Train{p, g, st, q1, q2, +l}:
      do IO<T.Train<B4P()>>:
        Unit <- print_or(should_print(i), l)
        loop(rest, (1n + i : Nat), x0, do_step(i, T.Train{p, g, st, q1, q2, l}, x0))

# ---- the sample: 64 fresh digits by the ancestral loop ----
#
# The parameters thread through a PS record (Layer.fwd returns them). The
# reverse update consumes the sample, the prediction and the noise in
# parallel, consing the new sample in reverse; the update is element-wise,
# so order only tracks position, and 64 steps (an even count) leave the
# pixels where they started.

type PS is Type:
  PS{p: B4P(), x: List<&2, F32>}

type PE is Type:
  PE{p: B4P(), eps: List<&2, F32>}

def flat_of(m: TA.ML<samples(), d()>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

def eps_list(m: SY()) -> List<&2, F32>:
  flat_of(TA.Mat.to_list(samples(), d(), m))

def fwd_eps_of(r: T.Fwd<B4P(), SY(), B4S(samples())>) -> PE:
  match r:
    case T.Fwd{p2, y, s}:
      PE{p2, eps_list(y)}

def predict(p: B4P(), +t: Nat, +x: List<&2, F32>) -> PE:
  fwd_eps_of(T.Layer.fwd(B4P(), SX(), SY(), B4S(samples()), Model(samples()), p,
    TA.Mat.from_list(samples(), inwidth(), sinput_of(t, x))))

def rev_elem(+coef: F32, +scale: F32, +sig: F32, +xv: F32, +eps: F32, +z: F32,
  last: Bool) -> F32:
  match last:
    case True{}:
      (scale * (xv - (coef * eps : F32) : F32) : F32)
    case False{}:
      (scale * ((xv - (coef * eps : F32) : F32) + (sig * z : F32) : F32) : F32)

def rev_go(+coef: F32, +scale: F32, +sig: F32, x: List<&2, F32>, eps: List<&2, F32>,
  z: List<&2, F32>, +last: Bool, left: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match x eps z left:
    case Con{xv, xt} Con{ev, et} Con{zv, zt} 1n+rest:
      rev_go(coef, scale, sig, xt, et, zt, last, rest,
        Con{rev_elem(coef, scale, sig, xv, ev, zv, last), acc})
    case _ _ _ 0n:
      acc
    case _ _ _ _:
      acc

def rev_pe(pe: PE, +coef: F32, +scale: F32, +sig: F32, +z: List<&2, F32>, +last: Bool,
  +x: List<&2, F32>) -> PS:
  match pe:
    case PE{p2, eps}:
      PS{p2, rev_go(coef, scale, sig, x, eps, z, last, 4096n, Nil{})}

def reverse_step(+t: Nat, last: Bool, st: PS) -> PS:
  match st:
    case PS{p, +x}:
      +beta = G.lin_beta_at(t, 64n)
      +coef = (beta / F32.sqrt((1.0 - abar(t) : F32)) : F32)
      +scale = (1.0 / F32.sqrt((1.0 - beta : F32)) : F32)
      +sig = F32.sqrt(beta)
      rev_pe(predict(p, t, x), coef, scale, sig, G.noise_of((1000000n + t : Nat), 4096n), last, x)

def sample_go(t: Nat, st: PS) -> PS:
  match t:
    case 0n:
      st
    case 1n++rest:
      last = Nat.is_eq(t, 1n)
      sample_go(rest, reverse_step(t, last, st))

def sample(p: B4P()) -> PS:
  sample_go(64n, PS{p, G.noise_of(2000000n, 4096n)})

def sample_x(st: PS) -> List<&2, F32>:
  match st:
    case PS{p, x}:
      x

def params_of(tr: T.Train<B4P()>) -> B4P():
  match tr:
    case T.Train{p, g, st, q1, q2, l}:
      p

# ---- the statistics and the grid, printed ----

type Pt is Data:
  Pt{x: F32, y: F32}

def sum_go(x: List<&2, F32>, left: Nat, +s: F32) -> F32:
  match x left:
    case Nil{} 0n:
      s
    case Con{h, t} 0n:
      s
    case Con{h, t} 1n+rest:
      sum_go(t, rest, (s + h : F32))
    case Nil{} 1n+rest:
      s

def mean_of(arr: List<&2, F32>, +count: Nat) -> F32:
  (sum_go(arr, count, 0.0) / F32.from_nat(count) : F32)

def var_go(x: List<&2, F32>, +m: F32, left: Nat, +v: F32) -> F32:
  match x left:
    case Nil{} 0n:
      v
    case Con{h, t} 0n:
      v
    case Con{h, t} 1n+rest:
      +dh = (h - m : F32)
      var_go(t, m, rest, (v + (dh * dh : F32) : F32))
    case Nil{} 1n+rest:
      v

def std_of(+arr: List<&2, F32>, +count: Nat) -> F32:
  m = mean_of(arr, count)
  F32.sqrt((var_go(arr, m, count, 0.0) / F32.from_nat(count) : F32))

def print_points_go(x: List<&2, F32>, left: Nat, +acc: String) -> String:
  match x left:
    case Nil{} 0n:
      acc
    case Con{h, t} 0n:
      acc
    case Con{h, t} 1n+rest:
      rest_txt = print_points_go(t, rest, acc)
      String.append(rest_txt, String.append(" ", F32.show(h)))
    case Nil{} 1n+rest:
      acc

def print_sample_x(+x: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    IO.print(String.append("samples:", print_points_go(x, 4096n, "")))
    return Unit{}

def digits_of(r: Result<&1, &1, Hf.Err, Datasets.Table>) -> IO(List<&2, F32>):
  match r:
    case Fail{e}:
      do IO<List<&2, F32>>:
        IO.print("the dataset failed")
        return Nil{}
    case Done{t}:
      do IO<List<&2, F32>>:
        return digits_rows(t)

def print_results(tr: T.Train<B4P()>) -> IO(Unit):
  +xs = sample_x(sample(params_of(tr)))
  sm = mean_of(xs, 4096n)
  ss = std_of(xs, 4096n)
  do IO<Unit>:
    IO.print(String.append("sample_mean: ", F32.show(sm)))
    IO.print(String.append("sample_std: ", F32.show(ss)))
    Unit <- print_sample_x(xs)
    return Unit{}

def print_data_stats(+x0: List<&2, F32>) -> IO(Unit):
  +dm = mean_of(x0, 65536n)
  +ds = std_of(x0, 65536n)
  do IO<Unit>:
    IO.print(String.append("data_mean: ", F32.show(dm)))
    IO.print(String.append("data_std: ", F32.show(ds)))
    return Unit{}

def train_and_report(+x0: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    Unit <- print_data_stats(x0)
    tr : T.Train<B4P()> <- loop(steps(), 0n, x0,
      T.Train.new(~B4P(), ~X(), ~Y(), ~B4S(n()), ~Model(n()), cfg(), params()))
    Unit <- print_results(tr)
    return Unit{}

def main() -> IO(Unit):
  do IO<Unit>:
    T.OptCfg.check(cfg())
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.csv_of(
      "mstz/optdigits", "4f20cc827b763fb69e2a80e18e892e4e11498aec", "optdigits.data", False{}, 0n)
    x0 : List<&2, F32> <- digits_of(t)
    Unit <- train_and_report(x0)
    return Unit{}
