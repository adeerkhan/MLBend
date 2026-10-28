# The ViT denoiser's type plumbing (issue #16, milestone M1): the attention
# block and the MLP block, composed from BendTorch's own layers over 1024
# token rows (64 digits x 16 tokens) of width 64, single head.
#
#   AttBlk : LayerNorm -> Residual( Attention )            : Mat<1024,64> -> itself
#   Mlp    : LayerNorm -> Residual( 64 -> 128 -> 64 GELU ) : Mat<1024,64> -> itself
#
# The facts that cost the iterations, for the next reader: `LinearNB(n, i, o)`
# takes its parameter as `Mat<o, i>` (out by in); `Seq` chains exactly two
# layers, so a three-layer chain is nested; `LayerNorm`'s parameter is
# `T.LN<d>`, not a row (that is `RMSNorm`); and a self-call must put its
# shrinking argument first. Forward, training and sampling are the next step;
# this file proves the types.
import Base
import bendtorch@1.0.1.0/main.bend as T
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import mlbend-datasets@0.1.6.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf
import ../examples/diffusion_digits/main.bend as D

def at_go(xs: List<&2, F32>, +k: Nat, +i: Nat) -> F32:
  match xs:
    case Nil{}:
      0.0
    case Con{h, t}:
      Bool.pick(F32, Nat.is_eq(i, k), h, at_go(t, k, (1n + i : Nat)))

def at(xs: List<&2, F32>, k: Nat) -> F32:
  at_go(xs, k, 0n)

def B() -> Nat:
  64n

def TK() -> Nat:
  16n

def RW() -> Nat:
  Nat.mul(B(), TK())

def W() -> Nat:
  64n

def HI() -> Nat:
  128n

def FE() -> Nat:
  20n

def PX() -> Nat:
  4n

def M(r: Nat, c: Nat) -> Type:
  TA.Mat<r, c>

def A(c: Nat) -> Type:
  TA.Mat<RW(), c>

# ---- the attention block, as the transformer example composes it ----

def KVP() -> Type:
  T.Two<M(W(), W()), M(W(), W())>

def KVY() -> Type:
  T.Two<A(W()), A(W())>

def KVS() -> Type:
  T.Two<A(W()), A(W())>

def KVL() -> T.Layer<KVP, A(W()), KVY, KVS>:
  T.Fork(~M(W(), W()), ~M(W(), W()), ~A(W()), ~A(W()), ~A(W()), ~A(W()), ~A(W()),
    ~(m => T.Mat.dup(RW(), W(), m)),
    ~(x => y => T.Mat.sum2(RW(), W(), x, y)),
    ~T.LinearNB(RW(), W(), W(), 0n),
    ~T.LinearNB(RW(), W(), W(), 0n))

def QKVP() -> Type:
  T.Two<M(W(), W()), KVP>

def QKVS() -> Type:
  T.Two<A(W()), KVS>

def QKV() -> T.Layer<QKVP, A(W()), T.Two<A(W()), KVY>, QKVS>:
  T.Fork(~M(W(), W()), ~KVP, ~A(W()), ~A(W()), ~KVY, ~A(W()), ~KVS,
    ~(m => T.Mat.dup(RW(), W(), m)),
    ~(x => y => T.Mat.sum2(RW(), W(), x, y)),
    ~T.LinearNB(RW(), W(), W(), 0n), ~KVL())

def AX() -> Type:
  T.Two<A(W()), KVY>

def CoreP() -> Type:
  T.Two<Unit, M(W(), W())>

def CoreS() -> Type:
  T.Two<T.AttnS, A(W())>

def Core() -> T.Layer<CoreP, AX, A(W()), CoreS>:
  T.Seq(~Unit, ~M(W(), W()), ~AX, ~A(W()), ~A(W()), ~T.AttnS, ~A(W()),
    ~T.Attention(RW(), W(), W(), B(), TK(), 1n, 1n, W(), 0.125, {==}, {==}, {==}),
    ~T.LinearNB(RW(), W(), W(), 0n))

def AttP() -> Type:
  T.Two<QKVP, CoreP>

def AttS() -> Type:
  T.Two<QKVS, CoreS>

def Att() -> T.Layer<AttP, A(W()), A(W()), AttS>:
  T.Seq(~QKVP, ~CoreP, ~A(W()), ~AX, ~A(W()), ~QKVS, ~CoreS, ~QKV(), ~Core())

# ---- one transformer block: norm -> attention -> residual, then
# norm -> MLP -> residual ----

def AttBlkP() -> Type:
  T.Two<T.LN<W()>, T.Two<T.Two<Unit, AttP>, Unit>>

def AttBlkS() -> Type:
  T.Two<A(W()), T.Two<T.Two<Unit, AttS>, Unit>>

def AttBlk() -> T.Layer<AttBlkP, A(W()), A(W()), AttBlkS>:
  T.Seq(~T.LN<W()>, ~T.Two<T.Two<Unit, AttP>, Unit>, ~A(W()), ~A(W()), ~A(W()),
    ~A(W()), ~T.Two<T.Two<Unit, AttS>, Unit>,
    ~T.LayerNorm(RW(), W(), 0.00001),
    ~T.Residual(RW(), W(), AttP(), AttS(), Att()))

# ---- the MLP half of a block: norm -> 64->128->64 -> residual ----

def MIP() -> Type:
  T.Two<T.Two<M(HI(), W()), Unit>, M(W(), HI())>

def MIS() -> Type:
  T.Two<T.Two<A(W()), A(HI())>, A(HI())>

def MlpP() -> Type:
  T.Two<T.LN<W()>, T.Two<T.Two<Unit, MIP>, Unit>>

def MlpS() -> Type:
  T.Two<A(W()), T.Two<T.Two<Unit, MIS>, Unit>>

def MlpHalf() -> T.Layer<T.Two<M(HI(), W()), Unit>, A(W()), A(HI()), T.Two<A(W()), A(HI())>>:
  T.Seq(~M(HI(), W()), ~Unit, ~A(W()), ~A(HI()), ~A(HI()), ~A(W()), ~A(HI()),
    ~T.LinearNB(RW(), W(), HI(), 0n), ~T.GELU(RW(), HI()))

def MlpInner() -> T.Layer<MIP, A(W()), A(W()), MIS>:
  T.Seq(~T.Two<M(HI(), W()), Unit>, ~M(W(), HI()), ~A(W()), ~A(HI()), ~A(W()),
    ~T.Two<A(W()), A(HI())>, ~A(HI()), ~MlpHalf(), ~T.LinearNB(RW(), HI(), W(), 0n))

def Mlp() -> T.Layer<MlpP, A(W()), A(W()), MlpS>:
  T.Seq(~T.LN<W()>, ~T.Two<T.Two<Unit, MIP>, Unit>, ~A(W()), ~A(W()), ~A(W()),
    ~A(W()), ~T.Two<T.Two<Unit, MIS>, Unit>,
    ~T.LayerNorm(RW(), W(), 0.00001),
    ~T.Residual(RW(), W(), MIP, MIS, MlpInner()))

# ---- a block is the two halves in sequence; three of them make the body ----

def B12P() -> Type:
  T.Two<AttBlkP, MlpP>

def B12S() -> Type:
  T.Two<AttBlkS, MlpS>

def B12() -> T.Layer<B12P, A(W()), A(W()), B12S>:
  T.Seq(~AttBlkP, ~MlpP, ~A(W()), ~A(W()), ~A(W()), ~AttBlkS, ~MlpS,
    ~AttBlk(), ~Mlp())

def B3P() -> Type:
  T.Two<T.Two<B12P, B12P>, B12P>

def B3S() -> Type:
  T.Two<T.Two<B12S, B12S>, B12S>

def B2() -> T.Layer<T.Two<B12P, B12P>, A(W()), A(W()), T.Two<B12S, B12S>>:
  T.Seq(~B12P, ~B12P, ~A(W()), ~A(W()), ~A(W()), ~B12S, ~B12S, ~B12(), ~B12())

def Body3() -> T.Layer<B3P, A(W()), A(W()), B3S>:
  T.Seq(~T.Two<B12P, B12P>, ~B12P, ~A(W()), ~A(W()), ~A(W()),
    ~T.Two<B12S, B12S>, ~B12S, ~B2(), ~B12())

# ---- the model: 20 features -> 64, three blocks, 4 noise values per patch ----

def EmbP() -> Type:
  TA.Mat<W(), FE()>

def ModelP() -> Type:
  T.Two<T.Two<EmbP, B3P>, TA.Mat<PX(), W()>>

def ModelS() -> Type:
  T.Two<T.Two<TA.Mat<RW(), FE()>, B3S>, TA.Mat<RW(), W()>>

def HeadP() -> Type:
  TA.Mat<PX(), W()>

def HeadS() -> Type:
  TA.Mat<RW(), W()>

def Embed() -> T.Layer<EmbP, TA.Mat<RW(), FE()>, A(W()), TA.Mat<RW(), FE()>>:
  T.LinearNB(RW(), FE(), W(), 0n)

def Head() -> T.Layer<HeadP, A(W()), TA.Mat<RW(), PX()>, HeadS>:
  T.LinearNB(RW(), W(), PX(), 0n)

def Model() -> T.Layer<ModelP, TA.Mat<RW(), FE()>, TA.Mat<RW(), PX()>, ModelS>:
  T.Seq(~T.Two<EmbP, B3P>, ~HeadP, ~TA.Mat<RW(), FE()>, ~A(W()), ~TA.Mat<RW(), PX()>,
    ~T.Two<TA.Mat<RW(), FE()>, B3S>, ~HeadS,
    ~T.Seq(~EmbP, ~B3P, ~TA.Mat<RW(), FE()>, ~A(W()), ~A(W()),
      ~TA.Mat<RW(), FE()>, ~B3S, ~Embed(), ~Body3()),
    ~Head())

def ones_of(+n: Nat, +i: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match i:
    case 0n:
      acc
    case 1n+rest:
      ones_of(n, rest, List.append(&2, F32, acc, Con{1.0, Nil{}}))

def flat_m(m: TA.ML<RW(), PX()>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

def show_y(y: TA.Mat<RW(), PX()>) -> IO(Unit):
  +ys = flat_m(TA.Mat.to_list(RW(), PX(), y))
  +np = T.Layer.count(ModelP(), TA.Mat<RW(), FE()>, TA.Mat<RW(), PX()>(),
    ModelS(), Model())
  do IO<Unit>:
    IO.print(String.append("out len ", Nat.show(List.length(&2, F32, ys))))
    IO.print(String.append("params ", U32.show(np)))
    IO.print(String.append("first ", F32.show(at(ys, 0n))))
    return Unit{}

def show_out(r: T.Fwd<ModelP, TA.Mat<RW(), PX()>, ModelS>) -> IO(Unit):
  match r:
    case T.Fwd{p, y, s}:
      show_y(y)


# ---- the token input: 4 patched pixels + 8 time + 8 position per token ----

def drop64_go(xs: List<&2, F32>, left: Nat) -> List<&2, F32>:
  match xs left:
    case Nil{} _:
      Nil{}
    case _ 0n:
      xs
    case Con{h, t} 1n+rest:
      drop64_go(t, rest)

def drop64(xs: List<&2, F32>) -> List<&2, F32>:
  drop64_go(xs, 64n)

# token k of one digit: rows 2r, 2r+1, columns 2c, 2c+1
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

# every digit of the batch, patched, in order
def patch_all(left: Nat, +xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      patch_all(rest, drop64(xs), List.append(&2, F32, acc, patch(xs)))
    case _:
      acc

# the position embedding: sin/cos of k * f, f = 0.001, 0.01, 0.1, 1
def pos_go(+k: F32, fs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match fs:
    case Nil{}:
      acc
    case Con{+f, t}:
      pos_go(k, t, List.append(&2, F32,
        List.append(&2, F32, acc, Con{F32.sin((k * f : F32)), Nil{}}),
        Con{F32.cos((k * f : F32)), Nil{}}))

def pos_of(+k: Nat) -> List<&2, F32>:
  pos_go(F32.from_nat(k), Con{0.001, Con{0.01, Con{0.1, Con{1.0, Nil{}}}}}, Nil{})

def snoc(xs: List<&2, F32>, +v: F32) -> List<&2, F32>:
  List.append(&2, F32, xs, Con{v, Nil{}})

def cat_go(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      cat_go(t, List.append(&2, F32, acc, Con{h, Nil{}}))

def rows_go(patched: List<&2, F32>, +te: List<&2, F32>, +k: Nat,
  +acc: List<&2, F32>) -> List<&2, F32>:
  match patched:
    case Nil{}:
      acc
    case Con{a, Con{b, Con{c, Con{d, t}}}}:
      r1 = snoc(snoc(snoc(snoc(Nil{}, a), b), c), d)
      r2 = cat_go(te, r1)
      r3 = cat_go(pos_of(k), r2)
      rows_go(t, te, Nat.mod((1n + k : Nat), 16n), List.append(&2, F32, acc, r3))
    case _:
      acc

# the model input for a batch of 64 digits at step t
def tinput(digits: List<&2, F32>, +t: Nat) -> List<&2, F32>:
  rows_go(patch_all(64n, digits, Nil{}), G.temb_of(t, 64n, 4n), 0n, Nil{})


# ---- training: a fixed window of 64 digits, the token input and target ----

def abar(+t: Nat) -> F32:
  at(G.lin_alphas(64n), Nat.sub(t, 1n))


def take_go(left: Nat, xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      match xs:
        case Nil{}:
          acc
        case Con{h, t}:
          take_go(rest, t, List.append(&2, F32, acc, Con{h, Nil{}}))

def mix_go(xs: List<&2, F32>, ys: List<&2, F32>, +a: F32, +b: F32,
  +acc: List<&2, F32>) -> List<&2, F32>:
  match xs ys:
    case Con{h, t} Con{e, u}:
      mix_go(t, u, a, b, List.append(&2, F32, acc,
        Con{((a * h : F32) + (b * e : F32) : F32), Nil{}}))
    case _ _:
      acc

def mse(y: TA.Mat<RW(), PX()>, tgt: TA.Mat<RW(), PX()>) -> T.LossOut<TA.Mat<RW(), PX()>>:
  T.MSE(RW(), PX(), y, tgt)

def cfg() -> T.OptCfg:
  T.CAdam{0.01, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

# the model input for one step: the noised pixels, patched, plus time and position
def step_in(+s: Nat, +t: Nat, +x0: List<&2, F32>) -> TA.Mat<RW(), FE()>:
  +a = F32.sqrt(abar(t))
  +b = F32.sqrt((1.0 - abar(t) : F32))
  +batch = take_go(4096n, x0, Nil{})
  +noise = take_go(4096n, G.noise_of(s, 65536n), Nil{})
  TA.Mat.from_list(RW(), FE(),
    rows_go(patch_all(64n, mix_go(batch, noise, a, b, Nil{}), Nil{}),
      G.temb_of(t, 64n, 4n), 0n, Nil{}))

def step_tgt(+s: Nat) -> TA.Mat<RW(), PX()>:
  TA.Mat.from_list(RW(), PX(),
    patch_all(64n, take_go(4096n, G.noise_of(s, 65536n), Nil{}), Nil{}))

def do_step(+s: Nat, tr: T.Train<ModelP()>, +x0: List<&2, F32>) -> T.Train<ModelP()>:
  +t = Nat.add(Nat.mod(s, 64n), 1n)
  T.Train.step(~ModelP(), ~TA.Mat<RW(), FE()>, ~TA.Mat<RW(), PX()>(), ~ModelS(),
    ~TA.Mat<RW(), PX()>(), ~Model(), ~mse, cfg(), tr,
    step_in(s, t, x0), step_tgt(s))

def should_print(+i: Nat) -> Bool:
  Bool.or(Bool.or(Nat.is_eq(i, 0n), Nat.is_eq(Nat.mod(i, 25n), 0n)),
    Nat.is_eq(i, 199n))

def print_or(sp: Bool, +l: F32) -> IO(Unit):
  match sp:
    case True{}:
      do IO<Unit>:
        IO.print(String.append("loss: ", F32.show(l)))
        return Unit{}
    case False{}:
      do IO<Unit>:
        return Unit{}

def train_go(left: Nat, +i: Nat, +x0: List<&2, F32>,
  tr: T.Train<ModelP()>) -> IO(Unit):
  match left tr:
    case 0n T.Train{p, g, st, p1, p2, +l}:
      do IO<Unit>:
        IO.print(String.append("final:", F32.show(l)))
        return Unit{}
    case 1n+rest T.Train{p, g, st, p1, p2, +l}:
      do IO<Unit>:
        Unit <- print_or(should_print(i), l)
        train_go(rest, (1n + i : Nat), x0,
          do_step(i, T.Train{p, g, st, p1, p2, l}, x0))
    case _ T.Train{p, g, st, p1, p2, l}:
      do IO<Unit>:
        return Unit{}


# ---- the starting weights: random, loaded through the layer's own loader ----

def scale_go(xs: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      scale_go(t, List.append(&2, F32, acc, Con{(0.1 * h : F32), Nil{}}))

def fill_go(left: Nat, xs: List<&2, F32>, +i: U32, arr: Array<F32>) -> Array<F32>:
  match left xs:
    case 0n _:
      arr
    case 1n+rest Con{h, t}:
      fill_go(rest, t, U32.add(i, U32.from_nat(1n)), Array.set(F32, arr, i, h))
    case _ _:
      arr

def un_pa(pa: T.PA<ModelP()>) -> ModelP():
  match pa:
    case T.PA{p, xs, off}:
      p

def load_p(p0: ModelP(), arr: Array<F32>) -> ModelP():
  un_pa(T.Layer.load(ModelP(), TA.Mat<RW(), FE()>, TA.Mat<RW(), PX()>(), ModelS(),
    Model(), p0, arr, U32.from_nat(0n)))

def init_params() -> ModelP():
  +n = T.Layer.count(ModelP(), TA.Mat<RW(), FE()>, TA.Mat<RW(), PX()>(), ModelS(), Model())
  +nn = U32.to_nat(n)
  arr = fill_go(nn, scale_go(G.noise_of(1n, nn), Nil{}), U32.from_nat(0n), Array.new(F32, T.depth(nn), 0.0))
  load_p(T.Layer.zeros(ModelP(), TA.Mat<RW(), FE()>, TA.Mat<RW(), PX()>(), ModelS(),
    Model()), arr)

def run(x0: List<&2, F32>, +n: Nat) -> IO(Unit):
  do IO<Unit>:
    Unit <- train_go(n, 0n, x0,
      T.Train.new(~ModelP(), ~TA.Mat<RW(), FE()>, ~TA.Mat<RW(), PX()>(), ~ModelS(),
        ~Model(), cfg(), init_params()))
    return Unit{}

def Body() -> IO(Unit):
  do IO<Unit>:
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.csv_of(
      "mstz/optdigits", "4f20cc827b763fb69e2a80e18e892e4e11498aec", "optdigits.data", False{}, 0n)
    x0 : List<&2, F32> <- D.digits_of(t)
    Unit <- run(x0, 400n)
    return Unit{}

def main() -> IO(Unit):
  Body()
