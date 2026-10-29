# A ViT denoiser for the digits, trained and sampled in Bend (issue #16, M1).
#
#   bendenv run -- bend examples/diffusion_vit/main.bend
#
# 64 digits a step, each 8x8 image cut into 16 tokens of 2x2 patches, 20
# features per token (4 pixels, 8 of time, 8 of position), width 64, one
# head, three blocks, one 4-value head per patch. Every layer is BendTorch's:
# `LinearNB`, `LayerNorm`, `Attention`, `GELU`, `Residual`, `Seq` — no tensor,
# no autodiff and no optimizer of ours. The task is the examples' usual one:
# predict the noise of the linear schedule of mlbend-generative, then sample
# by the 16-step DDIM walk with the clamped start.
#
# Two facts cost the most iterations, and both are real:
#
#   ISSUE-010  the pre-LN block is `Residual(x, Seq(ln, att))`, i.e.
#              x + att(ln(x)). Written `Seq(ln, Residual(att))` the residual
#              is taken on the *normalized* value — ln(x) + att(ln(x)) — and
#              the stream loses its identity path; the model then trains to a
#              plateau. The twin disagreed with Bend at O(1) until this was
#              found, and the probe's ladder (blk0) named it: 7.5e-08 with the
#              pre-LN wiring against 0.64 without.
#   ISSUE-011  BendTorch's `Attention` is causal by construction: query i
#              reads keys 0..i (`qrow` builds i+1 scores, `prow` softmaxes
#              exactly those) and the layer takes no mask. The 16 tokens are
#              therefore attended as a causal sequence; a bidirectional mask
#              is not expressible in this package today.
import Base
import bendtorch@1.0.1.0/main.bend as T
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import mlbend-datasets@0.1.6.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf
import ../diffusion_digits/main.bend as D

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

# The pre-LN block: the norm goes INSIDE the residual, so the path x is kept:
# x + att(ln(x)). Written the other way round -- Seq(ln, Residual(att)) -- the
# residual is taken on the normalized value, ln(x) + att(ln(x)), and the stream
# loses its identity path (ISSUE-010 in issues-hit.md).
def AttBlkP() -> Type:
  T.Two<T.Two<Unit, T.Two<T.LN<W()>, AttP>>, Unit>

def AttBlkS() -> Type:
  T.Two<T.Two<Unit, T.Two<A(W()), AttS>>, Unit>

def AttIn() -> T.Layer<T.Two<T.LN<W()>, AttP>, A(W()), A(W()), T.Two<A(W()), AttS>>:
  T.Seq(~T.LN<W()>, ~AttP, ~A(W()), ~A(W()), ~A(W()), ~A(W()), ~AttS,
    ~T.LayerNorm(RW(), W(), 0.00001), ~Att())

def AttBlk() -> T.Layer<AttBlkP, A(W()), A(W()), AttBlkS>:
  T.Residual(RW(), W(), T.Two<T.LN<W()>, AttP>, T.Two<A(W()), AttS>, AttIn())

# ---- the MLP half of a block: norm -> 64->128->64 -> residual ----

def MIP() -> Type:
  T.Two<T.Two<M(HI(), W()), Unit>, M(W(), HI())>

def MIS() -> Type:
  T.Two<T.Two<A(W()), A(HI())>, A(HI())>

def MlpP() -> Type:
  T.Two<T.Two<Unit, T.Two<T.LN<W()>, MIP>>, Unit>

def MlpS() -> Type:
  T.Two<T.Two<Unit, T.Two<A(W()), MIS>>, Unit>

def MlpHalf() -> T.Layer<T.Two<M(HI(), W()), Unit>, A(W()), A(HI()), T.Two<A(W()), A(HI())>>:
  T.Seq(~M(HI(), W()), ~Unit, ~A(W()), ~A(HI()), ~A(HI()), ~A(W()), ~A(HI()),
    ~T.LinearNB(RW(), W(), HI(), 0n), ~T.GELU(RW(), HI()))

def MlpInner() -> T.Layer<MIP, A(W()), A(W()), MIS>:
  T.Seq(~T.Two<M(HI(), W()), Unit>, ~M(W(), HI()), ~A(W()), ~A(HI()), ~A(W()),
    ~T.Two<A(W()), A(HI())>, ~A(HI()), ~MlpHalf(), ~T.LinearNB(RW(), HI(), W(), 0n))

def MlpIn() -> T.Layer<T.Two<T.LN<W()>, MIP>, A(W()), A(W()), T.Two<A(W()), MIS>>:
  T.Seq(~T.LN<W()>, ~MIP, ~A(W()), ~A(W()), ~A(W()), ~A(W()), ~MIS,
    ~T.LayerNorm(RW(), W(), 0.00001), ~MlpInner())

def Mlp() -> T.Layer<MlpP, A(W()), A(W()), MlpS>:
  T.Residual(RW(), W(), T.Two<T.LN<W()>, MIP>, T.Two<A(W()), MIS>, MlpIn())

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

def HP() -> Type:
  T.Two<T.LN<W()>, TA.Mat<PX(), W()>>

def HS() -> Type:
  T.Two<A(W()), A(W())>

def ModelP() -> Type:
  T.Two<T.Two<EmbP, B3P>, HP>

def ModelS() -> Type:
  T.Two<T.Two<TA.Mat<RW(), FE()>, B3S>, HS>

def HeadP() -> Type:
  TA.Mat<PX(), W()>

def HeadS() -> Type:
  TA.Mat<RW(), W()>

def Embed() -> T.Layer<EmbP, TA.Mat<RW(), FE()>, A(W()), TA.Mat<RW(), FE()>>:
  T.LinearNB(RW(), FE(), W(), 0n)

def Head() -> T.Layer<HeadP, A(W()), TA.Mat<RW(), PX()>, HeadS>:
  T.LinearNB(RW(), W(), PX(), 0n)

def HeadN() -> T.Layer<HP, A(W()), TA.Mat<RW(), PX()>, HS>:
  T.Seq(~T.LN<W()>, ~TA.Mat<PX(), W()>, ~A(W()), ~A(W()), ~TA.Mat<RW(), PX()>(),
    ~A(W()), ~A(W()), ~T.LayerNorm(RW(), W(), 0.00001), ~Head())

def Model() -> T.Layer<ModelP, TA.Mat<RW(), FE()>, TA.Mat<RW(), PX()>, ModelS>:
  T.Seq(~T.Two<EmbP, B3P>, ~HP, ~TA.Mat<RW(), FE()>, ~A(W()), ~TA.Mat<RW(), PX()>,
    ~T.Two<TA.Mat<RW(), FE()>, B3S>, ~HS,
    ~T.Seq(~EmbP, ~B3P, ~TA.Mat<RW(), FE()>, ~A(W()), ~A(W()),
      ~TA.Mat<RW(), FE()>, ~B3S, ~Embed(), ~Body3()),
    ~HeadN())

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

def abar_below(+t: Nat) -> F32:
  Bool.pick(F32, Nat.is_eq(t, 1n), 1.0, at(G.lin_alphas(64n), Nat.sub(t, 2n)))


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

# lr 3e-3, not the digits MLP's 1e-2: the ViT is deeper and the higher rate
# costs it legibility. Measured in torch on the same architecture (causal,
# one head, fixed position features, 3000 steps, batch 64): 1e-2 -> 0.721,
# 3e-3 -> 0.782, and 3e-3 with batch 128 -> 0.774. (ISSUE-012.)
def cfg() -> T.OptCfg:
  T.CAdam{0.003, 0.9, 0.999, 0.00000001, 0.0, 0.1, 0.001}

# the model input for one step: the noised pixels, patched, plus time and position
# the 16 chunks of 64 digits, one per step, in a fixed rotation: a
# deterministic minibatch that still sees all 1024 digits (the first 64 for
# every step would be a 16x smaller task, and it trains to a loss an order of
# magnitude below the arm that samples from all of them)
def drop_go(left: Nat, xs: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      xs
    case 1n+rest:
      drop_go(rest, drop64(xs))

def batch_of(+s: Nat, x0: List<&2, F32>) -> List<&2, F32>:
  take_go(4096n, drop_go(Nat.mul(Nat.mod(s, 16n), 64n), x0), Nil{})

def step_in(+s: Nat, +t: Nat, +x0: List<&2, F32>) -> TA.Mat<RW(), FE()>:
  +a = F32.sqrt(abar(t))
  +b = F32.sqrt((1.0 - abar(t) : F32))
  +batch = batch_of(s, x0)
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

# `l` is the loss of the step before the call, so i = 1 carries step 0's loss —
# printing at i = 0 prints the record's initial 0.0 instead (which is how the
# first loss went missing until 2026-10-29).
def should_print(+i: Nat) -> Bool:
  Bool.or(Nat.is_eq(Nat.mod(i, 25n), 1n), Nat.is_eq(i, 199n))

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
  tr: T.Train<ModelP()>) -> IO(ModelP()):
  match left tr:
    case 0n T.Train{p, g, st, p1, p2, +l}:
      do IO<ModelP()>:
        IO.print(String.append("final_loss: ", F32.show(l)))
        return p
    case 1n+rest T.Train{p, g, st, p1, p2, +l}:
      do IO<ModelP()>:
        Unit <- print_or(should_print(i), l)
        train_go(rest, (1n + i : Nat), x0,
          do_step(i, T.Train{p, g, st, p1, p2, l}, x0))
    case _ T.Train{p, g, st, p1, p2, l}:
      do IO<ModelP()>:
        return p


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


# ---- the sampler: the same token-ordered cloud, the clamped start ----

type VS is Type:
  VS{p: ModelP(), x: List<&2, F32>}

type VE is Type:
  VE{p: ModelP(), eps: List<&2, F32>}

def flat_m2(m: TA.ML<RW(), PX()>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

def eps_of(f: T.Fwd<ModelP, TA.Mat<RW(), PX()>, ModelS>) -> VE:
  match f:
    case T.Fwd{p, y, s}:
      VE{p, flat_m2(TA.Mat.to_list(RW(), PX(), y))}

def predict(p: ModelP(), +t: Nat, +x: List<&2, F32>) -> VE:
  eps_of(T.Layer.fwd(ModelP(), TA.Mat<RW(), FE()>, TA.Mat<RW(), PX()>(), ModelS(),
    Model(), p,
    TA.Mat.from_list(RW(), FE(), rows_go(x, G.temb_of(t, 64n, 4n), 0n, Nil{}))))

# the clamped start, then the posterior mean; sigma outside the scale
def rev_elem(+c1: F32, +c2: F32, +sa: F32, +sm: F32, +sig: F32,
  +xv: F32, +eps: F32, +z: F32) -> F32:
  x0 = ((xv - (sm * eps : F32) : F32) / sa : F32)
  xc = F32.clamp(x0, 0.0, 1.0)
  (((c1 * xc : F32) + (c2 * xv : F32) : F32) + (sig * z : F32) : F32)

def rev_go(+c1: F32, +c2: F32, +sa: F32, +sm: F32, +sig: F32,
  x: List<&2, F32>, eps: List<&2, F32>, z: List<&2, F32>, left: Nat,
  +acc: List<&2, F32>) -> List<&2, F32>:
  match x eps z left:
    case Con{xv, xt} Con{ev, et} Con{zv, zt} 1n+rest:
      rev_go(c1, c2, sa, sm, sig, xt, et, zt, rest,
        Con{rev_elem(c1, c2, sa, sm, sig, xv, ev, zv), acc})
    case _ _ _ _:
      acc

def rev_pe(ve: VE, +c1: F32, +c2: F32, +sa: F32, +sm: F32, +sig: F32,
  +z: List<&2, F32>, +x: List<&2, F32>) -> VS:
  match ve:
    case VE{p2, eps}:
      VS{p2, List.reverse(&2, F32, rev_go(c1, c2, sa, sm, sig, x, eps, z, 4096n, Nil{}))}

def rev_step(+t: Nat, st: VS) -> VS:
  match st:
    case VS{p, +x}:
      +beta = G.lin_beta_at(t, 64n)
      +abar_t = abar(t)
      +bar_prev = abar_below(t)
      +c1 = (beta * (F32.sqrt(bar_prev) / (1.0 - abar_t) : F32) : F32)
      +c2 = (F32.sqrt((1.0 - beta : F32)) * ((1.0 - bar_prev : F32) / (1.0 - abar_t : F32)) : F32)
      +sa = F32.sqrt(abar_t)
      +sm = F32.sqrt((1.0 - abar_t : F32))
      +sig = F32.sqrt((beta * ((1.0 - bar_prev : F32) / (1.0 - abar_t : F32)) : F32))
      rev_pe(predict(p, t, x), c1, c2, sa, sm, sig,
        G.noise_of((1000000n + t : Nat), 4096n), x)

def sample_go(t: Nat, st: VS) -> VS:
  match t:
    case 0n:
      st
    case 1n++rest:
      sample_go(rest, rev_step(t, st))

# the cloud starts in token order and stays there
def sample(p: ModelP()) -> VS:
  sample_go(64n, VS{p, patch_all(64n, take_go(4096n, G.noise_of(2000000n, 65536n), Nil{}), Nil{})})

# pixel order for the grid and the legibility statistic
def unpatch1(left: Nat, +t: List<&2, F32>, +i: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      +k = (Nat.add(Nat.mul(4n, Nat.div(Nat.div(i, 8n), 2n)),
        Nat.div(Nat.mod(i, 8n), 2n)) : Nat)
      +d = (Nat.add(Nat.mul(2n, Nat.mod(Nat.div(i, 8n), 2n)), Nat.mod(i, 2n)) : Nat)
      unpatch1(rest, t, (1n + i : Nat),
        List.append(&2, F32, acc, Con{at(t, Nat.add(Nat.mul(4n, k), d)), Nil{}}))
    case _:
      acc

def unpatch_all(left: Nat, +t: List<&2, F32>, +acc: List<&2, F32>) -> List<&2, F32>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      unpatch_all(rest, drop64(t), List.append(&2, F32, acc, unpatch1(64n, t, 0n, Nil{})))
    case _:
      acc

def show_cloud(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_cloud(t)))

def show_vs(st: VS) -> IO(Unit):
  match st:
    case VS{p2, x}:
      +px = unpatch_all(64n, x, Nil{})
      do IO<Unit>:
        IO.print(String.append("samples: ", show_cloud(px)))
        return Unit{}

def sample_report(p: ModelP()) -> IO(Unit):
  show_vs(sample(p))


# ---- DDIM: every 4th rung, eta = 0, the start clamped and eps re-derived ----

def clip_eps_go(left: Nat, x: List<&2, F32>, eps: List<&2, F32>, +sa: F32, +sm: F32,
  +acc: List<&2, F32>) -> List<&2, F32>:
  match left x eps:
    case 1n+rest Con{+xv, xt} Con{ev, et}:
      x0 = ((xv - (sm * ev : F32) : F32) / sa : F32)
      xc = F32.clamp(x0, 0.0, 1.0)
      clip_eps_go(rest, xt, et, sa, sm,
        Con{((xv - (sa * xc : F32) : F32) / sm : F32), acc})
    case _ _ _:
      acc

def clip_eps(+abar_now: F32, +x: List<&2, F32>, +eps: List<&2, F32>) -> List<&2, F32>:
  +sa = F32.sqrt(abar_now)
  +sm = F32.sqrt((1.0 - abar_now : F32))
  List.reverse(&2, F32, clip_eps_go(4096n, x, eps, sa, sm, Nil{}))

def rung_abars(+k: Nat) -> List<&2, F32>:
  List.reverse(&2, F32, G.strided_from(F32, G.lin_alphas(64n), k, Nat.mod(63n, k)))

def rung_ts(+k: Nat) -> List<&2, Nat>:
  List.reverse(&2, Nat, G.strided_from(Nat, List.range(64n), k, Nat.mod(63n, k)))

def head_or_one(xs: List<&2, F32>) -> F32:
  match xs:
    case Nil{}:
      1.0
    case Con{h, t}:
      h

def ddim_pe(ve: VE, +a: F32, +ar: List<&2, F32>, +x: List<&2, F32>) -> VS:
  match ve:
    case VE{p2, eps}:
      VS{p2, G.ddim(G.step_of(a, head_or_one(ar), 0.0), x, clip_eps(a, x, eps), Nil{})}

def ddim_go(abars: List<&2, F32>, ts: List<&2, Nat>, st: VS) -> VS:
  match abars:
    case Nil{}:
      st
    case Con{a, +ar}:
      match ts:
        case Nil{}:
          st
        case Con{t, tr}:
          match st:
            case VS{p, +x}:
              +now = (1n + t : Nat)
              ddim_go(ar, tr, ddim_pe(predict(p, now, x), a, ar, x))

def ddim_sample(p: ModelP()) -> VS:
  ddim_go(rung_abars(4n), rung_ts(4n), VS{p,
    patch_all(64n, take_go(4096n, G.noise_of(2000000n, 65536n), Nil{}), Nil{})})

def print_cloud(st: VS, tag: String) -> IO(ModelP()):
  match st:
    case VS{p, x}:
      +px = unpatch_all(64n, x, Nil{})
      do IO<ModelP()>:
        IO.print(String.append("sample_mean: ", F32.show(D.mean_of(px, 4096n))))
        IO.print(String.append("sample_std: ", F32.show(D.std_of(px, 4096n))))
        IO.print(String.append(tag, String.append(" ", show_cloud(px))))
        return p

# Both walks from the one trained model, on the one starting cloud: the
# 16-step DDIM cloud and the 64-step ancestral one, so a comparison holds
# the model and the noise fixed and varies only the sampler.
def run(+x0: List<&2, F32>, +n: Nat) -> IO(Unit):
  do IO<Unit>:
    p : ModelP() <- train_go(n, 0n, x0,
      T.Train.new(~ModelP(), ~TA.Mat<RW(), FE()>(), ~TA.Mat<RW(), PX()>(), ~ModelS(),
        ~Model(), cfg(), init_params()))
    Unit <- IO.print(String.append("data_mean: ", F32.show(D.mean_of(x0, 65536n))))
    p1 : ModelP() <- print_cloud(ddim_sample(p), "samples:")
    p2 : ModelP() <- print_cloud(sample(p1), "samples_ddpm:")
    return Unit{}

def Body() -> IO(Unit):
  do IO<Unit>:
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.csv_of(
      "mstz/optdigits", "4f20cc827b763fb69e2a80e18e892e4e11498aec", "optdigits.data", False{}, 0n)
    x0 : List<&2, F32> <- D.digits_of(t)
    Unit <- run(x0, 3000n)
    return Unit{}

def main() -> IO(Unit):
  Body()
