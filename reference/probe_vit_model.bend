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

def Body() -> IO(Unit):
  do IO<Unit>:
    IO.print("model compiles")
    return Unit{}

def main() -> IO(Unit):
  Body()
