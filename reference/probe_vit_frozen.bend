# The frozen-model probe of the ViT (issue #16, milestone M2): no training,
# just the starting weights, so the NumPy twin can compare the forward and
# the sampler element by element. Each line is one flat list for the twin.
#
#   eps    the model's noise at t = 64 on the starting cloud
#   step   one ancestral step from t = 64 (the clamp and the posterior mean)
#   ddim   the 16-step DDIM cloud, eta = 0
#
# `init_params` is deterministic (noise_of(1) scaled by 0.1, in the loader's
# order), so each output can consume its own copy of the weights.
import Base
import bendtorch@1.0.1.0/main.bend as T
import bend-ml-tensor-array@0.1.14.0/main.bend as TA
import mlbend-generative@0.1.1.0/main.bend as G
import ./probe_vit_model.bend as M

def show_go(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_go(t)))

def show(tag: String, xs: List<&2, F32>) -> String:
  String.append(tag, String.append(" ", show_go(xs)))

def start() -> List<&2, F32>:
  M.patch_all(64n, M.take_go(4096n, G.noise_of(2000000n, 65536n), Nil{}), Nil{})

def show_ve(ve: M.VE) -> IO(Unit):
  match ve:
    case M.VE{p, eps}:
      do IO<Unit>:
        IO.print(show("eps:", eps))
        return Unit{}

def show_vs(st: M.VS, tag: String) -> IO(Unit):
  match st:
    case M.VS{p, x}:
      do IO<Unit>:
        IO.print(show(tag, x))
        return Unit{}

def show_pl(pl: T.PL<M.ModelP()>) -> IO(Unit):
  match pl:
    case T.PL{p, xs}:
      do IO<Unit>:
        IO.print(show("weights:", xs))
        return Unit{}

def dump(p: M.ModelP()) -> IO(Unit):
  show_pl(T.Layer.dump(M.ModelP(), TA.Mat<M.RW(), M.FE()>(), TA.Mat<M.RW(), M.PX()>(),
    M.ModelS(), M.Model(), p))

def head20(xs: List<&2, F32>, left: Nat, +acc: List<&2, F32>) -> List<&2, F32>:
  match xs left:
    case Con{h, t} 1n+rest:
      head20(t, rest, List.append(&2, F32, acc, Con{h, Nil{}}))
    case _ _:
      acc

def row20() -> List<&2, F32>:
  head20(M.rows_go(start(), G.temb_of(64n, 64n, 4n), 0n, Nil{}), 20n, Nil{})

def flat_w(m: TA.ML<M.RW(), M.W()>) -> List<&2, F32>:
  match m:
    case TA.ML{mat, l}:
      l

def show_fwd64(f: T.Fwd<M.EmbP, M.A(M.W()), TA.Mat<M.RW(), M.FE()>>) -> IO(Unit):
  match f:
    case T.Fwd{p, y, s}:
      +ys = flat_w(TA.Mat.to_list(M.RW(), M.W(), y))
      do IO<Unit>:
        IO.print(show("emb0:", head20(ys, 64n, Nil{})))
        return Unit{}

def emb_out(p: M.ModelP()) -> IO(Unit):
  match p:
    case T.Two{T.Two{emb, body}, hd}:
      inp = TA.Mat.from_list(M.RW(), M.FE(),
        M.rows_go(start(), G.temb_of(64n, 64n, 4n), 0n, Nil{}))
      show_fwd64(T.Layer.fwd(M.EmbP, TA.Mat<M.RW(), M.FE()>(), M.A(M.W()),
        TA.Mat<M.RW(), M.FE()>(), M.Embed(), emb, inp))

def emb_f(emb: M.EmbP, inp: TA.Mat<M.RW(), M.FE()>) -> T.Fwd<M.EmbP, M.A(M.W()), TA.Mat<M.RW(), M.FE()>>:
  T.Layer.fwd(M.EmbP, TA.Mat<M.RW(), M.FE()>(), M.A(M.W()), TA.Mat<M.RW(), M.FE()>(),
    M.Embed(), emb, inp)

def then_body(body: M.B3P, f: T.Fwd<M.EmbP, M.A(M.W()), TA.Mat<M.RW(), M.FE()>>) -> T.Fwd<M.B3P, M.A(M.W()), M.B3S>:
  match f:
    case T.Fwd{e2, y, s2}:
      T.Layer.fwd(M.B3P, M.A(M.W()), M.A(M.W()), M.B3S(), M.Body3(), body, y)

def show_body(f: T.Fwd<M.B3P, M.A(M.W()), M.B3S>) -> IO(Unit):
  match f:
    case T.Fwd{p, y, s}:
      +ys = flat_w(TA.Mat.to_list(M.RW(), M.W(), y))
      do IO<Unit>:
        IO.print(show("body0:", head20(ys, 64n, Nil{})))
        return Unit{}

def body_out(p: M.ModelP()) -> IO(Unit):
  match p:
    case T.Two{T.Two{emb, body}, hd}:
      inp = TA.Mat.from_list(M.RW(), M.FE(),
        M.rows_go(start(), G.temb_of(64n, 64n, 4n), 0n, Nil{}))
      show_body(then_body(body, emb_f(emb, inp)))

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- dump(M.init_params())
    Unit <- emb_out(M.init_params())
    Unit <- body_out(M.init_params())
    do IO<Unit>:
      IO.print(show("row:", row20()))
      return Unit{}
    Unit <- show_ve(M.predict(M.init_params(), 64n, start()))
    Unit <- show_vs(M.rev_step(64n, M.VS{M.init_params(), start()}), "step:")
    Unit <- show_vs(M.ddim_sample(M.init_params()), "ddim:")
    return Unit{}
