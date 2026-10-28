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

def main() -> IO(Unit):
  do IO<Unit>:
    Unit <- show_ve(M.predict(M.init_params(), 64n, start()))
    Unit <- show_vs(M.rev_step(64n, M.VS{M.init_params(), start()}), "step:")
    Unit <- show_vs(M.ddim_sample(M.init_params()), "ddim:")
    return Unit{}
