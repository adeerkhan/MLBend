# The frozen-model probe of the digits sampler (issues #12 and #13): the
# samplers run with the deterministic starting weights (no training, so
# no optimizer divergence), each cloud printed as one line of 4096 numbers
# for the NumPy twin to compare element by element.
#
#   eps           the model's output for t = 64 alone: the probe's step
#                 gate feeds this to both sides, so what is gated is the
#                 sampler's arithmetic and the array's order — not the
#                 matmul (BendTorch's dot accumulates in four partial
#                 sums, numpy's BLAS does not, and the two differ by
#                 several ulps; reported, not gated)
#   step          ONE ancestral step from the starting noise at t = 64
#                 with that same eps — the tight gate: order is visible
#                 here at float32 rounding, with 64 steps of amplification
#                 not yet paid
#   frozen        the ancestral loop over all 64 rungs
#   frozen_ddim   DDIM at the report's settings (eta = 0, every 4th rung,
#                 no draw)
#   frozen_ddim2  the same walk with the draw turned on and another salt:
#                 at eta = 0 the step multiplies sigma = 0 into whatever
#                 it reads, so the cloud may not move at all
#
# The element-wise comparison is the point: the samplers' statistics
# (mean, std over the whole flat array) are order-invariant, so they
# cannot see the reversed-array issue #13 was about — only the reads
# can. Importing the example brings its pipeline along and no dataset is
# fetched, because the fetch lives in `main` and this file is the entry
# point.
import Base
import mlbend-generative@0.1.1.0/main.bend as G
import ../examples/diffusion_digits/main.bend as D

def show_fs_go(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_fs_go(t)))

def show_fs(tag: String, xs: List<&2, F32>) -> String:
  String.append(tag, String.append(" ", show_fs_go(xs)))

def eps_of(pe: D.PE) -> List<&2, F32>:
  match pe:
    case D.PE{p, eps}:
      eps

def main() -> IO(Unit):
  # bound before the `do`: an untyped let inside a `do` is refused, and
  # the two samplers that start from this same noise need it twice
  +x = G.noise_of(2000000n, 4096n)
  do IO<Unit>:
    IO.print(show_fs("eps:", eps_of(D.predict(D.params(), 64n, x))))
    IO.print(show_fs("step:",
      D.sample_x(D.reverse_step(64n, D.PS{D.params(), x}))))
    IO.print(show_fs("frozen:", D.sample_x(D.sample(D.params()))))
    IO.print(show_fs("frozen_ddim:",
      D.sample_x(D.ddim_sample(D.params(), D.eta(), D.stride(), False{}, 0n))))
    IO.print(show_fs("frozen_ddim2:",
      D.sample_x(D.ddim_sample(D.params(), D.eta(), D.stride(), True{}, 7n))))
    return Unit{}
