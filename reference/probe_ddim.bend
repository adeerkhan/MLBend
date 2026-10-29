# The frozen-model probe of reference/test_ddim.py: the two samplers of
# examples/diffusion run with the deterministic starting weights (no
# training), each printed as one line of numbers for the NumPy twin to
# compare. Importing the example brings its whole pipeline along — the
# data, the schedule, the model — and its `main` does not run, because
# this file is the entry point.
import Base
import ../examples/diffusion/2d/main.bend as D

def show_fs_go(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(F32.show(h), String.append(" ", show_fs_go(t)))

def show_ns_go(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(U32.show(U32.from_nat(h)), String.append(" ", show_ns_go(t)))

def show_fs(tag: String, xs: List<&2, F32>) -> String:
  String.append(tag, String.append(" ", show_fs_go(xs)))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(String.append("settings: ", String.append(F32.show(D.eta()),
      String.append(" ", U32.show(U32.from_nat(D.stride()))))))
    IO.print(show_fs("anc:", D.sample_x(D.sample(D.params()))))
    IO.print(show_fs("ddim1:", D.sample_x(D.ddim_sample(D.params(), 1.0, 1n, True{}, 0n))))
    IO.print(show_fs("ddim1s:", D.sample_x(D.ddim_sample(D.params(), 1.0, 1n, True{}, 999n))))
    IO.print(show_fs("ddim0:", D.sample_x(D.ddim_sample(D.params(), 0.0, 4n, False{}, 0n))))
    IO.print(show_fs("ddim0z:", D.sample_x(D.ddim_sample(D.params(), 0.0, 4n, True{}, 0n))))
    IO.print(show_fs("ddim0s:", D.sample_x(D.ddim_sample(D.params(), 0.0, 4n, True{}, 999n))))
    IO.print(show_fs("rungs4:", D.rung_abars(4n)))
    IO.print(String.append("rungts4: ", show_ns_go(D.rung_ts(4n))))
    IO.print(show_fs("rungs1:", D.rung_abars(1n)))
    IO.print(String.append("rungts1: ", show_ns_go(D.rung_ts(1n))))
    return Unit{}
