#!/usr/bin/env python3
"""Write examples/diffusion/flowers_flow/config.bend for a preset.

  scripts/flow-config.py --preset tiny
  scripts/flow-config.py --show

A Bend program's shapes are type-level constants, so choosing a smaller model is
choosing a different program to compile, not a runtime flag. This writes the
config module the example imports; `make doctor` says which presets the machine
can afford, and `make flow64 PRESET=...` calls it.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "examples/diffusion/flowers_flow/config.bend")

# blocks is 1 for every preset: the example's model composition is three blocks
# only in `full`, and no machine measured here builds that. See config.bend.
PRESETS = {
    "tiny":   {"B": 1, "W": 64,  "STEPS": 200,  "EULER": 32},
    "small":  {"B": 1, "W": 96,  "STEPS": 2000, "EULER": 32},
    "medium": {"B": 2, "W": 128, "STEPS": 2000, "EULER": 32},
    "full":   {"B": 8, "W": 128, "STEPS": 2000, "EULER": 32},
}

HEAD = """# The 64x64 flow's shape, fixed before the compile: every number here is a
# type-level constant, so a bigger model is a different program.
#
#   make flow64 PRESET=tiny        one image, 64 wide, one block
#   make flow64 PRESET=small       one image, 96 wide, one block
#   make flow64 PRESET=medium      two images, 128 wide, one block
#   make flow64 PRESET=full        eight images, 128 wide, three blocks
#
# `scripts/flow-config.py --preset <name>` writes this file; `make doctor` says
# which presets fit the machine. The compile is what costs memory, not the step:
# `bend -o` reached 4.3 GB on the 3-block model's data half alone and passed
# 8 GB once the training step was added, which is why the preset is chosen
# before the build and not at runtime. This file is the `{name}` preset.
"""

BODY = """
def B() -> Nat:
  {B}n

# one token per 64-pixel row, so the image is read in raster order
def TK() -> Nat:
  64n

def W() -> Nat:
  {W}n

def HI() -> Nat:
  Nat.mul(W(), 2n)

def PIL() -> Nat:
  Nat.mul(TK(), TK())

def NIMG() -> Nat:
  512n

def RW() -> Nat:
  Nat.mul(B(), TK())

def BATCH() -> Nat:
  Nat.mul(B(), PIL())

def PX() -> Nat:
  TK()

# 64 pixels, 8 time features, 8 position features
def FE() -> Nat:
  Nat.add(PX(), 16n)

def STEPS() -> Nat:
  {STEPS}n

def EULER() -> Nat:
  {EULER}n
"""


def main():
    args = sys.argv[1:]
    name = "tiny"
    show = False
    if "--show" in args:
        show = True
    if "--preset" in args:
        name = args[args.index("--preset") + 1]
    if name not in PRESETS:
        raise SystemExit(f"preset {name!r} is not one of {', '.join(PRESETS)}")
    p = PRESETS[name]
    text = HEAD.format(name=name) + BODY.format(**p)
    if show:
        print(text)
        return
    open(OUT, "w").write(text)
    print(f"flow-config: {name}  images/step {p['B']}  width {p['W']}  "
          f"rows {p['B'] * 64}  -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
