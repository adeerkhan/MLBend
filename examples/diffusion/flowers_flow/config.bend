# The 64x64 flow's shape, fixed before the compile: every number here is a
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
# before the build and not at runtime. This file is the `hi` preset.

def B() -> Nat:
  1n

# one token per 64-pixel row, so the image is read in raster order
def RES() -> Nat:
  256n

def DATA() -> String:
  "/home/adeerkhan/.cache/bend-lab/flowers/flowers256-512-gray.u8"

# one token per row of pixels, so RES tokens per image
def TK() -> Nat:
  RES()

def W() -> Nat:
  128n

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
  2000n

def EULER() -> Nat:
  32n
