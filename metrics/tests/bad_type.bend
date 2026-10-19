import Base
import ../main.bend as M

# MUST FAIL: the targets of a regression are F32 and the labels of a
# classification are Nat. The checker refuses the mix.

def mixed() -> Result<&1, &1, M.Bad, F32>:
  M.mse([1.0, 2.0], [0n, 1n])
