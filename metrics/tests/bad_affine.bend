import Base
import ../main.bend as M

# MUST FAIL: a list is affine, so it cannot be given to a metric twice:
# the data is consumed, not copied. The checker refuses it.

def twice(xs: List<&2, F32>) -> Result<&1, &1, M.Bad, F32>:
  M.mse(xs, xs)
