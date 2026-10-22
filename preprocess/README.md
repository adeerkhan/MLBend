# mlbend-preprocess

The transformers of scikit-learn, over bend-ml's matrix.

```python
import mlbend-preprocess@0.1.0.0/main.bend as P
```

A transformer takes a matrix and gives a matrix of the same shape. The
matrix is **bend-ml's**: `Mat<r, c>` from `bend-ml-tensor`, the one
bend-ml's autodiff takes, so what comes out of a transformer can be
handed to a model without a copy. MLBend writes no matrix of its own.

| function | what it does |
| --- | --- |
| `standardize(r, c, a)` | every column centred on its mean, scaled by its standard deviation |
| `min_max(r, c, a)` | every column mapped onto [0, 1] |
| `l1_normalize(r, c, a)` | every row divided by the sum of its absolute values |
| `l2_normalize(r, c, a)` | every row divided by the length of its arrow |
| `one_hot(k, n, labels)` | a row of `k` numbers for every label, with a one in the label's column |

`one_hot` is `Nothing` when there are not exactly `n` labels, the way
bend-ml's own `Mat.of` is `Nothing` when the rows do not fit.

## What is proved

- **The one-hot matrix has one row for every label** (`one_hot_count_length`).
  The kernel proves it; the proof is an induction over the labels.
- **The shape of every result** is in the type: `standardize` gives a
  `Mat<r, c>` for a `Mat<r, c>`, and the checker verifies it. That is
  bend-ml's own claim — the dimensions are erased type parameters, and
  the checker verifies them in every operation — so the transformer
  cannot change the shape of what it is given.

## What is only tested

- **The numbers.** F32 is not a real number, so a value cannot be
  proved: the checker treats every F32 operation as an axiom. The
  numbers are checked against scikit-learn instead
  (`reference/test_preprocess.py`): `StandardScaler`, `MinMaxScaler`,
  `Normalizer` and `np.eye`, on the same matrix, to the precision of an
  F32.
- **"The standardised column has mean 0 and variance 1."** This is the
  law the plan would like, and it is not proved here: it is a claim
  about F32 arithmetic. It is the debt of issue #5, and it comes back
  when the numeric tower can carry it.

## The arithmetic

The standardiser is bend-ml's own arithmetic: a column-wise
normalisation is a transpose, then `Mat.layernorm` (which centres a row
and scales it by its own standard deviation), then a transpose back.
The min-max scaler and the normalisers use `Vec.dot`, `Vec.scale` and
`Mat.to_rows`.

The standard deviation is the *population* one (divided by the count),
the same as scikit-learn's. A column whose values are all the same has
no spread; the scaler leaves it alone rather than dividing by zero.
