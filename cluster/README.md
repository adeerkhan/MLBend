# mlbend-cluster

k-means, functional: the centres after a fixed number of rounds, and the
assignment of every row to its nearest centre.

```python
import mlbend-cluster@0.1.0.0/main.bend as Cluster
```

A row is a `List<&2, F32>` and a centre is a row, so nothing here finds a
column by its place. `fit` takes the first `k` rows as the starting
centres and runs a fixed number of rounds: each round assigns every row
to its nearest centre and moves each centre to the mean of the rows that
chose it. A fixed number of rounds is a function of the arguments — there
is no loop waiting for the centres to settle — so the same rows give the
same centres and a run is reproducible.

| function | what it does |
| --- | --- |
| `fit(k, rounds, rows)` | the `k` centres after `rounds` rounds, and `k` beside them |
| `assign(centres, rows)` | the index of the nearest centre, for every row |
| `inertia(centres, rows, labels)` | the sum of the squared distances from every row to its centre |
| `sizes(labels, k)` | the number of rows of every cluster |
| `distance(a, b)` | the squared distance of two rows |
| `nearest(centres, row)` | the index of the nearest centre, or 0 when there is none |

The model is a value:

```python
type Model is Data:
  Model{centres: List<&2, List<&2, F32>>, k: Nat}
```

## What is proved

`cluster_covers`: every row gets exactly one cluster. The assignment
holds one label per row, so a row cannot be in two clusters and cannot be
left out. It is the same shape of law as `mlbend-split`'s `split_covers`,
and the proof is a recursion over the rows.

## What is tested

The labels, the cluster sizes, the inertia and the centres of six fixed
rows, against the same algorithm in NumPy (`reference/test_cluster.py`),
and the wine dataset through `examples/kmeans` against
`reference/test_kmeans.py`. `F32` is not a real number, so no value is
proved, only tested.

## The limits, stated

- The centres start at the first `k` rows. That is a starting rule, not an
  optimisation: another start can find another local minimum.
- The number of rounds is an argument. There is no convergence test.
- A cluster that loses every row keeps an empty centre, so the indices of
  the clusters never move under a model that is already fitted.
