# mlbend-datasets

The datasets of scikit-learn, fetched from HuggingFace at run time.

```python
import mlbend-datasets@0.1.0.0/main.bend as Datasets

def main() -> IO(Unit):
  do IO<Unit>:
    d : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.iris()
    ...
```

`iris()` gives the iris dataset: 150 rows of four features and the class
of each, as a `Mat<150n, 4n>` and a `List<&2, Nat>` of the labels. It is
the file the dataset's own repository holds, which holds the same numbers
scikit-learn ships.

The effect is at the edge and the data that comes back is a value, so an
estimator over it stays pure and a run is reproducible.
