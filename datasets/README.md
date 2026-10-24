# mlbend-datasets

The datasets of scikit-learn, fetched from HuggingFace, or read from a
local cache when `MLBEND_DATA` is set.

```python
import mlbend-datasets@0.1.5.0/main.bend as Datasets

def main() -> IO(Unit):
  do IO<Unit>:
    ...
```

A dataset comes back as a `Table`: the rows of the features and the
labels beside them.

Version 0.1.5.0 is published at
`0x508fc33b44130576819939e645eba7da` (tag `mlbend-datasets-v0.1.5.0`).

```python
type Table is Data:
  Table{rows: List<&2, List<&2, F32>>, labels: List<&2, F32>}
```

The labels are numbers. A classification label is the index of the
class, a regression label is the value, and the model decides which it
is.

Four datasets come from the `mstz` mirrors on HuggingFace, which hold
the UCI files as they are:

| function | rows | features | label |
| --- | --- | --- | --- |
| `iris()` | 150 | 4 | the species, as 0, 1 or 2 |
| `wine()` | 6497 | 11 | the quality, from 3 to 8 |
| `cancer()` | 683 | 9 | `is_cancer`, 0 or 1 |
| `heart()` | 299 | 12 | `is_dead`, 0 or 1 |

One loader reads them all: the file is fetched over the hub client, the
header is dropped, every cell is read as a number (`True` is 1 and
`False` is 0) and the label is the column before the ones that are left
out at the end of the row. A dataset with a different shape is one call
away:

```python
def csv_of(repo: String, rev: String, path: String, header: Bool, drop: Nat) ->
  IO(Result<&1, &1, Hf.Err, Table>)
```

`drop` is the number of columns to leave out at the end of a row. The
wine file ends with `quality,is_red`, so `wine()` asks for one column
to be left out and the quality is the label; `is_red` is not a feature.

The effect is at the edge and the data that comes back is a value, so an
estimator over it stays pure and a run is reproducible.

## The local cache

Set `MLBEND_DATA` to a directory laid out as `<repo>/<path>` — for example
`mstz/iris/iris.data` — and the loader reads the file from there instead of
the network. The experiment runner fills that directory with the exact bytes
it hashes, so a cached task measures compute and not the fetch. Without the
variable, or when the file is absent, the loader falls back to the hub.
