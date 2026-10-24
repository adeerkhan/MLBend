# k-means over the wine dataset, fetched from HuggingFace at run time.
#
#   bendenv run -- bend examples/kmeans/main.bend
#
# The clustering is `mlbend-cluster`'s: the centres start at the first
# three rows, and each of the ten rounds assigns every row to its nearest
# centre and moves the centre to the mean of its rows. What this program
# adds is the data and the standardising: the wine dataset mixes scales
# from 0 to 160, and a distance over the raw cells would be a distance
# between sulphates.
#
# The arithmetic is done on whole rows, so no column is found by its
# place. `mlbend-cluster` takes the rows as lists of F32; the
# standardising is here because it is about this dataset, not about
# k-means.
import Base
import mlbend-cluster@0.1.0.0/main.bend as Cluster
import mlbend-datasets@0.1.5.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf

# Two rows, cell by cell, the first minus the second.
def zip_sub(+a: List<&2, F32>, b: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      match b:
        case Con{hb, tb}:
          Con{F32.sub(ha, hb), zip_sub(ta, tb)}
        case Nil{}:
          Nil{}

# Two rows, cell by cell, the first divided by the second.
def zip_div(a: List<&2, F32>, b: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      match b:
        case Con{hb, tb}:
          Con{(ha / F32.max(hb, 0.0001) : F32), zip_div(ta, tb)}
        case Nil{}:
          Nil{}

# Two rows, added cell by cell.
def zip_add(a: List<&2, F32>, b: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      match b:
        case Con{hb, tb}:
          Con{F32.add(ha, hb), zip_add(ta, tb)}
        case Nil{}:
          Nil{}

# The cells of a row, squared.
def squares(+a: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{+ha, ta}:
      Con{F32.mul(ha, ha), squares(ta)}

# The sum of the squares of the deviations, row by row.
def deviation_sums_go(rows: List<&2, List<&2, F32>>, +means: List<&2, F32>,
  +acc: List<&2, F32>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      deviation_sums_go(t, means, zip_add(acc, squares(zip_sub(h, means))))

# The sum of the squares of the deviations of every column.
def deviation_sums(rows: List<&2, List<&2, F32>>, +means: List<&2, F32>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      deviation_sums_go(t, means, squares(zip_sub(h, means)))

# One row and a number, cell by cell.
def zip_scale(+a: List<&2, F32>, +k: F32) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      Con{F32.mul(ha, k), zip_scale(ta, k)}

# The square root of every cell of a row.
def roots(a: List<&2, F32>) -> List<&2, F32>:
  match a:
    case Nil{}:
      Nil{}
    case Con{ha, ta}:
      Con{F32.sqrt(ha), roots(ta)}

# The mean of every column.
def means_of(+rows: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match rows:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      zip_scale(Cluster.column_sums(rows),
        (1.0 / F32.from_nat(List.length(&2, List<&2, F32>, rows)) : F32))

# The standard deviation of every column: the population one, as the
# k-means comparison with NumPy uses.
def stds_of(+rows: List<&2, List<&2, F32>>, means: List<&2, F32>) -> List<&2, F32>:
  roots(zip_scale(deviation_sums(rows, means),
    (1.0 / F32.from_nat(List.length(&2, List<&2, F32>, rows)) : F32)))

# One row, centred and scaled.
def scale_row(row: List<&2, F32>, means: List<&2, F32>, stds: List<&2, F32>) -> List<&2, F32>:
  zip_div(zip_sub(row, means), stds)

# The rows, one at a time.
def standardise_go(+rows: List<&2, List<&2, F32>>, +means: List<&2, F32>, +stds: List<&2, F32>,
  +acc: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      standardise_go(t, means, stds,
        List.append(&2, List<&2, F32>, acc, Con{scale_row(h, means, stds), Nil{}}))

# The rows with every column centred and scaled.
def standardise(+rows: List<&2, List<&2, F32>>, means: List<&2, F32>, stds: List<&2, F32>) ->
  List<&2, List<&2, F32>>:
  standardise_go(rows, means, stds, Nil{})

# The centres of a model, out of it.
def centres_of(m: Cluster.Model) -> List<&2, List<&2, F32>>:
  match m:
    case Cluster.Model{centres, k}:
      centres

# A list of numbers, as text.
def sizes_text_go(+xs: List<&2, Nat>, +acc: String) -> String:
  match xs:
    case Nil{}:
      acc
    case Con{i, t}:
      sizes_text_go(t, String.append(acc, String.append(U32.show(U32.from_nat(i)), " ")))

# The sizes of the clusters, as text.
def sizes_text(+xs: List<&2, Nat>) -> String:
  sizes_text_go(xs, "")

# The clustering, over the standardised rows.
def show_rows(+rows: List<&2, List<&2, F32>>) -> IO(Unit):
  +means = means_of(rows)
  +prepared = standardise(rows, means, stds_of(rows, means))
  +model = Cluster.fit(3n, 10n, prepared)
  +centres = centres_of(model)
  +labels = Cluster.assign(centres, prepared)
  do IO<Unit>:
    IO.print(String.append("inertia: ", F32.show(Cluster.inertia(centres, prepared, labels))))
    IO.print(String.append("cluster sizes: ", sizes_text(Cluster.sizes(labels, 3n))))

# The clustering: three centres, ten rounds, and what it came to.
def show(t: Datasets.Table) -> IO(Unit):
  match t:
    case Datasets.Table{+rows, +labels}:
      show_rows(rows)

# The clustering, or the reason there is none.
def show_result(t: Result<&1, &1, Hf.Err, Datasets.Table>) -> IO(Unit):
  match t:
    case Fail{e}:
      IO.print("the dataset failed")
    case Done{v}:
      show(v)

def main() -> IO(Unit):
  do IO<Unit>:
    t : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.wine()
    show_result(t)
