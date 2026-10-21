# mlbend-datasets: the datasets of scikit-learn, fetched from the hub.
#
#   import mlbend-datasets@0.1.0.0/main.bend as Datasets
#
# The features of a dataset and its labels beside them: a `Mat<r, c>` of
# the features and a list of the labels.
#
# Every dataset here comes from HuggingFace, over the network, at run
# time: the effect is at the edge and the data that comes back is a
# value, so an estimator over it stays pure and a run is reproducible.
#
# The iris dataset is the one scikit-learn calls `load_iris`: 150 rows,
# four features, three classes. It is fetched as the file the dataset's
# own repository holds, which holds the same numbers scikit-learn ships.
#
# The file has one row per line, comma separated, the four features and
# then the class:
#
#   5.1,3.5,1.4,0.2,Iris-setosa

import Base
import mlbend-huggingface@0.1.1.0/main.bend as Hf
import bend-ml-tensor-array@0.1.14.0/main.bend as TA

# A dataset: the features and the labels beside them. The flat tensor is
# a Type, so the record that holds it is one too.
type Table is Type:
  Table{features: TA.Mat<150n, 4n>, labels: List<&2, Nat>}

# One row of the file: its four numbers and its class.
type Row is Data:
  Row{features: List<&2, F32>, label: Nat}

# The rows read so far.
type Rows is Data:
  Rows{rows: List<&2, Row>, labels: List<&2, Nat>}

# The class of a row: 0, 1 or 2 for the three iris species.
def add_features(acc: List<&2, F32>, r: Row) -> List<&2, F32>:
  match r:
    case Row{features, +label}:
      List.append(&2, F32, acc, features)

# The dataset the rows make.
def add_row_of(acc: Rows, r: Result<&1, &1, String, Row>) -> Rows:
  match acc:
    case Rows{rows, labels}:
      match r:
        case Fail{why}:
          Rows{rows, labels}
        case Done{row}:
          match row:
            case Row{features, +label}:
              Rows{List.append(&2, Row, rows, row <> Nil{}),
                List.append(&2, Nat, labels, label <> Nil{})}

# The rows of a file.
def number_or_zero(m: Maybe<&2, F32>) -> F32:
  match m:
    case Some{v}:
      v
    case None{}:
      0.0

# The text of a field, when the field is there.
def features4(a: Maybe<&2, F32>, b: Maybe<&2, F32>, c: Maybe<&2, F32>, d: Maybe<&2, F32>) ->
  List<&2, F32>:
  number_or_zero(a) <> (number_or_zero(b) <> (number_or_zero(c) <> (number_or_zero(d) <> Nil{})))

# The class of a row, when the row has one.
def number_of_text(t: Maybe<&2, String>) -> Maybe<&2, F32>:
  match t:
    case Some{s}:
      F32.read(s)
    case None{}:
      None{}

# The four features of a row.
def text_at(+xs: List<&2, String>, n: Nat) -> Maybe<&2, String>:
  List.get(&2, String, xs, n)

# The number of a field, when the field is there.
def number_at(+xs: List<&2, String>, n: Nat) -> Maybe<&2, F32>:
  number_of_text(text_at(xs, n))

# The number some text holds, when it is one that fits.
def features_of(+xs: List<&2, String>) -> List<&2, F32>:
  features4(number_at(xs, 0n), number_at(xs, 1n), number_at(xs, 2n), number_at(xs, 3n))

# The four features of a row, from the numbers.
def class_of(name: String) -> Nat:
  match name:
    case "Iris-setosa":
      0n
    case "Iris-versicolor":
      1n
    case "Iris-virginica":
      2n
    case _:
      0n

# A number that may be there, or zero when it is not.
def class_at(m: Maybe<&2, String>) -> Maybe<&2, Nat>:
  match m:
    case Some{s}:
      Some{class_of(s)}
    case None{}:
      None{}

# One row of the file.
def label_of(+xs: List<&2, String>) -> Maybe<&2, Nat>:
  class_at(text_at(xs, 4n))

# The class of a field, when the field is there.
def count_ok(n: Nat, fs: List<&2, F32>, label: Nat) -> Result<&1, &1, String, Row>:
  Bool.pick(Result<&1, &1, String, Row>, Nat.is_eq(n, 4n),
    Done{Row{fs, label}},
    Fail{"a row with the wrong number of features"})

# The rows of a file, and the row one line makes.
def row_of_parts(+fs: List<&2, F32>, l: Maybe<&2, Nat>) -> Result<&1, &1, String, Row>:
  match l:
    case Some{label}:
      count_ok(List.length(&2, F32, fs), fs, label)
    case None{}:
      Fail{"a row with no class"}

# A row with the right number of features, or the reason it has not.
def row_of(line: String) -> Result<&1, &1, String, Row>:
  +xs = String.split(line, Chr{44})
  row_of_parts(features_of(xs), label_of(xs))

# One row of the file, from its features and its class.
def add_row(acc: Rows, line: String) -> Rows:
  add_row_of(acc, row_of(line))

# The rows of a file, and the row one line makes.
def build_data(d: Result<&1, &1, String, Table>) -> Result<&1, &1, Hf.Err, Table>:
  match d:
    case Fail{why}:
      Fail{Hf.ErrJson{}}
    case Done{table}:
      Done{table}

# The text of a file that may be there, and the dataset it makes.
def matrix_of_mat(m: Maybe<&1, TA.Mat<150n, 4n>>, labels: List<&2, Nat>) ->
  Result<&1, &1, String, Table>:
  match m:
    case Some{matrix}:
      Done{Table{matrix, labels}}
    case None{}:
      Fail{"the rows are not 150 of four"}

# The rows of a file, and the dataset they make.
def matrix_of(fs: List<&2, F32>, labels: List<&2, Nat>) -> Result<&1, &1, String, Table>:
  matrix_of_mat(TA.Mat.of_list(150n, 4n, fs), labels)

# The matrix the features make, when it is one.
def data_of(acc: Rows) -> Result<&1, &1, String, Table>:
  match acc:
    case Rows{rows, labels}:
      matrix_of(List.foldl(&2, Row, List<&2, F32>, add_features, rows, Nil{}), labels)

# The matrix the features make, and the labels.
def build_rows(r: Result<&1, &1, String, Rows>) -> Result<&1, &1, Hf.Err, Table>:
  match r:
    case Fail{why}:
      Fail{Hf.ErrJson{}}
    case Done{rows}:
      build_data(data_of(rows))

# The dataset the rows make.
def rows_of(lines: List<&2, String>) -> Result<&1, &1, String, Rows>:
  Done{List.foldl(&2, String, Rows, add_row, lines, Rows{Nil{}, Nil{}})}

# The features of every row, one after the other.
def build(text: String) -> Result<&1, &1, Hf.Err, Table>:
  build_rows(rows_of(String.lines(text)))

# The rows of a file, and the dataset they make.
def build_of(t: Result<&1, &1, Hf.Err, String>) -> Result<&1, &1, Hf.Err, Table>:
  match t:
    case Fail{e}:
      Fail{e}
    case Done{text}:
      build(text)

# The iris dataset, fetched from the hub.
def iris() -> IO(Result<&1, &1, Hf.Err, Table>):
  do IO<Result<&1, &1, Hf.Err, Table>>:
    t : Result<&1, &1, Hf.Err, String> <- Hf.file_text("mstz/iris", "main", "iris.data")
    return build_of(t)
