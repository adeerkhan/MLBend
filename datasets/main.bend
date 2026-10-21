# mlbend-datasets: the datasets of scikit-learn, fetched from the hub.
#
#   import mlbend-datasets@0.1.3.0/main.bend as Datasets
#
# The features of a dataset and its labels beside them: the rows of the
# features, and one label per row.
#
# The labels are numbers. A classification label is the index of the
# class, a regression label is the value; which one it is, the model
# decides.
#
# Every dataset here comes from HuggingFace, over the network, at run
# time: the effect is at the edge and the data that comes back is a
# value, so an estimator over it stays pure and a run is reproducible.
#
# Two shapes of file. The iris dataset is the one scikit-learn calls
# `load_iris`: 150 rows, four features, three classes, and no header,
#
#   5.1,3.5,1.4,0.2,Iris-setosa
#
# The others are CSV with a header, and the last column is the label:
#
#   fixed_acidity,volatile_acidity,...,alcohol,quality
#   7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0

import Base
import mlbend-huggingface@0.1.1.0/main.bend as Hf

# A dataset: the rows of the features and the labels beside them.
type Table is Data:
  Table{rows: List<&2, List<&2, F32>>, labels: List<&2, F32>}

# One row of the file: its features and its label.
type Row is Data:
  Row{features: List<&2, F32>, label: F32}

# The rows read so far.
type Rows is Data:
  Rows{rows: List<&2, Row>, labels: List<&2, F32>}

# The class of a row, by name: 0, 1 or 2 for the three iris species.
def add_features(acc: List<&2, List<&2, F32>>, r: Row) -> List<&2, List<&2, F32>>:
  match r:
    case Row{features, label}:
      List.append(&2, List<&2, F32>, acc, features <> Nil{})

# The dataset a file of rows makes.
def add_row_of(acc: Rows, r: Row) -> Rows:
  match acc:
    case Rows{rows, labels}:
      match r:
        case Row{features, +label}:
          Rows{List.append(&2, Row, rows, r <> Nil{}),
            List.append(&2, F32, labels, label <> Nil{})}

# The rows of a file.
def number_or_zero(m: Maybe<&2, F32>) -> F32:
  match m:
    case Some{v}:
      v
    case None{}:
      0.0

# The text of a field of a row.
def cell(text: String) -> F32:
  match text:
    case "True":
      1.0
    case "False":
      0.0
    case _:
      number_or_zero(F32.read(text))

# A number that may be there, or zero when it is not.
def cells_of(+xs: List<&2, String>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      match t:
        case Nil{}:
          Nil{}
        case Con{h2, t2}:
          List.append(&2, F32, cell(h) <> Nil{}, cells_of(t))

# The features of a row, from the first field to the one before the
# label.
def features_of(+xs: List<&2, String>) -> List<&2, F32>:
  cells_of(xs)

# The features of a row, from the head: the last field is the label, so
# the walk stops one field short.
def class_of(name: String) -> F32:
  match name:
    case "Iris-setosa":
      0.0
    case "Iris-versicolor":
      1.0
    case "Iris-virginica":
      2.0
    case _:
      0.0

# A field of a row, as a number: a number is itself, True is 1, False is
# 0, and anything else is 0.
def class_or_number(text: String) -> F32:
  match text:
    case "Iris-setosa":
      class_of(text)
    case "Iris-versicolor":
      class_of(text)
    case "Iris-virginica":
      class_of(text)
    case _:
      number_or_zero(F32.read(text))

# One row of the file, from its fields.
def label_at(m: Maybe<&2, String>) -> F32:
  match m:
    case Some{s}:
      class_or_number(s)
    case None{}:
      0.0

# The label of a row: a class name is its class, a number is itself.
def text_at(+xs: List<&2, String>, n: Nat) -> Maybe<&2, String>:
  List.get(&2, String, xs, n)

# The features of a row: every field but the last.
def label_of(+xs: List<&2, String>) -> F32:
  label_at(text_at(xs, Nat.sub(List.length(&2, String, xs), 1n)))

# The label of a row, when the field is there.
def row_of_parts(+fs: List<&2, F32>, l: F32) -> Row:
  Row{fs, l}

# One row of the file.
def row_of(line: String) -> Row:
  +xs = String.split(line, Chr{44})
  row_of_parts(features_of(xs), label_of(xs))

# The rows of a file, folded together.
def add_row(acc: Rows, line: String) -> Rows:
  add_row_of(acc, row_of(line))

# The rows of a file, and the row one line makes.
def feature_rows(rs: List<&2, Row>) -> List<&2, List<&2, F32>>:
  List.foldl(&2, Row, List<&2, List<&2, F32>>, add_features, rs, Nil{})

# One row's features, appended to the rows so far.
def data_of(acc: Rows) -> Table:
  match acc:
    case Rows{rows, labels}:
      Table{feature_rows(rows), labels}

# The rows of the features, one after the other.
def not_empty(line: String) -> Bool:
  Bool.not(String.is_empty(line))

# The iris dataset, fetched from the hub.
def drop_empty(lines: List<&2, String>) -> List<&2, String>:
  List.filter(String, not_empty, lines)

# A line that holds something.
def rows_of(lines: List<&2, String>) -> Rows:
  List.foldl(&2, String, Rows, add_row, lines, Rows{Nil{}, Nil{}})

# The dataset the rows make.
def build(text: String) -> Table:
  data_of(rows_of(drop_empty(String.lines(text))))

# The dataset a file of CSV makes: the first line is the header and the
# last column of every other line is the label.
def build_csv_rows(lines: List<&2, String>) -> Table:
  data_of(rows_of(drop_empty(lines)))

# The lines of a file without its first.
def drop_first(lines: List<&2, String>) -> List<&2, String>:
  List.drop(&2, String, lines, 1n)

# The lines of a file without its empty ones: a file ends with a
# newline, and the line after it is empty.
def build_csv_lines(lines: List<&2, String>) -> Table:
  build_csv_rows(drop_first(lines))

# The dataset the rows of a CSV make.
def build_csv(text: String) -> Table:
  build_csv_lines(String.lines(text))

# The dataset the lines of a CSV make, without the header.
def csv_text(+text: String, header: Bool) -> Table:
  Bool.pick(Table, header, build_csv(text), build(text))

def csv_of_text(t: Result<&1, &1, Hf.Err, String>, header: Bool) ->
  Result<&1, &1, Hf.Err, Table>:
  match t:
    case Fail{e}:
      Fail{e}
    case Done{text}:
      Done{csv_text(text, header)}

# The dataset the text of a file makes.
def csv_of(repo: String, rev: String, path: String, header: Bool) ->
  IO(Result<&1, &1, Hf.Err, Table>):
  do IO<Result<&1, &1, Hf.Err, Table>>:
    t : Result<&1, &1, Hf.Err, String> <- Hf.file_text(repo, rev, path)
    return csv_of_text(t, header)

# The dataset a file of text makes.
def cancer() -> IO(Result<&1, &1, Hf.Err, Table>):
  csv_of("mstz/breast", "main", "cancer/train.csv", True{})

# The heart failure dataset: 299 rows, twelve features, and the label.
def cell_of(m: Maybe<&2, String>, rest: List<&2, F32>) -> List<&2, F32>:
  match m:
    case Some{s}:
      List.append(&2, F32, rest, cell(s) <> Nil{})
    case None{}:
      rest

# The label of a row: the last field.
def heart() -> IO(Result<&1, &1, Hf.Err, Table>):
  csv_of("mstz/heart_failure", "main", "death/train.csv", True{})

# A file of a dataset's repository, and the dataset it makes.
def iris() -> IO(Result<&1, &1, Hf.Err, Table>):
  csv_of("mstz/iris", "main", "iris.data", False{})

# The wine dataset: 178 rows, thirteen features, and the quality.
def wine() -> IO(Result<&1, &1, Hf.Err, Table>):
  csv_of("mstz/wine", "main", "wine/train.csv", True{})

# The breast cancer dataset: 569 rows, nine features, and the label.