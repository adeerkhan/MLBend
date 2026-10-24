# mlbend-datasets: the datasets of scikit-learn, fetched from the hub.
#
#   import mlbend-datasets@0.1.5.0/main.bend as Datasets
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
# The others are CSV with a header, and the label is one of the columns:
#
#   fixed_acidity,volatile_acidity,...,alcohol,quality,is_red
#   7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0
#
# The loader is told how many columns to leave out at the end of a row:
# the label is the column before them.

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

# One row's features, appended to the rows so far.
def add_features(acc: List<&2, List<&2, F32>>, r: Row) -> List<&2, List<&2, F32>>:
  match r:
    case Row{features, label}:
      List.append(&2, List<&2, F32>, acc, Con{features, Nil{}})

# The rows of a file, and the row one line makes.
def add_row_of(acc: Rows, +r: Row) -> Rows:
  match acc:
    case Rows{rows, labels}:
      match r:
        case Row{features, +label}:
          Rows{List.append(&2, Row, rows, r <> Nil{}),
            List.append(&2, F32, labels, label <> Nil{})}

# The rows of the features, one after the other.
def feature_rows(rs: List<&2, Row>) -> List<&2, List<&2, F32>>:
  List.foldl(&2, Row, List<&2, List<&2, F32>>, add_features, rs, Nil{})

# The dataset the rows make.
def data_of(acc: Rows) -> Table:
  match acc:
    case Rows{rows, labels}:
      Table{feature_rows(rows), labels}

# A line that holds something.
def not_empty(line: String) -> Bool:
  Bool.not(String.is_empty(line))

# The lines of a file without its empty ones: a file ends with a
# newline, and the line after it is empty.
def drop_empty(lines: List<&2, String>) -> List<&2, String>:
  List.filter(String, not_empty, lines)

# A number that may be there, or zero when it is not.
def number_or_zero(m: Maybe<&2, F32>) -> F32:
  match m:
    case Some{v}:
      v
    case None{}:
      0.0

# A field of a row, as a number: a number is itself, True is 1, False is
# 0, and anything else is 0.
def cell(+text: String) -> F32:
  match text:
    case "True":
      1.0
    case "False":
      0.0
    case _:
      number_or_zero(F32.read(text))

# The cells of a row, as numbers.
def cells_of(+xs: List<&2, String>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      List.append(&2, F32, Nil{}, Con{cell(h), cells_of(t)})

# The place of the label in a row: the last column, less the columns
# that are left out at the end.
def label_index(+xs: List<&2, String>, drop: Nat) -> Nat:
  Nat.sub(List.length(&2, String, xs), Nat.add(1n, drop))

# The fields before the label.
def fields_before(+xs: List<&2, String>, drop: Nat) -> List<&2, String>:
  List.take(&2, String, xs, label_index(xs, drop))

# The features of a row: the fields before the label, as numbers.
def features_of(+xs: List<&2, String>, drop: Nat) -> List<&2, F32>:
  cells_of(fields_before(xs, drop))

# The class of a row, by name: 0, 1 or 2 for the three iris species.
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

# The label of a row: a class name is its class, a number is itself.
def class_or_number(+text: String) -> F32:
  match text:
    case "Iris-setosa":
      class_of(text)
    case "Iris-versicolor":
      class_of(text)
    case "Iris-virginica":
      class_of(text)
    case _:
      number_or_zero(F32.read(text))

# The label of a row, when the field is there.
def label_at(m: Maybe<&2, String>) -> F32:
  match m:
    case Some{s}:
      class_or_number(s)
    case None{}:
      0.0

# The text of a field of a row.
def text_at(+xs: List<&2, String>, n: Nat) -> Maybe<&2, String>:
  List.get(&2, String, xs, n)

# The label of a row: the column at its place.
def label_of(+xs: List<&2, String>, drop: Nat) -> F32:
  label_at(text_at(xs, label_index(xs, drop)))

# One row of the file, from its fields.
def row_of_parts(+fs: List<&2, F32>, l: F32) -> Row:
  Row{fs, l}

# One row of the file.
def row_of(+line: String, +drop: Nat) -> Row:
  +xs = String.split(line, Chr{44})
  row_of_parts(features_of(xs, drop), label_of(xs, drop))

# The rows of a file, one line at a time.
def rows_go(lines: List<&2, String>, +drop: Nat, +acc: Rows) -> Rows:
  match lines:
    case Nil{}:
      acc
    case Con{h, t}:
      rows_go(t, drop, add_row_of(acc, row_of(h, drop)))

# The rows of a file.
def rows_of(lines: List<&2, String>, drop: Nat) -> Rows:
  rows_go(lines, drop, Rows{Nil{}, Nil{}})

# The dataset a file of rows makes.
def build(text: String, drop: Nat) -> Table:
  data_of(rows_of(drop_empty(String.lines(text)), drop))

# The dataset the rows of a CSV make.
def build_csv_rows(+lines: List<&2, String>, drop: Nat) -> Table:
  data_of(rows_of(drop_empty(lines), drop))

# The lines of a file without its first.
def drop_first(+lines: List<&2, String>) -> List<&2, String>:
  List.drop(&2, String, lines, 1n)

# The dataset a file of CSV makes: the first line is the header and the
# label is the column before the ones left out at the end.
def build_csv(text: String, drop: Nat) -> Table:
  build_csv_rows(drop_first(String.lines(text)), drop)

# The dataset the text of a file makes.
def csv_text(+text: String, header: Bool, +drop: Nat) -> Table:
  Bool.pick(Table, header, build_csv(text, drop), build(text, drop))

# The dataset a file of text makes.
def csv_of_text(t: Result<&1, &1, Hf.Err, String>, header: Bool, drop: Nat) ->
  Result<&1, &1, Hf.Err, Table>:
  match t:
    case Fail{e}:
      Fail{e}
    case Done{text}:
      Done{csv_text(text, header, drop)}

# ---- the local cache: MLBEND_DATA, a directory laid out like the hub ----
#
# The runner fills <dir>/<repo>/<path> with the exact bytes it hashed. When
# the variable is set and the file is there, the program reads it and does
# not touch the network; otherwise it falls back to the hub client.

def cached_path(dir: String, repo: String, path: String) -> String:
  String.append(dir, String.append("/", String.append(repo, String.append("/", path))))

def read_result(result: Result<&1, &1, U32 & String, String>) -> IO(Maybe<&1, String>):
  match result:
    case Fail{e}:
      IO.pure(Maybe<&1, String>, None{})
    case Done{text}:
      IO.pure(Maybe<&1, String>, Some{text})

def read_done(pair: File & Result<&1, &1, U32 & String, String>) -> IO(Maybe<&1, String>):
  (file, result) = pair
  do IO<Maybe<&1, String>>:
    File.close(file)
    read_result(result)

def sized_done(pair: File & Result<&1, &1, U32 & String, U32>) -> IO(Maybe<&1, String>):
  (file, result) = pair
  match result:
    case Fail{e}:
      do IO<Maybe<&1, String>>:
        File.close(file)
        return None{}
    case Done{size}:
      do IO<Maybe<&1, String>>:
        pair : File & Result<&1, &1, U32 & String, String> <- File.read(file, size)
        m : Maybe<&1, String> <- read_done(pair)
        return m

def opened(r: Result<&1, &1, U32 & String, File>) -> IO(Maybe<&1, String>):
  match r:
    case Fail{e}:
      IO.pure(Maybe<&1, String>, None{})
    case Done{file}:
      do IO<Maybe<&1, String>>:
        pair : File & Result<&1, &1, U32 & String, U32> <- File.size(file)
        m : Maybe<&1, String> <- sized_done(pair)
        return m

# The text of a cached file, or None when it is not there.
def read_cached(path: String) -> IO(Maybe<&1, String>):
  do IO<Maybe<&1, String>>:
    r : Result<&1, &1, U32 & String, File> <- File.open(path, "r")
    opened(r)

# A table from the cache when it is there, otherwise from the hub.
def fetch_csv(repo: String, rev: String, path: String, header: Bool, drop: Nat) ->
  IO(Result<&1, &1, Hf.Err, Table>):
  do IO<Result<&1, &1, Hf.Err, Table>>:
    t : Result<&1, &1, Hf.Err, String> <- Hf.file_text(repo, rev, path)
    return csv_of_text(t, header, drop)

def cached_or_fetch(m: Maybe<&1, String>, repo: String, rev: String, path: String,
  header: Bool, drop: Nat) -> IO(Result<&1, &1, Hf.Err, Table>):
  match m:
    case Some{text}:
      IO.pure(Result<&1, &1, Hf.Err, Table>, Done{csv_text(text, header, drop)})
    case None{}:
      fetch_csv(repo, rev, path, header, drop)

def from_env(e: Result<&1, &1, U32 & String, String>, +repo: String, rev: String, +path: String,
  header: Bool, drop: Nat) -> IO(Result<&1, &1, Hf.Err, Table>):
  match e:
    case Fail{code}:
      fetch_csv(repo, rev, path, header, drop)
    case Done{dir}:
      do IO<Result<&1, &1, Hf.Err, Table>>:
        m : Maybe<&1, String> <- read_cached(cached_path(dir, repo, path))
        t : Result<&1, &1, Hf.Err, Table> <- cached_or_fetch(m, repo, rev, path, header, drop)
        return t

# A file of a dataset's repository, and the dataset it makes.
def csv_of(repo: String, rev: String, path: String, header: Bool, drop: Nat) ->
  IO(Result<&1, &1, Hf.Err, Table>):
  do IO<Result<&1, &1, Hf.Err, Table>>:
    e : Result<&1, &1, U32 & String, String> <- IO.get_env("MLBEND_DATA")
    t : Result<&1, &1, Hf.Err, Table> <- from_env(e, repo, rev, path, header, drop)
    return t

# The breast cancer dataset: 683 rows, nine features, and the label.
def cancer() -> IO(Result<&1, &1, Hf.Err, Table>):
  csv_of("mstz/breast", "main", "cancer/train.csv", True{}, 0n)

# The heart failure dataset: 299 rows, twelve features, and the label.
def heart() -> IO(Result<&1, &1, Hf.Err, Table>):
  csv_of("mstz/heart_failure", "main", "death/train.csv", True{}, 0n)

# The iris dataset, fetched from the hub.
def iris() -> IO(Result<&1, &1, Hf.Err, Table>):
  csv_of("mstz/iris", "main", "iris.data", False{}, 0n)

# The wine dataset: 6497 rows, eleven features, and the quality.
def wine() -> IO(Result<&1, &1, Hf.Err, Table>):
  csv_of("mstz/wine", "main", "wine/train.csv", True{}, 1n)