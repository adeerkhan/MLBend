# Cross-validation of a logistic regression over the iris dataset.
#
#   bendenv run -- bend examples/crossval/main.bend
#
# The whole chain, end to end: the dataset comes from HuggingFace, the
# features are standardised by `mlbend-preprocess`, the model is trained
# with bend-ml's autodiff, the folds come from `mlbend-select`, and the
# score is `mlbend-metrics`. The only thing this program adds is the
# loop over the folds.
#
# The folds are stratified: the rows of each class are shuffled and then
# dealt out over the folds, so every fold holds the same classes in the
# same proportion — the same rule scikit-learn's StratifiedKFold uses.
# The shuffle is bend-kit-random's xoshiro128**, seeded: the same seed
# gives the same folds, and a different seed gives other ones. The split
# itself is `mlbend-split`'s, and `mlbend-select` walks the folds.
#
# The iris dataset has 150 rows of four features, fifty of each class, so
# a fold holds thirty rows (ten of each class) and the rest of the data
# is the training set of one hundred and twenty rows. The shapes are
# therefore fixed, and bend-ml's matrices carry them in their types: a
# fold that did not come out this way would not fit the matrix at all.
#
# The accuracy is checked against scikit-learn's LogisticRegression over
# stratified folds (reference/test_crossval.py).
import Base
import bend-ml-tensor@0.1.0.0/main.bend as T
import bend-ml-autograd@0.1.1.0/main.bend as AG
import mlbend-datasets@0.1.4.0/main.bend as Datasets
import mlbend-huggingface@0.1.1.0/main.bend as Hf
import mlbend-metrics@0.1.0.0/main.bend as M
import mlbend-preprocess@0.1.0.0/main.bend as P
import mlbend-select@0.1.0.0/main.bend as Select

# A row of the dataset: its features, its label, and the fold it is in.

type Row is Data:
  Row{features: List<&2, F32>, label: Nat, fold: Nat}

# The model: the weights of the layer and its bias.
type Model is Data:
  Model{+w: T.Mat<4n, 3n>, +b: T.Vec<3n>}

# The largest of the logits of a row, and where it is.
def argmax_go(+xs: List<&2, F32>, +i: Nat, +best: Nat, +best_v: F32) -> Nat:
  match xs:
    case Nil{}:
      best
    case Con{+h, t}:
      argmax_go(t, Nat.add(i, 1n),
        Bool.pick(Nat, F32.is_gt(h, best_v), i, best),
        Bool.pick(F32, F32.is_gt(h, best_v), h, best_v))

# The largest of the logits of a row, and where it is.
def argmax(+xs: List<&2, F32>) -> Nat:
  argmax_go(xs, 0n, 0n, (0.0 - 1.0e30 : F32))

# One row, with the fold it is in.
def with_fold(r: Row, f: Nat) -> Row:
  match r:
    case Row{features, label, fold}:
      Row{features, label, f}

# The rows, each with the fold it is in: the first rows of a class go to
# The class of a label.
def class_of(label: F32) -> Nat:
  U32.to_nat(F32.to_u32(label))

# The label of a row.
def label_of(+r: Row) -> Nat:
  match r:
    case Row{features, label, fold}:
      label

# The labels of the rows, as the class indices they are.
def labels_of_go(+rows: List<&2, Row>, +acc: List<&2, Nat>) -> List<&2, Nat>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      labels_of_go(t, List.append(&2, Nat, acc, Con{label_of(h), Nil{}}))

# The labels of the rows, as the class indices they are.
def labels_of(+rows: List<&2, Row>) -> List<&2, Nat>:
  labels_of_go(rows, Nil{})

# The class the model gives each test row: the largest logit.
def predictions_of_go(+rows: List<&2, List<&2, F32>>, +acc: List<&2, Nat>) -> List<&2, Nat>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      predictions_of_go(t, List.append(&2, Nat, acc, Con{argmax(h), Nil{}}))

# The class the model gives each test row: the largest logit.
def predictions_of(+rows: List<&2, List<&2, F32>>) -> List<&2, Nat>:
  predictions_of_go(rows, Nil{})

# The class the model gives each test row: the largest logit.
def predictions(+m: Model, x: T.Mat<30n, 4n>) -> List<&2, Nat>:
  match m:
    case Model{+w, +b}:
      predictions_of(T.Mat.to_rows(30n, 3n, AG.linear(30n, 4n, 3n, x, w, b)))

# The accuracy of one fold: the model's score on the test rows.
def fold_score(+m: Model, +x: T.Mat<30n, 4n>, +labels: List<&2, Nat>) ->
  Result<&1, &1, M.Bad, F32>:
  M.accuracy(labels, predictions(m, x))

# The accuracy of one fold, when the score came.
def scored_result(r: Result<&1, &1, M.Bad, F32>) -> F32:
  match r:
    case Done{score}:
      score
    case Fail{error}:
      (0.0 - 1.0 : F32)

# The accuracy of one fold, when the model is trained.
def scored_model(model: Model, t: T.Mat<30n, 4n>, +labels: List<&2, Nat>) -> F32:
  scored_result(fold_score(model, t, labels))

# The starting model: every weight zero, so every class scores the same.
def start() -> Model:
  Model{T.Mat.fill(4n, 3n, 0.0), T.Vec.fill(3n, 0.0)}

# The forward pass: the logits of every training row.
def logits(+m: Model, x: T.Mat<120n, 4n>) -> T.Mat<120n, 3n>:
  match m:
    case Model{+w, +b}:
      AG.linear(120n, 4n, 3n, x, w, b)

# The gradient of the loss with respect to the logits.
def gradient(+m: Model, x: T.Mat<120n, 4n>, labels: List<&2, Nat>) -> T.Mat<120n, 3n>:
  AG.ce_grad(120n, 3n, logits(m, x), labels)

# One step of SGD.
def stepped(w: T.Mat<4n, 3n>, b: T.Vec<3n>, +lr: F32,
  g: T.Mat<120n, 4n> & (T.Mat<4n, 3n> & T.Vec<3n>)) -> Model:
  (dx, (dw, db)) = g
  Model{AG.sgd(4n, 3n, lr, w, dw), AG.sgd_vec(3n, lr, b, db)}

# The step, from the gradient of the layer.
def step(+m: Model, +x: T.Mat<120n, 4n>, labels: List<&2, Nat>, lr: F32) -> Model:
  match m:
    case Model{+w, +b}:
      stepped(w, b, lr, AG.linear_bwd(120n, 4n, 3n, x, w, gradient(m, x, labels)))

# The model after a number of steps.
def training_go(steps: Nat, +m: Model, +x: T.Mat<120n, 4n>, +labels: List<&2, Nat>) -> Model:
  match steps:
    case 0n:
      m
    case 1n+rest:
      training_go(rest, step(m, x, labels, 0.5), x, labels)

# The model after a number of steps: the steps come first and are
# unpacked, so that the walk shrinks.
def training(steps: Nat, +x: T.Mat<120n, 4n>, +labels: List<&2, Nat>) -> Model:
  training_go(steps, start(), x, labels)

# The accuracy of one fold, when the test matrix is there.
def scored_test(m: Maybe<&2, T.Mat<30n, 4n>>, x: T.Mat<120n, 4n>, +labels: List<&2, Nat>,
  +test: List<&2, Row>) -> F32:
  match m:
    case Some{t}:
      scored_model(training(60n, x, labels), t, labels_of(test))
    case None{}:
      (0.0 - 1.0 : F32)

# The features of a row.
def features_of_row(+r: Row) -> List<&2, F32>:
  match r:
    case Row{features, label, fold}:
      features

# The features of the rows, as the matrix bend-ml's tensors take.
def features_of_go(+rows: List<&2, Row>, +acc: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      features_of_go(t, List.append(&2, List<&2, F32>, acc, Con{features_of_row(h), Nil{}}))

# The features of the rows, as the matrix bend-ml's tensors take.
def features_of(+rows: List<&2, Row>) -> List<&2, List<&2, F32>>:
  features_of_go(rows, Nil{})

# The test rows as bend-ml's matrix.
def test_mat(+rows: List<&2, Row>) -> Maybe<&2, T.Mat<30n, 4n>>:
  T.Mat.of(30n, 4n, features_of(rows))

# The accuracy of one fold, when the training matrix is there.
def scored_train(m: Maybe<&2, T.Mat<120n, 4n>>, +train: List<&2, Row>, +test: List<&2, Row>) -> F32:
  match m:
    case Some{x}:
      scored_test(test_mat(test), x, labels_of(train), test)
    case None{}:
      (0.0 - 1.0 : F32)

# The training rows as bend-ml's matrix.
def train_mat(+rows: List<&2, Row>) -> Maybe<&2, T.Mat<120n, 4n>>:
  T.Mat.of(120n, 4n, features_of(rows))

# The accuracy of one fold: the training rows and the test rows.
def scored(+train: List<&2, Row>, +test: List<&2, Row>) -> F32:
  scored_train(train_mat(train), train, test)

# The fold of a row.
def fold_of(+r: Row) -> Nat:
  match r:
    case Row{features, label, fold}:
      fold

# The row, kept when it is in the fold.
def keep_row(+h: Row, f: Nat, +acc: List<&2, Row>, same: Bool) -> List<&2, Row>:
  match same:
    case True{}:
      List.append(&2, Row, acc, Con{h, Nil{}})
    case False{}:
      acc

# The rows of one fold: the ones that test.
def test_rows_go(+rows: List<&2, Row>, +f: Nat, +acc: List<&2, Row>) -> List<&2, Row>:
  match rows:
    case Nil{}:
      acc
    case Con{+h, t}:
      test_rows_go(t, f, keep_row(h, f, acc, Nat.is_eq(fold_of(h), f)))

# The rows of one fold: the ones that test.
def test_rows(+rows: List<&2, Row>, f: Nat) -> List<&2, Row>:
  test_rows_go(rows, f, Nil{})

# The row, dropped when it is in the fold.
def drop_row(+h: Row, f: Nat, +acc: List<&2, Row>, same: Bool) -> List<&2, Row>:
  match same:
    case True{}:
      acc
    case False{}:
      List.append(&2, Row, acc, Con{h, Nil{}})

# The rows of one fold: the ones that train.
def train_rows_go(+rows: List<&2, Row>, +f: Nat, +acc: List<&2, Row>) -> List<&2, Row>:
  match rows:
    case Nil{}:
      acc
    case Con{+h, t}:
      train_rows_go(t, f, drop_row(h, f, acc, Nat.is_eq(fold_of(h), f)))

# The rows of one fold: the ones that train.
def train_rows(+rows: List<&2, Row>, f: Nat) -> List<&2, Row>:
  train_rows_go(rows, f, Nil{})

# The accuracy of one fold: the rows that train and the rows that test.
def fold_accuracy(+rows: List<&2, Row>, +f: Nat) -> F32:
  scored(train_rows(rows, f), test_rows(rows, f))

# The accuracy of every fold.
def crossval_go(fs: List<&2, Nat>, +rows: List<&2, Row>, +acc: List<&2, F32>) -> List<&2, F32>:
  match fs:
    case Nil{}:
      acc
    case Con{h, t}:
      crossval_go(t, rows, List.append(&2, F32, acc, Con{fold_accuracy(rows, h), Nil{}}))

# The number of folds.
def folds() -> Nat:
  5n

# The accuracy of every fold: the folds left come first and the walk
# shrinks on the list.
def crossval(+rows: List<&2, Row>) -> List<&2, F32>:
  crossval_go(List.range(folds()), rows, Nil{})

# The seed of the shuffle.
def seed() -> U32:
  42

# The rows of the dataset, each with the fold it is in: the folds come
# from `mlbend-select`, which makes them the way scikit-learn's
# StratifiedKFold does.
def folded_rows(+rows: List<&2, Row>, +fs: List<&2, Nat>, +acc: List<&2, Row>) -> List<&2, Row>:
  match rows fs:
    case Con{h, t} Con{f, ft}:
      folded_rows(t, ft, List.append(&2, Row, acc, Con{with_fold(h, f), Nil{}}))
    case _ _:
      acc

# The loss: the mean cross-entropy of the logits against the labels.
def loss(+m: Model, x: T.Mat<120n, 4n>, labels: List<&2, Nat>) -> F32:
  AG.ce_loss(120n, 3n, logits(m, x), labels)

# The features of the dataset, standardised, when there is a matrix of them.
def prepared_of(+rows: List<&2, List<&2, F32>>, m: Maybe<&2, T.Mat<150n, 4n>>) ->
  List<&2, List<&2, F32>>:
  match m:
    case Some{matrix}:
      T.Mat.to_rows(150n, 4n, P.standardize(150n, 4n, matrix))
    case None{}:
      rows

# The features of the dataset, standardised, over bend-ml's matrix.
def prepared(+rows: List<&2, List<&2, F32>>) -> List<&2, List<&2, F32>>:
  prepared_of(rows, T.Mat.of(150n, 4n, rows))

# The rows of the dataset, as rows of features and class index.
def rows_of_go(+rows: List<&2, List<&2, F32>>, +labels: List<&2, F32>, +acc: List<&2, Row>) ->
  List<&2, Row>:
  match rows:
    case Nil{}:
      acc
    case Con{h, t}:
      match labels:
        case Nil{}:
          acc
        case Con{l, ls}:
          rows_of_go(t, ls, List.append(&2, Row, acc, Con{Row{h, class_of(l), 0n}, Nil{}}))

# The rows of the dataset, as rows of features and class index.
def rows_of(+rows: List<&2, List<&2, F32>>, +labels: List<&2, F32>) -> List<&2, Row>:
  rows_of_go(rows, labels, Nil{})

# The mean of the accuracies.
def mean_of_go(+xs: List<&2, F32>, +sum: F32, +n: F32) -> F32:
  match xs:
    case Nil{}:
      (sum / n : F32)
    case Con{h, t}:
      mean_of_go(t, F32.add(sum, h), F32.add(n, 1.0))

# The mean of the accuracies.
def mean_of(+xs: List<&2, F32>) -> F32:
  mean_of_go(xs, 0.0, 0.0)

# The accuracies, one per line.
def show_scores_go(+scores: List<&2, F32>, +i: Nat) -> IO(Unit):
  match scores:
    case Nil{}:
      IO.print("")
    case Con{h, t}:
      do IO<Unit>:
        IO.print(String.append(String.append("fold ", U32.show(U32.from_nat(i))),
          String.append(": ", F32.show(h))))
        show_scores_go(t, Nat.add(i, 1n))

# The accuracies, one per line, and the mean.
def show_scores(+scores: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    show_scores_go(scores, 0n)
    IO.print(String.append("mean accuracy: ", F32.show(mean_of(scores))))

# The cross-validation of the dataset.
def show_all(+fs: List<&2, Row>) -> IO(Unit):
  do IO<Unit>:
    show_scores(crossval(fs))

# The rows of the dataset, with the fold of each row.
def show_rows(+rows: List<&2, List<&2, F32>>, +labels: List<&2, F32>) -> IO(Unit):
  +rs = rows_of(prepared(rows), labels)
  +fs = Select.folds(labels_of(rs), seed(), folds())
  show_all(folded_rows(rs, fs, Nil{}))

def show_table(t: Datasets.Table) -> IO(Unit):
  match t:
    case Datasets.Table{+rows, +labels}:
      show_rows(rows, labels)

def show(d: Result<&1, &1, Hf.Err, Datasets.Table>) -> IO(Unit):
  match d:
    case Fail{e}:
      IO.print("the dataset failed")
    case Done{table}:
      show_table(table)

def main() -> IO(Unit):
  do IO<Unit>:
    d : Result<&1, &1, Hf.Err, Datasets.Table> <- Datasets.iris()
    show(d)