# The scaler of the training rows, applied to both sides. The numbers are
# checked against scikit-learn's StandardScaler by reference/test_pipeline.py.
#
#| means: 3 4.3333335
#| stds: 1.6329932 2.0548048
#| train:
#| -1.2247448 -1.1355499
#| 0 -0.16222149
#| 1.2247448 1.2977712
#| test:
#| -1.8371172 -2.1088784
#| 4.286607 2.7577639
import Base
import mlbend-pipeline@0.1.0.0/main.bend as Pipeline

# A row of two numbers.
def row(a: F32, b: F32) -> List<&2, F32>:
  Con{a, Con{b, Nil{}}}

# The training rows: three of them.
def train() -> List<&2, List<&2, F32>>:
  Con{row(1.0, 2.0), Con{row(3.0, 4.0), Con{row(5.0, 7.0), Nil{}}}}

# The test rows: two of them, never seen by fit.
def test() -> List<&2, List<&2, F32>>:
  Con{row(0.0, 0.0), Con{row(10.0, 10.0), Nil{}}}

# The scaler of the training rows.
def scaler() -> Pipeline.Scaler:
  Pipeline.fit(train())

# The means of a scaler.
def means_of(s: Pipeline.Scaler) -> List<&2, F32>:
  match s:
    case Pipeline.Scaler{means, stds}:
      means

# The means of the scaler.
def means() -> List<&2, F32>:
  means_of(scaler())

# The standard deviations of a scaler.
def stds_of(s: Pipeline.Scaler) -> List<&2, F32>:
  match s:
    case Pipeline.Scaler{means, stds}:
      stds

# The standard deviations of the scaler.
def stds() -> List<&2, F32>:
  stds_of(scaler())

# A row, as text.
def row_text_go(+xs: List<&2, F32>, +acc: String) -> String:
  match xs:
    case Nil{}:
      acc
    case Con{h, t}:
      row_text_go(t, String.append(acc, String.append(F32.show(h), " ")))

# A row, as text.
def row_text(+xs: List<&2, F32>) -> String:
  row_text_go(xs, "")

# The rows after the first, each on its own line.
def rows_tail(+rows: List<&2, List<&2, F32>>) -> String:
  match rows:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(String.append("\n", row_text(h)), rows_tail(t))

# The rows, one line each, with no trailing newline.
def rows_text(+rows: List<&2, List<&2, F32>>) -> String:
  match rows:
    case Nil{}:
      ""
    case Con{h, t}:
      String.append(row_text(h), rows_tail(t))

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(String.append("means: ", row_text(means())))
    IO.print(String.append("stds: ", row_text(stds())))
    IO.print(String.append("train:\n", rows_text(Pipeline.transform(scaler(), train()))))
    IO.print(String.append("test:\n", rows_text(Pipeline.transform(scaler(), test()))))
