# The page of a dataset, and the rows of one that is stored as data.
import Base
import ../main.bend as Hf
import bend-kit-json@0.6.1.1/json.bend as Json

def page_name(v: Result<&1, &1, Hf.Err, Json.Val>) -> String:
  match v:
    case Fail{e}:
      "the dataset page failed"
    case Done{j}:
      String.append("dataset: ", Hf.text_field(j, "id"))

def why_of(e: Hf.Err) -> String:
  match e:
    case Hf.ErrNet{code, why}:
      String.append(String.append("network: ", why), String.append(" (", U32.show(code)))
    case Hf.ErrStatus{status}:
      String.append("the server answered ", U32.show(status))
    case Hf.ErrJson{}:
      "the answer was not JSON"

def rows_label(v: Result<&1, &1, Hf.Err, Json.Val>) -> String:
  match v:
    case Fail{e}:
      String.append("the rows failed: ", why_of(e))
    case Done{j}:
      String.append("first row label: ", Hf.text_item_field_field(Hf.array_field(j, "rows"), 0n, "row", "label"))

def main() -> IO(Unit):
  do IO<Unit>:
    page : Result<&1, &1, Hf.Err, Json.Val> <- Hf.dataset("mstz/iris")
    rows : Result<&1, &1, Hf.Err, Json.Val> <- Hf.rows("ylecun/mnist", "mnist", "train", 0n, 2n)
    IO.print(page_name(page))
    IO.print(rows_label(rows))
