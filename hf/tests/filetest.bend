import Base
import ../main.bend as Hf

def why_of(e: Hf.Err) -> String:
  match e:
    case Hf.ErrNet{code, why}:
      String.append(String.append("network: ", why), String.append(" (", U32.show(code)))
    case Hf.ErrStatus{status}:
      String.append("the server answered ", U32.show(status))
    case Hf.ErrJson{}:
      "the answer was not JSON"

def show(r: Result<&1, &1, Hf.Err, String>) -> String:
  match r:
    case Fail{e}:
      why_of(e)
    case Done{s}:
      String.append(String.append("got ", U32.show(U32.from_nat(String.length(s)))), " bytes")

def main() -> IO(Unit):
  do IO<Unit>:
    f : Result<&1, &1, Hf.Err, String> <- Hf.file_text("mstz/iris", "main", "iris.data")
    IO.print(show(f))
