# mlbend-huggingface: a HuggingFace client for Bend 2.
#
#   import mlbend-hf@0.1.0.0/main.bend as Hf
#
# Two APIs, both JSON over HTTP.
#
# The dataset's own host:
#
#   GET huggingface.co/api/datasets/<name>             the dataset's page
#   GET huggingface.co/api/datasets/<name>/tree/<rev>  the files in it
#   GET huggingface.co/datasets/<name>/resolve/<rev>/<path>  a file's bytes
#
# And the rows server, which answers a dataset that is stored as data as
# JSON rows, with the features beside them:
#
#   GET datasets-server.huggingface.co/rows?dataset=<name>&config=<c>&split=<s>&offset=<n>&length=<k>
#
# A dataset that is a script rather than data cannot be read this way. The
# rows server says so, and its message says what to do about it: convert the
# dataset to Parquet. The file tree and the raw file download work for those
# anyway.
#
# Everything is an effect at the edge, so the estimators that use a dataset
# stay pure and a run is reproducible.
#
# A note on the shape of the code: Bend's checker will not let a match
# scrutinize a computed value, so every helper that unwraps a Maybe takes it
# as a parameter and matches on that. The callers get one call each.

import Base
import bend-kit-http@0.36.0.0/http.bend as Http
import bend-kit-json@0.6.1.1/json.bend as Json
import bend-kit-bytes@0.3.2.0/bytes.bend as Bytes

# Why an answer did not come.
type Err is Data:
  ErrNet{code: U32, why: String}
  ErrStatus{status: U32}
  ErrJson{}

# The hosts.
def host() -> String:
  "https://huggingface.co"

def rows_host() -> String:
  "https://datasets-server.huggingface.co"

# The HTTP kit's error, as ours.
def why_of(e: Http.Err) -> Err:
  match e:
    case Http.ErrConnect{code, why}:
      ErrNet{code, why}
    case Http.ErrTls{code, why}:
      ErrNet{code, why}
    case Http.ErrRead{code, why}:
      ErrNet{code, why}
    case Http.ErrWrite{code, why}:
      ErrNet{code, why}
    case Http.ErrUrl{}:
      ErrNet{0, "a url bend cannot open"}
    case Http.ErrDns{}:
      ErrNet{0, "a name that does not resolve"}
    case Http.ErrTimeout{}:
      ErrNet{0, "the answer did not come in time"}
    case Http.ErrRedirect{}:
      ErrNet{0, "a redirect bend did not follow"}
    case Http.ErrBad{}:
      ErrNet{0, "a malformed answer"}

# The JSON of a parsed body.
def json_of_maybe(m: Maybe<&1, Json.Val>) -> Result<&1, &1, Err, Json.Val>:
  match m:
    case Some{v}:
      Done{v}
    case None{}:
      Fail{ErrJson{}}

# The JSON of a body, or the reason there is none.
def json_of_body(body: Bytes.Bytes) -> Result<&1, &1, Err, Json.Val>:
  json_of_maybe(Json.parse.bytes(body))

# The JSON of a response, or the reason there is none.
def json_of(r: Result<&1, &1, Http.Err, Http.Res>) -> Result<&1, &1, Err, Json.Val>:
  match r:
    case Fail{e}:
      Fail{why_of(e)}
    case Done{Http.Res{+status, headers, body}}:
      Bool.pick(Result<&1, &1, Err, Json.Val>, U32.is_lt(status, 400),
        json_of_body(body),
        Fail{ErrStatus{status}})

# The bytes of a response, or the reason there are none.
def bytes_of(r: Result<&1, &1, Http.Err, Http.Res>) -> Result<&1, &1, Err, Bytes.Bytes>:
  match r:
    case Fail{e}:
      Fail{why_of(e)}
    case Done{Http.Res{+status, headers, body}}:
      Bool.pick(Result<&1, &1, Err, Bytes.Bytes>, U32.is_lt(status, 400),
        Done{body},
        Fail{ErrStatus{status}})

# A GET, and the JSON it answered with.
def get_json(url: String) -> IO(Result<&1, &1, Err, Json.Val>):
  do IO<Result<&1, &1, Err, Json.Val>>:
    r : Result<&1, &1, Http.Err, Http.Res> <- Http.get(url)
    return json_of(r)

# A GET, and the bytes it answered with.
def get_bytes(url: String) -> IO(Result<&1, &1, Err, Bytes.Bytes>):
  do IO<Result<&1, &1, Err, Bytes.Bytes>>:
    r : Result<&1, &1, Http.Err, Http.Res> <- Http.get(url)
    return bytes_of(r)

# The page of a dataset: its name, its revision, and what it says about
# itself.
def dataset(name: String) -> IO(Result<&1, &1, Err, Json.Val>):
  get_json(String.append(String.append(host(), "/api/datasets/"), name))

# The files of a dataset, at a revision.
def tree(name: String, rev: String) -> IO(Result<&1, &1, Err, Json.Val>):
  get_json(String.append(String.append(String.append(host(), "/api/datasets/"), name),
    String.append("/tree/", rev)))

# A file of a dataset, at a revision: its bytes.
def file(name: String, rev: String, path: String) -> IO(Result<&1, &1, Err, Bytes.Bytes>):
  get_bytes(String.append(String.append(String.append(String.append(host(), "/datasets/"), name),
    String.append("/resolve/", rev)), String.append("/", path)))

# The rest of a query string, each pair behind an "&".
def query.go(pairs: List<&2, String>, acc: String) -> String:
  match pairs:
    case Nil{}:
      acc
    case Con{h, t}:
      query.go(t, String.append(acc, String.append("&", h)))

# A url with a query string: the base, the first pair, and the rest.
def query(base: String, first: String, rest: List<&2, String>) -> String:
  query.go(rest, String.append(base, first))

# The url of the rows server's answer for a dataset, a config and a
# split, a window of rows.
def rows_url(name: String, config: String, split: String, offset: Nat, length: Nat) -> String:
  query(String.append(rows_host(), "/rows?dataset="), name, [
    String.append("config=", config),
    String.append("split=", split),
    String.append("offset=", U32.show(U32.from_nat(offset))),
    String.append("length=", U32.show(U32.from_nat(length)))])

# The text of a file that may be there.
def text_of_file(r: Result<&1, &1, Err, Bytes.Bytes>) -> Result<&1, &1, Err, String>:
  match r:
    case Fail{e}:
      Fail{e}
    case Done{b}:
      Done{Bytes.to_string(b)}

# A file of a dataset, at a revision, as text.
def file_text(name: String, rev: String, path: String) -> IO(Result<&1, &1, Err, String>):
  do IO<Result<&1, &1, Err, String>>:
    b : Result<&1, &1, Err, Bytes.Bytes> <- file(name, rev, path)
    return text_of_file(b)

# The rows of a dataset, at a config and a split: the features beside them.
def rows(name: String, config: String, split: String, offset: Nat, length: Nat) ->
  IO(Result<&1, &1, Err, Json.Val>):
  get_json(rows_url(name, config, split, offset, length))

# Reading the JSON
# ----------------

# A field of a JSON object.
def field(v: Json.Val, key: String) -> Maybe<&1, Json.Val>:
  Json.get(v, key)

# An item of a JSON array.
def item(v: Json.Val, n: Nat) -> Maybe<&1, Json.Val>:
  Json.at(v, n)

# The text of a JSON value, when it is a string or a number.
def text_value(v: Json.Val) -> Maybe<&1, String>:
  match v:
    case Json.Str{s}:
      Some{Bytes.to_string(s)}
    case Json.Num{s}:
      Some{Bytes.to_string(s)}
    case _:
      None{}

# The text of a JSON string, or of a JSON number: the number keeps its
# exact text, so nothing is lost on the way.
def text_of(v: Json.Val) -> Maybe<&1, String>:
  text_value(v)

# The text of a value that may be there.
def text_of_value(m: Maybe<&1, Json.Val>) -> Maybe<&1, String>:
  match m:
    case Some{v}:
      text_value(v)
    case None{}:
      None{}

# The text of some text, or "" when there is none.
def text_maybe(m: Maybe<&1, String>) -> String:
  match m:
    case Some{s}:
      s
    case None{}:
      ""

# The field of an item of a JSON array, when the item is there.
def field_of(m: Maybe<&1, Json.Val>, key: String) -> Maybe<&1, Json.Val>:
  match m:
    case Some{it}:
      field(it, key)
    case None{}:
      None{}

# The field of a field of an item of a JSON array, when the item is there.
def field_of_field(m: Maybe<&1, Json.Val>, key1: String, key2: String) -> Maybe<&1, Json.Val>:
  match m:
    case Some{it}:
      field_of(field(it, key1), key2)
    case None{}:
      None{}

# The text of a field of a JSON object, or "" when it is not there.
def text_field(v: Json.Val, key: String) -> String:
  text_maybe(text_of_value(field(v, key)))

# The text of an item of a JSON array, or "" when it is not there.
def text_item(v: Json.Val, n: Nat) -> String:
  text_maybe(text_of_value(item(v, n)))

# The field of an item of a JSON array.
def field_item(v: Json.Val, n: Nat, key: String) -> Maybe<&1, Json.Val>:
  field_of(item(v, n), key)

# The text of a field of an item of a JSON array.
def text_item_field(v: Json.Val, n: Nat, key: String) -> String:
  text_maybe(text_of_value(field_item(v, n, key)))

# The field of a field of an item of a JSON array: the row of the rows
# server, which is two fields down from the row itself.
def field_item_field(v: Json.Val, n: Nat, key1: String, key2: String) -> Maybe<&1, Json.Val>:
  field_of_field(item(v, n), key1, key2)

# The text of a field of a field of an item of a JSON array.
def text_item_field_field(v: Json.Val, n: Nat, key1: String, key2: String) -> String:
  text_maybe(text_of_value(field_item_field(v, n, key1, key2)))

# A field of a JSON object, when it is there.
def array_of(m: Maybe<&1, Json.Val>) -> Json.Val:
  match m:
    case Some{a}:
      a
    case None{}:
      Json.Arr{Nil{}}

# The field of a JSON object, or an empty array when it is not there.
def array_field(v: Json.Val, key: String) -> Json.Val:
  array_of(field(v, key))

# The number some text holds, when it is one that fits.
def number_of_text(t: Maybe<&1, String>) -> Maybe<&2, F32>:
  match t:
    case Some{s}:
      F32.read(s)
    case None{}:
      None{}

# The number a JSON number holds, when it is a whole number that fits.
def number_of(v: Json.Val) -> Maybe<&2, F32>:
  number_of_text(text_of(v))

# The number a field of a JSON object holds, when it is one that fits.
def number_field(v: Json.Val, key: String) -> Maybe<&2, F32>:
  number_of_text(text_of_value(field(v, key)))
