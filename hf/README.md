# mlbend-hf

A HuggingFace client for Bend 2.

```python
import mlbend-hf@0.1.0.0/main.bend as Hf
```

Four calls, all JSON over HTTP:

| Call | What it answers |
|---|---|
| `Hf.dataset(name)` | the dataset's page: its id, its revision, what it says about itself |
| `Hf.tree(name, rev)` | the files in it |
| `Hf.file(name, rev, path)` | a file's bytes; `Hf.file_text` the same as text |
| `Hf.rows(name, config, split, offset, length)` | the rows server's JSON rows, with the features beside them |

The rows server answers a dataset that is stored as data. A dataset that is
a script rather than data cannot be read that way: the server says so, and
its message says what to do about it (convert the dataset to Parquet). The
file tree and the raw file download work for those anyway.

The HTTP is `bend-kit-http`'s, the JSON is `bend-kit-json`'s, and the bytes
are `bend-kit-bytes`'s. Nothing else.

## Why the shape is what it is

Bend's checker will not let a match scrutinize a computed value, so every
helper here that unwraps a `Maybe` takes it as a parameter and matches on
that. A caller gets one call each:

```python
Hf.text_field(json, "id")                                  # the id, or ""
Hf.text_item_field_field(Hf.array_field(json, "rows"), 0n, "row", "label")
Hf.number_field(json, "offset")
```

## The errors

The package has its own error type, so a caller needs no HTTP kit:

```python
type Err is Data:
  ErrNet{code: U32, why: String}
  ErrStatus{status: U32}
  ErrJson{}
```
