# The two ML stacks, in one environment

Bend has more than one tensor library on BendHub, and they do not mix.
This example is the proof that one environment can hold both of them
and that the program says which is which.

```console
$ bendenv sync
synced 6 packages, 196 files fetched, 204 linked, 0 already there

$ bendenv run -- bend examples/stacks/main.bend
bend-ml-tensor-array: [12, 12, 12, 12] | bendtorch-core: built a tensor of 4 words
```

## What it shows

**The environment resolves what the source names.** The program imports
`bend-ml-tensor-array@0.1.14.0`, `bend-ml-nat-lemmas@0.1.1.0` and
`bendtorch-core@0.1.0.0`. All three are in `bend.lock` with their
content hashes, and all three are in `.bend/env`, checked against those
hashes. A missing one is a failed import, not a silent difference.

**The two stacks do not mix, and the tool says so.** `bend-ml-autograd`
and `bend-ml-tensor-array` both declare a type named `G`;
`bend-ml-tensor` and `bend-ml-tensor-array` both declare `Mat`. A Bend
type belongs to the package that declares it, so those are two
different `Mat`, and a program that passes one to the other does not
compile. `bendenv lock` reports it:

```
two packages of the closure declare the same type. A Bend type
belongs to the package that declares it, so they are two
different types, and a program that mixes them does not compile:
  G: bend-ml-autograd, bend-ml-tensor-array
  Mat: bend-ml-tensor, bend-ml-tensor-array
```

So `bend-ml-autograd`, which is written against the list-based
`bend-ml-tensor`, cannot be used with the flat `bend-ml-tensor-array`.
The fast tensor library has no autodiff behind it yet.

**The versions skew, too.** `bend-ml-autograd@0.1.1.0` imports
`bend-ml-nat-lemmas@0.1.0.0`, while this repository asks for
`0.1.1.0`. Both are in the lock, and `bendenv lock` says so. It is not
a bug: a Bend source names its dependencies exactly, so two versions
can live side by side. It is the ecosystem's, and it is visible.

## The files

| | |
|---|---|
| `bend.toml` | the workspace: the members, and the dependencies they share |
| `metrics/bend.toml` | the metrics package, and the one dependency it names |
| `bend.lock` | every package in the closure, its version, its hash, its files, its types |
| `.bend/env/` | the environment itself, hardlinked out of the global cache (not committed) |

`bendenv` is at <https://github.com/adeerkhan/bend-env>.
