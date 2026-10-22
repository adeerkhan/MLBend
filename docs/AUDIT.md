# Audit guide

This project claims that some properties are **proved** and others only
**tested**. Tests cannot replace a review of the proved statements: a proof is
only as meaningful as the law it proves. This guide tells a reviewer what to
read, what is trusted, and how to re-check everything.

## 1. Read the specification, not the proofs

```bash
reference/.venv/bin/python reference/list_laws.py                  # every law: its claim
reference/.venv/bin/python reference/list_laws.py --check-readme   # every law is in its README
```

Questions to ask of each law:

- Is the claim what the README says it is (read the plain-language table next
  to it)?
- Is it **vacuous**? A law whose hypothesis is never satisfiable proves nothing.
- Do the **definitions** the law mentions mean what their names say?
- The quantity marks: `+x` means "may be used more than once", `-x` means
  "erased at run time"; they restrict how a proof may use a variable, they do
  not weaken the statement.

## 2. What is proved

| Package | Laws | What they guarantee |
|---|---|---|
| `mlbend-metrics` | `count_length` | the number of samples is the length of the list of labels |
| `mlbend-preprocess` | `one_hot_count_length` | the one-hot matrix has one row for every label |
| `mlbend-split` | `split_covers` | every row of the dataset is placed in a fold, so the folds partition the rows |
| `mlbend-select` | `select_covers` | every row of the dataset is in exactly one fold |

Besides laws, the **types** carry guarantees that no law restates: a metric's
two sequences are `List<&2, F32>` (the values may be read twice) and
`List<&2, Nat>` for the labels, and a metric that mixes them does not compile.

## 3. What is trusted (the trust base)

1. **Bend's type checker and the proof kernel.** `bend X/main.bend --verdict`
   re-checks a file with `bendtt.lean`, a small kernel with a Lean proof that
   no accepted definition has type `Empty` and that live code halts
   (`paper/BendTT.pdf` in bendlang/bend). The compiler and the runtime, which
   run the code, are trusted more than the proof: a bug in them is a bug in
   every claim.
2. **NumPy and scikit-learn**, for the numbers. `reference/test_metrics.py`
   compares the metrics with them; the comparison is within a relative `1e-5`,
   because `F32` is not a real number and the two round differently.
3. **The published packages this repository imports.** `mlbend-metrics`
   imports `bend-ml-nat-lemmas@0.1.1.0` for the arithmetic lemmas (its
   `add_zero` is what the `count_length` proof leans on). That package is
   trusted as far as its own audit says: its laws are proved by the same kernel,
   and its source is readable offline in the hub cache (`~/.bend/lib`).
4. **The host's floating point.** `F32` arithmetic is IEEE-754 single precision;
   the sums are in the order the code writes them, so a run is reproducible on
   one machine and one compiler.

## 4. What is not claimed

- No metric value is proved. `F32` has no algebraic laws to speak of.
- The count of a class is not proved to be a part of the total (the accounting
  law is debt; see `NOTES.md`).
- Nothing is proved about the *order* of the sums: they are fixed by the code,
  not by a law.
