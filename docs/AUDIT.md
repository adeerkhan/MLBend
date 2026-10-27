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
| `mlbend-cluster` | `cluster_covers` | every row gets exactly one cluster, so no row is left out and none is in two |
| `mlbend-pipeline` | `pipeline_preserves_rows` | transforming the rows keeps one output row for every input row |
| `mlbend-report` | `matrix_rows`, `row_length` | `n` rows of the confusion matrix are `n` rows, and `n` cells of a row are `n` cells |
| `mlbend-generative` | `noise_length`, `alphas_length`, `lin_alphas_length`, `temb_length`, `strided_length`, `ddim_length` | m numbers asked for are m numbers built, the schedules and embeddings have exactly as many entries as steps and rungs, the every-k-th walk emits exactly the count its own function says, and one DDIM step maps n elements to n elements |

### The experiment layer

The eight packages are the substrate; the experiment layer is
`reference/experiment.py`. It does not prove anything new: it makes the tested
numbers auditable. For every task it runs the Bend program, computes the same
number with NumPy or scikit-learn, and writes a report card with the delta and
the tolerance. The manifest beside it records the git commit (and whether the
tree was dirty), the Bend version, every `mlbend-*`, `bend-ml-*` and
`bendtorch-*` package with its hub hash, the machine, the seed and the
tolerances. `--wandb` sends the same manifest to Weights & Biases (offline when
there is no API key).

What the layer claims: *this program produced this number, and this reference
agrees within this tolerance, from a state that can be re-run.* What it does
not claim: that the folds are the same rows as the reference's (the generators
differ, so the comparison is procedural), or that a single run characterises a
model. Those limits are in `docs/PLAN.md`.

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
   because `F32` is not a real number and the two round differently. The
   diffusion twins are a mirror of the *whole* training loop, and their first
   loss is the only number gated: that gate runs one step and no update, so it
   cannot see a wrong backward pass. It could not, in fact — both twins
   evaluated the SiLU derivative at the activated value instead of the
   pre-activation, and no gate noticed until the Bend arm learned and the
   twins did not (`issues-hit.md`, ISSUE-006). The digits twins are now also
   compared with the vendored lucidrains reference, which is an outside
   implementation rather than a mirror of our own code.
3. **The published packages this repository imports.** `mlbend-metrics`
   imports `bend-ml-nat-lemmas@0.1.1.0` for the arithmetic lemmas (its
   `add_zero` is what the `count_length` proof leans on). That package is
   trusted as far as its own audit says: its laws are proved by the same kernel,
   and its source is readable offline in the hub cache (`~/.bend/lib`).
   `mlbend-split` and `mlbend-select` depend on `bend-kit-random@0.1.0.1`,
   which carries two defs that call the operating system (`entropy` and
   `from_os`). Neither package calls them — both seed the generator — but the
   verdict refuses a book that holds any foreign code, so **their laws are
   checked by Bend's checker and not certified by the kernel**. That is a
   weaker guarantee than `mlbend-metrics`'s, and it is the price of not
   writing a generator of our own.
4. **The host's floating point.** `F32` arithmetic is IEEE-754 single precision;
   the sums are in the order the code writes them, so a run is reproducible on
   one machine and one compiler.

## 4. What is not claimed

- No metric value is proved. `F32` has no algebraic laws to speak of.
- The count of a class is not proved to be a part of the total (the accounting
  law is debt; see `NOTES.md`).
- Nothing is proved about the *order* of the sums: they are fixed by the code,
  not by a law.
- No claim about the sampled cloud. The two samplers' numbers are tested
  against the NumPy twin (`reference/test_diffusion.py`, `reference/test_ddim.py`),
  and whether a step at η = 0 reads no noise counter at all is a gate on
  output, not a law: it is a claim about which terms the loop evaluates, and
  the loop lives in an example (issue #12 records why it is a test).
