# Notes

Every finding, every measurement and every discarded hypothesis, by date. Read the
relevant entry before changing performance-sensitive code or a law.

## 2026-10-20

- **v0.1: `mlbend-metrics`.** The first package: the evaluation surface (MSE,
  MAE, RMSE, accuracy, precision, recall, F1) and the counts they rest on. The
  ten metrics of `metrics/tests/ok.bend` match NumPy and scikit-learn within a
  relative 1e-5 (`reference/test_metrics.py`). One law proved so far:
  `count_length`, the number of samples is the length of the label list.
- **The plan** is in `docs/PLAN.md`: the packages, the order and the stop
  criteria. The order is metrics -> linalg -> preprocess -> split -> linear ->
  cluster -> pipeline -> datasets -> select.
- **Bend traps this package paid for** (they are in `AGENTS.md`): `match` only
  inspects a parameter, so every comparison is computed by the caller and passed
  in (`same_len_of(Nat.is_eq(n, m), n, m)`); there is no `let` keyword (`x = v`,
  and `+x = v` to reuse); a def cannot be used before it is defined, so the
  helpers come first; `Result` is a `Type` and cannot be reused with `+`; the
  comparison operators are not terms (`Nat.is_eq`, `F32.is_eq`); a negative
  literal is not a term (`(0.0 - 0.5 : F32)`).
- **Debt, not in the package.**
  - The accounting law ("the samples predicted right and the samples predicted
    wrong add up to the number of samples") is not in v0.1. It needs the
    hypothesis that the two label lists have the same length, and the proof
    needs the counts to come out of one walk, so the statement can name both.
    It is v0.2's first law, with the `mlbend-linalg` inner-product lemmas.
  - The classification metrics are stated per class (`precision(ys, ps, k)`);
    scikit-learn also has a macro and a weighted average over the classes.
    Those come with the confusion matrix, in v0.2.
  - `r2` and the log loss are not here: both need the variance of the targets
    (a second sum) and a zero-denominator case of their own. v0.2.
- **The process, and where the memory lives.** An agent's durable context is
  this repository, not a transcript: `AGENTS.md` (the rules, the read order, the
  traps), `docs/agents/handoff.md` (where we are, what was run, what was not),
  `docs/agents/domain.md` (the vocabulary and the design rules),
  `docs/agents/issue-tracker.md` (how the issues carry the work), `NOTES.md`
  (what was measured and what failed), `docs/AUDIT.md` (the trust base) and the
  issues themselves (the debt). The laws are the real memory: a claim that
  survives the Lean kernel needs no log to stay true.
- **The bend repository stays where it is.** It is not vendored here: it is a
  fork with two remotes, it moves daily, and the copy would be wrong tomorrow.
  The pinned install (`~/.bend`) is the versioned reference (the guide and
  `base.bend` come with it), the hub cache (`~/.bend/lib`) is an offline mirror
  of the published packages, and `make sync-bend` fetches the pinned tag into
  `../bend-ref` when a read-only copy of the compiler is wanted.
- **Housekeeping on the bend fork**: the `/understand` knowledge graph (`.ua/`)
  is generated and must never be committed — it is now in that repository's
  `.git/info/exclude`, because it would otherwise fail bend's own repo gate (the
  allow list) the first time an agent ran `git add -A`.
- **The base layer exists (2026-10-20).** The plan was wrong about
  `mlbend-linalg`: the ecosystem already has the containers, the CSV reader,
  the bytes, the integers, the soft float, the proved lemmas and the parser
  combinators (`bend-kit-csv`, `-bytes`, `-int`, `-f64`, `-lemmas`, `-parse`),
  and bend-ml has the tensors and the reverse-mode autodiff. MLBend writes the
  *machine-learning* surface and imports the rest. A second `Mat` and a second
  set of lemmas would be the third copy of the same thing in the ecosystem, so
  the `mlbend-linalg` package is off the plan.
- **The numbers gap, and the ask (2026-10-20).** Base can print a number but
  not read one: no `U32.of_string`, no `F32.of_string`, and `bend-kit-f64`
  says so outright ("Decimal text is not here"). The data layer needs it, so
  the ask is upstream: bendlang/bend#1789, with the shape (Base's own
  `Result<&1, &1, U32 & String, A>`) and the law (`U32.of_string
  (U32.show x) == x`, provable). Until it lands, v0.2 carries a small decimal
  parser and drops it when Base has one.
- **The environment (2026-10-20).** The repository is a workspace now: a
  root `bend.toml` names the members (`metrics`), one `bend.lock` holds
  every package in the closure, and one `.bend/env` holds them all.
  `bendenv` is at <https://github.com/adeerkhan/bend-env>; it is a Rust
  tool that takes uv's shape (a global content-addressed cache, hard
  links into the environment, parallel downloads, one lock per
  workspace). `make env` builds it, `make env-check` verifies it and
  runs `examples/stacks/main.bend` over it.
  - **The example is the proof.** One program, one environment, two
    stacks: `bend-ml-tensor-array@0.1.14.0` computes a 2x2 product
    (`[12, 12, 12, 12]`) and `bendtorch-core@0.1.0.0` builds a tensor
    of four words. Both come from the lock.
  - **The ecosystem's tensors do not mix, and the tool says so.** A
    Bend type belongs to the package that declares it, so
    `bend-ml-tensor` and `bend-ml-tensor-array` are two different
    `Mat`, and so are the `G` of `bend-ml-autograd` and of
    `bend-ml-tensor-array`. `bendenv` reports every type two packages
    of the closure declare. The consequence: **the flat tensor library
    has no autodiff behind it** — `bend-ml-autograd@0.1.1.0` is written
    against the list-based `bend-ml-tensor@0.1.0.0`. That is the gap
    the deep-learning half has to close.
  - **The versions skew too.** `bend-ml-autograd@0.1.1.0` imports
    `bend-ml-nat-lemmas@0.1.0.0` while this repository asks for
    `0.1.1.0`; both are in the lock, and the tool says so. Not a bug:
    a Bend source names its dependencies exactly, so two versions can
    live side by side.
  - **The publish path is verified.** `bendenv hash <file.bend>`
    computes the hash a package would publish under, and on
    `aaaxn/bendtorch`'s checkout it gives
    `0x1689e5deb278c16abd3bd45eac101ef3` — the hash the hub holds for
    `bendtorch-core@0.1.0.0`, over the same 196 files. So the file
    set, the paths, the digests and the hash are the compiler's, byte
    for byte.
- **The data (2026-10-20).** iris (150 rows, 4 features) and breast cancer
  (569 rows, 30 features) are in `mlbend-datasets/data/`, taken from
  scikit-learn's copies (which come from UCI). A hub package is one file, so
  the data lives with the program that uses it, not in the package: the loaders
  take a path.
