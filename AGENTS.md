# MLBend

Machine learning for Bend 2, in the shape scikit-learn has, with the shapes in
the types and the structure proved by the kernel.

## Where things are

- `metrics/`: the first package (`mlbend-metrics`), published to BendHub.
- `docs/PLAN.md`: the packages, the order and the stop criteria.
- `docs/AUDIT.md`: what is proved, what is tested, what is trusted.
- `NOTES.md`: every finding and every discarded hypothesis, by date.
- `reference/`: the NumPy and scikit-learn comparisons, and the runner.

## Branches

Work on `devel`: every commit and push goes there, and CI checks it. `main`
only receives releases, through a pull request `devel` → `main` merged after
its CI passes; the version tag and the GitHub release are made on `main` after
that merge.

## Rules

1. Bend is pinned to **2.0.37** (and Lean 4.34.0 for `--verdict`). A version
   bump is its own change, with an entry in `NOTES.md`.
2. Run the checker on every change; it takes a fraction of a second.
3. **Every proof is real.** No `@unsafe`, no `?TODO`, no axioms. A law that
   cannot be proved yet is recorded in `NOTES.md` as debt and left out of the
   package.
4. **Prove structure, test numbers.** Sizes, partitions and composition belong
   in the types and the laws; `F32` results are checked against NumPy and
   scikit-learn by the scripts in `reference/`.
5. **Every proved law is explained in plain language** in its package's README.
6. **Publishing to BendHub is public and permanent.** Four-number versions
   (`0.1.0.0`), names of 12 or more characters, a `LICENSE` next to the entry
   file. A published version never moves: a change is a new version.
7. **Benchmarks are honest**: the hardware, the versions, an idle machine, the
   losses and the noise, not only the best run.
8. Everything in the repository is in English. Commits are small, with clear
   messages and no attribution lines.
9. Public text (READMEs, `NOTES.md`, `docs/`) is written in the first person
   singular: "I measured", "my PR", never "we".

## Bend 2.0.37 pitfalls we paid for

The traps below are the ones this repository's code hit; they are the same
traps bend-ml paid for, and the fix is in each case.

- `match` only inspects a parameter or a pattern variable. A computed value
  (including a call) needs its own def, or a parameter computed by the caller:
  `same_len_of(Nat.is_eq(n, m), n, m)`, never `match Nat.is_eq(n, m):`.
- There is no `let` keyword: a let is `x = v`, and `+x = v` when the value is
  reused (and `Data`). A statement inside a case body needs the `;` that ends
  it, or the next `case` is parsed as part of it.
- No forward references: a def must be defined before it is used, so the
  helpers come first.
- Variables are affine; `+x` reuses a `Data` value. A parameter used in a type
  counts as a use. A `Result` is a `Type`, so it cannot be reused.
- `match` on two values takes two patterns: `case Con{y, yt} Con{p, pt}:` and
  `case _ _:` for the rest.
- `List<&2, F32>` is a list whose values may be used twice (the list is
  read-only); `List<&2, Nat>` likewise. `List.length(&2, F32, xs)` reads it
  without consuming it.
- `Nat` and `F32` are `Data`; `Result` and functions are `Type`.
- The comparison operators are not terms: use `Nat.is_eq`, `F32.is_eq` and
  `Nat.cmp`, and pass the `Bool` to a def that matches on it.
- A negative literal is not a term: write `(0.0 - 0.5 : F32)`.
