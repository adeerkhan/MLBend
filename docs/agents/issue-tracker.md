# The issue tracker

Issues are where the work lives: one issue per piece of work, with the
acceptance criteria in it, and the issue is the handoff to whoever picks it up
next.

## The rules

1. **A law that cannot be proved yet is an issue, not a package.** The package
   keeps only what the kernel checks. The issue says what the law would be, in
   plain language, and what is missing to prove it.
2. **One issue per work item.** Not per file, not per idea: per thing that can be
   merged on its own.
3. **The acceptance criteria go in the issue**, in the words of the stop
   criteria of `docs/PLAN.md`: the law is proved by `--verdict`, the numbers
   match NumPy and scikit-learn, the README explains the claim in plain
   language, and `make check` is green.
4. **The issue is the memory.** An agent that opens the repository and reads the
   open issues knows what is planned, what is blocked and what was deliberately
   left out. `NOTES.md` says why; the issue says what.
5. **Close the issue with the change.** The closing comment (or the pull
   request) says what was done, what was measured and what was left. That is the
   next reader's starting point.

## The labels

| Label | What it means |
|---|---|
| `law` | a claim that is not proved yet |
| `package` | a new package, or a change to a package's surface |
| `numbers` | a benchmark, a measurement, a performance question |
| `upstream` | something for bendlang/bend (an issue or a pull request there) |

## The relation to the branches

- Work happens on `main`, one commit (or pull request) per issue.
- `main` is the work branch; CI checks every push. The old `devel` branch was
  merged into it on 2026-10-23.
- A pull request closes the issue it names (`Closes #12`).

## Where the numbers live

A measurement is not a comment: it goes in `NOTES.md` (what was measured, on
what, and what it cost) and, when it is a claim about a package's speed, in the
package's README and in the pull request that changed it. The rule from
`AGENTS.md` applies: say what you ran yourself and what you did not.
