# Papers

A paper enters this repository as a **comparison**, not as a claim of
reproduction. Each folder holds:

```
papers/<name>/
  README.md        the paper, its claims, and the protocol used here
  data.py          the pinned data and the 64x64 cache
  <model>.py       the Python arm
  train.py         the Python run and its metrics
  results/         run.json, samples, report
```

The rule for every entry:

1. The paper's own numbers are quoted in a table and marked **paper**, with
   the resolution, data and compute they were measured on.
2. The numbers measured here are marked **this protocol**, with every
   deviation from the paper stated: resolution, dataset, model size, missing
   components, sampling budget.
3. The two are never presented as the same measurement. A 64x64 pixel run
   cannot be compared with an ImageNet 256 latent FID, and the folder says so.
4. Python comes first, because it is the reference the Bend arm is checked
   against. The Bend arm does not get to invent a different protocol.
5. Every number comes from a program and is written to `results/`.

## Entries

| paper | venue | Python | Bend | state |
|---|---|---|---|---|
| [TEM](TEM) | arXiv 2610.31889 | negative result | blocked | collapse recorded; Bend blocked on full attention, RoPE and adaLN |
