# The plumbing split of one training step

`make plumbing` builds `bench/plumbing/main.bend`, runs it three times, and
writes `results/plumbing.json` and `results/PLUMBING.md`.

The program imports the real ViT example and calls its own `init_params`,
`step_in`, `step_tgt` and `Model` functions. It times each phase with
`IO.now()` and prints it, so these are the numbers the model pays, not a
reimplementation.

```console
$ make plumbing
$ bendenv run -- bend bench/plumbing/main.bend   # the raw lines
```

A caveat paid for here: a phase that returns `Unit` is erased as unused and
times as 0. Every measured value is printed after its timer, so the compiler
cannot drop the work.

The result is in `results/PLUMBING.md`. In short: the input and target
construction dominate the step, the forward is about a tenth of it, and the
row build is O(n²) because it appends into a growing list. That is why moving
the matmul to the GPU changed nothing.
