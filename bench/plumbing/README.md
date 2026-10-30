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

The result is in `results/PLUMBING.md`. Before the fix, input and target
construction were 77% of a 543 ms step and `init_params` was 25.3 s: the
builders appended into a growing list (O(n²)) and the tokeniser scanned the
digit 64 times. With cons + one reverse and a linear tokeniser the step is
131 ms, the build is 7 ms, `init_params` is 6 ms, and the values are
bit-identical. The step is now forward and backward.
