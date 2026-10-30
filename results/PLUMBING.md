# The plumbing split of one ViT training step

`make plumbing` builds and runs `bench/plumbing/main.bend` 3 times and
keeps the median of each phase. It imports the real example (`examples/diffusion/vit`)
and calls its own functions, so these are the numbers the model actually pays.

Machine: Linux-7.2.9-200.fc44.x86_64-x86_64-with-glibc2.43 · 28 CPUs · Bend 2.0.38

| phase | ms |
|---|---:|
| `init_params` (once per process) | 6 |
| `rows` (1024 rows before the matrix) | 6 |
| `step_in` (batch, noise, mix, rows, matrix) | 6 |
| `Mat.from_list` (derived: step_in - rows) | 0 |
| `forward` (the model on that matrix) | 53 |
| `step_tgt` (noise, patches, matrix) | 1 |
| `train_new` (grad and optimizer buffers) | 0 |
| **one full step** | **131** |

Shares of the step: input and target construction **5%**,
forward 40%, loss + backward + optimizer
54%.

The step is now arithmetic. `step_in` and `step_tgt` together are
7 ms of 131 ms (5%); the forward is
53 ms and loss + backward + optimizer are 71 ms.
`init_params` costs 6 ms once per process.

Before the builder fix (2026-10-30, the same program) a step was 543 ms, with
77% in the input and target build, and `init_params` was 25.3 s. The builders
appended into a growing list (O(n²)) and the tokeniser called `at()` 64 times
per digit. With cons + one reverse and a linear 16-pixel tokeniser the build
falls to 7 ms, and the values are bit-identical: the full 20,480
value row list and the first three losses were compared against the old
version.

The device is worth re-measuring now: the forward and backward are
95% of the step, and the host no longer
hides them.
