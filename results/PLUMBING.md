# The plumbing split of one ViT training step

`make plumbing` builds and runs `bench/plumbing/main.bend` 3 times and
keeps the median of each phase. It imports the real example (`examples/diffusion/vit`)
and calls its own functions, so these are the numbers the model actually pays.

Machine: Linux-7.2.9-200.fc44.x86_64-x86_64-with-glibc2.43 · 28 CPUs · Bend 2.0.38

| phase | ms |
|---|---:|
| `init_params` (once per process) | 25029 |
| `rows` (1024 rows before the matrix) | 276 |
| `step_in` (batch, noise, mix, rows, matrix) | 279 |
| `Mat.from_list` (derived: step_in - rows) | 3 |
| `forward` (the model on that matrix) | 52 |
| `step_tgt` (noise, patches, matrix) | 139 |
| `train_new` (grad and optimizer buffers) | 0 |
| **one full step** | **543** |

Shares of the step: input and target construction **77%**,
forward 10%, loss + backward + optimizer
13%.

The step is plumbing. `step_in` and `step_tgt` together are
418 ms of 543 ms, and most of that is the list build: the
rows are assembled with `List.append` into a growing list, which is O(n²) in
the number of values, so it dominates even the forward. `init_params` pays the
same pattern over 100,736 parameters and costs 25.0 s
once per process.

That is why the device buys no speed: a matmul moved to the card is 10% of the
step, and the 77% around it stays on the host. It also reframes CIFAR: the
first fix is the row builder (cons and reverse, or build into an array once),
not the GPU.
