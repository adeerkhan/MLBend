# The lucidrains reference (vendored)

A verbatim copy of [`lucidrains/denoising-diffusion-pytorch`](https://github.com/lucidrains/denoising-diffusion-pytorch)
at the commit in [`VENDORED.md`](VENDORED.md), MIT licensed. It is a
**reference**, not a dependency: no Bend package imports it, `make check` does
not run it, and the Bend layer still writes no tensor, no autodiff and no
optimizer.

It exists because the diffusion examples needed a second opinion that is not
my own NumPy mirror. The mirror agrees with the example by construction; this
library is what the field actually uses, and comparing against it is how the
digits denoiser's failure was found (the local issue ledger, ISSUE-001).

## The environment

PyTorch is not in `reference/.venv` (that venv is NumPy-only, and it is what
`make check` uses). This reference has its own, gitignored:

```console
$ python3.14 -m venv reference/.venv-lucidrains
$ reference/.venv-lucidrains/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
$ reference/.venv-lucidrains/bin/pip install -r reference/lucidrains/requirements.txt
```

The CPU index keeps it to ~200 MB instead of pulling the CUDA build; the
versions that were used are pinned in `requirements.txt`.

## What runs against it

- `reference/lucidrains_compare.py` — trains the same denoiser task
  (noise prediction, T = 64, linear schedule) on the same 1024 optdigits rows
  as `examples/diffusion/digits/main.bend`, prints both schedules side by side,
  trains, samples with DDIM (16 calls) and with the ancestral loop (64), and
  writes `results/lucidrains-diffusion/`.

  ```console
  $ reference/.venv-lucidrains/bin/python reference/lucidrains_compare.py
  $ STEPS=200 reference/.venv-lucidrains/bin/python reference/lucidrains_compare.py   # smoke
  ```

- `reference/digits_ablation.py` — the controlled experiment that isolated
  the digits failure: our exact MLP and optimizer with init, input range and
  schedule changed one at a time.

  ```console
  $ reference/.venv-lucidrains/bin/python reference/digits_ablation.py
  ```

## What was taken from it (not copied, learned)

The parts that changed the Bend side, with the files they come from:

| Reference | Where | What Bend did |
|---|---|---|
| zero-mean init at `1/sqrt(fan_in)` | PyTorch `nn.Linear` default | `examples/diffusion/digits` `value()` rebuilt (ISSUE-001) |
| x̂₀ clamped to [-1, 1] before the posterior mean | `p_mean_variance`, `ddim_sample` (`maybe_clip`) | open — see the local issue ledger H2 |
| data normalized to [-1, 1] | `normalize_to_neg_one_to_one` | open — H3 |
| `linear_beta_schedule` = `linspace(1e-4·1000/T, 0.02·1000/T, T)` | `linear_beta_schedule` | divergence recorded (ISSUE-002) |
| sinusoidal embedding injected per block | `Unet` + `SinusoidalPosEmb` | ours concatenates 8 values into the input row — H4 |

The DDIM step itself was **already right**: `mlbend-generative`'s `sigma_of`
and `step_of` expand to the reference's `sigma` and `c` exactly (checked line
by line when the files were read, 2026-10-27).

Licence: MIT, Copyright (c) 2020 Phil Wang — see [`LICENSE`](LICENSE).
