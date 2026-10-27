# A DDPM denoiser on digits

The same denoiser over real data: the first 1024 rows of `mstz/optdigits` (pinned revision, its SHA-256 in the manifest), 8x8 digits scaled to [0, 1]. 64 pixels in, a 64-wide SiLU body, 64-step schedule, 3000 full-batch AdamW steps, then 64 fresh digits by the ancestral loop and DDIM over every 4th rung. The image shows real digits on the left and the samples on the right, and the samplers are checked element by element against the NumPy twin by `reference/test_diffusion_digits.py` (a frozen-weights probe, because mean and std over a grid cannot see a reordered array).

```console
$ bendenv run -- bend examples/diffusion_digits/main.bend
```

First loss **8062.69** against the NumPy twin's — the gate — and the data mean to 4.6e-9. The final loss, both clouds' statistics and the grid are reported, not gated: the underfit denoiser makes the sampler expand geometrically, which is the known follow-up (a residual head), and the variance is factored by its largest deviation so the statistics stay finite at that magnitude. The curve, the grid and the manifest are in [`results/diffusion-digits/`](../../results/diffusion-digits/).

![plot](plot.svg)
