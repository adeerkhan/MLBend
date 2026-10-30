# Rectified flow on two moons

Flow matching replaces the diffusion chain with one vector field and an ODE:

    forward   x_t = noise*(1-t) + data*t      (t = 0 noise, t = 1 data)
    target    v   = data - noise
    loss      MSE(v_theta(x_t, t), v)
    sample    x <- x + v_theta(x, t)*dt,  Euler over linspace(0, 1, steps+1)

There is no beta schedule, no posterior, no sigma and no eta. The level of each
point comes from a U32 Weyl sequence, one level per point. The model is three
linear layers with SiLU (2 inputs for x_t, 1 for t, width 128), trained
full-batch for 6000 steps with AdamW at 3e-3.

```console
$ bendenv run -- bend examples/diffusion/flow_matching/main.bend
$ make flow          # the same, then results/flow-matching/
```

## What it measures

The data is the DDPM example's two moons, normalised to [-1, 1] on both axes.
The sample is 32 Euler steps from `noise_of(2000000)`.

| | value |
|---|---|
| first loss | 1.4768 |
| final loss | 0.8360 |
| last 100 mean | **0.8859** |
| data mean / std | 0.000 / 0.000 · 0.577 / 0.659 |
| sample mean / std | +0.049 / -0.048 · 0.610 / 0.675 |
| inside the data's box | 112/128 |

The NumPy probe (`reference/flow_probe.py`) measured loss 2.1253 -> 0.8662,
samples mean [-0.111, 0.150] / std [0.546, 0.745] against its data's
[-0.002, -0.000] / [0.583, 0.741], and 465/512 inside the box. This example
runs the same method on the repository's own 128-point data, so its means and
box count are compared with that data. The loss scale differs because the
probe's data has 4096 points and a different init draw.

![loss](../../../results/flow-matching/loss.svg)

![data and samples](../../../results/flow-matching/scatter.svg)

The frozen probe and its NumPy twin are still owed: the example is measured
end to end, but there is no element-by-element gate yet.
