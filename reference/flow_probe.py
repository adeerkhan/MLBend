"""Rectified flow on two moons: the design probe before a Bend port.

  reference/.venv/bin/python reference/flow_probe.py

Mirrors github.com/lucidrains/rectified-flow-pytorch's path exactly, and unlike
DDPM there is almost nothing else to mirror:

    forward   x_t = noise*(1-t) + data*t      (t = 0 noise, t = 1 data)
    target    v   = data - noise
    loss      MSE(v_theta(x_t, t), v)
    sample    x <- x + v_theta(x, t)*dt,  Euler over linspace(0, 1, steps+1)

No beta schedule, no posterior, no sigma, no eta: one vector field and an ODE.
The level per step comes from a U32 Weyl sequence, not Nat (ISSUE-017).
"""
import numpy as np

F = np.float32
N, D, H, STEPS, BATCH, EULER, LR = 4096, 2, 128, 6000, 256, 32, 3e-3


def moons(n, seed=0):
    rng = np.random.RandomState(seed)
    u = rng.uniform(0, np.pi, n // 2).astype(F)
    x = np.concatenate([np.stack([np.cos(u), np.sin(u)], 1),
                        np.stack([np.cos(u), -np.sin(u)], 1) + np.array([1.0, -0.5], F)])
    lo, hi = x.min(0, keepdims=True), x.max(0, keepdims=True)
    return (2.0 * (x - lo) / (hi - lo) - 1.0).astype(F)


def init(seed=1):
    rng = np.random.RandomState(seed)
    lin = lambda i, o: (rng.randn(o, i) * np.sqrt(2.0 / i)).astype(F)
    return {"w1": lin(D + 1, H), "b1": np.zeros(H, F), "w2": lin(H, H),
            "b2": np.zeros(H, F), "w3": lin(H, D), "b3": np.zeros(D, F)}


def gelu(x):
    return (F(0.5) * x * (1.0 + np.tanh(F(0.7978845608) * (x + F(0.044715) * x ** 3)))).astype(F)


def dgelu(x):
    th = np.tanh(F(0.7978845608) * (x + F(0.044715) * x ** 3))
    return (F(0.5) * (1.0 + th) + F(0.5) * x * (1.0 - th ** 2) * F(0.7978845608)
            * (1.0 + 3 * F(0.044715) * x ** 2)).astype(F)


def levels(s, n):
    """One U32 Weyl level per point, moved on by the step."""
    a, c = 2654435761, 2246822519
    j = np.arange(n, dtype=np.uint64)
    mixed = (np.uint64(s * a) + j * np.uint64(c)) & np.uint64(0xFFFFFFFF)
    return (mixed.astype(np.float64) / 2.0 ** 32).astype(F)


def forward(w, x, t):
    a1 = np.concatenate([x, t[:, None]], 1).astype(F)
    z1 = a1 @ w["w1"].T + w["b1"]
    z2 = gelu(z1) @ w["w2"].T + w["b2"]
    return (gelu(z2) @ w["w3"].T + w["b3"]).astype(F)


def train(data, steps=STEPS, seed=2):
    rng = np.random.RandomState(seed)
    w = init(seed)
    m = {k: np.zeros_like(v) for k, v in w.items()}
    v = {k: np.zeros_like(v) for k, v in w.items()}
    b1, b2, eps, curve = 0.9, 0.999, 1e-8, []
    for s in range(steps):
        x1 = data[rng.randint(0, len(data), BATCH)]
        x0 = rng.randn(BATCH, D).astype(F)
        t = levels(s, BATCH)
        xt = (x0 * (1.0 - t[:, None]) + x1 * t[:, None]).astype(F)
        target = (x1 - x0).astype(F)
        a1 = np.concatenate([xt, t[:, None]], 1).astype(F)
        z1 = a1 @ w["w1"].T + w["b1"]; g1 = gelu(z1)
        z2 = g1 @ w["w2"].T + w["b2"]; g2 = gelu(z2)
        out = (g2 @ w["w3"].T + w["b3"]).astype(F)
        d = (2.0 * (out - target) / out.size).astype(F)
        gw3 = d.T @ g2; gb3 = d.sum(0)
        dz2 = (d @ w["w3"] * dgelu(z2)).astype(F)
        gw2 = dz2.T @ g1; gb2 = dz2.sum(0)
        dz1 = (dz2 @ w["w2"] * dgelu(z1)).astype(F)
        gw1 = dz1.T @ a1; gb1 = dz1.sum(0)
        grads = {"w1": gw1, "b1": gb1, "w2": gw2, "b2": gb2, "w3": gw3, "b3": gb3}
        for k in w:
            m[k] = (b1 * m[k] + (1 - b1) * grads[k]).astype(F)
            v[k] = (b2 * v[k] + (1 - b2) * grads[k] ** 2).astype(F)
            w[k] = (w[k] - LR * (m[k] / (1 - b1 ** (s + 1)))
                    / (np.sqrt(v[k] / (1 - b2 ** (s + 1))) + eps)).astype(F)
        curve.append(float(np.mean((out - target) ** 2)))
    return w, curve


def sample(w, n=512, steps=EULER, seed=3):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, D).astype(F)
    dt = F(1.0 / steps)
    for i in range(steps):
        x = (x + forward(w, x, np.full(n, i * dt, F)) * dt).astype(F)
    return x


def main():
    data = moons(N)
    w, curve = train(data)
    smp = sample(w)
    print(f"loss: first {curve[0]:.4f}  last100 {np.mean(curve[-100:]):.4f}")
    print(f"data   mean {data.mean(0).round(3)}  std {data.std(0).round(3)}")
    print(f"sample mean {smp.mean(0).round(3)}  std {smp.std(0).round(3)}")
    lo, hi = data.min(0), data.max(0)
    print(f"inside the data's box: {int(((smp >= lo).all(1) & (smp <= hi).all(1)).sum())}/{len(smp)}")


if __name__ == "__main__":
    main()
