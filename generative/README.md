# mlbend-generative

The generative skeleton, natively in Bend 2: randomness a verifier can draw
again, a noise schedule, a time embedding, and the two samplers' shared
walk — the strided subsequence of the schedule, and one DDIM step.

- Bend: **2.0.38**
- License: MIT
- No `@unsafe`, no `?TODO`: `bend generative/main.bend` prints `ALL PROOFS CHECK`,
  and `bend generative/main.bend --verdict` re-checks it with the Lean-proved kernel.

## Usage

```python
import Base
import mlbend-generative@0.1.1.0/main.bend as G

# three Gaussians from the counter 7: the twin draws the same three
def draw() -> List<&2, F32>:
  G.noise_of(7n, 3n)

# the schedule a sampler walks: every 4th rung of 64, top rung first
# (skip = (64-1) mod 4 lands the last entry on the last rung)
def rungs() -> List<&2, F32>:
  List.reverse(&2, F32, G.strided_from(F32, G.lin_alphas(64n), 4n, 3n))

# one DDIM step at eta = 0: sigma is 0, so the step reads no noise and
# the draw is the empty list
def one(x: List<&2, F32>, eps: List<&2, F32>) -> List<&2, F32>:
  G.ddim(G.step_of(0.0879891, 0.493844, 0.0), x, eps, Nil{})
```

## The randomness

Every number is a pure function of a counter — murmur3's fmix32 finaliser over
(step salt, index), Box-Muller over uniforms on a 24-bit grid — so a NumPy twin
draws the same numbers bit for bit.

| Function | What it answers |
|---|---|
| `mix32(z)` | murmur3's finaliser, over U32 arithmetic that wraps |
| `mix32_inv(z)` | its inverse, closed form (the bijection is tested) |
| `hash_of(s, i)` | the hash of counter `s` and index `i` |
| `uniform(s, i)` | the uniform on the 24-bit grid that hash names |
| `gauss_at(s, j)` | the j-th Gaussian of counter `s`, by Box-Muller |
| `noise_of(s, m)` | m Gaussians from counter `s`, as a list of m |

## The schedules

| Function | What it answers |
|---|---|
| `alpha_at(t, steps)` | ᾱ_t of the cosine schedule (Nichol & Dhariwal), closed form |
| `beta_at(t, steps)` | the cosine schedule's ratio 1 − ᾱ_t/ᾱ_{t−1}, clamped at 0.99 |
| `alphas(steps)` | the cosine schedule ᾱ_1 .. ᾱ_steps, in order |
| `lin_beta_at(t, steps)` | β_t of the linear schedule: 0.0001 + 0.2799·t/T |
| `lin_alphas(steps)` | the linear schedule ᾱ_1 .. ᾱ_steps, the running product |

`alpha_at` is a closed form per step, so the cosine table needs no running
product and every entry is reproducible on its own.

## The time embedding

`temb_of(t, steps, k)` is the sinusoidal embedding of step `t` over `k` rungs —
`2k` numbers (a sine and a cosine per rung), the input the denoiser sees beside
the sample. `freq_of(p)` is one rung's frequency.

## The strided walk

| Function | What it answers |
|---|---|
| `groups_of(n, k, skip)` | how many entries every k-th one of n gives, from `skip` |
| `strided_from(xs, k, skip)` | every k-th entry of `xs`, starting at index `skip` |
| `strided(xs, k)` | the same walk from index 0 |

The walk emits in order; `List.reverse` turns it into reverse time. The
`skip` argument is how a subsequence is anchored: `skip = (T−1) mod k` puts the
last rung on ᾱ_T, so the top of the walk is the top of the schedule for every k.

## The DDIM step

One step of Song et al. (2020), from the rung's ᾱ and the rung below it.

| Function | What it answers |
|---|---|
| `sigma_of(abar, abar_prev, eta)` | the noise scale σ of the step: 0 at η = 0, √β̃_t at η = 1 |
| `step_of(abar, abar_prev, eta)` | the five coefficients of one step (`Step{c0,c1,c2,c3,sig}`) |
| `ddim_elem(co, x, eps, z)` | one element's new value |
| `ddim(co, x, eps, z)` | one step over the whole sample |
| `head_or_zero(xs)` / `tail_or_nil(xs)` | the rung below a list: its head, or the empty answer at the end |

At η = 1 the step is the ancestral DDPM step over the exact posterior
(σ² = β̃_t) — the equivalence `reference/test_ddim.py` gates. At η = 0 σ is 0
and the step multiplies it into the draw, so a caller may pass `Nil{}` and the
step reads no counter at all.

## The laws

| Law | What it says |
|---|---|
| `noise_length` | the noise of a counter is exactly m numbers |
| `alphas_length` | a cosine schedule over t steps is exactly t entries |
| `lin_alphas_length` | a linear schedule over t steps is exactly t entries |
| `temb_length` | an embedding of k rungs is exactly 2k numbers |
| `strided_length` | the every-k-th walk over n entries emits exactly the count its own function says |
| `ddim_length` | one DDIM step over n elements is n elements |

In plain language, the two the samplers lean on:

- **`strided_length`** — the walk and its own counter agree: the list
  `strided_from` answers holds exactly as many entries as `groups_of` says it
  does, for every n, k and skip. A sampler counts its steps from that function
  (16 of 64 at stride 4) and pairs the ᾱ list with the timestep list entry by
  entry; if the walk could emit a different number than the count, the two
  lists would come apart — one would run out first and the remaining steps
  would have to run on a defaulted rung.
- **`ddim_length`** — a step maps the sample onto a sample of the same length:
  it can neither drop an element nor duplicate one. That is what lets a walk of
  steps chain — each step's output lines up element for element with the model's
  next input — and what makes the rung lists' lockstep (also `strided_length`)
  mean anything.

The shapes are proved; the numbers are not, because `F32` is not a real number.
The step's coefficients, the schedules and the noise are compared with the
NumPy twin by `reference/test_generative.py`, and `generative/tests/ok.bend`
pins them. Whether η = 0 consumes no counter at all is a claim about which
terms the loop evaluates rather than an equality over values, and the loop
lives in an example — it is gated as a test (`reference/test_ddim.py`: the
η = 0 cloud is bit-identical under a different draw flag and a different salt),
not stated as a law. The trust base is in [`docs/AUDIT.md`](../docs/AUDIT.md).
