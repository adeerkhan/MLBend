# mlbend-generative: the generative skeleton, natively in Bend 2.
#
#   import mlbend-generative@0.1.0.0/main.bend as G
#
# What a generative model needs besides the network: randomness that a
# verifier can draw again, a noise schedule, and a time embedding. The
# randomness is a pure function of counters — murmur3's fmix32 finaliser
# over (step salt · φ, index), Box-Muller over uniforms on a 24-bit grid —
# so a NumPy twin draws the same numbers bit for bit. The schedule is the
# cosine one (Nichol & Dhariwal): ᾱ_t = f(t)/f(0) in closed form, so every
# entry is computed on its own and the table needs no running product.
#
# What is proved: the shape — the noise of any counter is exactly m
# numbers (noise_length), a schedule over t steps is exactly t
# (alphas_length), an embedding of k rungs is exactly 2k (temb_length).
# The builders recurse on the count left, the current index riding after
# it, and the law's variables are ordered the same way — the structural
# variable before the riding one — which is what lets the checker follow
# the induction (NOTES.md, 2026-10-25). What is tested: every F32 number,
# against the NumPy twin (reference/test_generative.py), and the
# bijection of mix32: the inverse is closed form (the half-width xor-shift
# is an involution, the 13-shift inverts by the triple xor, the multiplies
# by their modular inverses mod 2^32) and generative/tests/ok.bend checks
# it on a grid — a law would need bit-level equational reasoning the
# kernel has not been asked to do, so the bijection is the debt of issue
# #10, not a claim.

import Base

# ---- the counter RNG ----

# murmur3's fmix32 finaliser, over U32 arithmetic that wraps
def mix32(+z: U32) -> U32:
  +a = U32.xor(z, U32.shrn(z, 16n))
  +b = (a * 2246822519 : U32)
  +c = U32.xor(b, U32.shrn(b, 13n))
  +e = (c * 3266489917 : U32)
  U32.xor(e, U32.shrn(e, 16n))

# the 13-shift undone: xor the shifted value twice more
def unxor13(+y: U32) -> U32:
  a = U32.xor(y, U32.shrn(y, 13n))
  U32.xor(a, U32.shrn(y, 26n))

# the inverse of mix32, in closed form: the 16-shift is an involution on
# 32 bits; the 13-shift inverts by the triple xor; the multiplies invert
# by their modular inverses mod 2^32
def mix32_inv(+z: U32) -> U32:
  +d = U32.xor(z, U32.shrn(z, 16n))
  +c = (d * 2828982549 : U32)
  +b = unxor13(c)
  +a = (b * 3066638151 : U32)
  U32.xor(a, U32.shrn(a, 16n))

# the hash of one counter: the step salt times the golden ratio, the
# index added beside it
def hash_of(+s: Nat, +i: Nat) -> U32:
  +k = (U32.from_nat(s) * 2654435769 : U32)
  (k + (U32.from_nat(i) + 305419898 : U32) : U32)

# a uniform in (0, 1]: the low 24 bits of the hash plus one, exactly
# representable, so log(u) is never log(0)
def uniform(+s: Nat, +i: Nat) -> F32:
  h = mix32(hash_of(s, i))
  m = ((U32.and(h, U32.from_nat(16777215n)) + 1 : U32) : U32)
  (F32.from_nat(U32.to_nat(m)) / 16777216.0 : F32)

# one Gaussian of a counter, by parity: the even index of the pair brings
# the radius, the odd one the angle
def gauss_body(is_even: Bool, +r: F32, +ang: F32) -> F32:
  match is_even:
    case True{}:
      (r * F32.cos(ang) : F32)
    case False{}:
      (r * F32.sin(ang) : F32)

def gauss_at(+s: Nat, +j: Nat) -> F32:
  +pair = Nat.sub(j, Nat.mod(j, 2n))
  +ua = uniform(s, pair)
  +ub = uniform(s, (1n + pair : Nat))
  +r = F32.sqrt((0.0 - (2.0 * F32.log(ua) : F32) : F32))
  gauss_body(Nat.is_eq(Nat.mod(j, 2n), 0n), r, (6.2831855 * ub : F32))

# The noise of counter s, left numbers from index `from` on. The count
# left is the recursion; the index rides after it.
def noise_from(+s: Nat, +left: Nat, +from: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{gauss_at(s, from), noise_from(s, rest, (1n + from : Nat))}

# the noise of counter s: m numbers, the index 0 first
def noise_of(+s: Nat, +m: Nat) -> List<&2, F32>:
  noise_from(s, m, 0n)

# LAW 1 (noise_length): m numbers asked for are m numbers built.
law noise_length:
  for +s: Nat
  for +m: Nat
  for +from: Nat
  {List.length(&2, F32, noise_from(s, m, from)) == m : Nat}

# m numbers asked for are m numbers built
def noise_length(s, m, from):
  match m:
    case 0n:
      {==}
    case 1n+rest:
      %noise_length(s, rest, (1n + from : Nat)) :
        {1n+List.length(&2, F32, noise_from(s, rest, (1n + from : Nat))) == 1n+_ : Nat}
      {==}

# ---- the cosine schedule ----
#
# ᾱ_t = f(t)/f(0) with f(t) = cos²(((t/T) + 0.008)/1.008 · π/2): a closed
# form per step, so the table needs no running product and every entry is
# reproducible on its own.

def cos_f(+t: Nat, +steps: Nat) -> F32:
  +x = (((F32.from_nat(t) / F32.from_nat(steps) : F32) + 0.008 : F32) / 1.008 : F32)
  +c = F32.cos((1.5707964 * x : F32))
  (c * c : F32)

# ᾱ_t of the cosine schedule over `steps` steps (t = 1 .. steps)
def alpha_at(+t: Nat, +steps: Nat) -> F32:
  (cos_f(t, steps) / cos_f(0n, steps) : F32)

# β_t, the ratio the reverse step removes: 1 - ᾱ_t/ᾱ_{t-1}, clamped at
# 0.99 — the cosine schedule's last ratio is 1 in F32 (ᾱ_T ~ 1e-15), and
# the reverse step's 1/sqrt(1-β) would amplify any prediction error
def beta_at(+t: Nat, +steps: Nat) -> F32:
  F32.min((1.0 - (alpha_at(t, steps) / alpha_at(Nat.sub(t, 1n), steps) : F32) : F32), 0.99)

# The schedule from step `from` on, `left` entries. The count left is the
# recursion; the step rides after it.
def alphas_from(+steps: Nat, +left: Nat, +from: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      Con{alpha_at(from, steps), alphas_from(steps, rest, (1n + from : Nat))}

# ᾱ_1 .. ᾱ_steps, in order
def alphas(+steps: Nat) -> List<&2, F32>:
  alphas_from(steps, steps, 1n)

# LAW 2 (alphas_length): a schedule of t entries is t entries.
law alphas_length:
  for +t: Nat
  for +left: Nat
  for +from: Nat
  {List.length(&2, F32, alphas_from(t, left, from)) == left : Nat}

# a schedule of t entries is t entries
def alphas_length(t, left, from):
  match left:
    case 0n:
      {==}
    case 1n+rest:
      %alphas_length(t, rest, (1n + from : Nat)) :
        {1n+List.length(&2, F32, alphas_from(t, rest, (1n + from : Nat))) == 1n+_ : Nat}
      {==}

# ---- the linear schedule ----
#
# For few-step samplers (tens of steps, not thousands): a linear β is
# gentle at the top (the cosine schedule's last ratios approach 1, and a
# reverse step with β near 1 multiplies any prediction error by
# 1/sqrt(1-β), so few-step cosine diverges). β runs from 0.0001 to 0.28,
# which lands ᾱ_T near 1e-4 — pure noise — over 64 steps with every
# reverse scale at most 1.18.

# β_t of the linear schedule over `steps` steps (t = 1 .. steps)
def lin_beta_at(+t: Nat, +steps: Nat) -> F32:
  (0.0001 + ((0.2799 * (F32.from_nat(t) / F32.from_nat(steps) : F32)) : F32) : F32)

# The schedule ᾱ_1 .. from `from` on, `left` entries, the running product
# riding beside. The count left is the recursion; the step and the
# product ride after it.
def lin_alphas_from(+steps: Nat, +left: Nat, +from: Nat, +acc: F32) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      +acc2 = (acc * (1.0 - lin_beta_at(from, steps) : F32) : F32)
      Con{acc2, lin_alphas_from(steps, rest, (1n + from : Nat), acc2)}

# ᾱ_1 .. ᾱ_steps of the linear schedule, in order
def lin_alphas(+steps: Nat) -> List<&2, F32>:
  lin_alphas_from(steps, steps, 1n, 1.0)

# LAW 4 (lin_alphas_length): a linear schedule of t entries is t entries.
law lin_alphas_length:
  for +t: Nat
  for +left: Nat
  for +from: Nat
  for +acc: F32
  {List.length(&2, F32, lin_alphas_from(t, left, from, acc)) == left : Nat}

# a linear schedule of t entries is t entries
def lin_alphas_length(t, left, from, acc):
  match left:
    case 0n:
      {==}
    case 1n+rest:
      %lin_alphas_length(t, rest, (1n + from : Nat),
        (acc * (1.0 - lin_beta_at(from, t) : F32) : F32)) :
        {1n+List.length(&2, F32, lin_alphas_from(t, rest, (1n + from : Nat),
          (acc * (1.0 - lin_beta_at(from, t) : F32) : F32))) == 1n+_ : Nat}
      {==}

# ---- the sinusoidal time embedding ----

# the frequency of rung p: a decade ladder 0.001 .. 1.0
def freq_of(+p: Nat) -> F32:
  match p:
    case 0n:
      0.001
    case 1n:
      0.01
    case 2n:
      0.1
    case _:
      1.0

# The embedding of step t over `steps`, from rung `p` on, `left` rungs —
# each rung a sine and a cosine. The rungs left are the recursion; the
# rung index rides after them.
def temb_from(+t: Nat, +steps: Nat, +left: Nat, +p: Nat) -> List<&2, F32>:
  match left:
    case 0n:
      Nil{}
    case 1n+rest:
      +tau = (F32.from_nat(t) / F32.from_nat(steps) : F32)
      +f = freq_of(p)
      Con{F32.sin((tau * f : F32)), Con{F32.cos((tau * f : F32)),
        temb_from(t, steps, rest, (1n + p : Nat))}}

# the embedding of step t: k rungs, the lowest frequency first
def temb_of(+t: Nat, +steps: Nat, +k: Nat) -> List<&2, F32>:
  temb_from(t, steps, k, 0n)

# LAW 3 (temb_length): an embedding of k rungs is 2k numbers.
law temb_length:
  for +t: Nat
  for +steps: Nat
  for +k: Nat
  for +p: Nat
  {List.length(&2, F32, temb_from(t, steps, k, p)) == Nat.mul(k, 2n) : Nat}

# an embedding of k rungs is 2k numbers
def temb_length(t, steps, k, p):
  match k:
    case 0n:
      {==}
    case 1n+rest:
      %temb_length(t, steps, rest, (1n + p : Nat)) :
        {1n+(1n+List.length(&2, F32, temb_from(t, steps, rest, (1n + p : Nat)))) == 1n+(1n+_) : Nat}
      {==}
