"""Compares mlbend-generative with the NumPy twin.

  reference/.venv/bin/python reference/test_generative.py

Every number of the package is a pure function of counters, so the twin
computes the same functions in float32 with the same operation order and
the two must agree: the uniforms and the Gaussians, the cosine schedule
and its betas, the time embedding, the strided walk and its count, and
one DDIM step (the coefficients and the elementwise update). The
bijection of mix32 is tested on a grid both here and in
generative/tests/ok.bend; proving it is the debt of issue #10. The count
is also checked against its closed form on a grid, which is what lets
strided_length (a statement about the walk) be read as a statement about
a number.
"""
import sys

import numpy as np

F = np.float32
np.seterr(over="ignore")  # uint32 wrap-around is the point, not a problem


def mix32(z):
    z = np.uint32(z)
    a = np.uint32(z ^ (z >> np.uint32(16)))
    b = np.uint32(a * np.uint32(2246822519))
    c = np.uint32(b ^ (b >> np.uint32(13)))
    e = np.uint32(c * np.uint32(3266489917))
    return np.uint32(e ^ (e >> np.uint32(16)))


def mix32_inv(z):
    z = np.uint32(z)
    d = np.uint32(z ^ (z >> np.uint32(16)))
    c = np.uint32(d * np.uint32(2828982549))
    b = np.uint32(c ^ (c >> np.uint32(13)) ^ (c >> np.uint32(26)))
    a = np.uint32(b * np.uint32(3066638151))
    return np.uint32(a ^ (a >> np.uint32(16)))


def hash_of(s, i):
    k = np.uint32(np.uint32(s) * np.uint32(2654435769))
    return np.uint32(k + np.uint32(np.uint32(i) + np.uint32(305419898)))


def uniform(s, i):
    h = mix32(hash_of(s, i))
    m = np.uint32((h & np.uint32(16777215)) + np.uint32(1))
    return F(F(m) / F(16777216.0))


def gauss_at(s, j):
    pair = j - (j % 2)
    ua = uniform(s, pair)
    ub = uniform(s, pair + 1)
    r = np.sqrt(F(0.0) - F(2.0 * np.log(ua)))
    ang = F(6.2831855 * ub)
    return F(r * np.cos(ang)) if j % 2 == 0 else F(r * np.sin(ang))


def cos_f(t, steps):
    x = F(F(F(F(t) / F(steps)) + F(0.008)) / F(1.008))
    c = np.cos(F(1.5707964 * x)).astype(F)
    return F(c * c)


def alpha_at(t, steps):
    return F(cos_f(t, steps) / cos_f(0, steps))


def beta_at(t, steps):
    return min(F(1.0 - F(alpha_at(t, steps) / alpha_at(t - 1, steps))), F(0.99))


def temb_of(t, steps, k):
    tau = F(F(t) / F(steps))
    freqs = [F(0.001), F(0.01), F(0.1), F(1.0)]
    out = []
    for p in range(k):
        f = freqs[p] if p < 4 else F(1.0)
        out += [np.sin(F(tau * f)).astype(F), np.cos(F(tau * f)).astype(F)]
    return out


def groups_of(n, k, skip):
    """The count of the strided walk, as the Bend recursion does it."""
    if n == 0:
        return 0
    if skip == 0:
        return 1 + groups_of(n - 1, k, max(k - 1, 0))
    return groups_of(n - 1, k, skip - 1)


def strided_from(xs, k, skip):
    """The walk itself: drop `skip`, emit one, drop k-1, emit one, ..."""
    if not xs:
        return []
    if skip == 0:
        return [xs[0]] + strided_from(xs[1:], k, max(k - 1, 0))
    return strided_from(xs[1:], k, skip - 1)


def sigma_of(abar, abar_prev, eta):
    above = np.maximum(F(0.0), F(F(1.0) - abar_prev))
    below = np.maximum(F(0.0), F(F(1.0) - F(abar / abar_prev)))
    return F(eta * F(np.sqrt(F(above / F(F(1.0) - abar))) * np.sqrt(below)))


def step_of(abar, abar_prev, eta):
    """The five coefficients, in the order step_of computes them."""
    sig = sigma_of(abar, abar_prev, eta)
    c0 = F(F(1.0) / np.sqrt(abar))
    c1 = np.sqrt(np.maximum(F(0.0), F(F(1.0) - abar)))
    c2 = np.sqrt(np.maximum(F(0.0), abar_prev))
    c3 = np.sqrt(np.maximum(F(0.0), F(F(1.0) - F(abar_prev + F(sig * sig)))))
    return c0, c1, c2, c3, sig


def ddim(co, x, eps, z):
    """One step over the lists: eps and z pad with 0.0 when short."""
    c0, c1, c2, c3, sig = co
    out = []
    for i in range(len(x)):
        e = eps[i] if i < len(eps) else F(0.0)
        zz = z[i] if i < len(z) else F(0.0)
        x0 = F(F(x[i] - F(c1 * e)) * c0)
        out.append(F(F(F(x0 * c2) + F(c3 * e)) + F(sig * zz)))
    return out


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def main():
    fail = 0
    checks = [
        ("u0", uniform(7, 0), 0.79354215),
        ("u1", uniform(7, 1), 0.047255337),
        ("g0", gauss_at(7, 0), 0.6503142),
        ("g1", gauss_at(7, 1), 0.19896899),
        ("g100", gauss_at(42, 100), 0.7640456),
        ("beta1", beta_at(1, 8), 0.04219544),
        ("beta8", beta_at(8, 8), 0.99),
    ]
    alphas = [alpha_at(t, 8) for t in range(1, 9)]
    want_alphas = [0.95780456, 0.8470122, 0.6842266, 0.49384364,
                   0.3043949, 0.14427207, 0.037471935, 1.9109826e-15]
    temb = temb_of(3, 8, 4)
    want_temb = [0.000375, 0.99999994, 0.003749991, 0.99999297,
                 0.037491214, 0.99929696, 0.36627254, 0.9305076]

    for name, got, want in checks:
        if close(got, want):
            print(f"ok   {name}: {float(got)} vs {want}")
        else:
            print(f"ERROR {name}: {float(got)} vs {want}")
            fail += 1
    for i, (a, w) in enumerate(zip(alphas, want_alphas)):
        if not close(a, w, 1e-3):
            print(f"ERROR alpha_{i+1}: {float(a)} vs {w}")
            fail += 1
    for i, (a, w) in enumerate(zip(temb, want_temb)):
        if not close(a, w):
            print(f"ERROR temb[{i}]: {float(a)} vs {w}")
            fail += 1
    if not (alphas and temb and not fail):
        pass

    # the strided walk, its count and one DDIM step, against the pins
    a8 = [alpha_at(t, 8) for t in range(1, 9)]
    xs = [F(1.0), F(2.0), F(-0.5)]
    es = [F(0.1), F(-0.2), F(0.3)]
    zs = [F(0.0), F(1.0), F(-1.0)]
    zbig = [F(1000.0), F(-1000.0), F(500.0)]
    walk_checks = [
        ("strided3", strided_from(a8, 3, 2), [0.6842266, 0.14427207]),
        ("strided0", strided_from(a8, 3, 0), [0.95780456, 0.49384364, 0.037471935]),
        ("stridedn", strided_from(list(range(10)), 3, 2), [2, 5, 8]),
        ("groups_a", [groups_of(10, 3, 0)], [4]),
        ("groups_b", [groups_of(10, 3, 2)], [3]),
        ("groups_c", [groups_of(64, 4, 3)], [16]),
        ("groups_d", [groups_of(5, 4, 3)], [1]),
        ("groups_e", [groups_of(0, 4, 0)], [0]),
        ("groups_f", [groups_of(8, 8, 7)], [1]),
        ("groups_g", [groups_of(7, 1, 0)], [7]),
        ("groups_h", [groups_of(3, 5, 0)], [1]),
        ("sigma1", [sigma_of(F(0.25), F(0.5), F(1.0))], [0.57735026]),
        ("sigma0", [sigma_of(F(0.25), F(0.5), F(0.0))], [0.0]),
        ("sigmac", [sigma_of(F(0.6), F(0.5), F(1.0))], [0.0]),
        ("step", list(step_of(F(0.25), F(0.5), F(1.0))),
         [2.0, 0.8660254, 0.70710677, 0.4082483, 0.57735026]),
        ("ddim", ddim(step_of(F(0.25), F(0.5), F(1.0)), xs, es, zs),
         [1.3325639, 3.5690765, -1.5294058]),
        ("ddimz", ddim(step_of(F(0.25), F(0.5), F(1.0)), xs, es, []),
         [1.3325639, 2.9917264, -0.95205563]),
        ("eta0a", ddim(step_of(F(0.25), F(0.5), F(0.0)), xs, es, zs),
         [1.3624496, 2.9319549, -0.8623981]),
        ("eta0b", ddim(step_of(F(0.25), F(0.5), F(0.0)), xs, es, zbig),
         [1.3624496, 2.9319549, -0.8623981]),
    ]
    bad = [c for c in walk_checks
           if not (len(c[1]) == len(c[2]) and all(close(g, w) for g, w in zip(c[1], c[2])))]
    if bad:
        for name, got, want in bad:
            print(f"ERROR {name}: {[float(g) for g in got]} vs {want}")
        fail += len(bad)
    else:
        print(f"ok   the walk, the count and one DDIM step ({len(walk_checks)} checks)")

    # the count against its closed form, and the walk against the count:
    # drop `skip`, then one every k, is ceil((n-skip)/k) whenever there
    # is anything left to emit. This is the arithmetic the proved
    # strided_length transfers onto the list walk.
    grid_bad = []
    for n in range(0, 40):
        for k in range(1, 7):
            for skip in range(0, 8):
                want = 0 if n <= skip else (n - skip - 1) // k + 1
                if (groups_of(n, k, skip) != want
                        or len(strided_from(list(range(n)), k, skip)) != want):
                    grid_bad.append((n, k, skip))
    if grid_bad:
        print(f"ERROR the walk and its count off the closed form: {grid_bad[:6]}")
        fail += 1
    else:
        print("ok   the walk and its count agree with ceil((n-skip)/k) on 1680 counts")
    roundtrip = sum(1 for z in range(1, 4097) if int(mix32_inv(mix32(np.uint32(z)))) == z)
    if roundtrip == 4096:
        print(f"ok   mix32 roundtrip on 4096 values")
    else:
        print(f"ERROR mix32 roundtrip: {roundtrip}/4096")
        fail += 1
    if fail:
        print(f"{fail} checks differ")
        return 1
    print("the generative skeleton matches the twin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
