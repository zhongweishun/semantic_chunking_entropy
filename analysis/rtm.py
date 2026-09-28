"""Finite-N entropy recurrence, seeded weak compositions, and scaling densities."""
import math
import random
import numpy as np
from scipy.signal import fftconvolve


def expected_entropy(k, nmax):
    """H(n)=ln Z_k(n)+k/Z_k(n) sum_{m=2}^{n-1} Z_{k-1}(n-m) H(m).

    H(0)=H(1)=0. Repeated cumulative sums evaluate the binomial convolution
    in O(k*nmax) time and O(nmax+k) memory, without a dense transition matrix.
    This is the expected labelled-outcome score used by size_and_logprob.
    """
    if k < 2 or nmax < 2:
        raise ValueError('Require k >= 2 and nmax >= 2')
    h, sums = np.zeros(nmax+1), np.zeros(k-1)
    for n in range(2, nmax+1):
        for j in range(1, k-1):
            sums[j] += sums[j-1]
        z = math.comb(n+k-1, k-1)
        h[n] = math.log(z) + k*sums[-1]/z
        sums += h[n]
    return h


def weak_composition(n, k, rng):
    bars = sorted(rng.sample(range(n+k-1), k-1))
    return [b-a-1 for a,b in zip([-1]+bars, bars+[n+k-1])]


def simulate_score(n, k, rng):
    """Stop if a sampled composition has at most one nonempty part."""
    pending, score = [n], 0.0
    while pending:
        size = pending.pop()
        if size <= 1:
            continue
        score += math.log(math.comb(size+k-1, k-1))
        parts = weak_composition(size, k, rng)
        if sum(v > 0 for v in parts) > 1:
            pending.extend(v for v in parts if v > 1)
    return score/n


def typicality(replicates=128, seed=20260928):
    rng = random.Random(seed)
    sizes = np.unique(np.geomspace(10, 10000, 14).astype(int))
    rates = np.asarray([[simulate_score(int(n), 4, rng) for _ in range(replicates)] for n in sizes])
    return sizes, rates


def scaling_densities(k=4, max_level=11, points=2**16, tmax=46.051701859880914):
    """Mellin product via convolution in t=-ln(s).

    R~Beta(1,k-1), s_L=product_{j=1}^{L-1} R_j. Densities in log size
    are computed on a uniform nonnegative t grid. The finite window is
    the same s>=1e-20 window used by the source FFT notebook.
    """
    t = np.linspace(0, tmax, points)
    dt = t[1]-t[0]
    s = np.exp(-t)
    base = (k-1)*s*(1-s)**(k-2)
    base /= np.trapz(base, t)
    current = base.copy()
    out = {}
    for level in range(2, max_level+1):
        if level > 2:
            current = fftconvolve(current, base)[:points]*dt
            current = np.maximum(current, 0)
        # Do not renormalize each level: record mass lost beyond the window.
        out[level] = current.copy()
    return t, out
