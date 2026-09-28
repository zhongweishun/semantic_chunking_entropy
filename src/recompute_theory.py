"""Recompute the RTM level-distribution theory curve for a given branching
factor k at story length N, writing theory/theory_dist_k=<k>.json.

    python recompute_theory.py --k 6 --N 2500 --lmax 11

Output format matches the shipped theory files: {L: {"distn": [[...]],
"dist0": [[...]], "dist01": [[...]]}} where distn[-1] is the level-L
distribution of node sizes n at N (used by reddit1000/make_chunksize_figure.py). The math is the
absorbing-Markov-chain formulation from rtm_theory_utilities.py, vectorised so
large N is tractable. Use this when a corpus needs a k/L range not already
shipped in theory/.
"""
import os
import argparse
import json
import numpy as np
from scipy.special import binom

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def Tij_matrix(N, k):
    """(N+1)x(N+1) matrix Tij = C(j-i+k-2, k-2) / C(j+k-1, k-1) for j >= i."""
    i = np.arange(N + 1)[:, None].astype(float)
    j = np.arange(N + 1)[None, :].astype(float)
    M = binom(j - i + k - 2, k - 2) / binom(j + k - 1, k - 1)
    M[j < i] = 0.0
    return M


def transition_matrix_n(N, k):
    T = np.zeros((2 * (N + 1), 2 * (N + 1)))
    T[:N + 1, :N + 1] = np.eye(N + 1)
    M = Tij_matrix(N, k)
    np.fill_diagonal(M, 0.0)          # strict inequality j > i
    T[N + 1:, N + 1:] = M
    T[:, N + 1] = 0.0
    T[:, N + 2] = 0.0
    idx = np.arange(N + 1)
    T[idx, N + 1 + idx] = 1.0 / binom(idx + k - 1, k - 1)
    T[0, N + 1] = 1.0
    T[1, N + 2] = 1.0
    return T


def occupation_transition_n(N, k):
    T = transition_matrix_n(N, k)
    diag = np.zeros(2 * (N + 1))
    diag[N + 3:] = k
    return T * diag[None, :]


def occupation_transition_01(N, k):
    T = k * Tij_matrix(N, k)
    T[:, 0] = 0.0
    T[:, 1] = 0.0
    return T


def compute(k, N, l_min, l_max):
    Qn = occupation_transition_n(N, k)
    Q01 = occupation_transition_01(N, k)
    T0 = Tij_matrix(N, k)
    mn = np.zeros(2 * (N + 1)); mn[-1] = 1.0
    m01 = np.zeros(N + 1); m01[-1] = 1.0
    p0 = np.zeros(N + 1); p0[-1] = 1.0
    e_n = np.ones(2 * (N + 1)); e_n[0] = 0.0; e_n[N + 1] = 0.0
    e_01 = np.ones(N + 1); e_01[0] = 0.0
    out = {}
    for L in range(2, l_max + 1):
        mn = Qn @ mn
        m01 = Q01 @ m01
        p0 = T0 @ p0
        if L < l_min:
            continue
        CL = e_n @ mn
        distn = (mn[1:N + 1] + mn[N + 2:]) / CL
        NL = e_01 @ m01
        dist01 = m01[1:] / NL
        dist0 = p0[1:] / (1 - p0[0])
        out[L] = {"distn": [distn.tolist()], "dist0": [dist0.tolist()],
                  "dist01": [dist01.tolist()]}
        print(f"L={L}: C_L={CL:.4g}, sum distn={distn.sum():.6f}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--N", type=int, default=2500)
    ap.add_argument("--lmin", type=int, default=2)
    ap.add_argument("--lmax", type=int, default=11)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = compute(a.k, a.N, a.lmin, a.lmax)
    dst = a.out or os.path.join(REPO, "theory", f"theory_dist_k={a.k}.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, "w", encoding="utf-8"))
    print("saved", dst)
