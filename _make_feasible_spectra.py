"""Phase-2 targets, re-specified with a realizability screen.

The original set A ("random convex combinations of real spectra") measured 1.87
from the nearest real spectrum in the evaluator's normalized-L1 units -- 27x the
corpus's own nearest-neighbour density (0.069) -- so the decode was being scored
against a target no corpus geometry can realize. Roughness does not catch that;
distance to the realizable set does.

Two metrics are used because a convex combination is FLATTER than any real
spectrum, which inflates mean|diff|/std(target) mechanically:
  * l1n  = mean|a-b| / std(b)      the evaluator's metric (sensitive to flatness)
  * corr = Pearson correlation     scale-free shape agreement

Sets produced:
  A' feasible   accepted only if l1n <= L1_MAX and corr >= CORR_MIN against the
                nearest real spectrum (i.e. genuinely on/near the data manifold)
  B' infeasible band-limited noise, as before, kept as the negative control --
                reported with the same two metrics so both sets are characterised
                by the same instrument.

Usage: python _make_feasible_spectra.py <out.npz> [n_each] [val_set.mat]
"""

import json
import os
import sys

import numpy as np
from scipy import io

REPO = os.path.dirname(os.path.abspath(__file__))
NFREQ = 301
L1_MAX = 0.5
CORR_MIN = 0.90
MAX_ATTEMPTS = 4000


def load_val(path):
    d = io.loadmat(path)
    return d["real"].astype(np.float64), d["imag"].astype(np.float64)


def smooth_noise(n, nf, rng, n_modes=6):
    u = np.linspace(0, 1, nf)
    out = np.empty((n, nf))
    for i in range(n):
        centres = rng.uniform(0.0, 1.0, n_modes)
        widths = rng.uniform(0.05, 0.25, n_modes)
        B = np.exp(-((u[None, :] - centres[:, None]) ** 2)
                   / (2.0 * widths[:, None] ** 2))
        w = rng.normal(size=(n_modes, nf))
        out[i] = (B * w).sum(axis=0) / (np.abs(B).sum(axis=0) + 1e-9)
    return out


def screen(t, P):
    """Nearest-real l1n and corr for one complex target (P complex [N,301])."""
    std = max(float(np.abs(t).std()), 1e-12)
    l1 = np.abs(P - t[None, :]).mean(axis=1) / std
    j = int(np.argmin(l1))
    a, b = np.abs(t), np.abs(P[j])
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    # also the best-correlated neighbour, and its l1
    cors = []
    return float(l1[j]), corr, int(j)


def main():
    out_path = sys.argv[1]
    n_each = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    mat = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
        REPO, "data", "metadit", "split_data", "val_set.mat")
    R, I = load_val(mat)
    N = len(R)
    rng = np.random.default_rng(0)
    P = R + 1j * I

    # corpus density reference (both metrics)
    sub = rng.choice(N, min(3000, N), replace=False)
    Ps = P[sub]
    dens_l1, dens_corr = [], []
    for a in range(0, min(400, len(Ps)), 40):
        blk = Ps[a:a + 40]
        d = np.abs(blk[:, None, :] - Ps[None, :, :]).mean(axis=2) \
            / np.maximum(np.abs(Ps).std(axis=1)[None, :], 1e-12)
        for i in range(len(blk)):
            d[i, a + i] = np.inf
        dens_l1.extend(d.min(axis=1).tolist())
    print(f"corpus density: l1n median {np.median(dens_l1):.4f} "
          f"(p95 {np.percentile(dens_l1, 95):.4f})")

    A_re, A_im, A_stat = [], [], []
    attempts = 0
    rejects = 0
    while len(A_re) < n_each and attempts < MAX_ATTEMPTS:
        attempts += 1
        k = int(rng.integers(3, 6))
        idx = rng.choice(N, k, replace=False)
        w = rng.dirichlet(np.ones(k))
        rr = (w[:, None] * R[idx]).sum(axis=0)
        ii = (w[:, None] * I[idx]).sum(axis=0)
        l1, corr, j = screen(rr + 1j * ii, Ps)
        if l1 <= L1_MAX and corr >= CORR_MIN:
            A_re.append(rr)
            A_im.append(ii)
            A_stat.append({"l1n": l1, "corr": corr, "mix": [int(x) for x in idx]})
        else:
            rejects += 1
    print(f"A' feasible: {len(A_re)} accepted after {attempts} draws "
          f"({rejects} rejected)")
    if A_stat:
        print(f"   accepted l1n median {np.median([s['l1n'] for s in A_stat]):.4f} "
              f"| corr median {np.median([s['corr'] for s in A_stat]):.4f}")

    env_r = np.abs(R).mean(axis=0)
    env_i = np.abs(I).mean(axis=0)
    B_re, B_im, B_stat = [], [], []
    while len(B_re) < n_each:
        rr = env_r * (1.0 + 0.5 * smooth_noise(1, NFREQ, rng)[0])
        ii = env_i * (1.0 + 0.5 * smooth_noise(1, NFREQ, rng)[0])
        m = np.abs(rr + 1j * ii)
        if m.max() > 1.0:
            rr, ii = rr / m.max(), ii / m.max()
        l1, corr, _ = screen(rr + 1j * ii, Ps)
        B_re.append(rr)
        B_im.append(ii)
        B_stat.append({"l1n": l1, "corr": corr})
    print(f"B' infeasible: {len(B_re)} built | l1n median "
          f"{np.median([s['l1n'] for s in B_stat]):.4f} | corr median "
          f"{np.median([s['corr'] for s in B_stat]):.4f}")

    np.savez(out_path,
             A_real=np.stack(A_re).astype(np.float32),
             A_imag=np.stack(A_im).astype(np.float32),
             B_real=np.stack(B_re).astype(np.float32),
             B_imag=np.stack(B_im).astype(np.float32),
             A_screen=json.dumps(A_stat), B_screen=json.dumps(B_stat),
             meta=json.dumps({"l1_max": L1_MAX, "corr_min": CORR_MIN,
                              "corpus_density_l1n_median":
                              float(np.median(dens_l1))}))
    print("saved", out_path)


if __name__ == "__main__":
    main()
