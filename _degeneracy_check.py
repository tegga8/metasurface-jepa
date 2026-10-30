"""Is the spectrum-to-geometry inverse degenerate on MetaDiT's own corpus?

The symmetric + full-mask stratum came out statistically null (median |T|
correlation margin +0.0025, sign test p = 1.0 against a shuffled pairing). Two
explanations:

  (a) the model fails there, or
  (b) the condition is ill-posed -- many distinct geometries share nearly the
      same spectrum, so the target simply does not identify one geometry.

This measures (b) directly on real MetaDiT data, with no model involved. For
every sampled geometry it finds the OTHER geometry whose true CST spectrum is
closest, then asks how similar those two geometries actually are.

If the nearest-spectrum neighbour is a geometrically DIFFERENT design, the
inverse is not injective on the data manifold and no decoder could have
recovered the specific input. Reference points: random pairs and self-pairs.

Reported per stratum (occupancy band, and mirror-symmetry-constrained subsets)
so the symmetric case can be compared against the whole corpus.

Usage: python _degeneracy_check.py [n_items]
"""

import os
import sys

import numpy as np
from scipy import io

REPO = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(REPO, "data", "metadit", "split_data", "val_set.mat")
N_ITEMS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
CHUNK = 256


def iou(a, b):
    a = a > 0.5
    b = b > 0.5
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def mirror_score(p):
    ts = [p, np.fliplr(p), np.flipud(p), np.rot90(p, 2)]
    return float(np.mean([iou(p, t) for t in ts]))


def nn_analysis(pat, spec_mag, idx_pool, label):
    """For each i in idx_pool, the nearest OTHER j by spectrum distance."""
    # pattern is stored (64, 64, N): the ITEM axis is last, not first.
    P = pat[:, :, idx_pool].transpose(2, 0, 1).astype(np.float32)  # [M,64,64]
    S = spec_mag[idx_pool].astype(np.float32)     # [M,301]
    M = len(idx_pool)
    sd = S.std(axis=1, keepdims=True)
    S = S / np.maximum(sd, 1e-12)                # per-item normalised

    nn_j = np.empty(M, dtype=np.int64)
    nn_d = np.empty(M, dtype=np.float64)
    for a in range(0, M, CHUNK):
        b = min(a + CHUNK, M)
        D = np.abs(S[a:b, None, :] - S[None, :, :]).mean(axis=2)
        rows = np.arange(b - a)
        D[rows, np.arange(a, b)] = np.inf          # exclude self
        j = np.argmin(D, axis=1)
        nn_j[a:b] = j
        nn_d[a:b] = D[rows, j]

    # random-pair baseline on the same items
    rng = np.random.default_rng(0)
    ri = rng.integers(0, M, M)
    rj = rng.integers(0, M, M)
    keep = ri != rj
    rand_d = np.abs(S[ri[keep]] - S[rj[keep]]).mean(axis=1)
    # geometry IoU between each item and its nearest-spectrum neighbour
    nn_iou = np.array([iou(P[i], P[nn_j[i]]) for i in range(M)])

    print(f"\n=== {label} (n={M}) ===")
    print(f"  spectrum distance   NN: med {np.median(nn_d):.4f}  "
          f"p10 {np.percentile(nn_d, 10):.4f}")
    print(f"                      random pairs: med {np.median(rand_d):.4f}")
    ratio = np.median(nn_d) / max(np.median(rand_d), 1e-12)
    print(f"  NN/random distance ratio: {ratio:.3f}"
          f"   (1.0 = spectrum no more informative than chance)")
    print(f"  geometry IoU with that neighbour: med {np.median(nn_iou):.3f}  "
          f"p10 {np.percentile(nn_iou, 10):.3f}  "
          f"frac<0.8 {float((nn_iou < 0.8).mean()):.3f}  "
          f"frac<0.5 {float((nn_iou < 0.5).mean()):.3f}")
    return nn_iou, nn_d


def main():
    try:
        d = io.loadmat(SPLIT)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read {SPLIT}: {exc}")
    pat, par, real, imag = d["pattern"], d["parameter"], d["real"], d["imag"]
    n_items = pat.shape[-1]
    occ = pat.reshape(-1, n_items).sum(axis=0)

    mag = np.sqrt(real ** 2 + imag ** 2)            # |T|, [N,301]
    rng = np.random.default_rng(0)
    pool = rng.choice(n_items, min(N_ITEMS, n_items), replace=False)

    print("=" * 68)
    print("Spectrum->geometry degeneracy on MetaDiT's own validation split")
    print(f"sampled {len(pool)} of {n_items}; spectrum = |real+i*imag| over 301")
    print("=" * 68)

    all_iou, all_d = nn_analysis(pat, mag, pool, "whole sampled corpus")

    band = pool[(occ[pool] >= 1040) & (occ[pool] <= 2240)]
    nn_analysis(pat, mag, band, "occupancy band 1040-2240 (our novel range)")

    # The symmetric designs were generated with a partial mirror; check whether
    # the most symmetric and least symmetric halves behave differently, since
    # mirror symmetry is a second way the spectrum can lose information.
    sub = band[:600]
    ms = np.array([mirror_score(pat[:, :, i]) for i in sub])
    order = np.argsort(ms)
    hi = sub[order[-300:]]
    lo = sub[order[:300]]
    nn_analysis(pat, mag, lo, "least mirror-symmetric 300")
    nn_analysis(pat, mag, hi, "most mirror-symmetric 300")

    print("\nInterpretation: if the most mirror-symmetric subset has a LOW")
    print("nearest-neighbour IoU (many distinct geometries share a spectrum),")
    print("then that stratum is ill-posed and no decoder could identify the")
    print("specific input geometry from its spectrum alone.")


if __name__ == "__main__":
    main()