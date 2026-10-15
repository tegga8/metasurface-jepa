"""Phase-2 feasibility check: how realizable are the random target spectra?

The phase-2 verdict ("the decode does not radiate the requested spectrum, no better
than a mismatched target") is only a model failure if the requested spectra are
realizable by SOME geometry. The released corpus is the set of spectra we know are
realizable, so the decisive, model-free measurement is:

    for each target t:  d_nn(t) = min over the corpus of  normalized-L1(t, r)

in the same units as every reported phase-2 error (mean|diff| / std(target)).

Reference scales computed in the same metric:
  * a real spectrum's own distance to its nearest OTHER real spectrum  -> corpus density
  * two unrelated real spectra                                          -> "mismatched target" scale
  * a real spectrum against its own true target                         -> 0 by construction

Usage: python _phase2_feasibility.py <val_set.mat> <random_spectra.npz> [n_ref]
"""

import sys

import numpy as np
from scipy import io


def norm_l1(a, b):
    """Evaluator's normalized L1: mean|a-b| / std(b), a and b complex [301]."""
    return float(np.abs(a - b).mean() / max(float(np.abs(b).std()), 1e-12))


def main():
    mat_path = sys.argv[1] if len(sys.argv) > 1 else None
    npz_path = sys.argv[2] if len(sys.argv) > 2 else None
    n_ref = int(sys.argv[3]) if len(sys.argv) > 3 else 3000

    d = io.loadmat(mat_path)
    R = d["real"].astype(np.float64)
    I = d["imag"].astype(np.float64)
    C = R + 1j * I
    rng = np.random.default_rng(0)
    idx = rng.choice(len(C), min(n_ref, len(C)), replace=False)
    P = C[idx]
    print(f"corpus reference set: {len(P)} real spectra from {mat_path}")

    z = np.load(npz_path, allow_pickle=True)
    targets = {
        "A interpolated": (z["A_real"] + 1j * z["A_imag"]).astype(np.float64),
        "B synthetic": (z["B_real"] + 1j * z["B_imag"]).astype(np.float64),
    }

    # corpus density: each real spectrum vs its nearest OTHER real spectrum
    self_nn = []
    for a in range(0, len(P), 64):
        blk = P[a:a + 64]
        dmat = np.abs(blk[:, None, :] - P[None, :, :]).mean(axis=2) \
            / np.maximum(P.std(axis=1, keepdims=True)[None, :, 0], 1e-12)
        for i in range(len(blk)):
            dmat[i, a + i] = np.inf        # exclude self
        self_nn.extend(dmat.min(axis=1).tolist())

    # mismatched-target scale: random real pairs
    perm = rng.permutation(len(P))
    mism = [norm_l1(P[i], P[perm[i]]) for i in range(len(P))]

    print(f"\n{'scale':34s} {'median':>8s} {'mean':>8s}")
    print(f"{'corpus self-NN (density)':34s} {np.median(self_nn):8.4f} "
          f"{np.mean(self_nn):8.4f}")
    print(f"{'unrelated real spectra':34s} {np.median(mism):8.4f} "
          f"{np.mean(mism):8.4f}")

    for name, T in targets.items():
        nn = []
        for t in T:
            n = len(P)
            dist = np.abs(P - t[None, :]).mean(axis=1) / max(float(np.abs(t).std()), 1e-12)
            nn.append(float(dist.min()))
        nn = np.array(nn)
        print(f"{name + ' -> nearest real':34s} {np.median(nn):8.4f} "
              f"{np.mean(nn):8.4f}   (n={len(nn)})")
        print(f"{'   per-target:':34s} {np.round(nn, 3).tolist()}")


if __name__ == "__main__":
    main()
