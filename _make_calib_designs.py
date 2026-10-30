"""Build the Meep calibration design set from MetaDiT's own validation split.

Picks N val items stratified by occupancy fraction across the dataset's actual
range, and writes two npz files for the fixed harness and the convention fit:

  meep_calib_designs.npz  patterns (N,64,64) uint8, scalars (N,3), names
  meep_calib_targets.npz  real (N,301), imag (N,301)  <- MetaDiT's CST spectra

The harness output for these geometries is what the Meep->CST convention is
fitted and validated on, so this set replaces the old 4-item fit whose
residuals reached 3898 and which was produced by the defective harness.

Usage: python _make_calib_designs.py [n_items]
"""

import os
import sys

import numpy as np
from scipy import io

REPO = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(REPO, "data", "metadit", "split_data", "val_set.mat")

N_ITEMS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
# The split's occupancy is tightly clustered (5th-99th pct = 1040..2240 of
# 4096), so stratify inside that band rather than over the full range.
OCC_LO, OCC_HI = 1040, 2240


def main():
    d = io.loadmat(SPLIT)
    pat, par, real, imag = d["pattern"], d["parameter"], d["real"], d["imag"]
    n_items = pat.shape[-1]
    occ = pat.reshape(-1, n_items).sum(axis=0)

    cand = np.where((occ >= OCC_LO) & (occ <= OCC_HI))[0]
    # Order by occupancy so quantile picks span the fill-fraction range evenly.
    order = cand[np.argsort(occ[cand])]
    fracs = [(i + 0.5) / N_ITEMS for i in range(N_ITEMS)]
    picks = sorted({int(order[int(f * (len(order) - 1))]) for f in fracs})
    print(f"val items={n_items} candidates={len(cand)} picked={len(picks)}")

    pats, scs, names = [], [], []
    for idx in picks:
        m = (pat[:, :, idx] == 1).astype(np.uint8)
        l, h, r = (float(par[idx, 0]), float(par[idx, 1]), float(par[idx, 2]))
        pats.append(m)
        scs.append([l, h, r])
        names.append(f"val{idx}_occ{int(m.sum())}")
        print(f"  {names[-1]:22s} l={l:.4f} h={h:.4f} r={r:.4f}")

    designs = os.path.join(REPO, "meep_calib_designs.npz")
    np.savez(designs, patterns=np.stack(pats),
             scalars=np.array(scs, dtype=np.float64),
             names=np.array(names))
    targets = os.path.join(REPO, "meep_calib_targets.npz")
    np.savez(targets, real=np.array(real[picks], dtype=np.float64),
             imag=np.array(imag[picks], dtype=np.float64),
             idx=np.array(picks))
    print("saved", designs, np.stack(pats).shape)
    print("saved", targets, real[picks].shape)
    print("scalar spread l %.3f-%.3f  h %.3f-%.3f  r %.3f-%.3f" % (
        np.array(scs)[:, 0].min(), np.array(scs)[:, 0].max(),
        np.array(scs)[:, 1].min(), np.array(scs)[:, 1].max(),
        np.array(scs)[:, 2].min(), np.array(scs)[:, 2].max()))


if __name__ == "__main__":
    main()