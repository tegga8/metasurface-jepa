"""Seed test, step 5: the strongest form of the same question, from the same 90 solves.

The selection test asks "does picking by the correct target beat picking by a wrong
one?" -- n=18, so it is coarse. But every candidate was measured, and every target
spectrum is known, so we can score *every* candidate against *every* target for free:
90 x 18 = 1,620 pairs. That turns the question into a retrieval one:

    for the geometry decoded for target k, is target k its best match among the 18?

If the model carries target-specific information into the design, the own-target error
should sit at the bottom of that row. If it does not, the own target is an ordinary
member of the row and the rank is uniform on 1..18 (mean 9.5).
"""

import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import _compare_phase2 as p2  # noqa: E402
from _compare_phase2 import (  # noqa: E402
    curve, inverse_convention, metrics, target_to_meep, _load_json, _load_npz,
)

ROOT = os.path.expanduser("~/runs/seedtest")
W_LIST = ("w050", "w075", "w100", "w125", "w150")


def main():
    conv = _load_json(os.path.join(REPO, "_meep_convention.json"))
    inv = inverse_convention(conv)
    if inv[0] == "full":
        p2.conv_bias = np.array(conv["coef"]).reshape(-1, 2, 3)[:, :, 2]
    conj = conv.get("conj", False)
    z = _load_npz(os.path.expanduser("~/runs/phase3f/random_spectra.npz"))
    n_t = 18
    tgts = [np.abs(target_to_meep(z["A_real"][i], z["A_imag"][i], inv, conj))
            for i in range(n_t)]

    rows = {}          # (w, slot) -> error against every target
    for w in W_LIST:
        for f in sorted(glob.glob(os.path.join(ROOT, "meep", f"real_{w}", "e_0_*.json"))):
            j = json.load(open(f, encoding="utf-8"))
            for e in j.get("designs", []):
                if not e.get("passivity_ok"):
                    continue
                m = np.abs(curve(e))
                rows[(w, int(e["idx"]))] = np.array(
                    [metrics(m, t)[0] for t in tgts])

    keys = sorted(rows)
    R = np.stack([rows[k] for k in keys])              # [n_design, n_target]
    own = np.array([k[1] for k in keys])
    own_err = R[np.arange(len(keys)), own]
    # rank of the own target within the row (1 = best)
    rank = (R < own_err[:, None]).sum(axis=1) + 1
    # margin over the best *wrong* target
    wrong = R.copy()
    wrong[np.arange(len(keys)), own] = np.inf
    margin = wrong.min(axis=1) - own_err

    print(f"designs scored: {len(keys)}   pairs: {R.size}")
    print("\n=== own-target rank among the 18 targets (1 = best) ===")
    print(f"  mean rank      : {rank.mean():.2f}   (uniform = 9.5)")
    print(f"  median rank    : {np.median(rank):.1f}")
    print(f"  rank == 1      : {int((rank == 1).sum())}/{len(rank)}  "
          f"(chance {len(rank)/n_t:.1f})")
    print(f"  rank <= 3      : {int((rank <= 3).sum())}/{len(rank)}  "
          f"(chance {3*len(rank)/n_t:.1f})")
    print(f"  rank in top half: {int((rank <= n_t // 2).sum())}/{len(rank)}")

    print("\n=== margin: best wrong target - own target (positive = own wins) ===")
    print(f"  mean   : {margin.mean():+.4f}")
    print(f"  median : {np.median(margin):+.4f}")
    print(f"  own wins: {int((margin > 0).sum())}/{len(margin)}")
    try:
        from scipy.stats import wilcoxon
        print(f"  Wilcoxon signed-rank p: {wilcoxon(margin).pvalue:.4g}")
    except Exception as exc:                                   # pragma: no cover
        print(f"  (scipy unavailable: {exc})")

    print("\n=== per guidance strength ===")
    print(f"{'w':6s} {'n':>3s} {'mean rank':>10s} {'mean margin':>12s} {'own wins':>9s}")
    for w in W_LIST:
        sel = [i for i, k in enumerate(keys) if k[0] == w]
        if sel:
            print(f"{w:6s} {len(sel):3d} {rank[sel].mean():10.2f} "
                  f"{margin[sel].mean():+12.4f} "
                  f"{int((margin[sel] > 0).sum()):4d}/{len(sel)}")


if __name__ == "__main__":
    main()
