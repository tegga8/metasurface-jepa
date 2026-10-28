"""Seed test, step 4: is the null informative? Diagnostics on the 90 measured designs.

The selection test returned a null (margin +0.041, CI spanning zero, 10/18 wins).
A null is only meaningful if the experiment had something to select between, so this
checks, without any new solves:

  1. Geometric diversity -- mean pairwise IoU between the five candidates of the SAME
     target vs. between candidates of DIFFERENT targets. If within-target diversity is
     near the between-target value, the guidance knob barely moves the design and the
     selection has little to choose from.
  2. Error spread -- std of the five measured errors per target, against the observed
     real-vs-shuffled margin. If the spread dwarfs the margin, selection is noise-driven.
  3. Selection overlap -- how often the real target's argmin coincides with the shuffled
     target's argmin, and the rank correlation between the two error vectors.
"""

import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

from _compare_phase2 import (  # noqa: E402
    curve, inverse_convention, metrics, target_to_meep, _load_json, _load_npz,
)
from runtime.physics_controls import derangement_permutation  # noqa: E402

ROOT = os.path.expanduser("~/runs/seedtest")
W_LIST = ("w050", "w075", "w100", "w125", "w150")


def iou(a, b):
    a = np.asarray(a).astype(bool).ravel()
    b = np.asarray(b).astype(bool).ravel()
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def main():
    conv = _load_json(os.path.join(REPO, "_meep_convention.json"))
    inv = inverse_convention(conv)
    if inv[0] == "full":
        import _compare_phase2 as p2
        p2.conv_bias = np.array(conv["coef"]).reshape(-1, 2, 3)[:, :, 2]
    conj = conv.get("conj", False)
    z = _load_npz(os.path.expanduser("~/runs/phase3f/random_spectra.npz"))
    idx = json.load(open(os.path.join(ROOT, "dec", "index.json"), encoding="utf-8"))
    n_t = len({d["slot"] for d in idx["designs"].values()})
    perm = derangement_permutation(n_t, "cpu", seed=0).numpy()

    def tgt(i):
        return np.abs(target_to_meep(z["A_real"][i], z["A_imag"][i], inv, conj))

    occ, e_real, e_shuf, clean = {}, {}, {}, {}
    for w in W_LIST:
        clean[w] = set()
        # the JSON's "occ" field is an occupied-pixel COUNT (a scalar), not the pattern,
        # so the geometry has to come from the decoded npz the worker simulated
        zz = np.load(os.path.join(ROOT, "dec", f"decoded_real_{w}.npz"), allow_pickle=True)
        pats = np.asarray(zz["patterns"]).astype(bool)
        for f in sorted(glob.glob(os.path.join(ROOT, "meep", f"real_{w}", "e_0_*.json"))):
            j = json.load(open(f, encoding="utf-8"))
            for e in j.get("designs", []):
                slot = int(e["idx"])
                occ[(w, slot)] = pats[slot]
                if not e.get("passivity_ok"):
                    continue
                clean[w].add(slot)
                m = np.abs(curve(e))
                e_real[(w, slot)] = metrics(m, tgt(slot))[0]
                e_shuf[(w, slot)] = metrics(m, tgt(int(perm[slot])))[0]

    # --- 1. geometric diversity ------------------------------------------------
    within, between = [], []
    slots = range(n_t)
    for s in slots:
        have = [w for w in W_LIST if (w, s) in occ]
        for i, a in enumerate(have):
            for b in have[i + 1:]:
                within.append(iou(occ[(a, s)], occ[(b, s)]))
    for s in slots:
        t = (s + 1) % n_t
        for w in W_LIST:
            if (w, s) in occ and (w, t) in occ:
                between.append(iou(occ[(w, s)], occ[(w, t)]))
    print("=== 1. geometric diversity (occupancy IoU) ===")
    print(f"  within-target  (same slot, different w): mean {np.mean(within):.3f}  "
          f"min {np.min(within):.3f}  n={len(within)}")
    print(f"  between-target (same w, adjacent slot): mean {np.mean(between):.3f}  "
          f"min {np.min(between):.3f}  n={len(between)}")
    print(f"  diversity retained = (1-within)/(1-between): "
          f"{(1 - np.mean(within)) / max(1e-9, 1 - np.mean(between)):.3f}")

    # --- 2. error spread vs the margin ----------------------------------------
    print("\n=== 2. per-target spread of the five measured errors ===")
    spreads, margins = [], []
    for s in slots:
        r = [e_real[(w, s)] for w in W_LIST if (w, s) in e_real]
        h = [e_shuf[(w, s)] for w in W_LIST if (w, s) in e_shuf]
        if len(r) >= 2:
            spreads.append(np.std(r))
        # same rule as _seedtest_score.py, so the margin matches the reported one
        if r and h:
            margins.append(min(h) - min(r))
    print(f"  mean within-target std of error : {np.mean(spreads):.4f}  "
          f"(n={len(spreads)} slots with >=2 clean candidates)")
    print(f"  mean selection margin           : {np.mean(margins):+.4f}  "
          f"(n={len(margins)} slots, as reported)")
    print(f"  ratio (spread / |margin|)       : "
          f"{np.mean(spreads) / max(1e-9, abs(np.mean(margins))):.1f}x")

    # --- 3. selection overlap --------------------------------------------------
    print("\n=== 3. do real and shuffled targets pick the same candidate? ===")
    same, rho = 0, []
    n_cmp = 0
    for s in slots:
        ws = [w for w in W_LIST if (w, s) in e_real and (w, s) in e_shuf]
        if len(ws) < 2:
            continue
        n_cmp += 1
        r = [e_real[(w, s)] for w in ws]
        h = [e_shuf[(w, s)] for w in ws]
        if ws[int(np.argmin(r))] == ws[int(np.argmin(h))]:
            same += 1
        rr = np.argsort(np.argsort(r)).astype(float)
        hh = np.argsort(np.argsort(h)).astype(float)
        if rr.std() > 0 and hh.std() > 0:
            rho.append(np.corrcoef(rr, hh)[0, 1])
    print(f"  same argmin                  : {same}/{n_cmp}")
    print(f"  mean rank corr (real vs shuf): {np.mean(rho):+.3f}  (n={len(rho)})")

    print("\n=== 4. passivity-clean per set ===")
    for w in W_LIST:
        print(f"  {w}: {len(clean[w])}/{n_t}")


if __name__ == "__main__":
    main()
