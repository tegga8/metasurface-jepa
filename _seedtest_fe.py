"""Seed test, step 6: target-fixed-effects version of the retrieval check.

Step 5 scored every measured design against every target (77 x 18) and read the rank of
its own target. That rank is confounded: some targets are intrinsically harder to reach,
so they rank low for *every* design. The clean version removes the per-target effect
before ranking (a two-way fixed-effects decomposition), and gets a p-value from a
permutation null that keeps the same designs and the same targets.
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
N_T = 18


def main():
    conv = _load_json(os.path.join(REPO, "_meep_convention.json"))
    inv = inverse_convention(conv)
    if inv[0] == "full":
        p2.conv_bias = np.array(conv["coef"]).reshape(-1, 2, 3)[:, :, 2]
    conj = conv.get("conj", False)
    z = _load_npz(os.path.expanduser("~/runs/phase3f/random_spectra.npz"))
    tgts = [np.abs(target_to_meep(z["A_real"][i], z["A_imag"][i], inv, conj))
            for i in range(N_T)]

    rows, own = [], []
    for w in W_LIST:
        for f in sorted(glob.glob(os.path.join(ROOT, "meep", f"real_{w}", "e_0_*.json"))):
            j = json.load(open(f, encoding="utf-8"))
            for e in j.get("designs", []):
                if not e.get("passivity_ok"):
                    continue
                m = np.abs(curve(e))
                rows.append([metrics(m, t)[0] for t in tgts])
                own.append(int(e["idx"]))
    R = np.array(rows)
    own = np.array(own)
    n = len(own)

    # remove the per-target (column) effect: some targets are simply harder
    Rc = R - R.mean(axis=0, keepdims=True)

    def mean_rank(labels):
        vals = Rc[np.arange(n), labels]
        return float((Rc < vals[:, None]).sum(axis=1).mean() + 1)

    obs = mean_rank(own)

    rng = np.random.default_rng(0)
    null = np.empty(20000)
    for i in range(null.size):
        lab = rng.permutation(N_T)
        # derangement only: no design keeps its own target
        while np.any(lab[own] == own):
            lab = rng.permutation(N_T)
        null[i] = mean_rank(lab[own])

    print(f"designs {n}   targets {N_T}   pairs {R.size}")
    print("\n=== target-fixed-effects retrieval ===")
    print(f"  observed mean rank of own target : {obs:.2f}   (uniform = 9.50)")
    print(f"  permutation null mean            : {null.mean():.2f}  "
          f"sd {null.std():.2f}")
    p_low = (null <= obs).mean()
    p_high = (null >= obs).mean()
    print(f"  p(better than chance)            : {p_low:.4g}")
    print(f"  p(worse than chance)             : {p_high:.4g}")

    # own target at the top of the residual row
    vals = Rc[np.arange(n), own]
    top1 = (Rc < vals[:, None]).sum(axis=1) == 0
    print(f"\n  own target is the residual-best : {int(top1.sum())}/{n}  "
          f"(chance {n/N_T:.1f})")

    print("\n=== raw (un-centred) for reference ===")
    vals_raw = R[np.arange(n), own]
    rank_raw = (R < vals_raw[:, None]).sum(axis=1) + 1
    print(f"  mean raw rank {rank_raw.mean():.2f}   "
          f"own is best {int((rank_raw == 1).sum())}/{n}  (chance {n/N_T:.1f})")


if __name__ == "__main__":
    main()
