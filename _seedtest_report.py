"""Seed test, step 7: emit the complete result set as markdown.

Everything below is computed from the 90 measured designs in ~/runs/seedtest/meep and the
targets they were decoded from -- no new solves, no model calls. Output is the table set
that goes into the paper appendix.
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
from _compare_decoded_meep import boot_margin, sign_p  # noqa: E402
from runtime.physics_controls import derangement_permutation  # noqa: E402

ROOT = os.path.expanduser("~/runs/seedtest")
W = ("w050", "w075", "w100", "w125", "w150")
LBL = {"w050": "0.50", "w075": "0.75", "w100": "1.00", "w125": "1.25", "w150": "1.50"}
N_T = 18


def iou(a, b):
    a, b = np.asarray(a).astype(bool).ravel(), np.asarray(b).astype(bool).ravel()
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def main():
    conv = _load_json(os.path.join(REPO, "_meep_convention.json"))
    inv = inverse_convention(conv)
    if inv[0] == "full":
        p2.conv_bias = np.array(conv["coef"]).reshape(-1, 2, 3)[:, :, 2]
    conj = conv.get("conj", False)
    z = _load_npz(os.path.expanduser("~/runs/phase3f/random_spectra.npz"))
    tgts = [np.abs(target_to_meep(z["A_real"][i], z["A_imag"][i], inv, conj)) for i in range(N_T)]
    perm = derangement_permutation(N_T, "cpu", seed=0).numpy()

    e_real, e_shuf, clean, cap, pat, mag = {}, {}, {}, {}, {}, {}
    for w in W:
        zz = np.load(os.path.join(ROOT, "dec", f"decoded_real_{w}.npz"), allow_pickle=True)
        pats = np.asarray(zz["patterns"]).astype(bool)
        for f in sorted(glob.glob(os.path.join(ROOT, "meep", f"real_{w}", "e_0_*.json"))):
            j = json.load(open(f, encoding="utf-8"))
            for e in j.get("designs", []):
                s = int(e["idx"])
                pat[(w, s)] = pats[s]
                cap[(w, s)] = bool(e.get("hit_cap"))
                if not e.get("passivity_ok"):
                    continue
                clean[(w, s)] = True
                m = np.abs(curve(e))
                mag[(w, s)] = m
                e_real[(w, s)] = metrics(m, tgts[s])[0]
                e_shuf[(w, s)] = metrics(m, tgts[int(perm[s])])[0]

    def cell(d, w, s):
        if (w, s) not in d:
            return "—"
        return f"{d[(w, s)]:.4f}"

    print("## Appendix — seed test, full table set\n")
    print(f"Derangement (slot → shuffled target): "
          f"{', '.join(f'{i}→{int(perm[i])}' for i in range(N_T))}\n")

    print("### A. Measured error against the design's OWN target\n")
    print("| tgt | w=0.50 | w=0.75 | w=1.00 | w=1.25 | w=1.50 | best (w, err) |")
    print("|---|---|---|---|---|---|---|")
    for s in range(N_T):
        best = min((e_real[(w, s)], w) for w in W if (w, s) in e_real)
        print(f"| {s:02d} | " + " | ".join(cell(e_real, w, s) for w in W)
              + f" | **{LBL[best[1]]}, {best[0]:.4f}** |")

    print("\n### B. Same designs, error against the SHUFFLED target\n")
    print("| tgt | w=0.50 | w=0.75 | w=1.00 | w=1.25 | w=1.50 | best (w, err) | margin vs A |")
    print("|---|---|---|---|---|---|---|---|")
    sel_r, sel_h = [], []
    for s in range(N_T):
        best = min((e_shuf[(w, s)], w) for w in W if (w, s) in e_shuf)
        br = min(e_real[(w, s)] for w in W if (w, s) in e_real)
        sel_r.append(br)
        sel_h.append(best[0])
        print(f"| {s:02d} | " + " | ".join(cell(e_shuf, w, s) for w in W)
              + f" | **{LBL[best[1]]}, {best[0]:.4f}** | {best[0] - br:+.4f} |")

    r, h = np.array(sel_r), np.array(sel_h)
    d = h - r
    mean, lo, hi = boot_margin(h, r)
    print("\n### C. Selection summary (paired, n=18)\n")
    print("| quantity | value |")
    print("|---|---|")
    print(f"| real mean / median | {r.mean():.4f} / {np.median(r):.4f} |")
    print(f"| shuffled mean / median | {h.mean():.4f} / {np.median(h):.4f} |")
    print(f"| paired margin | {mean:+.4f} |")
    print(f"| 95 % bootstrap CI | [{lo:+.4f}, {hi:+.4f}] |")
    print(f"| median margin | {np.median(d):+.4f} |")
    print(f"| win rate | {int((d > 0).sum())}/{len(d)} |")
    print(f"| sign test p | {sign_p(d):.4g} |")

    print("\n### D. Per guidance strength, no selection\n")
    print("| w | clean | real err | shuffled err | margin |")
    print("|---|---|---|---|---|")
    for w in W:
        rr = [e_real[(w, s)] for s in range(N_T) if (w, s) in e_real]
        hh = [e_shuf[(w, s)] for s in range(N_T) if (w, s) in e_shuf]
        print(f"| {LBL[w]} | {len(rr)}/18 | {np.mean(rr):.4f} | {np.mean(hh):.4f} | "
              f"{np.mean(hh) - np.mean(rr):+.4f} |")

    print("\n### E. Passivity census\n")
    print("| w | clean | passivity fail | ringdown-capped |")
    print("|---|---|---|---|")
    for w in W:
        n = sum(1 for s in range(N_T) if (w, s) in pat)
        c = sum(1 for s in range(N_T) if (w, s) in clean)
        k = sum(1 for s in range(N_T) if cap.get((w, s)))
        print(f"| {LBL[w]} | {c}/{n} | {n - c} | {k} |")

    within, between = [], []
    for s in range(N_T):
        have = [w for w in W if (w, s) in pat]
        for i, a in enumerate(have):
            for b in have[i + 1:]:
                within.append(iou(pat[(a, s)], pat[(b, s)]))
    for s in range(N_T):
        t = (s + 1) % N_T
        for w in W:
            if (w, s) in pat and (w, t) in pat:
                between.append(iou(pat[(w, s)], pat[(w, t)]))
    print("\n### F. Candidate-pool geometry\n")
    print("| check | value |")
    print("|---|---|")
    print(f"| within-target IoU (same target, different w) | {np.mean(within):.3f} "
          f"(min {np.min(within):.3f}, n={len(within)}) |")
    print(f"| between-target IoU (same w, adjacent target) | {np.mean(between):.3f} "
          f"(min {np.min(between):.3f}, n={len(between)}) |")
    spreads = [np.std([e_real[(w, s)] for w in W if (w, s) in e_real])
               for s in range(N_T)
               if sum((w, s) in e_real for w in W) >= 2]
    print(f"| mean within-target std of the five errors | {np.mean(spreads):.4f} |")

    print("\n### G. All-pairs retrieval (77 clean designs x 18 targets)\n")
    keys = sorted(e_real)
    R = np.stack([[metrics(mag[k], t)[0] for t in tgts] for k in keys])
    own = np.array([k[1] for k in keys])
    Rc = R - R.mean(axis=0, keepdims=True)
    vals = Rc[np.arange(len(own)), own]
    rank = (Rc < vals[:, None]).sum(axis=1) + 1
    raw = (R < R[np.arange(len(own)), own][:, None]).sum(axis=1) + 1
    print("| quantity | value |")
    print("|---|---|")
    print(f"| pairs | {R.size} |")
    print(f"| own-target mean rank, raw | {raw.mean():.2f} (uniform 9.50) |")
    print(f"| own-target mean rank, target fixed effects | {rank.mean():.2f} |")
    print(f"| own target is residual-best | {int((rank == 1).sum())}/{len(rank)} "
          f"(chance {len(rank)/N_T:.1f}) |")


if __name__ == "__main__":
    main()
