"""Is the empty-cell reference itself resonant? (decides what |T| <= 1 means)

Passivity (|T| <= 1) is a statement about the field the sample sees, i.e. the
INCIDENT field. The harness instead divides by an empty-cell run -- and that
reference is an air path between two PML faces, which is a Fabry-Perot cavity:
it can have deep in-band notches whose positions move with the cell length, i.e.
with (l, h, r). If so, then "sample/reference > 1" at a notch is the reference's
own resonance moving, not the sample creating energy, and the gate is ill-posed
as implemented rather than the solver being broken.

Cost-free: every run stored its reference flux array, so the answer is in the
JSONs already on disk. Reports, per scalar combo, the band-median-normalised
reference spectrum, its notches, and whether the notch positions shift between
combos (which is what separates "resonant reference" from "smooth source
roll-off").
"""

import glob
import json
import os
import sys

import numpy as np

DIRS = sys.argv[1:] or [r"D:\metamaterials_scratch\t3"]
NOTCH = 1e-2


def report(base):
    print(f"\n######## {base} ########")
    seen = {}
    for tag in sorted(os.listdir(base)):
        d = os.path.join(base, tag)
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "e_0_*.json"))):
            j = json.load(open(f, encoding="utf-8"))
            for key, r in j.get("refs", {}).items():
                fe = np.abs(np.asarray(r["flux"], dtype=float))
                if fe.size == 0:
                    continue
                med = float(np.median(fe))
                norm = fe / med if med else fe
                notches = [i for i in range(1, len(norm) - 1)
                           if norm[i] < NOTCH and norm[i] <= norm[i - 1]
                           and norm[i] <= norm[i + 1]]
                seen.setdefault(key, []).append((tag, norm, notches))
    for key, rows in sorted(seen.items()):
        tag, norm, notches = rows[0]
        # only interesting if these are notches INSIDE the band, not just edges
        inner = [i for i in notches if 5 < i < len(norm) - 5]
        print(f"\ncombo {key:22s} (from {tag})")
        print(f"   norm min {norm.min():.2e}  median 1  max {norm.max():.2f}")
        print(f"   low  quartile {np.percentile(norm, 25):.2e} "
              f"| high {np.percentile(norm, 75):.2e}")
        print(f"   notches in band: {inner if inner else 'none'}"
              f"   (edges excluded: {[i for i in notches if i not in inner]})")
        # how much of the band is well below the median?
        for thr in (1e-3, 1e-2, 1e-1):
            print(f"   fraction of band < {thr:g}: {100.0 * (norm < thr).mean():5.1f}%")


if __name__ == "__main__":
    for b in DIRS:
        report(b)
