"""Per-channel null diagnosis for the task-3 |T| > 1 peaks.

Earlier pass looked only at the FLUX denominator and concluded the peaks sit in
reference nulls. The null guard then failed to move them, which says the peak's
own channel may be the other one: the harness takes max(flux ratio, mode ratio),
and each has its own denominator (the reference's flux, the reference's
eigenmode amplitude). This version reports, per channel, the peak, where it sits,
and how deep that channel's own denominator is there -- so the guard is applied
to the denominator that actually produced the number.

Also reports the sample/reference amplitude ratio at the peak, which is the
quantity a passivity test is really about: |T| <= 1 is a statement about the
SAMPLE relative to the INCIDENT field, not relative to a resonant empty cell.
"""

import glob
import json
import os
import sys

import numpy as np

DIRS = sys.argv[1:] or [r"D:\metamaterials_scratch\t3", r"D:\metamaterials_scratch\t3win"]
FLOOR = 1e-2


def med(a):
    a = np.abs(np.asarray(a, dtype=float))
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


for base in DIRS:
    print(f"\n######## {base} ########")
    for tag in sorted(os.listdir(base)):
        d = os.path.join(base, tag)
        if not os.path.isdir(d):
            continue
        print(f"\n=== {tag} ===")
        for f in sorted(glob.glob(os.path.join(d, "e_0_*.json"))):
            j = json.load(open(f, encoding="utf-8"))
            refs = j.get("refs", {})
            for e in j.get("designs", []):
                key = f"{round(e['l'], 4)}_{round(e['h'], 4)}_{round(e['r'], 4)}"
                ref = refs.get(key)
                if ref is None:
                    continue
                fm = np.asarray(e["flux_mag"], dtype=float)
                mm = np.asarray(e["modes"]["def"]["mag"], dtype=float)
                fe = np.abs(np.asarray(ref["flux"], dtype=float))
                ae = np.abs(np.asarray(ref["abs_alpha"], dtype=float))
                part = []
                for lbl, arr, den in (("flux", fm, fe), ("mode", mm, ae)):
                    i = int(np.nanargmax(arr))
                    dep = den[i] / med(den) if med(den) else float("nan")
                    flags = []
                    if dep < FLOOR:
                        flags.append("NULL")
                    if den[i] <= 0:
                        flags.append("ZERO")
                    part.append(f"{lbl} {arr[i]:.4f} @f{i} depth={dep:.1e}"
                                f"{'*' if flags else ''}")
                print(f"  {e.get('name', e.get('idx')):20s} "
                      f"peak={e['peak_ratio']:.4f} | " + " | ".join(part))
