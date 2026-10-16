"""Pick the shared fixed window for the task-3 re-run (two-pass recipe, pass 1).

The adaptive runs already recorded when each run's decay test stopped, so pass 1
does not need re-running: read `ringdown` from every design entry and from every
empty-cell reference, take the maximum, and add a margin. The re-run then uses
one window for all four affected sets, which also makes the m20/p20 rows
comparable with each other.

Also reports the excess each set currently shows, so the effect of the fix is
visible against the pre-fix numbers.
"""

import glob
import json
import os
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else r"D:\metamaterials_scratch\t3"
SETS = ["jepa_v1_spec_m20", "jepa_v1_spec_p20",
        "jepa_v2_spec_m20", "jepa_v2_spec_p20"]
MARGIN = 1.05

worst = 0.0
for tag in SETS:
    p = os.path.join(BASE, tag)
    ds, rf, caps, peaks = [], [], 0, []
    for f in sorted(glob.glob(os.path.join(p, "e_0_*.json"))):
        j = json.load(open(f, encoding="utf-8"))
        for e in j.get("designs", []):
            if e.get("ringdown") is not None:
                ds.append(float(e["ringdown"]))
            caps += bool(e.get("hit_cap"))
            if e.get("peak_ratio") is not None:
                peaks.append(float(e["peak_ratio"]))
        for r in j.get("refs", {}).values():
            t = (r.get("meta") or {}).get("ringdown")
            if t is not None:
                rf.append(float(t))
    if not ds and not rf:
        print(f"{tag}: no ringdown recorded (nothing to derive)")
        continue
    m = max(ds + rf + [0.0])
    worst = max(worst, m)
    print(f"{tag}: designs n={len(ds)} ringdown max {max(ds):.1f} | "
          f"refs n={len(rf)} max {max(rf, default=float('nan')):.1f} | "
          f"capped={caps} | peaks {min(peaks):.3f}-{max(peaks):.3f}")

T = worst * MARGIN
print(f"\nshared window T = {worst:.1f} x {MARGIN} = {T:.1f}")
print(f"--ring-fixed {T:.1f}")
