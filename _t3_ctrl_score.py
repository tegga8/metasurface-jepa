"""Score the empty-cell control: the passivity statistic applied to itself.

Sample and reference are physically identical (both all-zero patterns at the same
scalars), so every number here is the instrument's own error. The gate calls
anything above 1.02 a solver failure, so:

  - a reading near 1.000 means the tolerance is resolvable and the surviving
    v1_spec_p20 excess (1.026-1.193) is a real difference from the instrument;
  - a reading at or above 1.02 means the tolerance is tighter than the
    instrument's own reproducibility, and no passivity claim can be made at that
    level at all.

Reports the raw statistic and the corrected one (flux-only, conditioned band),
plus the generalised version the harness now computes, and the run-to-run spread
between the two independent runs of each combo.
"""

import glob
import json
import os

import numpy as np

BASE = r"D:\metamaterials_scratch\t3ctrl"
TOL = 1.02


def peaks(e, ref):
    fm = np.asarray(e["flux_mag"], dtype=float)
    mm = np.asarray(e["modes"]["def"]["mag"], dtype=float)
    fe = np.abs(np.asarray(ref["flux"], dtype=float))
    ae = np.abs(np.asarray(ref["abs_alpha"], dtype=float))
    raw = float(max(np.nanmax(fm), np.nanmax(mm)))
    out = {"raw": raw, "mode": float(np.nanmax(mm))}
    for label, floor in (("1e-2", 1e-2), ("5e-2", 5e-2)):
        keep = fe >= floor * np.median(fe)
        sel = fm[keep]
        sel = sel[np.isfinite(sel)]
        out[f"flux@{label}"] = float(sel.max()) if sel.size else float("nan")
        out[f"exc@{label}"] = float(1.0 - keep.mean())
        m = mm[keep]
        m = m[np.isfinite(m)]
        out[f"mode@{label}"] = float(m.max()) if m.size else float("nan")
    _ = ae
    return out


files = sorted(glob.glob(os.path.join(BASE, "*.json")))
print(f"{'file':12s} {'raw':>8s} {'mode':>8s} {'flux@1e-2':>10s} "
      f"{'mode@1e-2':>10s} {'flux@5e-2':>10s} {'exc':>6s}")
for f in files:
    j = json.load(open(f, encoding="utf-8"))
    for e in j.get("designs", []):
        for ref in j.get("refs", {}).values():
            p = peaks(e, ref)
            print(f"{os.path.basename(f):12s} {p['raw']:8.4f} {p['mode']:8.4f} "
                  f"{p['flux@1e-2']:10.4f} {p['mode@1e-2']:10.4f} "
                  f"{p['flux@5e-2']:10.4f} {p['exc@1e-2']:6.1%}")

# run-to-run: the flux peak of two independent runs of the same empty cell
print("\nrun-to-run reproducibility (same physical system, independent runs):")
for tag in ("m20", "p20"):
    vals = []
    for suf in ("a", "b"):
        f = os.path.join(BASE, f"{tag}_{suf}.json")
        if not os.path.exists(f):
            continue
        j = json.load(open(f, encoding="utf-8"))
        for e in j.get("designs", []):
            for ref in j.get("refs", {}).values():
                vals.append(peaks(e, ref)["flux@1e-2"])
    if len(vals) > 1:
        print(f"  {tag}: {['%.4f' % v for v in vals]}  "
              f"spread {max(vals) - min(vals):.4f}")
