"""Score the resolution test: v1_spec_p20 designs at res 32 vs 48.

res-32 rows are a same-code, same-setting replication of the stored originals, so
they also test reproducibility: if they land on the stored corrected values the
comparison is sound. The question is whether the surviving excess shrinks with
resolution. It must NOT be read as "higher res is better" -- it is one variable to
see whether the >1 excess is a discretisation/PML artefact (shrinks) or a
property of the decodes (holds).
"""

import glob
import json
import os

import numpy as np

NEW = r"D:\metamaterials_scratch\t3res"
OLD = r"D:\metamaterials_scratch\t3\jepa_v1_spec_p20"


def cond_peak(e, ref, floor=1e-2):
    fm = np.asarray(e["flux_mag"], dtype=float)
    fe = np.abs(np.asarray(ref["flux"], dtype=float))
    keep = fe >= floor * np.median(fe)
    sel = fm[keep]
    sel = sel[np.isfinite(sel)]
    raw = float(max(np.nanmax(fm),
                    np.nanmax(np.asarray(e["modes"]["def"]["mag"], float))))
    return float(sel.max()) if sel.size else float("nan"), raw


def scan(base, pattern):
    rows = {}
    for f in sorted(glob.glob(os.path.join(base, pattern))):
        j = json.load(open(f, encoding="utf-8"))
        tag = os.path.basename(f).replace(".json", "")
        for e in j.get("designs", []):
            for ref in j.get("refs", {}).values():
                c, r = cond_peak(e, ref)
                rows.setdefault(e.get("name", tag), {})[tag] = (c, r)
    return rows


new = scan(NEW, "res*_k*.json")
old = scan(OLD, "e_0_*.json")

print("stored originals (res 32, adaptive) - corrected flux / raw:")
for name, d in sorted(old.items()):
    for tag, (c, r) in sorted(d.items()):
        print(f"  {name:20s} {tag:14s} corrected={c:.4f} raw={r:.4f}")

print("\nresolution test:")
seen = {}
for name, d in sorted(new.items()):
    print(f"  {name}")
    for tag, (c, r) in sorted(d.items()):
        res = tag.split("_")[0]
        seen.setdefault(name, {})[res] = c
        print(f"      {tag:16s} corrected={c:.4f} raw={r:.4f}")
    pair = seen.get(name, {})
    if len(pair) > 1:
        a, b = sorted(pair.items())
        print(f"      -> {a[0]}={a[1]:.4f}  {b[0]}={b[1]:.4f}  "
              f"change {b[1] - a[1]:+.4f}")
