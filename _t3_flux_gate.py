"""Corrected task-3 passivity verdicts, computed from stored curves only.

Established by measurement, in order:
  1. The shared fixed window changed every peak by <0.001  -> the window-mismatch
     explanation in the ledger is refuted.
  2. Every stored reference (two files, two scalar combos) is the SAME smooth
     curve: 2.8e-4 of its band median at the edge, 12.2x at the centre, peak at
     omega = 0.1517. That is the source spectrum - an empty air cell has no
     resonance, and the curve does not move with (l, h, r).
  3. Therefore the gate's max-ratio is taken across ~4.4 decades of denominator,
     including frequencies where the denominator is 1e-4 of the band centre. There
     the ratio is noise over noise: it reads ~0.999 for near-empty decodes and
     1.16-2.11 for v1's +20 % row, and neither number is a physics statement.
  4. The mode channel is a second, independent problem: it is an eigenmode
     COEFFICIENT ratio, not an energy ratio, and MEEP normalises the coefficient
     by the mode of the geometry at the monitor -- which differs between the air
     reference and a cell containing a slab. It is the channel that exceeds 1 at
     well-conditioned frequencies for v2 (depth 2-2.7 x median).

So two corrections: score the FLUX channel only (an energy quantity, |T| <= 1 is
a statement about it), and only where the reference denominator is well
conditioned. Reports the verdict as the floor is swept, so the choice is visible
rather than hidden.
"""

import glob
import json
import os
import sys

import numpy as np

DIRS = sys.argv[1:] or [r"D:\metamaterials_scratch\t3"]
FLOORS = [0.0, 1e-3, 1e-2, 3e-2, 5e-2]
TOL = 1.02


def med(a):
    a = np.abs(np.asarray(a, dtype=float))
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def rows(base):
    for tag in sorted(os.listdir(base)):
        d = os.path.join(base, tag)
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "e_0_*.json"))):
            j = json.load(open(f, encoding="utf-8"))
            refs = j.get("refs", {})
            for e in j.get("designs", []):
                k = f"{round(e['l'], 4)}_{round(e['h'], 4)}_{round(e['r'], 4)}"
                if k not in refs:
                    continue
                yield tag, e, refs[k]


for base in DIRS:
    print(f"\n######## {base} ########")
    for floor in FLOORS:
        print(f"\n--- flux channel only | denominator floor {floor:g} x median ---")
        agg = {}
        for tag, e, ref in rows(base):
            fm = np.asarray(e["flux_mag"], dtype=float)
            fe = np.abs(np.asarray(ref["flux"], dtype=float))
            mm = np.asarray(e["modes"]["def"]["mag"], dtype=float)
            ae = np.abs(np.asarray(ref["abs_alpha"], dtype=float))
            ok_ref = fe >= floor * med(fe) if floor > 0 else np.ones_like(fe, bool)
            ok_amp = ae >= floor * med(ae) if floor > 0 else np.ones_like(ae, bool)
            band = ok_ref & ok_amp
            sel = fm[band]
            sel = sel[np.isfinite(sel)]
            s = agg.setdefault(tag, {"pk": [], "n": 0, "ok": 0, "mode": []})
            if sel.size:
                pk = float(sel.max())
                s["pk"].append(pk)
                s["n"] += 1
                s["ok"] += bool(pk <= TOL)
                m = mm[band]
                m = m[np.isfinite(m)]
                if m.size:
                    s["mode"].append(float(m.max()))
        for tag, s in sorted(agg.items()):
            if not s["pk"]:
                continue
            print(f"  {tag:22s} n={s['n']:2d} flux-ok={s['ok']:2d} | "
                  f"flux {min(s['pk']):.3f}-{max(s['pk']):.3f} | "
                  f"mode(channel, not a gate) {min(s['mode']):.3f}-{max(s['mode']):.3f}")
