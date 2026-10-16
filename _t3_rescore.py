"""Re-score the task-3 passivity gate with a reference-null guard.

The fixed-window re-run left every peak unchanged to four decimals, so the
window-mismatch explanation in the ledger is REFUTED. The stored curves show the
real mechanism: the harness divides by the empty-cell reference and guards only
EXACT zeros, but an empty cell has deep interference nulls -- at every recorded
peak the reference denominator sits 2.8e-4 to 4.8e-4 of its band median, i.e.
~3000x below the band, and the eigenmode channel's nulls are deeper still. The
gate was reporting the denominator's null, not the sample's passivity.

The guard: a frequency counts only where the reference's own denominator is at
least NULL_FLOOR x its band median. The verdict is then taken over the trusted
band, with the raw verdict kept alongside so nothing is silently redefined.

Everything here runs on the stored JSONs (both the adaptive originals and the
fixed-window copies) -- no FDTD, so the corrected verdicts cost nothing.
"""

import glob
import json
import os
import sys

import numpy as np

ORIG = r"D:\metamaterials_scratch\t3"
WINFIX = r"D:\metamaterials_scratch\t3win"
FLOORS = [0.0, 1e-3, 1e-2, 3e-2]


def med(a):
    a = np.abs(np.asarray(a, dtype=float))
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def score(entry, ref, floor):
    """(peak_trusted, peak_raw, n_total, n_excluded) for one design."""
    fm = np.asarray(entry["flux_mag"], dtype=float)
    mm = np.asarray(entry["modes"]["def"]["mag"], dtype=float)
    fe = np.abs(np.asarray(ref["flux"], dtype=float))
    ae = np.abs(np.asarray(ref["abs_alpha"], dtype=float))
    raw = float(max(np.nanmax(fm), np.nanmax(mm)))
    if floor <= 0:
        return raw, raw, int(fm.size), 0
    tf = fe >= floor * med(fe)
    ta = ae >= floor * med(ae)
    trust = tf & ta
    vals = [v for v, ok in ((fm, tf), (mm, ta))]
    pk = []
    for arr, ok in ((fm, tf), (mm, ta)):
        sel = arr[ok]
        sel = sel[np.isfinite(sel)]
        if sel.size:
            pk.append(float(sel.max()))
    trusted = max(pk) if pk else float("nan")
    return trusted, raw, int(fm.size), int((~trust).sum())


def run(base, label, floor):
    print(f"\n--- {label} | null floor {floor:g} ---")
    for tag in sorted(os.listdir(base)):
        d = os.path.join(base, tag)
        if not os.path.isdir(d):
            continue
        pk, raw, excl, nt, ok = [], [], [], 0, 0
        for f in sorted(glob.glob(os.path.join(d, "e_0_*.json"))):
            j = json.load(open(f, encoding="utf-8"))
            refs = j.get("refs", {})
            for e in j.get("designs", []):
                key = f"{round(e['l'], 4)}_{round(e['h'], 4)}_{round(e['r'], 4)}"
                ref = refs.get(key)
                if ref is None:
                    continue
                t, r, n, x = score(e, ref, floor)
                pk.append(t)
                raw.append(r)
                excl.append(x)
                nt += n
                ok += bool(np.isfinite(t) and t <= 1.02)
        if not pk:
            continue
        print(f"  {tag:22s} n={len(pk):2d} ok={ok:2d} | "
              f"raw {min(raw):.3f}-{max(raw):.3f} | "
              f"trusted {min(pk):.3f}-{max(pk):.3f} | "
              f"excluded {100.0 * sum(excl) / max(nt, 1):5.1f}% of band")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "orig"):
        for fl in FLOORS:
            run(ORIG, "adaptive originals (as quoted in the ledger)", fl)
    if which in ("all", "winfix"):
        for fl in FLOORS:
            run(WINFIX, "fixed-window re-run (the refuted hypothesis)", fl)
