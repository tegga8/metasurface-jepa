"""Score the phase-3 physics rows (scalar sensitivity) for JEPA v1/v2 + MetaDiT.

Rows produced by _phase3_models.py and run through _meep_novel.py. Each Meep
output design carries a name of the form "<row>/<design>" (B_true, spec_m20,
spec_p20, spec_true, A_interpolated, B_synthetic), so rows are grouped by that
prefix.

Target per row, and why:
  B_true              regime-B decode was conditioned on the ORIGINAL geometry's
                      Meep->CST spectrum, so the target is that original's own
                      Meep spectrum -- Meep-vs-Meep, same instrument, no
                      convention. Directly comparable to the task-1 rows.
  spec_true/m20/p20   conditioned on the frozen surrogate's spectrum of the SAME
                      original geometry at (true / -20% / +20%) scalars, i.e. a
                      CST-domain target. Meep measures the decode, so the
                      comparison needs the INVERSE convention (inherits its
                      0.1138 held-out floor). Identical conditions across v1, v2
                      and MetaDiT, so the rows are directly comparable.
  A/B sets            use _compare_phase2.py (targets come from random_spectra).

Shuffled null on every row: smooth transmission curves correlate generically, so
a raw correlation says nothing without it.

Usage:
  python _compare_phase3_physics.py <meep_runs_dir> [--cond-dir DIR] [--orig DIR]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _compare_decoded_meep import boot_margin, sign_p  # noqa: E402
from _compare_phase2 import inverse_convention, target_to_meep  # noqa: E402
import _compare_phase2 as p2  # noqa: E402
from _compare_decoded_meep import curve, load_dir_or_file, metrics  # noqa: E402


def load_rows(path):
    """All passivity-clean designs under path, grouped by name prefix."""
    designs, _ = load_dir_or_file(path)
    rows = {}
    for key, e in designs.items():
        nm = e.get("name") or key
        row = nm.split("/")[0] if "/" in nm else "single"
        rows.setdefault(row, {})[nm.split("/")[-1]] = e
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meep_dir", help="dir of per-row Meep output dirs, or one dir")
    ap.add_argument("--cond-dir", default=None,
                    help="dir with <tag>_<row>_cond.npz (jepa v1 rows)")
    ap.add_argument("--orig", default=None, help="b40_merged.json (Meep targets)")
    ap.add_argument("--orig-rows", default="B_true",
                    help="comma list of rows scored against the originals")
    ap.add_argument("--conv", default=os.path.join(REPO, "_meep_convention.json"))
    args = ap.parse_args()
    orig_rows = {r.strip() for r in args.orig_rows.split(",") if r.strip()}

    orig = load_dir_or_file(args.orig)[0] if args.orig else {}
    conv = json.load(open(args.conv, encoding="utf-8"))
    inv = inverse_convention(conv)
    if inv[0] == "full":
        c = np.array(conv["coef"]).reshape(-1, 2, 3)
        p2.conv_bias = c[:, :, 2]

    # gather every Meep dir under the given path (per-row dirs) or the dir itself.
    # The MODEL tag lives in the directory name (jepa_v1_B_true, metadit_spec_m20,
    # ...) while the design names carry only the ROW ("B_true/sym00"), so rows must
    # be keyed by (dir, row) -- grouping by row alone silently merged v1 and v2.
    sub = sorted(d for d in glob.glob(os.path.join(args.meep_dir, "*"))
                 if os.path.isdir(d))
    rows = {}
    for d in [args.meep_dir] + sub:
        tag = os.path.basename(os.path.normpath(d))
        for row, designs in load_rows(d).items():
            rows.setdefault((tag, row), {}).update(designs)

    rng = np.random.default_rng(0)
    print("rows found:", sorted(f"{t}:{r}" for t, r in rows), "\n")
    for (tag, row) in sorted(rows):
        ds = rows[(tag, row)]
        if row in orig_rows:
            common = sorted(set(ds) & set(orig))
            if len(common) < 4:
                print(f"[{tag}:{row}] not enough overlap with the originals "
                      f"({len(common)})")
                continue
            tgt = {k: curve(orig[k]) for k in common}
        else:
            conds = None
            if args.cond_dir:
                p = os.path.join(args.cond_dir, f"{tag}_cond.npz")
                if os.path.exists(p):
                    conds = np.load(p, allow_pickle=True)
            if conds is None:
                print(f"[{tag}:{row}] no conditioning spectra "
                      f"({tag}_cond.npz in --cond-dir); skipping")
                continue
            names = [str(n) for n in conds["names"]]
            ok = [n for n in sorted(ds) if n in names]
            tgt = {}
            for n in ok:
                i = names.index(n)
                t = target_to_meep(conds["real"][i], conds["imag"][i], inv,
                                   conv.get("conj", False))
                tgt[n] = np.abs(t)
            ds = {n: ds[n] for n in ok}
            common = ok

        if len(common) < 4:
            print(f"[{tag}:{row}] too few comparable designs ({len(common)})")
            continue

        re_e, re_c, sh_e, sh_c = [], [], [], []
        perm = rng.permutation(len(common))
        for i, k in enumerate(common):
            m = curve(ds[k])
            er, cr = metrics(m, tgt[k])
            es, cs = metrics(m, tgt[common[perm[i]]])
            re_e.append(er); re_c.append(cr); sh_e.append(es); sh_c.append(cs)
        me, lo_e, hi_e = boot_margin(re_e, sh_e)
        mc, lo_c, hi_c = boot_margin(re_c, sh_c)
        de = np.asarray(re_e) - np.asarray(sh_e)
        dc = np.asarray(re_c) - np.asarray(sh_c)
        print(f"=== {tag}:{row}: n={len(common)} ===")
        print(f"  REAL     err={np.mean(re_e):.4f} corr={np.mean(re_c):+.4f}")
        print(f"  SHUFFLED err={np.mean(sh_e):.4f} corr={np.mean(sh_c):+.4f}")
        print(f"  MEAN margin  err {me:+.4f} CI [{lo_e:+.4f},{hi_e:+.4f}] | "
              f"corr {mc:+.4f} CI [{lo_c:+.4f},{hi_c:+.4f}]")
        print(f"  MEDIAN margin err {np.median(de):+.4f} | "
              f"corr {np.median(dc):+.4f}")
        print(f"  win rate: err {int((de < 0).sum())}/{len(de)}, corr "
              f"{int((dc > 0).sum())}/{len(dc)}   "
              f"sign p: err {sign_p(-de):.3g}, corr {sign_p(dc):.3g}")
        if row not in orig_rows:
            print("  (target path inherits the inverse convention's 0.1138 "
                  "held-out error; the null is the load-bearing comparison)")


if __name__ == "__main__":
    main()
