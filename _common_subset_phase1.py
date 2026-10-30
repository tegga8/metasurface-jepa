"""Phase-1 physics, all three models on the IDENTICAL design subset.

base v1, v2 and MetaDiT-S each had a different set of passivity-clean designs, so
their individual scores are not directly comparable. This restricts all three to
the intersection and re-scores with the same shuffled permutation.

Usage: python _common_subset_phase1.py <orig> <v1_full_dir> <v2_full_dir> <md_dir>
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _compare_decoded_meep import (boot_margin, curve, load_dir_or_file,  # noqa: E402, I001
                                   metrics, sign_p)


def main():
    orig_p, v1_p, v2_p, md_p = sys.argv[1:5]
    orig, _ = load_dir_or_file(orig_p)
    sets = {
        "base v1": load_dir_or_file(v1_p)[0],
        "v2": load_dir_or_file(v2_p)[0],
        "MetaDiT-S": load_dir_or_file(md_p)[0],
    }
    common = sorted(set(orig) & set(sets["base v1"]) & set(sets["v2"])
                    & set(sets["MetaDiT-S"]))
    print(f"common passivity-clean designs: {len(common)}  {common[:6]}...\n")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(common))       # ONE permutation for all three
    print(f"{'model':11s} {'real err':>9s} {'shuf err':>9s} {'med m_err':>10s} "
          f"{'real corr':>10s} {'shuf corr':>10s} {'med m_corr':>11s} "
          f"{'win err':>8s} {'win corr':>9s} {'p err':>9s} {'p corr':>8s}")
    for name, ds in sets.items():
        re_e, re_c, sh_e, sh_c = [], [], [], []
        for i, k in enumerate(common):
            m = curve(ds[k])
            er, cr = metrics(m, curve(orig[k]))
            es, cs = metrics(m, curve(orig[common[perm[i]]]))
            re_e.append(er); re_c.append(cr); sh_e.append(es); sh_c.append(cs)
        de = np.asarray(re_e) - np.asarray(sh_e)
        dc = np.asarray(re_c) - np.asarray(sh_c)
        me, lo_e, hi_e = boot_margin(re_e, sh_e)
        print(f"{name:11s} {np.mean(re_e):9.4f} {np.mean(sh_e):9.4f} "
              f"{np.median(de):+10.4f} {np.mean(re_c):+10.4f} "
              f"{np.mean(sh_c):+10.4f} {np.median(dc):+11.4f} "
              f"{int((de < 0).sum()):4d}/{len(de):<3d} "
              f"{int((dc > 0).sum()):4d}/{len(dc):<4d} "
              f"{sign_p(-de):9.3g} {sign_p(dc):8.3g}")
        print(f"{'':11s}   mean err margin {me:+.4f} CI [{lo_e:+.4f},{hi_e:+.4f}]")
    print("\n(err: lower is better, so a negative error margin favours the model; "
          "corr: higher is better)")


if __name__ == "__main__":
    main()
