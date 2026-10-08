"""Task 1, step 4: Meep-vs-Meep check on the decoded geometries.

Answers the actual question: does the geometry the model produced radiate the
same field as the geometry it was given? Both spectra come from the same Meep
harness, so the fitted Meep->CST convention cancels entirely and cannot flatter
the result -- this is independent of how good that convention is.

Compared per design, against the original novel geometry's own spectrum:

  real      Meep(decoded_i)   vs Meep(original_i)
  shuffled  Meep(decoded_i)   vs Meep(original_perm(i))   <- null baseline

The shuffled control is not optional. A correlation of 0.8 on its own says
nothing: smooth transmission curves correlate with each other even when
physically unrelated, so without a null we cannot tell a real match from a
generic one. A result only means something if real beats shuffled by more than
the noise on the difference.

Reported: normalized L1 error (mean|diff|/std, evaluator definition), Pearson
correlation of the |T| curves, and the paired real-minus-shuffled margin with a
bootstrap CI on it.

Usage: python _compare_decoded_meep.py <orig_meep.json> <dec_full.json> <dec_half.json>
"""

import glob
import json
import os
import sys

import numpy as np

RNG = np.random.default_rng(0)
NFREQ = 301


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        sys.exit(f"could not read {path}: {exc}")


def load_dir_or_file(path):
    """Accept either a single harness JSON or a directory of per-design JSONs.

    Keyed by NAME, not idx. The decoded npz files contain only the passivity-
    clean subset of the originals, so their harness idx runs 0..37 while the
    original run's idx is 0..39 -- the two index spaces do not correspond.
    Decoded names carry a mask prefix ("full/sym00"), which is stripped.
    """
    if not os.path.exists(path):
        sys.exit(f"no such file or directory: {path}")
    if os.path.isdir(path):
        designs, refs = [], {}
        for f in sorted(glob.glob(os.path.join(path, "*.json"))):
            d = _load_json(f)
            designs += d.get("designs", [])
            refs.update(d.get("refs", {}))
    else:
        d = _load_json(path)
        designs, refs = d.get("designs", []), d.get("refs", {})
    out = {}
    for e in designs:
        if not e.get("passivity_ok"):
            continue
        nm = e.get("name") or str(e["idx"])
        out[nm.split("/")[-1]] = e
    return out, refs


def curve(e):
    return np.abs(np.array(e["modes"]["def"]["re"])
                  + 1j * np.array(e["modes"]["def"]["im"]))


def metrics(a, b):
    std = max(float(b.std()), 1e-12)
    err = float(np.abs(a - b).mean() / std)
    corr = float(np.corrcoef(np.abs(a), np.abs(b))[0, 1]) if a.std() > 0 else 0.0
    return err, corr


def boot_margin(real, shuf, n=5000):
    """Bootstrap CI on mean(real) - mean(shuffled), paired per design."""
    d = np.asarray(real) - np.asarray(shuf)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def sign_p(d):
    """Two-sided sign-test p-value on paired differences (exact binomial)."""
    from math import comb
    d = np.asarray(d)
    k = int((d > 0).sum())
    n = len(d)
    lo = min(k, n - k)
    tail = sum(comb(n, i) for i in range(lo + 1)) / 2.0 ** n
    return min(1.0, 2 * tail)


def main():
    orig_path, full_path, half_path = sys.argv[1:4]
    orig, _ = load_dir_or_file(orig_path)
    print(f"original novel geometries (passivity-clean): {len(orig)}")

    for tag, path in (("full mask", full_path), ("half mask", half_path)):
        dec, _ = load_dir_or_file(path)
        common = sorted(set(orig) & set(dec))
        print(f"\n=== {tag}: {len(dec)} decoded passivity-clean, "
              f"{len(common)} comparable ===")
        dropped = sorted(set(orig) - set(dec))
        if dropped:
            print(f"  excluded (decoded spectrum failed passivity): {dropped}")
        if len(common) < 4:
            print("  not enough comparable designs")
            continue

        re_r, re_c, sh_r, sh_c = [], [], [], []
        perm = RNG.permutation(len(common))
        for i, k in enumerate(common):
            o, d = curve(orig[k]), curve(dec[k])
            er, cr = metrics(d, o)
            es_, cs = metrics(d, curve(orig[common[perm[i]]]))
            re_r.append(er); re_c.append(cr)
            sh_r.append(es_); sh_c.append(cs)
            print(f"  {k:10s} real err={er:.4f} corr={cr:+.3f}   |   "
                  f"shuffled err={es_:.4f} corr={cs:+.3f}")

        m_err, lo_e, hi_e = boot_margin(re_r, sh_r)
        m_cor, lo_c, hi_c = boot_margin(re_c, sh_c)
        d_err = np.asarray(re_r) - np.asarray(sh_r)
        d_cor = np.asarray(re_c) - np.asarray(sh_c)
        print(f"\n  REAL     err={np.mean(re_r):.4f}  corr={np.mean(re_c):+.4f}")
        print(f"  SHUFFLED err={np.mean(sh_r):.4f}  corr={np.mean(sh_c):+.4f}"
              f"   (median err {np.median(sh_r):.4f})")
        print(f"  MEAN margin  err {m_err:+.4f} CI [{lo_e:+.4f},{hi_e:+.4f}] | "
              f"corr {m_cor:+.4f} CI [{lo_c:+.4f},{hi_c:+.4f}]")
        print("  (lower err is better, so a NEGATIVE err margin favours real; "
              "higher corr is better, so a POSITIVE corr margin favours real)")
        print(f"  MEDIAN margin err {np.median(d_err):+.4f} | "
              f"corr {np.median(d_cor):+.4f}   <- outlier-robust")
        print(f"  PER-DESIGN win rate: real err better on "
              f"{int((d_err < 0).sum())}/{len(d_err)}, "
              f"real corr higher on {int((d_cor > 0).sum())}/{len(d_cor)}")
        print(f"  SIGN TEST   err p={sign_p(-d_err):.4g} | "
              f"corr p={sign_p(d_cor):.4g}")
        print(f"  worst 3 shuffled errs: "
              f"{np.round(np.sort(sh_r)[-3:], 2).tolist()}  "
              f"(outliers where the shuffled target has near-zero std)")


if __name__ == "__main__":
    main()