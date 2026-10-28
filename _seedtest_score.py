"""Seed test, step 3: Meep-based candidate selection, paired real vs shuffled target.

Design (corrected after the first draft, which had a degenerate control):

  For every target k we hold five candidate geometries -- one per guidance strength
  w in {0.5, 0.75, 1.0, 1.25, 1.5} -- all decoded from target k. Meep measures all
  five. The SELECTION procedure is identical in both arms; only the target the
  errors are computed against changes:

      real arm      select the candidate with the lowest error vs target k
      shuffled arm  select the candidate with the lowest error vs target perm(k)

  where perm is a derangement of the same 18 targets. So each arm searches the same
  five geometries; the question is whether choosing by the *correct* target
  achieves a lower measured error than choosing by a *wrong* one. That is the
  question the seed test is supposed to answer, and unlike "decode from shuffled
  targets and select", it is not degenerate: a permutation of the candidates cannot
  produce the same statistic, and a model that ignores the target gives a margin of
  exactly zero (the sharp null).

Metric and convention are the phase-2 ones, so the errors are on the same scale as
every phase-2 number.
"""

import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import _compare_phase2 as p2  # noqa: E402
from _compare_phase2 import (  # noqa: E402
    curve, inverse_convention, metrics, target_to_meep, _load_json, _load_npz,
)
from _compare_decoded_meep import boot_margin, sign_p  # noqa: E402

ROOT = os.path.expanduser("~/runs/seedtest")
W_LIST = ("w050", "w075", "w100", "w125", "w150")


def main():
    conv = _load_json(os.path.join(REPO, "_meep_convention.json"))
    inv = inverse_convention(conv)
    if inv[0] == "full":
        p2.conv_bias = np.array(conv["coef"]).reshape(-1, 2, 3)[:, :, 2]
    conj = conv.get("conj", False)
    z = _load_npz(os.path.expanduser("~/runs/phase3f/random_spectra.npz"))
    idx = json.load(open(os.path.join(ROOT, "dec", "index.json"), encoding="utf-8"))
    n_t = len({d["slot"] for d in idx["designs"].values()})

    # derangement of the target assignment, same seed the decode used
    from runtime.physics_controls import derangement_permutation  # noqa: E402
    perm = derangement_permutation(n_t, "cpu", seed=0).numpy()

    def tgt(i):
        return target_to_meep(z["A_real"][i], z["A_imag"][i], inv, conj)

    # (w, slot) -> error against own target and against the shuffled target
    e_real, e_shuf = {}, {}
    for w in W_LIST:
        files = sorted(glob.glob(os.path.join(ROOT, "meep", f"real_{w}", "e_0_*.json")))
        clean = 0
        for f in files:
            j = json.load(open(f, encoding="utf-8"))
            for e in j.get("designs", []):
                if not e.get("passivity_ok"):
                    continue
                slot = int(e["idx"])
                m = np.abs(curve(e))
                e_real[(w, slot)] = metrics(m, np.abs(tgt(slot)))[0]
                e_shuf[(w, slot)] = metrics(m, np.abs(tgt(int(perm[slot]))))[0]
                clean += 1
        print(f"real_{w}: passivity-clean designs {clean}/{n_t}")

    print("\n=== per-guidance mean error (no selection) ===")
    print(f"{'w':6s} {'vs own target':>14s} {'vs shuffled':>13s} {'margin':>9s}")
    for w in W_LIST:
        r = [e_real[(w, s)] for s in range(n_t) if (w, s) in e_real]
        h = [e_shuf[(w, s)] for s in range(n_t) if (w, s) in e_shuf]
        if r and h:
            print(f"{w:6s} {np.mean(r):14.4f} {np.mean(h):13.4f} "
                  f"{np.mean(h) - np.mean(r):+9.4f}")

    print("\n=== SELECTED: best of 5 by measured error, paired per target ===")
    sel_r, sel_h, slot_used = [], [], []
    for s in range(n_t):
        r = [(e_real[(w, s)], w) for w in W_LIST if (w, s) in e_real]
        h = [e_shuf[(w, s)] for w in W_LIST if (w, s) in e_shuf]
        if r and h:
            best_r, w_best = min(r)
            sel_r.append(best_r)
            sel_h.append(min(h))
            slot_used.append((s, w_best))
    if len(sel_r) >= 4:
        r, h = np.array(sel_r), np.array(sel_h)
        d = h - r                                  # positive = real target better
        mean, lo, hi = boot_margin(h, r)
        print(f"  targets compared : {len(r)}")
        print(f"  real    mean {r.mean():.4f}  median {np.median(r):.4f}")
        print(f"  shuffled mean {h.mean():.4f}  median {np.median(h):.4f}")
        print(f"  paired margin   mean {mean:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  median margin   {np.median(d):+.4f}")
        print(f"  win rate        {int((d > 0).sum())}/{len(d)}")
        print(f"  sign test p     {sign_p(d):.4g}")
        print(f"  CI excludes 0?  {'YES' if lo > 0 else 'no'}")
        from collections import Counter
        print(f"  guidance chosen : {dict(Counter(w for _, w in slot_used))}")

    print("\n=== reference: no selection (single shot, w = 1.0) ===")
    r = [e_real[("w100", s)] for s in range(n_t) if ("w100", s) in e_real]
    h = [e_shuf[("w100", s)] for s in range(n_t) if ("w100", s) in e_shuf]
    if r and h:
        n = min(len(r), len(h))
        d = np.array(h[:n]) - np.array(r[:n])
        print(f"  vs own target {np.mean(r):.4f}  vs shuffled {np.mean(h):.4f}  "
              f"margin {d.mean():+.4f}  win {int((d > 0).sum())}/{n}  "
              f"p {sign_p(d):.4g}")


if __name__ == "__main__":
    main()
