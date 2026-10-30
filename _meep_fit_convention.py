"""Robust Meep->dataset convention fit (v3b/v4).

Only calibration items whose Meep spectrum actually agrees with the dataset
(flux-magnitude corr > 0.9) inform the convention — a genuine physics
disagreement (e.g. a resonance-shifted item) must not drag the fit. Per item,
fit (conj flag, a, b) for data ~= exp(i(a + b*u)) * t (u = idx/(n-1), robust
frequency mask); combine good items by median. Report everything, including
the excluded items and the residual each item achieves under the FINAL params.

Usage: python _meep_fit_convention.py <calib.json> <outdir>
"""

import json
import os
import sys

import numpy as np

calib_path, out_dir = sys.argv[1], sys.argv[2]
items = json.load(open(calib_path))
print(f"items: {len(items)}")

tags = [t for t in ("def", "ez", "oz")
        if all(t in it.get("modes", {}) for it in items)]
best_tag, best_score = None, -1
for t in tags:
    cs = [float(np.dot(np.asarray(it["modes"][t]["mag"]),
                       np.asarray(it["flux_mag"])) /
                (np.linalg.norm(it["modes"][t]["mag"])
                 * np.linalg.norm(it["flux_mag"]) + 1e-30)) for it in items]
    if np.mean(cs) > best_score:
        best_tag, best_score = t, float(np.mean(cs))
print("chosen mode tag:", best_tag)


def item_fit(it, flag):
    t = np.array(it["modes"][best_tag]["re"]) + 1j * np.array(it["modes"][best_tag]["im"])
    d = np.array(it["d_re"]) + 1j * np.array(it["d_im"])
    n = len(t)
    u = np.arange(n) / (n - 1)
    th = np.conj(t) if flag else t
    Tm = d / np.where(np.abs(th) < 1e-30, 1e-30, th)
    good = (np.abs(t) > 0.25 * np.abs(t).max()) & (np.abs(d) > 0.25 * np.abs(d).max())
    ang = np.unwrap(np.angle(Tm))[good]
    U = u[good]
    A = np.vstack([np.ones_like(U), U]).T
    coef, *_ = np.linalg.lstsq(A, ang, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    T = np.exp(1j * (a + b * u))
    res = float(np.linalg.norm(d - T * th) / (np.linalg.norm(d) + 1e-30))
    return a, b, res


good_items = [it for it in items if it.get("corr_fluxmag", 0.0) > 0.9]
bad_items = [it for it in items if it.get("corr_fluxmag", 0.0) <= 0.9]
print(f"agreement gate (>0.9): good={len(good_items)} excluded={len(bad_items)} "
      f"(excluded idx: {[it['idx'] for it in bad_items]})")
assert good_items, "no calibration item agrees with the dataset — cannot fit"

per = {0: [], 1: []}
for it in good_items:
    for flag in (0, 1):
        a, b, res = item_fit(it, flag)
        per[flag].append((a, b, res))
        print(f"  item{it['idx']} flag={flag}: a={a:.4f} b={b:.4f} res={res:.4f}")

flag = 0 if np.median([r for _, _, r in per[0]]) <= np.median([r for _, _, r in per[1]]) else 1
a = float(np.median([a for a, _, _ in per[flag]]))
b = float(np.median([b for _, b, _ in per[flag]]))
print(f"chosen flag={flag} a={a:.4f} b={b:.4f} (median over {len(per[flag])} good items)")

# residuals under the FINAL params for EVERY item (good and excluded)
final_res = []
for it in items:
    t = np.array(it["modes"][best_tag]["re"]) + 1j * np.array(it["modes"][best_tag]["im"])
    d = np.array(it["d_re"]) + 1j * np.array(it["d_im"])
    n = len(t)
    u = np.arange(n) / (n - 1)
    th = np.conj(t) if flag else t
    T = np.exp(1j * (a + b * u))
    final_res.append(float(np.linalg.norm(d - T * th) / (np.linalg.norm(d) + 1e-30)))
print("final-params residuals per item:", np.round(final_res, 4).tolist())

out = {"conj": bool(flag), "a": a, "b": b, "mode_tag": best_tag,
       "good_item_idx": [int(it["idx"]) for it in good_items],
       "excluded_item_idx": [int(it["idx"]) for it in bad_items],
       "final_residuals": final_res,
       "good_mean_residual": float(np.mean([final_res[i] for i, it in enumerate(items)
                                            if it in good_items]))}
json.dump(out, open(os.path.join(out_dir, "_meep_convention.json"), "w"), indent=1)
print("chosen:", json.dumps(out, indent=1))
