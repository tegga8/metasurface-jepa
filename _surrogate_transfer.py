"""Does the surrogate predict what Meep measures for the SAME decoded geometry?

Phase 2's own numbers already hint at this: the decode hits the target well under
the surrogate (imp2's A' surrogate error 0.427) while under Meep it is no better
than a shuffled target (0.689 vs 0.645). Those two statements are only compatible
if the surrogate's prediction for the decoded geometry disagrees with the
full-wave measurement of that same geometry.

This measures exactly that, per design, with the phase-2 metric and convention:

  transfer err = metrics(|Meep(decoded)|, |target_to_meep(surrogate(decoded))|)
  target   err = metrics(|Meep(decoded)|, |target_to_meep(phase-2 target)|)

If the transfer error is of the same order as the whole phase-2 error, the frozen
surrogate -- not the target generation and not the architecture -- is the
bottleneck: the model optimises a spectrum the surrogate predicts but Meep does
not measure.

NOTE: the convention's own held-out error (0.1138) applies to both numbers equally,
so the comparison between them is unaffected by it.
"""

import glob
import json
import os
import sys

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

from data.factorize import assemble_metadit_geometry  # noqa: E402
from physics.physics_loop import load_surrogate  # noqa: E402
import _compare_phase2 as p2  # noqa: E402
from _compare_phase2 import (  # noqa: E402
    curve, inverse_convention, metrics, target_to_meep, _load_json, _load_npz,
)

W = os.path.expanduser("~/runs/p23imp2")
SETS = (
    ("A_interpolated", "jepa_imp2_A", "A_real", "A_imag"),
    ("B_synthetic", "jepa_imp2_B", "B_real", "B_imag"),
)


def main():
    cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "unified.yaml"),
                              encoding="utf-8"))
    w = cfg["weights"]["surrogate"]
    surrogate = load_surrogate(w if os.path.isabs(w) else os.path.join(REPO, w),
                               device="cpu")
    surrogate.eval()
    conv = _load_json(os.path.join(REPO, "_meep_convention.json"))
    inv = inverse_convention(conv)
    if inv[0] == "full":
        # target_to_meep reads a module-level bias; the phase-2 scorer sets it the
        # same way at startup, so mirror that rather than reimplementing the map.
        p2.conv_bias = np.array(conv["coef"]).reshape(-1, 2, 3)[:, :, 2]
    conj = conv.get("conj", False)
    zt = _load_npz(os.path.expanduser("~/runs/phase3f/random_spectra.npz"))

    for row, tag, rk, ik in SETS:
        dec = _load_npz(os.path.join(W, "dec_imp2", f"decoded_{row}.npz"))
        pats = dec["patterns"]
        sc = dec["scalars"]
        files = sorted(glob.glob(os.path.join(W, "meep", tag, "e_0_*.json")))
        trans, target_e, corr_t = [], [], []
        for f in files:
            j = json.load(open(f, encoding="utf-8"))
            for e in j.get("designs", []):
                if not e.get("passivity_ok"):
                    continue
                k = int(e["idx"])
                occ = torch.from_numpy(pats[k].astype(np.float32))[None, None]
                l, h, r = (float(v) for v in sc[k])
                geom = assemble_metadit_geometry(
                    occ, torch.tensor([l]), torch.tensor([h]), torch.tensor([r]))
                with torch.no_grad():
                    pred = surrogate(geom).prediction.cpu().numpy()[0]
                p_meep = target_to_meep(pred[0], pred[1], inv, conj)
                t_meep = target_to_meep(zt[rk][k], zt[ik][k], inv, conj)
                meas = curve(e)
                er_t, _ = metrics(np.abs(meas), np.abs(p_meep))
                er_g, cr_g = metrics(np.abs(meas), np.abs(t_meep))
                trans.append(er_t)
                target_e.append(er_g)
                corr_t.append(cr_g)
        if trans:
            print(f"\n=== {row} ({len(trans)} passivity-clean designs) ===")
            print(f"  surrogate(decoded) vs Meep(decoded) : err "
                  f"{np.mean(trans):.4f}  (median {np.median(trans):.4f})")
            print(f"  phase-2 target    vs Meep(decoded) : err "
                  f"{np.mean(target_e):.4f}  corr {np.mean(corr_t):+.4f}")
            print(f"  -> the surrogate's own misfit is "
                  f"{np.mean(trans) / max(np.mean(target_e), 1e-9):.2f}x "
                  f"the phase-2 error")
        else:
            print(f"\n=== {row}: no passivity-clean designs found ===")


if __name__ == "__main__":
    main()
