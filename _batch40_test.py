"""Batch40 evaluation: random designs decoded from their Meep spectra under
FULL mask vs 50% mask (same samples, same masker seed), all four models.

Outputs per model and mask setting:
  * IoU / F1 of the decoded geometry vs the true design (sym / asym / all);
  * spectrum similarity: decoded geometry -> assembled [B,3,64,64] -> frozen
    surrogate -> spectrum vs the design's Meep spectrum (normalized L1 error +
    |T| magnitude correlation);
  * decoded occupancy maps saved for the Meep-on-decodes confirmation.
Renders a paired per-design IoU figure (full vs half).

Usage: python _batch40_test.py <scratchpad_dir>
"""

import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "train"))
sys.path.insert(0, os.path.join(REPO, "scripts", "eval"))
sys.path.insert(0, REPO)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eval_scenarios as es
import _novel_modeltest as nmt
from data.factorize import assemble_metadit_geometry
from data.mask import BlockMasker

SP = sys.argv[1]
conv = json.load(open(os.path.join(SP, "_meep_convention.json")))
meep = json.load(open(os.path.join(SP, "batch40_meep.json")))
z = np.load(os.path.join(REPO, "batch40_designs.npz"), allow_pickle=True)
names = z["names"].tolist()
pats = torch.from_numpy(z["patterns"].astype(np.float32)).unsqueeze(1)  # [40,1,64,64]
l, h, r = (float(v) for v in z["scalars"])
n = len(names)
sv = torch.tensor([[l, h, r]] * n, dtype=torch.float32)
spec = torch.stack([nmt.meep_spec_from_entry(dd, conv)
                    for dd in meep["designs"]])  # [40,2,301]
target_mag = spec.pow(2).sum(1).sqrt().numpy()

models = [
    ("S2", "configs/scaling/unified_s2_slim.yaml",
     os.path.join(SP, "s2_out", "s2", "seed0.pt")),
    ("S1", "configs/scaling/unified_s1_small.yaml",
     os.path.join(SP, "s1_out", "s1", "seed0.pt")),
    ("L1", "configs/scaling/unified_l1_wide.yaml",
     os.path.join(SP, "l1a_ckpt", "l1", "seed0.pt")),
    ("base", "configs/unified.yaml",
     os.path.join(SP, "schedfix_ckpt", "schedfix", "seed0.pt")),
]

sym = np.array([nm.startswith("sym") for nm in names])
all_iou = {}
for model_name, cfg, ckpt in models:
    model, surrogate = nmt.load_model(os.path.join(REPO, cfg), ckpt)
    for ratio, tag in ((1.0, "full"), (0.5, "half")):
        masker = BlockMasker(placement="random", grid=16, min_side=3,
                             k_range=(1, 4), seed=42)
        M = masker.sample(pats, ratio=ratio)
        with torch.no_grad():
            sk = torch.zeros(n, 3, dtype=torch.bool)
            out = model(pats, sv, sk, spec, M, goal_mode="real",
                        with_target=False)
            prob = model.decode_occupancy_prob(out["z_hat"], out["scalar_pred"],
                                               scalar_known=sk, scalar_values=sv)
            hard = (prob > 0.5).float()
            geom = assemble_metadit_geometry(
                hard, torch.full((n,), l), torch.full((n,), h),
                torch.full((n,), r))
            pred = surrogate(geom).prediction  # [40,2,301]
        err = es._spectrum_error_per_sample(pred, spec).numpy()
        pred_mag = pred.pow(2).sum(1).sqrt().numpy()
        corr = np.array([np.corrcoef(target_mag[i], pred_mag[i])[0, 1]
                         if pred_mag[i].std() > 0 else 0.0 for i in range(n)])

        ious = np.zeros(n)
        for k in range(n):
            iou, _, _, _ = nmt.iou_f1(prob[k], pats[k])
            ious[k] = iou
        all_iou[(model_name, tag)] = ious
        np.savez(os.path.join(SP, f"batch40_dec_{model_name}_{tag}.npz"),
                 prob=prob.detach().cpu().numpy())
        print(f"[{model_name} | {tag:4s}] IoU all={ious.mean():.3f} "
              f"sym={ious[sym].mean():.3f} asym={ious[~sym].mean():.3f} | "
              f"spec err={err.mean():.3f} corr={corr.mean():.3f}", flush=True)

fig, ax = plt.subplots(figsize=(16, 5))
mean_full = np.mean([all_iou[(m, "full")] for m, _, _ in models], axis=0)
mean_half = np.mean([all_iou[(m, "half")] for m, _, _ in models], axis=0)
x = np.arange(n)
ax.bar(x - 0.2, mean_full, 0.4, label="full mask", color="black")
ax.bar(x + 0.2, mean_half, 0.4, label="50% mask", color="tab:blue")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=90, fontsize=7)
ax.axvline(19.5, color="gray", ls="--")
ax.set_ylabel("IoU (mean over 4 models)")
ax.set_title("batch40 decode: full mask vs 50% mask (left 20 = symmetric, "
             "right 20 = asymmetric)")
ax.legend()
fig.tight_layout()
out_png = os.path.join(REPO, "batch40_masks_summary.png")
fig.savefig(out_png, dpi=115)
print("saved", out_png)
