"""Mirror-leak diagnostic: masked-region reconstruction, leaky vs leak-free masks.

The corpus is exactly mirror-symmetric by construction, so under independent
block masking a masked token often has a VISIBLE mirror partner and its content
is determined by the visible geometry alone. This measures what a checkpoint's
masked-region score is worth once that path is removed: same checkpoint, same
designs, same ratio — only the mask placement changes.

Reported per placement ("random" = shipped baseline, "mirror_pairs" = imp2):
  * leak fraction (masked tokens with a visible mirror partner),
  * IoU over the MASKED region of the deployed occupancy,
  * occupancy fraction (collapse check).

Usage:
  python scripts/diagnostics/mirror_leak.py <config> <ckpt> [--n 40] [--ratio 0.5]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

import _novel_modeltest as nmt  # noqa: E402
from _compare_decoded_meep import load_dir_or_file  # noqa: E402
from _fit_meep_convention import APPLY  # noqa: E402
from data.mask import (BlockMasker, mirror_leak_fraction,  # noqa: E402
                       mirror_pair_masks, random_masks)


def _iou(a, b):
    a, b = a > 0.5, b > 0.5
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--targets", default="",
                    help="b40_merged.json (Meep targets); default: next to the "
                         "checkpoint, else ~/runs/phase3/")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    conv = json.load(open(os.path.join(REPO, "_meep_convention.json"),
                          encoding="utf-8"))
    b = np.load(os.path.join(REPO, "batch40_designs.npz"), allow_pickle=True)
    names_all = b["names"].tolist()
    occ_all = b["patterns"].astype(np.float32)
    l0, h0, r0 = (float(v) for v in b["scalars"])
    cand = []
    if args.targets:
        cand.append(args.targets)
    cand.append(os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                             "b40_merged.json"))
    cand.append(os.path.expanduser("~/runs/phase3/b40_merged.json"))
    orig = None
    for p in cand:
        if p and os.path.exists(p):
            orig = load_dir_or_file(p)[0]
            print("targets:", p)
            break
    if orig is None:
        sys.exit(f"no b40_merged.json found in {cand}")

    picks, targets = [], []
    for nm in names_all:
        if orig is not None and nm not in orig:
            continue
        picks.append(names_all.index(nm))
        if orig is not None:
            t = APPLY[conv["kind"]](orig[nm], conv)
            targets.append(np.stack([t.real, t.imag]).astype(np.float32))
        if len(picks) >= args.n:
            break
    if not targets:
        sys.exit("no usable targets (need b40_merged.json)")
    occ = torch.from_numpy(occ_all[picks]).unsqueeze(1)          # (N,1,64,64)
    target = torch.from_numpy(np.stack(targets))                 # (N,2,301)
    sv = torch.tensor(np.tile([l0, h0, r0], (len(picks), 1)),
                      dtype=torch.float32)
    known = torch.zeros(len(picks), 3, dtype=torch.bool)

    model, _surrogate = nmt.load_model(args.config, args.ckpt)
    model.eval()

    print(f"{os.path.basename(args.ckpt)} | n={len(picks)} | ratio={args.ratio}")
    rows = {}
    for placement in ("random", "mirror_pairs"):
        rng = torch.Generator()
        rng.manual_seed(0)
        if placement == "random":
            M = random_masks(rng, len(picks), args.ratio, grid=16)
        else:
            M = mirror_pair_masks(rng, len(picks), args.ratio, grid=16)
        leaked = mirror_leak_fraction(M)
        up = M.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2).unsqueeze(1)
        up_bool = (up > 0.5)
        with torch.no_grad():
            out = model(occ, sv, known, target, M, goal_mode="real",
                        with_target=False)
            prob = model.decode_occupancy_prob(out["z_hat"], out["scalar_pred"],
                                               scalar_known=known,
                                               scalar_values=sv)
        # deployed retention rule: visible pixels come from the input
        deployed = torch.where(up_bool, occ, (prob > 0.5).float())
        masked = ~up_bool.expand_as(deployed)
        ious = []
        for i in range(len(picks)):
            m = masked[i, 0]
            a = deployed[i, 0][m].numpy()
            t = occ[i, 0][m].numpy()
            ious.append(_iou(a, t))
        occfrac = float(deployed.mean())
        rows[placement] = dict(leak=leaked, masked_iou=float(np.mean(ious)),
                               occ_frac=occfrac)
        print(f"  {placement:13s} leak={leaked:5.3f}  "
              f"masked-region IoU={np.mean(ious):.4f}  occ={occfrac:.3f}")
    d = rows["mirror_pairs"]["masked_iou"] - rows["random"]["masked_iou"]
    print(f"  delta IoU (mirror_pairs - random) = {d:+.4f}  "
          f"<- negative means the random-mask score was inflated by the leak")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"ckpt": args.ckpt, "ratio": args.ratio,
                       "n": len(picks), "rows": rows, "delta_iou": d}, f, indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
