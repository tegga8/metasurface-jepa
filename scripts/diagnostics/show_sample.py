"""Show one test sample: pick a random test spectrum, generate the geometry with the
model (Scenario A: full occupancy mask, all scalars unknown), run the frozen surrogate
on the generated geometry, and save the figures + metrics.

Run:
    python scripts/diagnostics/show_sample.py --config configs/unified.yaml \
        --checkpoint <ckpt.pt> --seed 12345 --out-dir docs/benchmarking/sample_figures
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "eval"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "benchmark"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import eval_scenarios as es
from data.dataset import MetaDiTDataset
from data.factorize import factorize_geometry, assemble_metadit_geometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out-dir", default=os.path.join("docs", "benchmarking",
                                                      "sample_figures"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(args.device)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(
        REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # data: one random test item (fixed seed -> reproducible)
    tg = cfg["data"].get(f"{args.split}_split",
                         os.path.join("data", "metadit", "split_data",
                                      f"{args.split}_set.mat"))
    ds = MetaDiTDataset(os.path.join(REPO_ROOT, tg) if not os.path.isabs(tg) else tg)
    n = len(ds)
    idx = int(np.random.RandomState(args.seed).randint(0, n))
    G, S = ds[idx]
    G = G.unsqueeze(0); S = S.unsqueeze(0)
    occ, sv = factorize_geometry(G)
    occ, sv, S = occ.to(device), sv.to(device), S.to(device)

    model, surrogate = es._load_eval(cfg, args.checkpoint, device)
    model.eval()
    b = 1
    sk = torch.zeros(b, 3, dtype=torch.bool, device=device)      # all scalars unknown
    M = torch.ones(b, 16, 16, device=device)                     # FULL occupancy mask

    with torch.no_grad():
        out = model(occ, sv, sk, S, M, goal_mode="real", with_target=False)
        geom, occ_delivered = model.decode_geometry(
            out["z_hat"], out["scalar_pred"], occ_input=occ, mask=M,
            scalar_known=sk, scalar_values=sv, hard_forward=True)
        pred = surrogate(geom).prediction                        # (1,2,301)

    # metrics (MetaDiT units)
    mae = float((pred - S).abs().mean().item())
    aae = float((pred - S).abs().sum().item())
    true_bin = (occ[:, 0] > 0.5)
    pred_bin = (occ_delivered[:, 0] > 0.5)
    tp = float((pred_bin & true_bin).sum()); fp = float((pred_bin & ~true_bin).sum())
    fn = float((~pred_bin & true_bin).sum())
    iou = tp / max(1.0, tp + fp + fn)
    f1 = 2 * tp / max(1.0, 2 * tp + fp + fn)
    scal = {"l_true": float(sv[0, 0]), "h_true": float(sv[0, 1]), "r_true": float(sv[0, 2]),
            "l_pred": float(out["scalar_pred"][0, 0]), "h_pred": float(out["scalar_pred"][0, 1]),
            "r_pred": float(out["scalar_pred"][0, 2])}
    metrics = {"split": args.split, "test_index": idx, "n_test": n,
               "MAE": mae, "AAE": aae, "occupancy_iou": iou, "occupancy_f1": f1,
               "pred_occupancy_fraction": float(pred_bin.float().mean()),
               "true_occupancy_fraction": float(true_bin.float().mean()),
               "scalars": scal}

    # figure: true occupancy | generated occupancy | spectra
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    ax[0].imshow(true_bin[0].cpu(), cmap="gray_r"); ax[0].set_title("TRUE occupancy (64x64)")
    ax[1].imshow(pred_bin[0].cpu(), cmap="gray_r")
    ax[1].set_title(f"GENERATED occupancy\nIoU={iou:.3f}  F1={f1:.3f}")
    tr = S[0, 0].cpu().numpy(); ti = S[0, 1].cpu().numpy()
    pr = pred[0, 0].cpu().numpy(); pi = pred[0, 1].cpu().numpy()
    ax[2].plot(tr, label="target real", lw=1.2)
    ax[2].plot(pr, "--", label="surrogate(gen) real", lw=1.2)
    ax[2].plot(ti, label="target imag", lw=1.2, alpha=0.6)
    ax[2].plot(pi, "--", label="surrogate(gen) imag", lw=1.2, alpha=0.6)
    ax[2].set_title(f"spectrum: MAE={mae:.4f}  AAE={aae:.2f}")
    ax[2].legend(fontsize=7)
    for a in ax[:2]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Test item #{idx} (of {n}) — Scenario A: full occupancy mask, "
                 f"all scalars unknown", fontsize=10)
    fig.tight_layout()
    figpath = os.path.join(out_dir, f"sample_{args.split}_{idx}_fig.png")
    fig.savefig(figpath, dpi=130)

    # geometry channels figure (the 3-channel MetaDiT tensor the surrogate consumes)
    gg = geom[0].detach().cpu().numpy()
    fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4))
    for c, name in enumerate(("ch0 = r/5 (occupied)", "ch1 = h (occupied)",
                              "ch2 = l/3 (dense)")):
        ax2[c].imshow(gg[c], cmap="viridis"); ax2[c].set_title(name); ax2[c].set_xticks([]); ax2[c].set_yticks([])
    fig2.suptitle("Generated geometry (assembled 3x64x64 fed to the frozen surrogate)",
                  fontsize=10)
    fig2.tight_layout()
    gpath = os.path.join(out_dir, f"sample_{args.split}_{idx}_geometry.png")
    fig2.savefig(gpath, dpi=130)

    with open(os.path.join(out_dir, f"sample_{args.split}_{idx}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({"figure": figpath, "geometry_figure": gpath, **metrics}, indent=2))


if __name__ == "__main__":
    main()
