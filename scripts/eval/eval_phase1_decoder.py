"""Phase 1 decoder evaluation: side-by-side metric comparison.

Evaluates both the frozen-JEPA latent decoder and scalar-only baseline
on the same fixed validation set. Reports occupancy IoU/F1, per-channel
MAE, and overall geometry MAE.

Usage:
    python scripts/eval/eval_phase1_decoder.py \
        --checkpoint checkpoints/phase1_decoder/best.pt \
        --base-jepa-checkpoint checkpoints/milestone_b/minimal_jepa_vicreg_latest.pt \
        --config configs/milestone_b.yaml \
        --device cuda:0
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from data.dataset import MetaDiTDataset, collate_batch
from assembly import build_model, load_into_model
from decoders.geometry_decoder import GeometryDecoder
from losses.geometry_reconstruction import GeometryReconstructionLoss
from runtime.device import resolve_device
from runtime.reproducibility import set_seed

OUT_DIR = os.path.join(REPO_ROOT, "checkpoints", "phase1_decoder")


def _derive_scalars(G):
    """Extract [l_lattice, h_atom, r_atom] from a geometry tensor."""
    l_lattice = G[:, 2, 0, 0] * 3.0
    occ = (G[:, 0] != 0) | (G[:, 1] != 0)
    B = G.shape[0]
    h_atom = torch.zeros(B, device=G.device)
    r_atom = torch.zeros(B, device=G.device)
    for i in range(B):
        if occ[i].any():
            h_atom[i] = G[i, 1][occ[i]].mean()
            r_atom[i] = G[i, 0][occ[i]].mean() * 5.0
    return torch.stack([l_lattice, h_atom, r_atom], dim=1)


class _JepaDecoderWrapper(nn.Module):
    def __init__(self, jepa_model, decoder):
        super().__init__()
        self._jepa = jepa_model
        self._decoder = decoder
        self._ema_ref = jepa_model.ema

    def forward(self, x):
        with torch.no_grad():
            z_y_raw = self._ema_ref(x)
        return self._decoder(z_y_raw)

    def eval(self):
        self._decoder.eval()
        return self

    def train(self, mode=True):
        self._decoder.train(mode)
        return self


@torch.no_grad()
def _compute_detailed_metrics(decoder, val_loader, device, decoder_type="jepa"):
    """Compute detailed per-channel metrics with standard errors."""
    decoder_was_training = decoder.training
    decoder.eval()

    occ_correct = 0
    occ_total = 0
    occ_pred_pos = 0
    occ_pred_neg = 0
    occ_true_pos = 0
    tp_sum = fp_sum = fn_sum = tn_sum = 0
    mae_r_sum = mae_h_sum = mae_lattice_sum = mae_overall_sum = 0.0
    n_occ_r_sum = n_occ_h_sum = 0
    n_batches = 0
    n_samples = 0

    for G, _ in val_loader:
        G = G.to(device)
        B = G.shape[0]
        n_samples += B

        if decoder_type == "jepa":
            geom_pred, occ_logits = decoder(G)
        else:
            scalars = _derive_scalars(G)
            geom_pred, occ_logits = decoder(scalars)

        occ_prob = torch.sigmoid(occ_logits)
        occ_pred = (occ_prob > 0.5).float()
        occ_target = GeometryReconstructionLoss.occupancy_target(G)

        # Pixel-level accuracy
        occ_correct += (occ_pred == occ_target).float().sum().item()
        occ_total += occ_target.numel()

        # Confusion matrix
        tp = ((occ_pred == 1) & (occ_target == 1)).float().sum().item()
        fp = ((occ_pred == 1) & (occ_target == 0)).float().sum().item()
        fn = ((occ_pred == 0) & (occ_target == 1)).float().sum().item()
        tn = ((occ_pred == 0) & (occ_target == 0)).float().sum().item()
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        tn_sum += tn

        # Per-channel MAE on occupied pixels
        mask = occ_target.bool()
        n_occ = mask[:, 0].sum().item()
        if n_occ > 0:
            mae_r_sum += (geom_pred[:, 0][mask[:, 0]] -
                          G[:, 0][mask[:, 0]]).abs().sum().item()
            n_occ_r_sum += n_occ
            mae_h_sum += (geom_pred[:, 1][mask[:, 0]] -
                          G[:, 1][mask[:, 0]]).abs().sum().item()
            n_occ_h_sum += n_occ

        mae_lattice_sum += (geom_pred[:, 2] - G[:, 2]).abs().mean().item()
        mae_overall_sum += (geom_pred - G).abs().mean().item()
        n_batches += 1

    decoder.train(decoder_was_training)

    precision = tp_sum / max(tp_sum + fp_sum, 1e-8)
    recall = tp_sum / max(tp_sum + fn_sum, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp_sum / max(tp_sum + fp_sum + fn_sum, 1e-8)

    return {
        "occupancy_accuracy": occ_correct / max(occ_total, 1),
        "occupancy_iou": iou,
        "occupancy_f1": f1,
        "occupancy_precision": precision,
        "occupancy_recall": recall,
        "r_atom_mae": mae_r_sum / max(n_occ_r_sum, 1),
        "h_atom_mae": mae_h_sum / max(n_occ_h_sum, 1),
        "lattice_mae": mae_lattice_sum / max(n_batches, 1),
        "overall_mae": mae_overall_sum / max(n_batches, 1),
        "n_samples": n_samples,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 decoder evaluation")
    p.add_argument("--checkpoint", required=True,
                   help="Phase-1 decoder checkpoint (best.pt)")
    p.add_argument("--base-jepa-checkpoint", required=True,
                   help="Milestone-B checkpoint for building the JEPA model")
    p.add_argument("--config", required=True,
                   help="Path to milestone_b.yaml")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-samples", type=int, default=0,
                   help="0 = full validation set")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: checkpoints/phase1_decoder/eval_report.json)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(args.device or cfg["train"].get("device", "auto"))
    set_seed(args.seed)

    # ---- load validation data ----
    val_ds = MetaDiTDataset(os.path.join(REPO_ROOT, cfg["data"]["val_split"]))
    if args.val_samples > 0:
        val_ds = torch.utils.data.Subset(val_ds, range(min(args.val_samples, len(val_ds))))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
        collate_fn=collate_batch,
    )
    print(f"[eval] Validation samples: {len(val_ds)}")

    # ---- load Phase-1 decoder ----
    decoder = GeometryDecoder().to(device)
    print(f"[eval] Loading decoder checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    decoder.load_state_dict(ckpt["decoder_state"])
    decoder.eval()
    print(f"[eval] Decoder loaded from step={ckpt.get('step')}, "
          f"epoch={ckpt.get('epoch')}")

    # ---- load frozen JEPA model ----
    print(f"[eval] Building frozen JEPA model from: {args.base_jepa_checkpoint}")
    base_obj = torch.load(args.base_jepa_checkpoint, map_location="cpu",
                          weights_only=False)
    if "model" in base_obj:
        model_sd = base_obj["model"]
    elif "state_dict" in base_obj:
        model_sd = base_obj["state_dict"]
    else:
        model_sd = base_obj

    jepa_model = build_model(
        cfg["model"],
        os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
        device=device,
        init_from_metadit=cfg["model"].get("init_from_metadit", True),
        metadit_weights=os.path.join(REPO_ROOT, cfg["weights"]["metadit"]),
    )
    load_into_model(jepa_model, model_sd, device, strict=False)
    if "ema_state" in base_obj and "target" in base_obj["ema_state"]:
        jepa_model.ema.target.load_state_dict(base_obj["ema_state"]["target"])

    jepa_wrapper = _JepaDecoderWrapper(jepa_model, decoder)

    # ---- build scalar baseline from checkpoint's decoder ----
    scalar_decoder = GeometryDecoder().to(device)
    # The scalar baseline uses the SAME decoder backbone. We need to train it
    # separately, but for evaluation we load its trained weights.
    scalar_ckpt_path = os.path.join(OUT_DIR, "scalar_baseline_best.pt")
    if os.path.exists(scalar_ckpt_path):
        scalar_obj = torch.load(scalar_ckpt_path, map_location=device,
                                weights_only=False)
        # The scalar checkpoint stores the full ScalarBaselineDecoder state
        from train.train_phase1_decoder import ScalarBaselineDecoder
        scalar_decoder_full = ScalarBaselineDecoder().to(device)
        scalar_decoder_full.load_state_dict(scalar_obj["decoder_state"])
        scalar_decoder_full.eval()
    else:
        # No separate scalar checkpoint; use untrained baseline for reference
        print("[eval] WARNING: no trained scalar baseline checkpoint found; "
              "using untrained decoder as reference")
        scalar_decoder_full = None

    # ---- compute metrics ----
    print("\n[eval] Computing JEPA latent decoder metrics ...")
    jepa_metrics = _compute_detailed_metrics(jepa_wrapper, val_loader, device,
                                             decoder_type="jepa")

    print("[eval] Computing scalar baseline metrics ...")
    if scalar_decoder_full is not None:
        scalar_metrics = _compute_detailed_metrics(scalar_decoder_full,
                                                   val_loader, device,
                                                   decoder_type="scalar")
    else:
        scalar_metrics = {k: float("nan") for k in jepa_metrics if k != "n_samples"}
        scalar_metrics["n_samples"] = jepa_metrics["n_samples"]

    # ---- print side-by-side table ----
    print()
    print(f"{'metric':<25} {'scalar baseline':>16} {'JEPA latent':>16}")
    print("-" * 60)
    for key in ["occupancy_iou", "occupancy_f1", "occupancy_precision",
                "occupancy_recall", "occupancy_accuracy",
                "r_atom_mae", "h_atom_mae", "lattice_mae", "overall_mae"]:
        s = scalar_metrics.get(key, float("nan"))
        j = jepa_metrics.get(key, float("nan"))
        print(f"{key:<25} {s:>16.4f} {j:>16.4f}")
    print("-" * 60)
    print(f"{'n_samples':<25} {scalar_metrics.get('n_samples', 0):>16} "
          f"{jepa_metrics.get('n_samples', 0):>16}")
    print()

    # ---- Verdict ----
    j_iou = jepa_metrics.get("occupancy_iou", 0)
    s_iou = scalar_metrics.get("occupancy_iou", 0)
    j_mae = jepa_metrics.get("r_atom_mae", float("inf"))
    s_mae = scalar_metrics.get("r_atom_mae", float("inf"))

    if j_iou > s_iou and j_mae < s_mae:
        verdict = "JEPA latent decoder beats scalar baseline on both gates"
    elif j_iou > s_iou:
        verdict = ("JEPA wins on occupancy IoU but scalar wins on r_atom MAE — "
                   "ambiguous, consider a second seed")
    elif j_mae < s_mae:
        verdict = ("JEPA wins on r_atom MAE but scalar wins on occupancy IoU — "
                   "ambiguous, consider a second seed")
    else:
        verdict = ("Scalar baseline matches or beats JEPA latent decoder — "
                   "STOP and diagnose before proceeding")
    print(f"[verdict] {verdict}")

    # ---- save report ----
    output = args.output or os.path.join(OUT_DIR, "eval_report.json")
    report = {
        "jepa_latent": jepa_metrics,
        "scalar_baseline": scalar_metrics,
        "verdict": verdict,
        "checkpoint": args.checkpoint,
        "base_jepa_checkpoint": args.base_jepa_checkpoint,
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n[eval] Report saved to {output}")


if __name__ == "__main__":
    main()
