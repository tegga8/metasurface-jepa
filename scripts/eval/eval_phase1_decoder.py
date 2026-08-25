"""Phase 1 decoder evaluation: side-by-side metric comparison.

Evaluates both the frozen-JEPA latent decoder and scalar-only baseline
on the same fixed validation set. Reports occupancy IoU/F1, per-channel
MAE, normalized combined occupied MAE, and a strict Phase-1 gate verdict.

Prerequisites (all MUST be present or the script raises):
  - EMA state with target in the base JEPA checkpoint
  - Trained scalar baseline checkpoint
  - Stored loss scales in the Phase-1 decoder checkpoint
  - Strict checkpoint compatibility

Usage:
    python scripts/eval/eval_phase1_decoder.py \
        --checkpoint checkpoints/phase1_decoder/best.pt \
        --base-jepa-checkpoint checkpoints/milestone_b/minimal_jepa_vicreg_latest.pt \
        --scalar-baseline-checkpoint checkpoints/phase1_decoder/scalar_baseline_best.pt \
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
EPSILON = 1e-8


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
def _compute_detailed_metrics(decoder, val_loader, device, decoder_type="jepa",
                              loss_config=None):
    """Compute detailed per-channel metrics with normalized errors."""
    decoder_was_training = decoder.training
    decoder.eval()

    tp_sum = fp_sum = fn_sum = 0
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

        # Confusion matrix
        tp = ((occ_pred == 1) & (occ_target == 1)).float().sum().item()
        fp = ((occ_pred == 1) & (occ_target == 0)).float().sum().item()
        fn = ((occ_pred == 0) & (occ_target == 1)).float().sum().item()
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

        # Per-channel MAE on occupied pixels
        mask = occ_target.squeeze(1).bool()  # (B, 64, 64)
        n_occ = mask.float().sum().item()
        if n_occ > 0:
            mae_r_sum += (geom_pred[:, 0][mask] -
                          G[:, 0][mask]).abs().sum().item()
            n_occ_r_sum += n_occ
            mae_h_sum += (geom_pred[:, 1][mask] -
                          G[:, 1][mask]).abs().sum().item()
            n_occ_h_sum += n_occ

        mae_lattice_sum += (geom_pred[:, 2] - G[:, 2]).abs().mean().item()
        mae_overall_sum += (geom_pred - G).abs().mean().item()
        n_batches += 1

    decoder.train(decoder_was_training)

    precision = tp_sum / max(tp_sum + fp_sum, 1e-8)
    recall = tp_sum / max(tp_sum + fn_sum, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp_sum / max(tp_sum + fp_sum + fn_sum, 1e-8)

    r_atom_mae = mae_r_sum / max(n_occ_r_sum, 1)
    h_atom_mae = mae_h_sum / max(n_occ_h_sum, 1)

    # Normalized combined occupied MAE using stored training scales
    scale_r = loss_config["scale_r"] if loss_config else 1.0
    scale_h = loss_config["scale_h"] if loss_config else 1.0
    norm_r = r_atom_mae / scale_r
    norm_h = h_atom_mae / scale_h
    combined_occupied_mae = 0.5 * norm_r + 0.5 * norm_h

    return {
        "occupancy_iou": iou,
        "occupancy_f1": f1,
        "occupancy_precision": precision,
        "occupancy_recall": recall,
        "r_atom_mae": r_atom_mae,
        "h_atom_mae": h_atom_mae,
        "combined_occupied_mae": combined_occupied_mae,
        "normalized_r_mae": norm_r,
        "normalized_h_mae": norm_h,
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
    p.add_argument("--scalar-baseline-checkpoint", default=None,
                   help="Trained scalar baseline checkpoint. If omitted, "
                        "tries checkpoints/phase1_decoder/scalar_baseline_best.pt")
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
        val_ds = torch.utils.data.Subset(val_ds,
                                         range(min(args.val_samples, len(val_ds))))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
        collate_fn=collate_batch,
    )
    print(f"[eval] Validation samples: {len(val_ds)}")

    # ================================================================
    # 1. Load Phase-1 decoder checkpoint (strict, with stored scales)
    # ================================================================
    decoder = GeometryDecoder().to(device)
    print(f"[eval] Loading decoder checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Require stored loss_config
    if "loss_config" not in ckpt:
        raise RuntimeError(
            f"Phase-1 checkpoint {args.checkpoint} is missing 'loss_config'. "
            "This checkpoint was saved before loss-scale storage was added. "
            "Re-run training to produce a valid checkpoint."
        )
    loss_config = ckpt["loss_config"]

    # Require stored scales
    for key in ("scale_r", "scale_h"):
        if key not in loss_config:
            raise RuntimeError(
                f"loss_config in checkpoint is missing '{key}'. "
                "Cannot evaluate without stored training scales."
            )
        if not (isinstance(loss_config[key], (int, float))
                and loss_config[key] > 0):
            raise RuntimeError(
                f"loss_config['{key}'] must be a positive number, "
                f"got {loss_config[key]!r}"
            )
    print(f"[eval] Using stored scales: scale_r={loss_config['scale_r']:.6f}, "
          f"scale_h={loss_config['scale_h']:.6f}")

    # Strict decoder state-dict loading
    decoder.load_state_dict(ckpt["decoder_state"], strict=True)
    decoder.eval()
    print(f"[eval] Decoder loaded (strict) from step={ckpt.get('step')}, "
          f"epoch={ckpt.get('epoch')}")

    # ================================================================
    # 2. Load frozen JEPA model — EMA state is MANDATORY
    # ================================================================
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

    # Strict JEPA checkpoint loading
    load_into_model(jepa_model, model_sd, device, strict=True)

    # EMA state is MANDATORY
    if "ema_state" not in base_obj:
        raise RuntimeError(
            "Base JEPA checkpoint is missing 'ema_state'. "
            "The EMA target encoder is required for Phase-1 evaluation. "
            "Cannot proceed without a trained EMA target."
        )
    ema_state = base_obj["ema_state"]
    if "target" not in ema_state:
        raise RuntimeError(
            "Base JEPA checkpoint has 'ema_state' but it is missing 'target'. "
            "The EMA target state dict is required for Phase-1 evaluation."
        )
    jepa_model.ema.target.load_state_dict(ema_state["target"])
    if "total_steps" in ema_state:
        jepa_model.ema.set_total_steps(ema_state["total_steps"])

    # Freeze and enforce eval mode on EMA
    jepa_model.ema.eval()
    for p in jepa_model.ema.parameters():
        p.requires_grad_(False)
    assert all(not p.requires_grad for p in jepa_model.ema.parameters()), \
        "EMA parameters are not frozen after loading"
    print("[eval] EMA state loaded, frozen, and in eval mode")

    jepa_wrapper = _JepaDecoderWrapper(jepa_model, decoder)

    # ================================================================
    # 3. Load trained scalar baseline — MANDATORY, no untrained fallback
    # ================================================================
    scalar_ckpt_path = args.scalar_baseline_checkpoint or os.path.join(
        OUT_DIR, "scalar_baseline_best.pt")
    if not os.path.exists(scalar_ckpt_path):
        raise RuntimeError(
            f"Trained scalar baseline checkpoint not found: {scalar_ckpt_path}. "
            "The scalar baseline is part of the Phase-1 gate and MUST be "
            "trained before evaluation. Run the training script to produce "
            "checkpoints/phase1_decoder/scalar_baseline_best.pt, or provide "
            "an explicit --scalar-baseline-checkpoint path."
        )

    print(f"[eval] Loading scalar baseline checkpoint: {scalar_ckpt_path}")
    scalar_obj = torch.load(scalar_ckpt_path, map_location=device,
                            weights_only=False)

    # Verify the scalar checkpoint has a trained decoder state
    if "decoder_state" not in scalar_obj:
        raise RuntimeError(
            f"Scalar baseline checkpoint {scalar_ckpt_path} is missing "
            "'decoder_state'. Cannot evaluate an untrained baseline."
        )

    from train.train_phase1_decoder import ScalarBaselineDecoder
    scalar_decoder_full = ScalarBaselineDecoder().to(device)
    scalar_decoder_full.load_state_dict(scalar_obj["decoder_state"],
                                        strict=True)
    scalar_decoder_full.eval()
    print("[eval] Scalar baseline loaded (strict)")

    # ================================================================
    # 4. Compute metrics
    # ================================================================
    print("\n[eval] Computing JEPA latent decoder metrics ...")
    jepa_metrics = _compute_detailed_metrics(jepa_wrapper, val_loader, device,
                                             decoder_type="jepa",
                                             loss_config=loss_config)

    print("[eval] Computing scalar baseline metrics ...")
    scalar_metrics = _compute_detailed_metrics(scalar_decoder_full,
                                               val_loader, device,
                                               decoder_type="scalar",
                                               loss_config=loss_config)

    # ================================================================
    # 5. Validate all required metrics are finite
    # ================================================================
    required_keys = [
        "occupancy_iou", "occupancy_f1", "r_atom_mae", "h_atom_mae",
        "combined_occupied_mae", "normalized_r_mae", "normalized_h_mae",
        "lattice_mae", "overall_mae",
    ]
    missing_metrics = []
    for k in required_keys:
        if k not in jepa_metrics or not torch.isfinite(
                torch.tensor(jepa_metrics[k])):
            missing_metrics.append(f"jepa_{k}")
        if k not in scalar_metrics or not torch.isfinite(
                torch.tensor(scalar_metrics[k])):
            missing_metrics.append(f"scalar_{k}")
    if missing_metrics:
        raise RuntimeError(
            f"Required metrics are missing or non-finite: {missing_metrics}. "
            "Cannot produce a Phase-1 verdict."
        )

    # ================================================================
    # 6. Print side-by-side table
    # ================================================================
    print()
    print(f"{'metric':<28} {'scalar baseline':>16} {'JEPA latent':>16}")
    print("-" * 64)
    for key in ["occupancy_iou", "occupancy_f1", "occupancy_precision",
                "occupancy_recall",
                "r_atom_mae", "h_atom_mae",
                "combined_occupied_mae",
                "normalized_r_mae", "normalized_h_mae",
                "lattice_mae", "overall_mae"]:
        s = scalar_metrics.get(key, float("nan"))
        j = jepa_metrics.get(key, float("nan"))
        print(f"{key:<28} {s:>16.4f} {j:>16.4f}")
    print("-" * 64)
    print(f"{'n_samples':<28} {scalar_metrics.get('n_samples', 0):>16} "
          f"{jepa_metrics.get('n_samples', 0):>16}")
    print(f"{'scale_r':<28} {loss_config['scale_r']:>16.6f}")
    print(f"{'scale_h':<28} {loss_config['scale_h']:>16.6f}")
    print()

    # ================================================================
    # 7. Phase-1 gate: IoU + combined occupied MAE
    # ================================================================
    j_iou = jepa_metrics["occupancy_iou"]
    s_iou = scalar_metrics["occupancy_iou"]
    j_comb = jepa_metrics["combined_occupied_mae"]
    s_comb = scalar_metrics["combined_occupied_mae"]

    iou_pass = j_iou > s_iou
    mae_pass = j_comb < s_comb

    if iou_pass and mae_pass:
        verdict = (
            "PASS — JEPA latent decoder beats scalar baseline on both gates "
            f"(IoU {j_iou:.4f} > {s_iou:.4f}, "
            f"combined MAE {j_comb:.4f} < {s_comb:.4f}). "
            "Proceed to Phase 2."
        )
    elif iou_pass:
        verdict = (
            "STOP — JEPA wins occupancy IoU "
            f"({j_iou:.4f} > {s_iou:.4f}) but loses combined occupied MAE "
            f"({j_comb:.4f} >= {s_comb:.4f}). "
            "Do not proceed to Phase 2."
        )
    elif mae_pass:
        verdict = (
            "STOP — JEPA wins combined occupied MAE "
            f"({j_comb:.4f} < {s_comb:.4f}) but loses occupancy IoU "
            f"({j_iou:.4f} <= {s_iou:.4f}). "
            "Do not proceed to Phase 2."
        )
    else:
        verdict = (
            "STOP — Scalar baseline matches or beats JEPA latent decoder on "
            f"both gates (IoU {j_iou:.4f} <= {s_iou:.4f}, "
            f"combined MAE {j_comb:.4f} >= {s_comb:.4f}). "
            "Do not proceed to Phase 2."
        )
    print(f"[verdict] {verdict}")

    # ================================================================
    # 8. Save report
    # ================================================================
    output = args.output or os.path.join(OUT_DIR, "eval_report.json")
    report = {
        "jepa_latent": jepa_metrics,
        "scalar_baseline": scalar_metrics,
        "loss_config": loss_config,
        "verdict": verdict,
        "gate": {
            "iou_pass": iou_pass,
            "mae_pass": mae_pass,
            "jepa_iou": j_iou,
            "scalar_iou": s_iou,
            "jepa_combined_mae": j_comb,
            "scalar_combined_mae": s_comb,
        },
        "checkpoint": args.checkpoint,
        "base_jepa_checkpoint": args.base_jepa_checkpoint,
        "scalar_baseline_checkpoint": scalar_ckpt_path,
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n[eval] Report saved to {output}")


if __name__ == "__main__":
    main()
