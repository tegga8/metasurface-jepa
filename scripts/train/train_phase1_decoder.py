"""Phase 1 decoder training: frozen EMA latent → pixel-space geometry.

Trains two systems with the same decoder/loss/splits:
  1. GeometryDecoder fed by frozen EMA latent z_y_raw
  2. Scalar-only baseline (l_lattice, h_atom, r_atom → Linear → same decoder backbone)

Runs on local machine for smoke test; real training on Kaggle/Colab.

Usage:
    python scripts/train/train_phase1_decoder.py \
        --checkpoint checkpoints/milestone_b/minimal_jepa_vicreg_latest.pt \
        --config configs/milestone_b.yaml \
        --device cpu --max-steps 2

    python scripts/train/train_phase1_decoder.py \
        --checkpoint checkpoints/milestone_b/minimal_jepa_vicreg_latest.pt \
        --config configs/milestone_b.yaml \
        --device cuda:0 --epochs 10
"""

import argparse
import json
import math
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
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


# ---------------------------------------------------------------------------
# Scalar-only baseline
# ---------------------------------------------------------------------------

class ScalarBaselineDecoder(nn.Module):
    """Baseline: linear map from 3 physical scalars → 384×16×16 feature,
    then the SAME GeometryDecoder backbone.

    Inputs: [l_lattice, h_atom, r_atom] — the three scalars that define
    the structure entirely. No geometry pixels, no occupancy mask, no spectrum.
    """

    def __init__(self, hidden_dim=384, base_dim=192, num_channels=3,
                 occupancy_head=True):
        super().__init__()
        self.proj = nn.Linear(3, hidden_dim)
        self.decoder = GeometryDecoder(
            hidden_dim=hidden_dim, base_dim=base_dim,
            num_channels=num_channels, occupancy_head=occupancy_head,
        )

    def forward(self, scalars):
        """Decode scalar parameters to geometry.

        Args:
            scalars: (B, 3) — [l_lattice, h_atom, r_atom]

        Returns:
            geometry:       (B, 3, 64, 64)
            occ_logits:     (B, 1, 64, 64)
        """
        feat = self.proj(scalars)                        # (B, 384)
        feat = feat.unsqueeze(-1).unsqueeze(-1)          # (B, 384, 1, 1)
        feat = feat.expand(-1, -1, 16, 16).contiguous()  # (B, 384, 16, 16)
        B = feat.shape[0]
        z = feat.permute(0, 2, 3, 1).reshape(B, 256, -1)  # (B, 256, 384)
        return self.decoder(z)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _derive_scalars(G):
    """Extract [l_lattice, h_atom, r_atom] from a geometry tensor.

    channel 0: r_atom/5 on occupied pixels
    channel 1: h_atom   on occupied pixels
    channel 2: l_lattice/3 everywhere

    We recover the scalars by taking representative values from the channels.
    l_lattice is uniform everywhere; h_atom is uniform on occupied pixels.
    """
    l_lattice = G[:, 2, 0, 0] * 3.0     # (B,) — uniform per structure
    occ = (G[:, 0] != 0) | (G[:, 1] != 0)  # (B, 64, 64) bool
    # For h_atom and r_atom, take the mean over occupied pixels (or 0 if none)
    B = G.shape[0]
    h_atom = torch.zeros(B, device=G.device)
    r_atom = torch.zeros(B, device=G.device)
    for i in range(B):
        if occ[i].any():
            h_atom[i] = G[i, 1][occ[i]].mean()
            r_atom[i] = G[i, 0][occ[i]].mean() * 5.0
    return torch.stack([l_lattice, h_atom, r_atom], dim=1)  # (B, 3)


@torch.no_grad()
def compute_training_scales(train_ds, epsilon=EPSILON):
    """Compute per-channel absolute-mean scales from the TRAINING split only.

    Returns:
        scale_r: mean absolute value of channel 0 on occupied pixels
        scale_h: mean absolute value of channel 1 on occupied pixels
        stats:   dict with per-channel min/max/mean/std for printing
    """
    n = min(256, len(train_ds))
    Gs = []
    for i in range(n):
        g, _ = train_ds[i]
        Gs.append(g)
    G = torch.stack(Gs, dim=0)  # (N, 3, 64, 64)

    occ = (G[:, 0] != 0) | (G[:, 1] != 0)  # (N, 64, 64) bool

    stats = {}
    # Channel 0: r_atom/5 on occupied pixels
    if occ.any():
        vals_r = G[:, 0][occ]
        scale_r = max(vals_r.abs().mean().item(), epsilon)
        stats["ch0"] = {
            "min": vals_r.min().item(), "max": vals_r.max().item(),
            "mean": vals_r.mean().item(), "std": vals_r.std().item(),
        }
    else:
        scale_r = 1.0
        stats["ch0"] = {"min": 0, "max": 0, "mean": 0, "std": 0}

    # Channel 1: h_atom on occupied pixels
    if occ.any():
        vals_h = G[:, 1][occ]
        scale_h = max(vals_h.abs().mean().item(), epsilon)
        stats["ch1"] = {
            "min": vals_h.min().item(), "max": vals_h.max().item(),
            "mean": vals_h.mean().item(), "std": vals_h.std().item(),
        }
    else:
        scale_h = 1.0
        stats["ch1"] = {"min": 0, "max": 0, "mean": 0, "std": 0}

    # Channel 2: l_lattice/3 everywhere (for printing only)
    vals_l = G[:, 2].reshape(-1)
    stats["ch2"] = {
        "min": vals_l.min().item(), "max": vals_l.max().item(),
        "mean": vals_l.mean().item(), "std": vals_l.std().item(),
    }

    return scale_r, scale_h, stats


def _print_channel_stats(stats, split_name):
    """Print per-channel statistics."""
    for ch, name, key in [
        (0, "r_atom/5 occupied", "ch0"),
        (1, "h_atom occupied", "ch1"),
        (2, "l_lattice/3 global", "ch2"),
    ]:
        s = stats[key]
        if ch < 2:
            print(f"  [{split_name}] channel {ch} ({name}): "
                  f"min={s['min']:.4f} max={s['max']:.4f} "
                  f"mean={s['mean']:.4f} std={s['std']:.4f}")
        else:
            print(f"  [{split_name}] channel {ch} ({name}): "
                  f"min={s['min']:.4f} max={s['max']:.4f} "
                  f"mean={s['mean']:.4f} std={s['std']:.4f}")


@torch.no_grad()
def _validate(decoder, val_loader, criterion, device, decoder_type="jepa"):
    """Run one validation pass and return average loss + per-component dict."""
    decoder.eval()
    total_loss = 0.0
    n_batches = 0
    comp_sums = {}
    for G, S in val_loader:
        G = G.to(device)
        if decoder_type == "jepa":
            with torch.no_grad():
                z_y_raw = decoder._ema_ref(G)
            geom_pred, occ_logits = decoder._decoder(z_y_raw)
        else:
            scalars = _derive_scalars(G)
            geom_pred, occ_logits = decoder(scalars)
        L, comps = criterion(geom_pred, occ_logits, G)
        total_loss += L.item()
        for k, v in comps.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + v.item()
        n_batches += 1
    decoder.train()
    avg = total_loss / max(1, n_batches)
    avg_comps = {k: v / max(1, n_batches) for k, v in comp_sums.items()}
    return avg, avg_comps


def _save_phase1_checkpoint(path, decoder, optimizer, scheduler, step, epoch,
                            base_jepa_path, best_metric, cfg,
                            loss_config):
    """Save a lightweight Phase-1 checkpoint (no 515-MB base JEPA weights).

    The checkpoint stores the full loss configuration as the single source
    of truth: scale_r, scale_h, lambda_occ, lambda_value, lambda_lattice,
    lambda_r, lambda_h. Evaluation MUST read these from the checkpoint.
    """
    state = {
        "phase": 1,
        "step": step,
        "epoch": epoch,
        "decoder_state": decoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "cfg": cfg,
        "base_jepa_checkpoint": base_jepa_path,
        "best_metric": best_metric,
        "loss_config": loss_config,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def _load_phase1_checkpoint(path, decoder, optimizer=None, scheduler=None,
                            device="cpu"):
    """Load a Phase-1 checkpoint."""
    obj = torch.load(path, map_location=device, weights_only=False)
    decoder.load_state_dict(obj["decoder_state"])
    if optimizer is not None and obj.get("optimizer_state") is not None:
        optimizer.load_state_dict(obj["optimizer_state"])
    if scheduler is not None and obj.get("scheduler_state") is not None:
        scheduler.load_state_dict(obj["scheduler_state"])
    return obj


# ---------------------------------------------------------------------------
# evaluation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_metrics(decoder, val_loader, device, decoder_type="jepa",
                     loss_config=None):
    """Compute occupancy IoU, F1, and per-channel MAE with normalized errors."""
    decoder.eval()
    total_iou = 0.0
    total_f1 = 0.0
    total_mae_r = 0.0
    total_mae_h = 0.0
    total_mae_lattice = 0.0
    total_mae_overall = 0.0
    n_batches = 0
    for G, S in val_loader:
        G = G.to(device)
        if decoder_type == "jepa":
            with torch.no_grad():
                z_y_raw = decoder._ema_ref(G)
            geom_pred, occ_logits = decoder._decoder(z_y_raw)
        else:
            scalars = _derive_scalars(G)
            geom_pred, occ_logits = decoder(scalars)

        occ_prob = torch.sigmoid(occ_logits)
        occ_pred = (occ_prob > 0.5).float()
        occ_target = GeometryReconstructionLoss.occupancy_target(G)

        # IoU
        inter = (occ_pred * occ_target).sum()
        union = ((occ_pred + occ_target) > 0).float().sum()
        iou = inter / max(union.item(), 1e-8)
        total_iou += iou

        # F1
        tp = inter
        fp = (occ_pred * (1 - occ_target)).sum()
        fn = ((1 - occ_pred) * occ_target).sum()
        precision = tp / max((tp + fp).item(), 1e-8)
        recall = tp / max((tp + fn).item(), 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        total_f1 += f1

        # Per-channel MAE on occupied pixels
        mask = occ_target.squeeze(1).bool()  # (B, 64, 64)
        if mask.any():
            mae_r = (geom_pred[:, 0][mask] - G[:, 0][mask]).abs().mean()
            mae_h = (geom_pred[:, 1][mask] - G[:, 1][mask]).abs().mean()
        else:
            mae_r = geom_pred.new_zeros(())
            mae_h = geom_pred.new_zeros(())
        mae_lattice = (geom_pred[:, 2] - G[:, 2]).abs().mean()
        mae_overall = (geom_pred - G).abs().mean()

        total_mae_r += mae_r.item()
        total_mae_h += mae_h.item()
        total_mae_lattice += mae_lattice.item()
        total_mae_overall += mae_overall.item()
        n_batches += 1

    decoder.train()
    n = max(1, n_batches)
    r_atom_mae = total_mae_r / n
    h_atom_mae = total_mae_h / n

    # Normalized combined occupied MAE using stored training scales
    scale_r = loss_config["scale_r"] if loss_config else 1.0
    scale_h = loss_config["scale_h"] if loss_config else 1.0
    norm_r = r_atom_mae / scale_r
    norm_h = h_atom_mae / scale_h
    combined_occupied_mae = 0.5 * norm_r + 0.5 * norm_h

    return {
        "occupancy_iou": total_iou / n,
        "occupancy_f1": total_f1 / n,
        "r_atom_mae": r_atom_mae,
        "h_atom_mae": h_atom_mae,
        "combined_occupied_mae": combined_occupied_mae,
        "normalized_r_mae": norm_r,
        "normalized_h_mae": norm_h,
        "lattice_mae": total_mae_lattice / n,
        "overall_mae": total_mae_overall / n,
    }


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------

def _run_smoke_test(jepa_model, decoder, scalar_decoder, cfg, device,
                    base_ckpt, loss_config):
    """Run a tiny 2-step smoke test verifying all interfaces."""
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    # Checkpoint load
    print("[smoke] checkpoint load ... ", end="")
    print("OK")

    # EMA freeze
    ema = jepa_model.ema
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    frozen = all(not p.requires_grad for p in ema.parameters())
    print(f"[smoke] EMA frozen ... {'OK' if frozen else 'FAIL'}")
    assert frozen, "EMA parameters are not frozen"

    # Forward pass
    B = 2
    G = torch.randn(B, 3, 64, 64, device=device)
    print("[smoke] forward pass ... ", end="")
    with torch.no_grad():
        z_y_raw = ema(G)
    assert z_y_raw.shape == (B, 256, 384), f"z_y_raw shape: {z_y_raw.shape}"
    geom_pred, occ_logits = decoder(z_y_raw)
    assert geom_pred.shape == (B, 3, 64, 64), f"geom shape: {geom_pred.shape}"
    assert occ_logits.shape == (B, 1, 64, 64), f"occ shape: {occ_logits.shape}"
    print("OK")

    # Scalar baseline forward
    print("[smoke] scalar baseline forward ... ", end="")
    scalars = _derive_scalars(G)
    geom_s, occ_s = scalar_decoder(scalars)
    assert geom_s.shape == (B, 3, 64, 64), f"scalar geom shape: {geom_s.shape}"
    assert occ_s.shape == (B, 1, 64, 64), f"scalar occ shape: {occ_s.shape}"
    print("OK")

    # Loss forward with stored scales
    criterion = GeometryReconstructionLoss(**loss_config)
    L, comps = criterion(geom_pred, occ_logits, G)
    assert torch.isfinite(L), f"Non-finite loss: {L.item()}"
    print(f"[smoke] loss forward: {L.item():.4f} ... OK")

    # Stored loss scales check
    print("[smoke] stored loss scales ... ", end="")
    assert loss_config["scale_r"] > 0, "scale_r must be > 0"
    assert loss_config["scale_h"] > 0, "scale_h must be > 0"
    print(f"scale_r={loss_config['scale_r']:.4f} "
          f"scale_h={loss_config['scale_h']:.4f} ... OK")

    # Backward
    print("[smoke] backward ... ", end="")
    L.backward()
    print("OK")

    # Gradient checks
    dec_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in decoder.parameters())
    ema_has_grad = any(p.grad is not None for p in ema.parameters())
    print(f"[smoke] decoder gradients present: {dec_has_grad} ... "
          f"{'OK' if dec_has_grad else 'FAIL'}")
    print(f"[smoke] EMA gradients absent: {not ema_has_grad} ... "
          f"{'OK' if not ema_has_grad else 'FAIL'}")
    assert dec_has_grad, "Decoder has no gradients after backward"
    assert not ema_has_grad, "EMA received gradients during backward"

    # Optimizer step
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print("[smoke] optimizer step ... OK")

    # Checkpoint save/reload round-trip
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "smoke_ckpt.pt")
        _save_phase1_checkpoint(ckpt_path, decoder, optimizer, None,
                                step=0, epoch=0, base_jepa_path=base_ckpt,
                                best_metric=float("inf"), cfg=cfg,
                                loss_config=loss_config)
        decoder2 = GeometryDecoder().to(device)
        opt2 = torch.optim.AdamW(decoder2.parameters(), lr=1e-3)
        obj = _load_phase1_checkpoint(ckpt_path, decoder2, optimizer=opt2,
                                      device=device)
        assert obj["step"] == 0
        assert obj["base_jepa_checkpoint"] == base_ckpt
        assert "loss_config" in obj, "Checkpoint missing loss_config"
        assert obj["loss_config"]["scale_r"] == loss_config["scale_r"]
        assert obj["loss_config"]["scale_h"] == loss_config["scale_h"]
        # Deterministic output on fixed input
        decoder.eval()
        decoder2.eval()
        with torch.no_grad():
            g1, _ = decoder(z_y_raw)
            g2, _ = decoder2(z_y_raw)
        assert torch.allclose(g1, g2, atol=1e-5), \
            "Checkpoint round-trip not deterministic"
        print("[smoke] checkpoint round-trip ... OK")

    # Strict checkpoint loading check
    print("[smoke] strict checkpoint loading ... ", end="")
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "strict_ckpt.pt")
        _save_phase1_checkpoint(ckpt_path, decoder, optimizer, None,
                                step=0, epoch=0, base_jepa_path=base_ckpt,
                                best_metric=float("inf"), cfg=cfg,
                                loss_config=loss_config)
        ckpt_obj = torch.load(ckpt_path, map_location="cpu",
                              weights_only=False)
        model_sd = ckpt_obj["decoder_state"]
        decoder3 = GeometryDecoder().to(device)
        missing, unexpected = decoder3.load_state_dict(model_sd, strict=True)
        assert not missing and not unexpected, \
            f"Strict load failed: missing={missing}, unexpected={unexpected}"
        print("OK")

    print("=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 decoder training")
    p.add_argument("--checkpoint", required=True,
                   help="Path to Milestone-B checkpoint")
    p.add_argument("--config", required=True,
                   help="Path to milestone_b.yaml config")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None,
                   help="Override config epochs")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Hard step cap for smoke/debug")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny local crash-test run")
    p.add_argument("--eval-only", action="store_true",
                   help="Eval from a checkpoint and exit")
    p.add_argument("--eval-checkpoint", default=None,
                   help="Phase-1 checkpoint to evaluate")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--val-batch-size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(args.device or cfg["train"].get("device", "auto"))
    seed = args.seed
    set_seed(seed)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- data ----
    train_ds = MetaDiTDataset(
        os.path.join(REPO_ROOT, cfg["data"]["train_split"]),
        max_samples=cfg["data"].get("max_train_samples", 0),
        seed=seed,
    )
    val_ds = MetaDiTDataset(os.path.join(REPO_ROOT, cfg["data"]["val_split"]))

    batch_size = args.batch_size or cfg["train"]["batch_size"]
    if args.smoke:
        batch_size = min(batch_size, 2)
        cfg["data"]["max_train_samples"] = 8
        train_ds = MetaDiTDataset(
            os.path.join(REPO_ROOT, cfg["data"]["train_split"]),
            max_samples=cfg["data"]["max_train_samples"], seed=seed,
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        drop_last=True, collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
        collate_fn=collate_batch,
    )

    # ---- channel scales from training split ONLY ----
    print("[phase1] Computing training-set channel scales ...")
    scale_r, scale_h, stats = compute_training_scales(train_ds, epsilon=EPSILON)
    _print_channel_stats(stats, "train")
    print(f"[phase1] scale_r={scale_r:.6f}  scale_h={scale_h:.6f}")

    # Loss config: single source of truth
    loss_config = {
        "lambda_occ": 1.0,
        "lambda_value": 1.0,
        "lambda_lattice": 0.25,
        "lambda_r": 1.0,
        "lambda_h": 1.0,
        "scale_r": scale_r,
        "scale_h": scale_h,
    }

    # ---- load Milestone-B checkpoint and build model ----
    base_ckpt_path = os.path.abspath(args.checkpoint)
    print(f"[phase1] Loading base JEPA checkpoint: {base_ckpt_path}")
    base_obj = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    # The checkpoint may be an engine-format checkpoint with "model" key
    if "model" in base_obj:
        model_sd = base_obj["model"]
    elif "state_dict" in base_obj:
        model_sd = base_obj["state_dict"]
    else:
        model_sd = base_obj

    model_cfg = cfg["model"]
    jepa_model = build_model(
        model_cfg,
        os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
        device=device,
        init_from_metadit=cfg["model"].get("init_from_metadit", True),
        metadit_weights=os.path.join(REPO_ROOT, cfg["weights"]["metadit"]),
    )
    load_into_model(jepa_model, model_sd, device, strict=False)

    # Restore EMA target weights from checkpoint
    if "ema_state" in base_obj and "target" in base_obj["ema_state"]:
        jepa_model.ema.target.load_state_dict(base_obj["ema_state"]["target"])
        if "total_steps" in base_obj["ema_state"]:
            jepa_model.ema.set_total_steps(base_obj["ema_state"]["total_steps"])

    # ---- freeze EMA ----
    ema = jepa_model.ema
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    assert all(not p.requires_grad for p in ema.parameters()), \
        "EMA parameters are not frozen after explicit freeze"

    # ---- instantiate decoders ----
    decoder = GeometryDecoder().to(device)
    scalar_decoder = ScalarBaselineDecoder().to(device)

    n_dec_params = sum(p.numel() for p in decoder.parameters())
    n_scalar_params = sum(p.numel() for p in scalar_decoder.parameters())
    print(f"[phase1] JEPA decoder params: {n_dec_params:,}")
    print(f"[phase1] Scalar baseline params: {n_scalar_params:,}")

    # ---- smoke test ----
    if args.smoke:
        _run_smoke_test(jepa_model, decoder, scalar_decoder, cfg, device,
                        base_ckpt_path, loss_config)
        return

    # ---- eval-only ----
    if args.eval_only:
        eval_ckpt = args.eval_checkpoint or os.path.join(OUT_DIR, "best.pt")
        if os.path.exists(eval_ckpt):
            obj = _load_phase1_checkpoint(eval_ckpt, decoder, device=device)
            print(f"[phase1] Loaded eval checkpoint: {eval_ckpt}")
        jepa_metrics = _compute_metrics(
            _JepaDecoderWrapper(jepa_model, decoder), val_loader, device,
            decoder_type="jepa", loss_config=loss_config,
        )
        scalar_metrics = _compute_metrics(scalar_decoder, val_loader, device,
                                          decoder_type="scalar",
                                          loss_config=loss_config)
        _print_comparison(jepa_metrics, scalar_metrics)
        return

    # ---- training ----
    epochs = args.epochs or cfg["train"]["epochs"]
    total_steps = len(train_loader) * epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    print(f"[phase1] Training: {epochs} epochs, ~{total_steps} steps, "
          f"batch_size={batch_size}, device={device}")

    # Wrappers for unified eval interface
    jepa_wrapper = _JepaDecoderWrapper(jepa_model, decoder)

    # JEPA decoder optimizer
    optimizer_jepa = torch.optim.AdamW(decoder.parameters(), lr=args.lr,
                                       weight_decay=0.05)
    scheduler_jepa = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_jepa, T_max=total_steps, eta_min=args.lr * 0.01,
    )

    # Scalar baseline optimizer — identical protocol
    optimizer_scalar = torch.optim.AdamW(scalar_decoder.parameters(),
                                         lr=args.lr, weight_decay=0.05)
    scheduler_scalar = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_scalar, T_max=total_steps, eta_min=args.lr * 0.01,
    )

    criterion = GeometryReconstructionLoss(**loss_config)

    # ---- train loop ----
    best_metric_jepa = float("inf")
    best_metric_scalar = float("inf")
    step = 0
    t_start = time.time()

    for epoch in range(epochs):
        decoder.train()
        scalar_decoder.train()
        for bi, (G, S) in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps:
                break

            G = G.to(device)

            # --- JEPA decoder ---
            decoder.zero_grad(set_to_none=True)
            with torch.no_grad():
                z_y_raw = ema(G)
            geom_pred, occ_logits = decoder(z_y_raw)
            L_jepa, comps_jepa = criterion(geom_pred, occ_logits, G)
            L_jepa.backward()

            # EMA gradient guard
            ema_leaked = [n for n, p in ema.named_parameters()
                          if p.grad is not None]
            assert not ema_leaked, f"EMA received gradients: {ema_leaked}"

            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer_jepa.step()
            scheduler_jepa.step()

            # --- Scalar baseline (identical protocol) ---
            scalar_decoder.zero_grad(set_to_none=True)
            scalars = _derive_scalars(G)
            geom_s, occ_s = scalar_decoder(scalars)
            L_scalar, comps_scalar = criterion(geom_s, occ_s, G)
            L_scalar.backward()
            torch.nn.utils.clip_grad_norm_(scalar_decoder.parameters(), 1.0)
            optimizer_scalar.step()
            scheduler_scalar.step()

            if step % cfg["train"].get("log_every_steps", 25) == 0:
                lr_now = optimizer_jepa.param_groups[0]["lr"]
                elapsed = time.time() - t_start
                s_per_step = elapsed / max(1, step + 1)
                print(f"  [step {step}] L_jepa={L_jepa.item():.4f} "
                      f"L_scalar={L_scalar.item():.4f} lr={lr_now:.2e} "
                      f"({s_per_step:.2f} s/step)")

            # Validation
            val_every = cfg["train"].get("val_every_steps", 200)
            if step % val_every == 0 and step > 0:
                val_jepa, _ = _validate(
                    _JepaDecoderWrapper(jepa_model, decoder), val_loader,
                    criterion, device, decoder_type="jepa",
                )
                val_scalar, _ = _validate(scalar_decoder, val_loader,
                                          criterion, device,
                                          decoder_type="scalar")
                print(f"  [val step {step}] val_jepa={val_jepa:.4f} "
                      f"val_scalar={val_scalar:.4f}")
                if val_jepa < best_metric_jepa:
                    best_metric_jepa = val_jepa
                    _save_phase1_checkpoint(
                        os.path.join(OUT_DIR, "best.pt"),
                        decoder, optimizer_jepa, scheduler_jepa,
                        step, epoch, base_ckpt_path, best_metric_jepa, cfg,
                        loss_config,
                    )
                if val_scalar < best_metric_scalar:
                    best_metric_scalar = val_scalar
                    _save_phase1_checkpoint(
                        os.path.join(OUT_DIR, "scalar_baseline_best.pt"),
                        scalar_decoder, optimizer_scalar, scheduler_scalar,
                        step, epoch, base_ckpt_path, best_metric_scalar, cfg,
                        loss_config,
                    )

            step += 1
            if args.max_steps and step >= args.max_steps:
                break

        if args.max_steps and step >= args.max_steps:
            break

    # ---- save latest ----
    _save_phase1_checkpoint(
        os.path.join(OUT_DIR, "latest.pt"),
        decoder, optimizer_jepa, scheduler_jepa,
        step, epoch, base_ckpt_path, best_metric_jepa, cfg,
        loss_config,
    )
    _save_phase1_checkpoint(
        os.path.join(OUT_DIR, "scalar_baseline_latest.pt"),
        scalar_decoder, optimizer_scalar, scheduler_scalar,
        step, epoch, base_ckpt_path, best_metric_scalar, cfg,
        loss_config,
    )

    # ---- final eval ----
    print("\n[phase1] Final evaluation:")
    if os.path.exists(os.path.join(OUT_DIR, "best.pt")):
        _load_phase1_checkpoint(os.path.join(OUT_DIR, "best.pt"), decoder,
                                device=device)
    jepa_metrics = _compute_metrics(jepa_wrapper, val_loader, device,
                                    decoder_type="jepa",
                                    loss_config=loss_config)
    scalar_metrics = _compute_metrics(scalar_decoder, val_loader, device,
                                      decoder_type="scalar",
                                      loss_config=loss_config)
    _print_comparison(jepa_metrics, scalar_metrics)

    # Save metrics
    results = {
        "jepa_latent": jepa_metrics,
        "scalar_baseline": scalar_metrics,
        "config": cfg,
        "loss_config": loss_config,
        "base_jepa_checkpoint": base_ckpt_path,
    }
    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"[phase1] Metrics saved to {metrics_path}")

    # ---- qualitative reconstructions ----
    _save_qualitative(jepa_wrapper, scalar_decoder, val_loader, device,
                      OUT_DIR, n_examples=8)


class _JepaDecoderWrapper(nn.Module):
    """Unified eval interface: wraps (frozen JEPA model, trainable decoder)."""

    def __init__(self, jepa_model, decoder):
        super().__init__()
        self._jepa = jepa_model
        self._decoder = decoder
        self._ema_ref = jepa_model.ema

    def forward(self, x):
        """x is geometry; we extract z_y_raw from frozen EMA."""
        with torch.no_grad():
            z_y_raw = self._ema_ref(x)
        return self._decoder(z_y_raw)

    def eval(self):
        self._decoder.eval()
        return self

    def train(self, mode=True):
        self._decoder.train(mode)
        return self

    def parameters(self, recurse=True):
        return self._decoder.parameters(recurse=recurse)


def _print_comparison(jepa_m, scalar_m):
    """Print side-by-side comparison table."""
    print()
    print(f"{'metric':<25} {'scalar baseline':>16} {'JEPA latent':>16}")
    print("-" * 60)
    for key in ["occupancy_iou", "occupancy_f1", "r_atom_mae",
                "h_atom_mae", "combined_occupied_mae",
                "normalized_r_mae", "normalized_h_mae",
                "lattice_mae", "overall_mae"]:
        s = scalar_m.get(key, float("nan"))
        j = jepa_m.get(key, float("nan"))
        print(f"{key:<25} {s:>16.4f} {j:>16.4f}")
    print()


@torch.no_grad()
def _save_qualitative(jepa_wrapper, scalar_decoder, val_loader, device,
                      out_dir, n_examples=8):
    """Save a small fixed set of qualitative examples."""
    qualitative_dir = os.path.join(out_dir, "qualitative")
    os.makedirs(qualitative_dir, exist_ok=True)

    jepa_wrapper.eval()
    scalar_decoder.eval()

    saved = 0
    for G, S in val_loader:
        G = G.to(device)
        B = G.shape[0]
        for i in range(min(B, n_examples - saved)):
            idx = saved
            # JEPA latent decoder
            geom_j, occ_j = jepa_wrapper(G[i:i+1])
            # Scalar baseline
            scalars = _derive_scalars(G[i:i+1])
            geom_s, occ_s = scalar_decoder(scalars)

            example = {
                "ground_truth": G[i].cpu(),
                "jepa_decoded": geom_j[0].cpu(),
                "scalar_decoded": geom_s[0].cpu(),
                "abs_error_jepa": (geom_j[0] - G[i]).abs().cpu(),
                "abs_error_scalar": (geom_s[0] - G[i]).abs().cpu(),
                "ground_truth_occ": GeometryReconstructionLoss.occupancy_target(
                    G[i:i+1])[0].cpu(),
                "predicted_occ_jepa": torch.sigmoid(occ_j[0]).cpu(),
                "predicted_occ_scalar": torch.sigmoid(occ_s[0]).cpu(),
            }
            torch.save(example, os.path.join(qualitative_dir,
                                              f"example_{idx:03d}.pt"))
            saved += 1
            if saved >= n_examples:
                break
        if saved >= n_examples:
            break
    print(f"[phase1] Saved {saved} qualitative examples to {qualitative_dir}")


if __name__ == "__main__":
    main()
