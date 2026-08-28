"""Phase 3 — Unified JEPA training script (architecture_v5.md §3-§5,
unified_jepa/Phase 3 MD).

Run (local smoke test, no real data required):
    python scripts/train/train_unified.py --config configs/unified.yaml --no-train

Run (with real MetaDiT data):
    python scripts/train/train_unified.py --config configs/unified.yaml
    python scripts/train/train_unified.py --config configs/unified.yaml --resume checkpoints/unified/latest.pt

Staged training per Phase 3 MD §5:
  A. forward-only smoke (--no-train)
  B. tiny training with physics loss disabled (lambda_phys = 0)
  C. tiny training with small physics loss (Phase 4)
  D. controlled run
  E. scale only after gates pass

Per AGENTS.md Compute Environment section: real training runs happen on
cloud GPU (Kaggle/Colab) per CLOUD_TRAINING.md, not on the local 4GB-VRAM
machine. This script supports both local smoke tests and cloud runs.
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
from torch.utils.data import DataLoader

from data.dataset import MetaDiTDataset, collate_batch
from data.factorize import factorize_geometry, assemble_geometry
from data.mask import BlockMasker
from assembly import build_unified_model, load_into_model, set_spectrum_path
from predictor.guidance import goal_dropout
from physics.physics_loop import load_surrogate, physics_loss
from losses.unified_losses import UnifiedJEPALoss
from runtime.reproducibility import set_seed, collect_rng_state, restore_rng_state
from runtime.device import resolve_device
from train.engine import save_checkpoint, load_checkpoint, collect_ema_state


def _ensure_spectrum_weights(path, device):
    """Create a dummy spectrum encoder checkpoint if the real one is absent.

    For local smoke testing only — the real checkpoint is staged on cloud GPU
    per CLOUD_TRAINING.md. The dummy uses the released VanillaSpectrumEncoder's
    own random init (frozen in practice during training).
    """
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        from model.spec_encoder import VanillaSpectrumEncoder
    except ImportError:
        return path  # let build_unified_model fail with a clear error
    enc = VanillaSpectrumEncoder()
    torch.save(enc.state_dict(), path)
    print(f"[smoke] Created dummy spectrum encoder checkpoint at {path}")
    return path


# ---------------------------------------------------------------------------
# synthetic data (local smoke test, Phase 3 MD §8 — no large Kaggle run)
# ---------------------------------------------------------------------------

def synthetic_batch(b, device):
    """Generate a synthetic batch of factorized geometry + spectrum.

    Returns:
        occupancy: [B, 1, 64, 64] binary float
        scalars:   [B, 3] (l_lattice, h_atom, r_atom)
        spectrum:  [B, 2, 301]
    """
    torch.manual_seed(int(time.time() * 1000) % 2**31)
    occ = (torch.rand(b, 1, 64, 64) > 0.5).float().to(device)
    occ[:, :, :32, :32] = 1.0  # ensure some occupied region
    scalars = (torch.rand(b, 3) * 10 + 1).to(device)  # [1, 11]
    spectrum = torch.randn(b, 2, 301).to(device)
    return occ, scalars, spectrum


def make_synthetic_dataset(n, device):
    """Pre-generate n batches for reproducibility in smoke tests."""
    batches = []
    for _ in range(n):
        batches.append(synthetic_batch(2, device))
    return batches


# ---------------------------------------------------------------------------
# curriculum sampling (Phase 3 MD §4)
# ---------------------------------------------------------------------------

def sample_mask_ratio(cfg, rng):
    """Sample an occupancy mask ratio from the curriculum distribution."""
    ratios = cfg["curriculum"]["mask_ratios"]
    probs = cfg["curriculum"]["mask_ratio_probs"]
    p = torch.tensor(probs)
    idx = torch.multinomial(p, 1, generator=rng).item()
    return ratios[idx]


def sample_scalar_known(B, cfg, rng):
    """Sample scalar known/unknown flags per the curriculum regime.

    Returns:
        scalar_known: (B, 3) bool
        regime: str
    """
    regimes = cfg["curriculum"]["scalar_regimes"]
    probs = cfg["curriculum"]["scalar_regime_probs"]
    p = torch.tensor(probs)
    idx = torch.multinomial(p, 1, generator=rng).item()
    regime = regimes[idx]

    if regime == "all_known":
        return torch.ones(B, 3, dtype=torch.bool), regime
    elif regime == "all_unknown":
        return torch.zeros(B, 3, dtype=torch.bool), regime
    else:  # "mixed"
        return torch.rand(B, 3, generator=rng) > 0.5, regime


class RegimeLogger:
    """Log scalar regime and mask-ratio frequencies per Phase 3 MD §4."""

    def __init__(self, cfg):
        self.mask_ratios = cfg["curriculum"]["mask_ratios"]
        self.scalar_regimes = cfg["curriculum"]["scalar_regimes"]
        self.mask_counts = {r: 0 for r in self.mask_ratios}
        self.regime_counts = {r: 0 for r in self.scalar_regimes}
        self._total = 0

    def record(self, ratio, regime):
        self.mask_counts[ratio] += 1
        self.regime_counts[regime] += 1
        self._total += 1

    def report(self):
        n = max(1, self._total)
        return {
            "mask_freq": {r: c / n for r, c in self.mask_counts.items()},
            "regime_freq": {r: c / n for r, c in self.regime_counts.items()},
        }


# ---------------------------------------------------------------------------
# per-step EMA-frozen guard (Phase 3 MD §6)
# ---------------------------------------------------------------------------

def _assert_no_ema_gradients(model, step):
    """Per-step guard: EMA targets must receive no gradient (spec §6)."""
    leaked = []
    for name, p in model.ema.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            leaked.append(f"ema.{name}")
    if hasattr(model, "scalar_mlp_ema"):
        for name, p in model.scalar_mlp_ema.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                leaked.append(f"scalar_mlp_ema.{name}")
    released = getattr(model.spectrum_path, "released", None)
    if released is not None:
        for name, p in released.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                leaked.append(f"released.{name}")
    if leaked:
        raise RuntimeError(
            f"Step {step}: EMA/released params received gradient: {leaked}")


# ---------------------------------------------------------------------------
# cosine warmup scheduler (identical to train_milestone_b.py)
# ---------------------------------------------------------------------------

class CosineWarmup:
    def __init__(self, base_lr, warmup_steps, total_steps):
        self.base_lr = base_lr
        self.warmup = warmup_steps
        self.total = total_steps

    def factor(self, step):
        if step < self.warmup:
            return (step + 1) / max(1, self.warmup)
        t = (step - self.warmup) / max(1, self.total - self.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))


def build_scheduler(optimizer, base_lr, warmup_steps, total_steps):
    cos = CosineWarmup(base_lr, warmup_steps, total_steps)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: cos.factor(max(0, int(e))))


# ---------------------------------------------------------------------------
# training step (Phase 3 MD §1-§3)
# ---------------------------------------------------------------------------

def training_step(model, objective, occ, sv, spec, cfg, device, step,
                  masker, rng, regime_logger, surrogate=None):
    """One forward + loss + backward step.

    Args:
        occ: [B, 1, 64, 64] occupancy
        sv:  [B, 3] true scalar values
        spec: [B, 2, 301] spectrum
        surrogate: optional frozen surrogate for half_sensitivity mask placement.

    Returns:
        result dict from objective, mask, scalar_known
    """
    B = occ.shape[0]

    # Sample scalar known/unknown regime
    sk, regime = sample_scalar_known(B, cfg, rng)

    # Sample occupancy mask ratio from curriculum
    ratio = sample_mask_ratio(cfg, rng)

    # Generate block mask (BlockMasker works with 1-channel input —
    # it only reads spatial dimensions; half_sensitivity placement uses
    # the frozen surrogate for sensitivity maps, architecture_v5.md §2)
    M = masker.sample(occ, ratio, surrogate)

    # Forward + loss
    # Phase 4 MD §3.5.1: goal dropout — replace A_goal with null token ~10%
    gd_p = cfg.get("train", {}).get("guidance_dropout", 0.0)
    goal_mode = goal_dropout("real", gd_p, rng)
    result = objective(model, occ, sv, sk, spec, M, goal_mode=goal_mode)
    loss = result["total_loss"]

    regime_logger.record(ratio, regime)

    return result, M, sk


def validate(model, objective, val_batches, cfg, device):
    """Run validation on a list of pre-built (occ, sv, spec) batches."""
    model.eval()
    objective.eval()
    val_mask_ratio = cfg["curriculum"].get("val_mask_ratio", 0.5)
    val_masker = BlockMasker(
        placement="random", grid=16, min_side=3, k_range=(1, 4),
        seed=12345)
    metrics = {"cos_err": [], "scalar_err": [], "L_total": []}
    try:
        with torch.no_grad():
            for occ, sv, spec in val_batches:
                B = occ.shape[0]
                sk = torch.ones(B, 3, dtype=torch.bool, device=device)  # all known for val
                M = val_masker.sample(occ, val_mask_ratio)  # fixed val mask
                result = objective(model, occ, sv, sk, spec, M, goal_mode="real")
                m = result["components"]["L_total"]
                metrics["L_total"].append(float(m))
                # cos error between prediction and target
                cos_err = (1 - torch.nn.functional.cosine_similarity(
                    result["out"]["z_hat"].flatten(0, 1),
                    result["out"]["z_y_raw"].flatten(0, 1),
                    dim=-1).clamp(min=0)).mean()
                metrics["cos_err"].append(float(cos_err))
                se = (result["out"]["scalar_pred"] - sv).abs().mean()
                metrics["scalar_err"].append(float(se))
    finally:
        model.train()
        objective.train()

    out = {k: float(np.mean(v)) for k, v in metrics.items() if v}

    # Phase 4 MD §20.3: guidance gap diagnostic at validation time
    try:
        from diagnostics.guidance_gap import compute_guidance_gap
        occ_v, sv_v, spec_v = val_batches[0]
        B = occ_v.shape[0]
        sk_v = torch.ones(B, 3, dtype=torch.bool, device=device)
        M = val_masker.sample(occ_v, val_mask_ratio)
        gap_info = compute_guidance_gap(
            model, occ_v, sv_v, sk_v, spec_v, M, device=device)
        out["guidance_gap"] = gap_info["guidance_gap"]
        out["normalized_guidance_gap"] = gap_info["normalized_guidance_gap"]
    except Exception as e:
        out["guidance_gap_error"] = str(e)

    return out


# ---------------------------------------------------------------------------
# main training loop
# ---------------------------------------------------------------------------

def train(cfg, resume_path=None, no_train=False, device=None):
    """Main entry point. Returns a summary dict."""
    from train.engine import collect_ema_state

    set_seed(cfg["train"].get("seed", 42))
    device = device or resolve_device(cfg["train"].get("device", "cpu"))
    total_steps = cfg["train"].get("total_steps", 1500)

    # --- model ---
    spec_weights = _ensure_spectrum_weights(
        cfg["weights"]["spectrum"], device)
    model = build_unified_model(cfg, spec_weights, device=device)
    cfg.setdefault("_architecture_id", model.architecture_id)

    # --- objective ---
    loss_cfg = cfg.get("loss", {})
    # Phase 4 MD §4: load frozen surrogate when physics loss is active
    lambda_phys = loss_cfg.get("lambda_phys", 0.0)
    ramp_steps = cfg.get("staging", {}).get("lambda_phys_ramp_steps", 0)
    surrogate = None
    if lambda_phys > 0:
        surrogate_path = cfg.get("weights", {}).get("surrogate")
        if surrogate_path and os.path.exists(surrogate_path):
            surrogate = load_surrogate(surrogate_path, device=device)
            print(f"[phase4] Loaded frozen surrogate from {surrogate_path}")
        elif surrogate_path:
            print(f"[phase4] WARNING: surrogate not found at {surrogate_path}, "
                  f"physics loss will be inactive")
    objective = UnifiedJEPALoss(
        hidden=cfg["hidden"],
        lambda_inv=loss_cfg.get("lambda_inv", 25.0),
        lambda_var=loss_cfg.get("lambda_var", 25.0),
        lambda_cov=loss_cfg.get("lambda_cov", 1.0),
        lambda_scalar=loss_cfg.get("lambda_scalar", 1.0),
        lambda_phys=lambda_phys,
        gamma=loss_cfg.get("gamma", 1.0),
        eps=loss_cfg.get("eps", 1e-4),
        surrogate=surrogate,
        physics_use_ste=cfg.get("staging", {}).get("physics_use_ste", True),
    ).to(device)

    # --- optimizer + scheduler ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": trainable, "lr": cfg["train"]["lr"]},
         {"params": objective.parameters(), "lr": cfg["train"]["lr"]}],
        weight_decay=cfg["train"].get("wd", 1e-4),
    )
    scheduler = build_scheduler(
        optimizer, cfg["train"]["lr"],
        cfg["train"].get("warmup_steps", 100), total_steps)

    # --- masker ---
    mask_placement = cfg.get("curriculum", {}).get(
        "mask_placement", "random")
    masker = BlockMasker(
        placement=mask_placement, grid=16, min_side=3, k_range=(1, 4),
        seed=cfg["train"].get("seed", 42))
    # If half_sensitivity placement is configured, the frozen surrogate is
    # required for sensitivity maps (architecture_v5.md §2) — load it even
    # when physics loss is inactive.
    if mask_placement == "half_sensitivity" and surrogate is None:
        surrogate_path = cfg.get("weights", {}).get("surrogate")
        if surrogate_path and os.path.exists(surrogate_path):
            surrogate = load_surrogate(surrogate_path, device=device)
            print(f"[phase4] Loaded frozen surrogate for half_sensitivity "
                  f"masking from {surrogate_path}")

    # --- data ---
    use_synthetic = cfg.get("data", {}).get("use_synthetic", False)
    train_cfg = cfg["train"]
    if use_synthetic or not os.path.exists(cfg["data"].get("train_split", "")):
        n_train = max(train_cfg.get("batch_size", 2) * 4, 8)
        train_data = make_synthetic_dataset(n_train, device)
    else:
        ds = MetaDiTDataset(
            cfg["data"]["train_split"],
            max_samples=cfg["data"].get("max_train_samples", 0),
            seed=cfg["train"].get("seed", 42),
        )
        loader = DataLoader(ds, batch_size=train_cfg["batch_size"],
                            shuffle=True, num_workers=0,
                            collate_fn=collate_batch)

    val_batches = []
    if cfg.get("data", {}).get("use_synthetic", True) or \
       not os.path.exists(cfg["data"].get("val_split", "")):
        val_batches = make_synthetic_dataset(
            cfg["train"].get("val_batches", 1), device)
    else:
        vds = MetaDiTDataset(
            cfg["data"]["val_split"],
            max_samples=cfg["train"].get("val_batches", 1) * train_cfg["batch_size"],
            seed=cfg["train"].get("seed", 42) + 1000,
        )
        vloader = DataLoader(vds, batch_size=train_cfg["batch_size"],
                             shuffle=False, num_workers=0,
                             collate_fn=collate_batch)
        for G, S in vloader:
            occ, sv = factorize_geometry(G)
            val_batches.append((occ.to(device), sv.to(device), S.to(device)))

    # --- resume ---
    start_step = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = load_checkpoint(
            resume_path, model, objective, optimizer, scheduler, device,
            strict_objective=True, strict_optimizer=True, masker=masker)
        start_step = ckpt.get("step", 0)
        print(f"Resumed at step {start_step}")

    # --- no-train smoke ---
    if no_train:
        model.train()
        rng = torch.Generator().manual_seed(cfg["train"].get("seed", 42))
        regime_logger = RegimeLogger(cfg)
        if use_synthetic or not isinstance(train_data, DataLoader):
            occ, sv, spec = train_data[0]
        else:
            G, S = next(iter(loader))
            occ, sv = factorize_geometry(G)
            occ, sv = occ.to(device), sv.to(device)
            spec = S.to(device)
        result, M, sk = training_step(
            model, objective, occ, sv, spec, cfg, device, 0, masker, rng,
            regime_logger, surrogate=surrogate)
        loss = result["total_loss"]
        loss.backward()
        _assert_no_ema_gradients(model, 0)
        components = result["components"]
        print(f"[smoke] step=0  loss={float(loss.detach()):.4f}  "
              f"L_inv={components['L_inv']:.4f} "
              f"L_var={components['L_var']:.4f} "
              f"L_cov={components['L_cov']:.4f} "
              f"L_scalar={components['L_scalar']:.4f}")
        assert torch.isfinite(loss), "smoke loss must be finite"
        return {"final_step": 0, "final_loss": float(loss.detach()),
                "components": components, "regime_report": regime_logger.report()}

    # --- training loop ---
    model.train()
    objective.train()
    rng = torch.Generator().manual_seed(cfg["train"].get("seed", 42))
    regime_logger = RegimeLogger(cfg)
    batch_size = train_cfg.get("batch_size", 2)
    grad_accum = train_cfg.get("grad_accum", 1)
    log_every = train_cfg.get("log_every_steps", 10)
    val_every = train_cfg.get("val_every_steps", 50)
    ckpt_every = train_cfg.get("ckpt_every_steps", 100)
    clip_norm = train_cfg.get("clip_grad_norm", 1.0)

    data_iter = iter(train_data) if use_synthetic or not isinstance(train_data, DataLoader) \
        else iter(loader)

    last_loss = None
    for step in range(start_step, total_steps):
        # Phase 4 MD §4.1: ramp lambda_phys from 0 to target over ramp steps
        if ramp_steps > 0:
            ramp_frac = min(1.0, (step + 1) / ramp_steps)
            objective.lambda_phys = lambda_phys * ramp_frac

        micro_losses = []
        for _ in range(grad_accum):
            # Get batch (auto-reset on exhaustion)
            try:
                if use_synthetic or not isinstance(train_data, DataLoader):
                    occ, sv, spec = next(data_iter)
                else:
                    G, S = next(data_iter)
                    occ, sv = factorize_geometry(G)
                    occ, sv = occ.to(device), sv.to(device)
                    spec = S.to(device)
            except StopIteration:
                data_iter = iter(train_data) if use_synthetic or not isinstance(train_data, DataLoader) \
                    else iter(loader)
                if use_synthetic or not isinstance(train_data, DataLoader):
                    occ, sv, spec = next(data_iter)
                else:
                    G, S = next(data_iter)
                    occ, sv = factorize_geometry(G)
                    occ, sv = occ.to(device), sv.to(device)
                    spec = S.to(device)

            result, M, sk = training_step(
                model, objective, occ, sv, spec, cfg, device, step,
                masker, rng, regime_logger, surrogate=surrogate)
            loss = result["total_loss"]
            (loss / grad_accum).backward()
            micro_losses.append(float(loss.detach()))

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] +
            [p for p in objective.parameters() if p.requires_grad],
            clip_norm)

        # Guard: EMA must not receive gradients
        _assert_no_ema_gradients(model, step)

        optimizer.step()
        scheduler.step()
        objective.on_optimizer_step(model, step)

        last_loss = np.mean(micro_losses)

        if step % log_every == 0:
            c = result["components"]
            print(f"step {step:5d}  loss={last_loss:.4f}  "
                  f"L_inv={c['L_inv']:.4f} L_var={c['L_var']:.4f} "
                  f"L_cov={c['L_cov']:.4f} L_scalar={c['L_scalar']:.4f} "
                  f"L_phys={c['L_phys']:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        if step % val_every == 0 and step > 0:
            val_metrics = validate(model, objective, val_batches, cfg, device)
            print(f"  [val] {json.dumps(val_metrics)}")
            model.train()
            objective.train()

        if step % ckpt_every == 0 and step > 0:
            ckpt_path = os.path.join(
                REPO_ROOT, "checkpoints", "unified", "latest.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            ema_state = collect_ema_state(model)
            save_checkpoint(
                ckpt_path, model, objective, optimizer, scheduler, cfg,
                global_step=step, epoch=0, micro_step=0, batch_index=0,
                is_epoch_end=False, metrics={"L_total": last_loss},
                health={}, ema_state=ema_state,
                masker_rng_state=masker.get_rng_state() if hasattr(masker, "get_rng_state") else None,
                device=device, artifact_type="latest")
            print(f"  [ckpt] saved to {ckpt_path}")

    # Final checkpoint
    ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "unified", "final.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    ema_state = collect_ema_state(model)
    save_checkpoint(
        ckpt_path, model, objective, optimizer, scheduler, cfg,
        global_step=total_steps - 1, epoch=0, micro_step=0, batch_index=0,
        is_epoch_end=True, metrics={"L_total": last_loss if last_loss else 0.0},
        health={}, ema_state=ema_state, device=device, artifact_type="final")

    report = {
        "final_step": total_steps - 1,
        "final_loss": last_loss if last_loss else 0.0,
        "regime_report": regime_logger.report(),
    }
    return report


# ---------------------------------------------------------------------------
# eval (Phase 3 MD §8 — forward-only smoke)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_forward(model, occ, sv, spec, mask, cfg, device):
    """Forward-only evaluation: verify shapes and finiteness without backward."""
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=999)
    sk = torch.ones(occ.shape[0], 3, dtype=torch.bool, device=occ.device)
    out = model(occ, sv, sk, spec, mask, goal_mode="real")
    return {
        "z_x_shape": list(out["z_x"].shape),
        "z_hat_shape": list(out["z_hat"].shape),
        "scalar_pred_shape": list(out["scalar_pred"].shape),
        "z_y_raw_shape": list(out["z_y_raw"].shape),
        "z_x_finite": bool(torch.isfinite(out["z_x"]).all()),
        "z_hat_finite": bool(torch.isfinite(out["z_hat"]).all()),
        "z_y_raw_finite": bool(torch.isfinite(out["z_y_raw"]).all()),
        "scalar_pred_finite": bool(torch.isfinite(out["scalar_pred"]).all()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified JEPA training (Phase 3)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--no-train", action="store_true",
                        help="Forward-only smoke test (Phase 3 MD §5 stage A)")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g. 'cpu' or 'cuda')")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or cfg["train"].get("device", "cpu")

    report = train(cfg, resume_path=args.resume, no_train=args.no_train,
                   device=device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
