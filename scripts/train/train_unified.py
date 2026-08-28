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


def _ensure_spectrum_weights(path, device, allow_dummy=False):
    """Resolve the released spectrum encoder checkpoint.

    Args:
        path:        configured spectrum encoder checkpoint path.
        allow_dummy: if True (explicit smoke mode), create a random-init
                     VanillaSpectrumEncoder checkpoint when the released one
                     is absent. Must NEVER be an implicit fallback for real
                     training.

    Returns:
        resolved path.

    Raises:
        RuntimeError: in real mode when the released checkpoint is missing.
    """
    if os.path.exists(path):
        return path
    if allow_dummy:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            from model.spec_encoder import VanillaSpectrumEncoder
        except ImportError:
            raise RuntimeError(
                f"spectrum checkpoint {path} missing and VanillaSpectrumEncoder "
                "unavailable — cannot create smoke dummy") from None
        enc = VanillaSpectrumEncoder()
        torch.save(enc.state_dict(), path)
        print(f"[smoke] Created dummy spectrum encoder checkpoint at {path}")
        return path
    raise RuntimeError(
        f"released spectrum encoder checkpoint not found at {path}. "
        "Real training requires the released weights; pass --use-synthetic-smoke "
        "only for controlled local smoke tests.")


# ---------------------------------------------------------------------------
# synthetic data (local smoke test, Phase 3 MD §8 — no large Kaggle run)
# ---------------------------------------------------------------------------

def synthetic_batch(b, device, seed=None, generator=None):
    """Generate a synthetic batch of factorized geometry + spectrum.

    Fix 6: seeding is REPRODUCIBLE — either an explicit seed or a caller-
    supplied torch.Generator. No wall-clock-derived seeding: two calls with
    the same seed produce identical scalars/spectrum.

    Returns:
        occupancy: [B, 1, 64, 64] binary float
        scalars:   [B, 3] (l_lattice, h_atom, r_atom)
        spectrum:  [B, 2, 301]
    """
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(seed if seed is not None else 0)
    occ = (torch.rand(b, 1, 64, 64, generator=generator) > 0.5).float().to(device)
    occ[:, :, :32, :32] = 1.0  # ensure some occupied region
    scalars = (torch.rand(b, 3, generator=generator) * 10 + 1).to(device)  # [1, 11]
    spectrum = torch.randn(b, 2, 301, generator=generator).to(device)
    return occ, scalars, spectrum


def make_synthetic_dataset(n, device, seed=0):
    """Pre-generate n batches for reproducibility in smoke tests.

    Fix 6: each batch is derived from the SAME seed via an advancing
    generator, so a smoke run with a fixed seed is fully reproducible.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    batches = []
    for _ in range(n):
        batches.append(synthetic_batch(2, device, generator=generator))
    return batches


# ---------------------------------------------------------------------------
# curriculum sampling (Phase 3 MD §4)
# ---------------------------------------------------------------------------

def sample_mask_ratio(cfg, rng):
    """Sample an occupancy mask ratio from the TRAINING curriculum distribution.

    Cleanup item 3: the config now distinguishes train_mask_ratios (which
    EXCLUDE 0.0 — with zero masked tokens the masked-token VICReg/JEPA
    objective is undefined) from eval_mask_ratios (which MAY include 0.0 as
    the genuinely unmasked reference condition). This function samples from
    the train list only; the 0.0-exclusion is explicit in config, not a
    silent runtime filter.

    Falls back to the legacy mask_ratios/mask_ratio_probs keys (with the
    runtime 0.0 filter) if a config predates the split.
    """
    cur = cfg["curriculum"]
    ratios = cur.get("train_mask_ratios", cur.get("mask_ratios"))
    probs = cur.get("train_mask_ratio_probs", cur.get("mask_ratio_probs"))
    # Defensive: even if 0.0 is present in the train list, exclude it — the
    # masked-token objective is undefined at 0.0 (no masked tokens).
    pairs = [(r, p) for r, p in zip(ratios, probs) if r > 0.0]
    rs, ps = zip(*pairs) if pairs else ([1.0], [1.0])
    p = torch.tensor(ps, dtype=torch.float32)
    p = p / p.sum()
    idx = torch.multinomial(p, 1, generator=rng).item()
    return rs[idx]


def _build_scalar_masker_bank(cfg, seed=0):
    """Build one persistent ScalarMasker per configured curriculum regime.

    Fix 3: scalar masking must use PERSISTENT RNG state that evolves across
    batches (a fresh seed=0 ScalarMasker per call would replay the same
    "mixed" pattern every batch). One masker per regime, all created once at
    training start.
    """
    from data.scalar_mask import ScalarMasker
    regime_to_masker = {
        "all_known": "all_known",
        "all_unknown": "all_unknown",
        "mixed": "independent",
    }
    bank = {}
    for regime in cfg["curriculum"]["scalar_regimes"]:
        if regime not in regime_to_masker:
            raise ValueError(
                f"unknown scalar regime {regime!r}; expected one of "
                f"{list(regime_to_masker)}")
        bank[regime] = ScalarMasker(
            regime=regime_to_masker[regime],
            p_independent=0.5, seed=seed)
    return bank


def sample_scalar_known(B, cfg, rng, device=None, masker_bank=None):
    """Sample scalar known/unknown flags via the CANONICAL ScalarMasker.

    Fix 3: masker_bank is the PERSISTENT per-regime ScalarMasker set created
    once at training start (RNG state evolves across batches and is
    checkpointed). If None (isolated test calls), a fresh bank is created —
    tests that need reproducible cross-batch progression must pass a bank.

    The curriculum regime is sampled from config (all_known / all_unknown /
    mixed), and "mixed" maps to ScalarMasker's "independent" regime (each
    scalar known with p=0.5 — the same per-scalar Bernoulli semantics the old
    sampler used).

    ScalarMasker constructs flags on the scalar_values device (CPU generator,
    then .to(device)), so CUDA batches stay device-correct by construction.

    Returns:
        scalar_known: (B, 3) bool on `device`
        regime: str (curriculum regime name for logging)
    """
    regimes = cfg["curriculum"]["scalar_regimes"]
    probs = cfg["curriculum"]["scalar_regime_probs"]
    p = torch.tensor(probs)
    idx = torch.multinomial(p, 1, generator=rng).item()
    regime = regimes[idx]

    if masker_bank is None:
        masker_bank = _build_scalar_masker_bank(cfg, seed=0)
    masker = masker_bank[regime]
    # ScalarMasker.sample returns (masked_values, known_flags) on the device
    # of scalar_values; we only need the flags here.
    sv_dummy = torch.zeros(B, 3, device=device)
    _, known = masker.sample(sv_dummy)
    return known, regime


def collect_scalar_masker_bank_state(masker_bank):
    """Collect per-regime ScalarMasker RNG states for checkpointing (Fix 3)."""
    return {regime: m.get_rng_state() for regime, m in masker_bank.items()}


def restore_scalar_masker_bank_state(masker_bank, state):
    """Restore per-regime ScalarMasker RNG states from a checkpoint (Fix 3)."""
    if not state:
        return
    for regime, m in masker_bank.items():
        if regime in state:
            m.set_rng_state(state[regime])


class RegimeLogger:
    """Log scalar regime and mask-ratio frequencies per Phase 3 MD §4."""

    def __init__(self, cfg):
        cur = cfg["curriculum"]
        # Train ratios (excludes 0.0); falls back to legacy keys.
        self.mask_ratios = cur.get("train_mask_ratios", cur.get("mask_ratios"))
        self.scalar_regimes = cur["scalar_regimes"]
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
                  masker, rng, regime_logger, surrogate=None,
                  scalar_masker_bank=None):
    """One forward + loss + backward step.

    Args:
        occ: [B, 1, 64, 64] occupancy
        sv:  [B, 3] true scalar values
        spec: [B, 2, 301] spectrum
        surrogate: optional frozen surrogate for half_sensitivity mask placement.
        scalar_masker_bank: persistent per-regime ScalarMasker set (Fix 3).

    Returns:
        result dict from objective, mask, scalar_known
    """
    B = occ.shape[0]

    # Fix 2 (spec §4): every tensor handed to the model must be on the model
    # device. Sample scalar known flags on the device, and move the block mask
    # to the device (masker.sample returns CPU tensors by construction).
    sk, regime = sample_scalar_known(
        B, cfg, rng, device=device, masker_bank=scalar_masker_bank)

    # Sample occupancy mask ratio from curriculum
    ratio = sample_mask_ratio(cfg, rng)

    # Generate block mask. For half_sensitivity placement, the frozen
    # surrogate's sensitivity map needs the COMPLETE [B,3,64,64] MetaDiT
    # broadcast geometry — assemble it from the TRUE training sample
    # (true occupancy + true l/h/r). The mask is used ONLY to decide which
    # occupancy tokens are hidden; the unified model still receives the
    # factorized occupancy [B,1,64,64], scalar values + known flags, spectrum
    # (Fix 2: never reintroduce the 3-channel tensor as the internal
    # representation, never use model predictions to determine their own mask).
    if getattr(masker, "placement", "random") == "half_sensitivity":
        from data.factorize import assemble_metadit_geometry
        assert surrogate is not None, (
            "half_sensitivity masking requires the frozen surrogate")
        geo_true = assemble_metadit_geometry(
            occ, sv[:, 0], sv[:, 1], sv[:, 2])
        M = masker.sample(geo_true, ratio, surrogate).to(device)
    else:
        M = masker.sample(occ, ratio, surrogate).to(device)

    # Forward + loss
    # Phase 4 MD §3.5.1: goal dropout — replace A_goal with null token ~10%
    gd_p = cfg.get("train", {}).get("guidance_dropout", 0.0)
    goal_mode = goal_dropout("real", gd_p, rng)
    result = objective(model, occ, sv, sk, spec, M, goal_mode=goal_mode)
    loss = result["total_loss"]

    regime_logger.record(ratio, regime)

    return result, M, sk


def validate(model, objective, val_batches, cfg, device):
    """Run validation on a list of pre-built (occ, sv, spec) batches.

    Unified model signature: model(occupancy, scalar_values, scalar_known,
    spectrum, mask). Reports BOTH raw-latent and projected-latent diagnostics
    on the exact masked-token population used by the training objective, so
    a raw-vs-projected discrepancy (projector/statistics problem) can be
    distinguished from a genuine representation failure.

    Field-name convention (item 7):
      - raw_*      : computed on z_hat / z_y_raw (encoder output space)
      - proj_*     : computed on p_hat / p_y (objective projector space,
                     the space in which L_inv/L_var/L_cov are trained)
      - L_*        : the training objective's loss components
      - L_total    : the FULL training objective (all weighted terms) — NOT a
                     reconstruction/physics metric; do not read it as one.
    """
    model.eval()
    objective.eval()
    val_mask_ratio = cfg["curriculum"].get("val_mask_ratio", 0.5)
    val_masker = BlockMasker(
        placement="random", grid=16, min_side=3, k_range=(1, 4),
        seed=12345)
    metrics = {
        "raw_mse": [], "raw_cos_err": [], "raw_z_hat_norm": [],
        "raw_z_y_norm": [],
        "proj_mse": [], "proj_cos_err": [], "proj_p_hat_norm": [],
        "proj_p_y_norm": [],
        "L_total": [], "L_inv": [], "L_var": [], "L_cov": [],
        "L_scalar": [], "L_phys": [], "L_phys_weighted": [],
        "scalar_err": [],
    }
    try:
        with torch.no_grad():
            for occ, sv, spec in val_batches:
                B = occ.shape[0]
                sk = torch.ones(B, 3, dtype=torch.bool, device=device)  # all known for val
                # Validation masks are deterministic (fixed seed) and
                # explicitly transferred to the model device.
                M = val_masker.sample(occ, val_mask_ratio).to(device)
                assert M.device == occ.device, (
                    "validation mask must be on the model device")
                result = objective(model, occ, sv, sk, spec, M, goal_mode="real")
                out = result["out"]
                mask_bool = out["mask"]
                z_hat, z_y = out["z_hat"], out["z_y_raw"]

                # --- RAW latent space diagnostics (masked tokens only) ---
                z_hat_m = z_hat[mask_bool]
                z_y_m = z_y[mask_bool]
                raw_mse = torch.nn.functional.mse_loss(z_hat_m, z_y_m)
                raw_cos = (1 - torch.nn.functional.cosine_similarity(
                    z_hat_m, z_y_m, dim=-1).clamp(min=0)).mean()
                metrics["raw_mse"].append(float(raw_mse))
                metrics["raw_cos_err"].append(float(raw_cos))
                metrics["raw_z_hat_norm"].append(
                    float(z_hat_m.norm(dim=-1).mean()))
                metrics["raw_z_y_norm"].append(
                    float(z_y_m.norm(dim=-1).mean()))

                # --- PROJECTED latent space diagnostics (same tokens) ---
                # p_hat/p_y are exactly the tensors L_inv uses.
                p_hat_full = result["projector_outputs"]["p_hat"]
                p_y_full = result["projector_outputs"]["p_y"]
                p_hat_m = p_hat_full[mask_bool]
                p_y_m = p_y_full[mask_bool]
                proj_mse = torch.nn.functional.mse_loss(p_hat_m, p_y_m)
                proj_cos = (1 - torch.nn.functional.cosine_similarity(
                    p_hat_m, p_y_m, dim=-1).clamp(min=0)).mean()
                metrics["proj_mse"].append(float(proj_mse))
                metrics["proj_cos_err"].append(float(proj_cos))
                metrics["proj_p_hat_norm"].append(
                    float(p_hat_m.norm(dim=-1).mean()))
                metrics["proj_p_y_norm"].append(
                    float(p_y_m.norm(dim=-1).mean()))

                # --- Loss components (composition is explicit) ---
                c = result["components"]
                for k in ("L_total", "L_inv", "L_var", "L_cov", "L_scalar",
                          "L_phys", "L_phys_weighted"):
                    metrics[k].append(float(c[k]))
                se = (out["scalar_pred"] - sv).abs().mean()
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
        M = val_masker.sample(occ_v, val_mask_ratio).to(device)
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

def train(cfg, resume_path=None, no_train=False, device=None,
          use_synthetic_smoke=False):
    """Main entry point. Returns a summary dict.

    Args:
        use_synthetic_smoke: explicit smoke mode — synthetic data and dummy
            spectrum weights allowed. Real mode (default) requires the real
            dataset and released weights and fails loudly if they are missing.
    """
    from train.engine import collect_ema_state

    set_seed(cfg["train"].get("seed", 42))
    device = device or resolve_device(cfg["train"].get("device", "cpu"))
    total_steps = cfg["train"].get("total_steps", 1500)

    # --- data mode banner (Fix 16) ---
    def _resolved(path):
        return os.path.join(REPO_ROOT, path) if not os.path.isabs(path) else path

    train_split = _resolved(cfg["data"]["train_split"])
    val_split = _resolved(cfg["data"]["val_split"])
    spec_path = _resolved(cfg["weights"]["spectrum"])
    surr_path = _resolved(cfg["weights"]["surrogate"])
    mode = "SMOKE (synthetic)" if use_synthetic_smoke else "REAL"
    print(f"DATA MODE: {mode}")
    print(f"TRAIN SPLIT: {train_split}")
    print(f"VAL SPLIT: {val_split}")
    print(f"SPECTRUM ENCODER: {spec_path}")
    print(f"SURROGATE: {surr_path}")
    print(f"ARCHITECTURE ID: unified_occ_param_spectrum_jepa_v1")

    # --- strict real-data requirement (Fix 5) ---
    if not use_synthetic_smoke:
        missing = [p for p in (train_split, val_split) if not os.path.exists(p)]
        if missing:
            raise RuntimeError(
                f"real dataset split(s) missing: {missing}. Refusing to "
                "silently fall back to synthetic data in real training mode. "
                "Pass --use-synthetic-smoke for an explicit local smoke run.")
    # Synthetic data requires the explicit smoke flag (never an implicit fallback).
    use_synthetic = use_synthetic_smoke

    # --- model (Fix 6: released spectrum weights required in real mode) ---
    spec_weights = _ensure_spectrum_weights(
        spec_path, device, allow_dummy=use_synthetic_smoke)
    model = build_unified_model(cfg, spec_weights, device=device)
    cfg.setdefault("_architecture_id", model.architecture_id)

    # --- objective ---
    loss_cfg = cfg.get("loss", {})
    # Phase 4 MD §4: load frozen surrogate when physics loss is active
    lambda_phys = loss_cfg.get("lambda_phys", 0.0)
    ramp_steps = cfg.get("staging", {}).get("lambda_phys_ramp_steps", 0)
    surrogate = None
    if lambda_phys > 0:
        # Fix 3 (spec §5): a real-mode run that requests physics loss but
        # cannot load the surrogate must FAIL LOUDLY, never silently continue
        # with a zero placeholder physics term. Smoke mode keeps its explicit
        # permissive contract (documented in the --use-synthetic-smoke CLI).
        surrogate_path = _resolved(cfg.get("weights", {}).get("surrogate", ""))
        if surrogate_path and os.path.exists(surrogate_path):
            surrogate = load_surrogate(surrogate_path, device=device)
            print(f"[phase4] Loaded frozen surrogate from {surrogate_path}")
        elif not use_synthetic_smoke:
            raise RuntimeError(
                f"lambda_phys={lambda_phys} > 0 but surrogate checkpoint "
                f"missing at {surrogate_path!r}. Refusing to run a physics "
                "training step with a silent zero physics placeholder in real "
                "mode. Supply the surrogate weights, or set lambda_phys=0, or "
                "pass --use-synthetic-smoke for an explicit local smoke run.")
        else:
            print(f"[phase4] SMOKE: surrogate not found at {surrogate_path}, "
                  f"physics loss inactive (explicit smoke mode only)")
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
    # Objective optimizer group must contain ONLY trainable objective params.
    # UnifiedJEPALoss registers the frozen MetaDiT surrogate as self.surrogate;
    # its params have requires_grad=False and must NOT enter the optimizer
    # (otherwise Phase-C optimizer ownership differs from Phase-B and resume
    # fingerprints mismatch). Filtering by requires_grad keeps the group to the
    # objective projector (9 params) in both phases.
    objective_trainable = [
        p for p in objective.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [{"params": trainable, "lr": cfg["train"]["lr"]},
         {"params": objective_trainable, "lr": cfg["train"]["lr"]}],
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

    # --- scalar masker bank (Fix 3) ---
    # One persistent ScalarMasker per curriculum regime, created ONCE here so
    # RNG state evolves across batches and can be checkpointed/restored.
    scalar_masker_bank = _build_scalar_masker_bank(
        cfg, seed=cfg["train"].get("seed", 42))

    # --- data ---
    # use_synthetic is bound from use_synthetic_smoke above (never an implicit
    # fallback from a missing path — that now raises in real mode).
    train_cfg = cfg["train"]
    if use_synthetic:
        n_train = max(train_cfg.get("batch_size", 2) * 4, 8)
        train_data = make_synthetic_dataset(
            n_train, device, seed=cfg["train"].get("seed", 42))
    else:
        ds = MetaDiTDataset(
            train_split,
            max_samples=cfg["data"].get("max_train_samples", 0),
            seed=cfg["train"].get("seed", 42),
        )
        loader = DataLoader(ds, batch_size=train_cfg["batch_size"],
                            shuffle=True, num_workers=0,
                            collate_fn=collate_batch)
        # train_data is the loader in real mode so the training loop's
        # isinstance(train_data, DataLoader) dispatch works uniformly.
        train_data = loader

    val_batches = []
    if use_synthetic:
        val_batches = make_synthetic_dataset(
            cfg["train"].get("val_batches", 1), device,
            seed=cfg["train"].get("seed", 42) + 1000)
    else:
        vds = MetaDiTDataset(
            val_split,
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
        # Checkpoint `step` is the optimizer step that has ALREADY completed.
        # Resume at step+1 so a checkpoint saved at step 1499 resumes at step
        # 1500 (the next un-run step), not re-running step 1499.
        start_step = ckpt.get("step", -1) + 1
        # Fix 3: restore the persistent scalar-masker RNG state so resumed
        # training continues the same scalar-masking sequence (not restarting
        # from seed).
        restore_scalar_masker_bank_state(
            scalar_masker_bank, ckpt.get("scalar_masker_rng_state", {}))
        print(f"Resumed at step {start_step}")

    # --- no-train smoke ---
    if no_train:
        model.train()
        rng = torch.Generator().manual_seed(cfg["train"].get("seed", 42))
        regime_logger = RegimeLogger(cfg)
        if use_synthetic:
            occ, sv, spec = train_data[0]
        else:
            G, S = next(iter(loader))
            occ, sv = factorize_geometry(G)
            occ, sv = occ.to(device), sv.to(device)
            spec = S.to(device)
        result, M, sk = training_step(
            model, objective, occ, sv, spec, cfg, device, 0, masker, rng,
            regime_logger, surrogate=surrogate,
            scalar_masker_bank=scalar_masker_bank)
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

        # Explicitly reset gradients at the START of each optimizer step
        # (Fix 1, spec §3): gradients must accumulate only across the
        # grad_accum microbatches of THIS optimizer step, never leak from the
        # previous step. Not inside the microbatch loop; not left implicit.
        optimizer.zero_grad(set_to_none=True)

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
                masker, rng, regime_logger, surrogate=surrogate,
                scalar_masker_bank=scalar_masker_bank)
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
            # L_phys is the RAW physics term; L_phys_weighted is
            # lambda_phys * L_phys — the actual contribution to the objective
            # (item 10: report both, never describe Phase C by the raw value
            # alone).
            print(f"step {step:5d}  loss={last_loss:.4f}  "
                  f"L_inv={c['L_inv']:.4f} L_var={c['L_var']:.4f} "
                  f"L_cov={c['L_cov']:.4f} L_scalar={c['L_scalar']:.4f} "
                  f"L_phys={c['L_phys']:.4f} "
                  f"L_phys_w={c['L_phys_weighted']:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

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
                extra={"scalar_masker_rng_state": collect_scalar_masker_bank_state(
                    scalar_masker_bank)},
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
        health={}, ema_state=ema_state,
        masker_rng_state=masker.get_rng_state() if hasattr(masker, "get_rng_state") else None,
        extra={"scalar_masker_rng_state": collect_scalar_masker_bank_state(
            scalar_masker_bank)},
        device=device, artifact_type="final")

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
# real-data preflight (Fix 17)
# ---------------------------------------------------------------------------

def preflight(cfg, device=None):
    """End-to-end real-data preflight: one real sample through the full path.

    real sample → factorize_geometry → mask → scalar masking → unified forward
    → occupancy decode → known-scalar substitution → assemble_metadit_geometry
    → frozen surrogate → finite physics loss → backward.

    Verifies shapes, finiteness, and gradient ownership. Must pass before the
    first real training run.
    """
    device = device or resolve_device(cfg["train"].get("device", "cpu"))
    set_seed(cfg["train"].get("seed", 42))

    train_split = os.path.join(REPO_ROOT, cfg["data"]["train_split"])
    if not os.path.exists(train_split):
        raise RuntimeError(f"preflight requires the real training split: {train_split}")

    spec_weights = _ensure_spectrum_weights(
        os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]), device,
        allow_dummy=False)
    model = build_unified_model(cfg, spec_weights, device=device)
    model.train()

    surrogate_path = os.path.join(REPO_ROOT, cfg["weights"]["surrogate"])
    if not os.path.exists(surrogate_path):
        raise RuntimeError(f"preflight requires the released surrogate: {surrogate_path}")
    surrogate = load_surrogate(surrogate_path, device=device)

    # Preflight objective: physics loss FORCED ON (lambda_phys > 0) so the
    # surrogate gradient path is actually exercised — the config default is
    # lambda_phys=0 (stage B), which would leave the physics path untested.
    objective = UnifiedJEPALoss(
        hidden=cfg["hidden"],
        lambda_inv=cfg.get("loss", {}).get("lambda_inv", 25.0),
        lambda_var=cfg.get("loss", {}).get("lambda_var", 25.0),
        lambda_cov=cfg.get("loss", {}).get("lambda_cov", 1.0),
        lambda_scalar=cfg.get("loss", {}).get("lambda_scalar", 1.0),
        lambda_phys=max(cfg.get("loss", {}).get("lambda_phys", 0.0), 1.0),
        surrogate=surrogate,
        physics_use_ste=cfg.get("staging", {}).get("physics_use_ste", True),
    ).to(device)

    # One real batch.
    ds = MetaDiTDataset(train_split, max_samples=2, seed=0)
    G, S = collate_batch([ds[0], ds[1]])
    occ, sv = factorize_geometry(G)
    occ, sv = occ.to(device), sv.to(device)
    S = S.to(device)
    b = occ.shape[0]

    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    # Fix 1: the mask must be on the ACTIVE device (masker.sample returns a
    # CPU tensor; the model forward requires M.device == occ.device).
    M = masker.sample(occ, ratio=0.5, surrogate=surrogate).to(device)
    assert M.device == occ.device, "preflight: mask must be on the model device"
    sk = torch.zeros(b, 3, dtype=torch.bool, device=device)  # all unknown (hard stratum)
    assert sk.device == occ.device, "preflight: scalar_known must be on the model device"

    result = objective(model, occ, sv, sk, S, M, goal_mode="real")
    loss = result["total_loss"]
    if not torch.isfinite(loss):
        raise RuntimeError(f"preflight: non-finite total loss {loss.item()}")
    loss.backward()

    out = result["out"]
    geometry, _ = model.decode_geometry(
        out["z_hat"], out["scalar_pred"], occ_input=occ, mask=M,
        scalar_known=sk, scalar_values=sv)
    spec_pred = surrogate(geometry).prediction

    # Cleanup item 7: geometry broadcast invariants on the assembled tensor.
    # The invariant check uses the HARD (binary) occupancy assembly — the
    # broadcast invariants (constant occupied values per channel) are defined
    # for the deterministic binary MetaDiT convention. The soft-occupancy
    # path is the differentiable training path and legitimately has
    # continuously-valued channels; it is checked separately via
    # hard_forward=True here.
    from data.factorize import validate_geometry_broadcast
    geometry_hard, _ = model.decode_geometry(
        out["z_hat"], out["scalar_pred"], occ_input=occ, mask=M,
        scalar_known=sk, scalar_values=sv, hard_forward=True)
    invariant_violations = validate_geometry_broadcast(geometry_hard)
    if invariant_violations:
        raise RuntimeError(
            f"preflight: assembled geometry violates broadcast invariants: "
            f"{invariant_violations}")

    # Cleanup item 7 / Fix 5: scalar precedence — the assembled geometry must
    # use scalar_values where known and the PREDICTION where unknown. Both
    # cases use the SAME deliberately-wrong prediction (999.0) so the test is
    # CAUSAL: it does not depend on whether an untrained model happens to
    # predict a value close to ground truth. All three scalars are verified:
    #   l — via channel 2 (spatially dense, l/3 everywhere);
    #   h — via channel 1 (h on occupied pixels);
    #   r — via channel 0 (r/5 on occupied pixels).
    # Occupied pixels are located from the HARD binary occupancy support, not
    # assumed to be at [0,0]. Hard assembly (binary occupancy) so channel
    # values are exactly the scalar values.
    with torch.no_grad():
        wrong_pred = torch.full_like(out["scalar_pred"], 999.0)

        # Locate an occupied pixel per sample from the true occupancy.
        occ_pixels = occ[:, 0] > 0.5  # (B, 64, 64)
        occ_idx = occ_pixels.nonzero()
        assert occ_idx.shape[0] >= b, (
            "preflight: each sample needs at least one occupied pixel for "
            "h/r precedence verification")

        # KNOWN case: all scalars known → assembly must use scalar_values.
        sk_known = torch.ones(b, 3, dtype=torch.bool, device=device)
        geom_known, _ = model.decode_geometry(
            out["z_hat"], wrong_pred, occ_input=occ, mask=M,
            scalar_known=sk_known, scalar_values=sv, hard_forward=True)
        # l via channel 2 (dense).
        l_used = geom_known[:, 2, 0, 0]
        if not torch.allclose(l_used, sv[:, 0] / 3.0, atol=1e-5):
            raise RuntimeError(
                "preflight: known-scalar precedence violated for l — assembly "
                "did not use scalar_values for known scalars")
        # h via channel 1 on an occupied pixel of each sample.
        for i in range(b):
            px = occ_idx[occ_idx[:, 0] == i][0]
            h_used = geom_known[i, 1, px[1], px[2]].item()
            if abs(h_used - sv[i, 1].item()) > 1e-5:
                raise RuntimeError(
                    f"preflight: known-scalar precedence violated for h "
                    f"(sample {i}: got {h_used}, expected {sv[i,1].item()})")
            r_used = geom_known[i, 0, px[1], px[2]].item()
            if abs(r_used - sv[i, 2].item() / 5.0) > 1e-5:
                raise RuntimeError(
                    f"preflight: known-scalar precedence violated for r "
                    f"(sample {i}: got {r_used}, expected {sv[i,2].item()/5.0})")

        # UNKNOWN case: all scalars unknown → assembly must use wrong_pred
        # (999.0), NOT scalar_values.
        sk_unknown = torch.zeros(b, 3, dtype=torch.bool, device=device)
        geom_unknown, _ = model.decode_geometry(
            out["z_hat"], wrong_pred, occ_input=occ, mask=M,
            scalar_known=sk_unknown, scalar_values=sv, hard_forward=True)
        # l via channel 2 (dense): must be 999/3, not sv/3.
        l_pred_used = geom_unknown[:, 2, 0, 0] * 3.0
        if not torch.allclose(l_pred_used, torch.full_like(l_pred_used, 999.0),
                              atol=1e-3):
            raise RuntimeError(
                "preflight: unknown-scalar precedence violated for l — "
                "assembly did not use scalar_pred for unknown scalars")
        # h/r via occupied pixels: must be 999 (h) / 999/5 (r), not sv.
        for i in range(b):
            px = occ_idx[occ_idx[:, 0] == i][0]
            h_used = geom_unknown[i, 1, px[1], px[2]].item()
            if abs(h_used - 999.0) > 1e-3:
                raise RuntimeError(
                    f"preflight: unknown-scalar precedence violated for h "
                    f"(sample {i}: got {h_used}, expected 999.0)")
            r_used = geom_unknown[i, 0, px[1], px[2]].item()
            if abs(r_used - 999.0 / 5.0) > 1e-3:
                raise RuntimeError(
                    f"preflight: unknown-scalar precedence violated for r "
                    f"(sample {i}: got {r_used}, expected 999.0/5)")

    checks = {
        "occupancy_shape": list(occ.shape),
        "z_x_shape": list(out["z_x"].shape),
        "z_hat_shape": list(out["z_hat"].shape),
        "scalar_pred_shape": list(out["scalar_pred"].shape),
        "assembled_geometry_shape": list(geometry.shape),
        "surrogate_prediction_shape": list(spec_pred.shape),
        "loss_finite": bool(torch.isfinite(loss)),
        "geometry_finite": bool(torch.isfinite(geometry).all()),
        "spec_pred_finite": bool(torch.isfinite(spec_pred).all()),
        "geometry_invariants_ok": len(invariant_violations) == 0,
        "known_scalar_precedence_ok": True,
        "unknown_scalar_precedence_ok": True,
    }

    # Gradient ownership.
    student_grads = sum(
        1 for p in model.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    decoder_grads = sum(
        1 for p in model.geometry_decoder.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0)
    predictor_grads = sum(
        1 for p in model.predictor.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0)
    surrogate_grads = sum(
        1 for p in surrogate.parameters() if p.grad is not None)
    ema_grads = sum(
        1 for p in model.ema.parameters() if p.grad is not None)
    scalar_ema_grads = sum(
        1 for p in model.scalar_mlp_ema.parameters() if p.grad is not None)
    released = getattr(model.spectrum_path, "released", None)
    released_grads = sum(
        1 for p in released.parameters() if p.grad is not None) if released else 0

    ownership = {
        "student_params_with_grad": student_grads,
        "decoder_params_with_grad": decoder_grads,
        "predictor_params_with_grad": predictor_grads,
        "surrogate_params_with_grad": surrogate_grads,
        "ema_params_with_grad": ema_grads,
        "scalar_mlp_ema_params_with_grad": scalar_ema_grads,
        "released_params_with_grad": released_grads,
    }

    if student_grads == 0:
        raise RuntimeError("preflight: no student parameters received gradients")
    if decoder_grads == 0 or predictor_grads == 0:
        raise RuntimeError("preflight: decoder/predictor received no gradients")
    if surrogate_grads != 0 or ema_grads != 0 or scalar_ema_grads != 0 or released_grads != 0:
        raise RuntimeError(
            f"preflight: frozen params received gradients: {ownership}")

    return {"checks": checks, "gradient_ownership": ownership,
            "loss": float(loss.detach())}


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
    parser.add_argument("--use-synthetic-smoke", action="store_true",
                        help="EXPLICIT smoke mode: synthetic data + dummy "
                             "spectrum weights allowed. Never used for real "
                             "training; normal invocation requires the real "
                             "dataset and released weights.")
    parser.add_argument("--preflight", action="store_true",
                        help="Run the real-data end-to-end preflight (Fix 17) "
                             "and exit.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or cfg["train"].get("device", "cpu")

    if args.preflight:
        result = preflight(cfg, device=device)
        print(json.dumps(result, indent=2))
        return

    report = train(cfg, resume_path=args.resume, no_train=args.no_train,
                   device=device, use_synthetic_smoke=args.use_synthetic_smoke)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
