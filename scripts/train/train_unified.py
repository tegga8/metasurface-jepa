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
import contextlib
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
from train.engine import (
    collect_ema_state, load_checkpoint, restore_ema_state, save_checkpoint,
)


# Curriculum regime vocabulary -> ScalarMasker regime (documented alias,
# audit B17: the config's "mixed" means an independent per-scalar draw).
SCALAR_REGIME_ALIASES = {"mixed": "independent"}

# Top-level config keys the unified trainer understands (audit B17: unknown
# keys are warned about instead of being silently ignored).
_KNOWN_TOP_LEVEL_KEYS = frozenset({
    "scalar_convention", "hidden", "num_heads", "geo_depth", "predictor_depth",
    "goal_tokens", "num_predictor_heads", "scalar_hidden", "n_film_blocks",
    "spec_dim", "ema_momentum_start", "ema_momentum_end", "loss", "curriculum",
    "staging", "weights", "data", "eval", "train", "_architecture_id",
    "scalar_predictor_film", "objective",
})


def _validate_config(cfg):
    """Validate the unified config against what the trainer actually reads.

    Audit B17: invalid curriculum entries and incoherent staging were silently
    ignored (e.g. scalar_regimes could name a regime ScalarMasker does not
    implement). Raises ValueError on invalid values; returns a list of
    non-fatal warnings (also printed).
    """
    from data.scalar_mask import ScalarMasker

    warnings = []
    unknown = sorted(set(cfg) - _KNOWN_TOP_LEVEL_KEYS)
    if unknown:
        warnings.append(f"unknown top-level config keys (ignored): {unknown}")

    cur = cfg.get("curriculum", {})
    for regime in cur.get("scalar_regimes", []):
        if regime not in ScalarMasker.REGIMES and regime not in SCALAR_REGIME_ALIASES:
            raise ValueError(
                f"curriculum.scalar_regimes contains {regime!r}; valid regimes "
                f"are {list(ScalarMasker.REGIMES)} (plus the documented alias "
                f"{sorted(SCALAR_REGIME_ALIASES)})")
    for name in ("train_mask_ratios", "eval_mask_ratios"):
        bad = [r for r in cur.get(name, []) if not (0.0 <= float(r) <= 1.0)]
        if bad:
            raise ValueError(f"curriculum.{name} must be in [0, 1], got {bad}")
    if any(float(r) <= 0.0 for r in cur.get("train_mask_ratios", [])):
        warnings.append(
            "curriculum.train_mask_ratios contains 0.0 — excluded at runtime: "
            "the masked-token objective is undefined with no masked tokens")

    staging_phase = str(cfg.get("staging", {}).get("phase", "")).upper()
    lambda_phys = float(cfg.get("loss", {}).get("lambda_phys", 0.0))
    for key in ("lambda_inv", "lambda_var", "lambda_cov", "lambda_scalar",
                "lambda_occ", "lambda_phys", "lambda_summary"):
        val = cfg.get("loss", {}).get(key)
        if val is not None and float(val) < 0:
            raise ValueError(f"loss.{key} must be >= 0, got {val}")
    if lambda_phys > 0 and staging_phase in ("A", "B"):
        raise ValueError(
            f"staging.phase={staging_phase!r} is a no-physics stage but "
            f"loss.lambda_phys={lambda_phys} > 0 — make the staging explicit "
            "(either set phase >= 'C' or keep lambda_phys = 0)")
    if lambda_phys == 0 and staging_phase in ("C", "D", "E"):
        warnings.append(
            f"staging.phase={staging_phase!r} expects physics but "
            "loss.lambda_phys = 0 — the physics loss stays inactive")

    # Audit B24: lambda_phys > 0 with STE disabled is a SILENT NO-OP. Measured
    # on the released surrogate (2026-09-13 probe): the soft-occupancy path
    # computes a perfectly healthy-looking L_phys (18.58) while ZERO student
    # parameters receive gradient — the soft field is ~96% out of distribution
    # for the surrogate (spectrum_rel_diff = 0.9599). Nothing else in the
    # pipeline refuses this combination, and the failure is invisible in the
    # loss logs, so the config must be rejected rather than warned about.
    physics_use_ste = cfg.get("staging", {}).get("physics_use_ste", True)
    if lambda_phys > 0 and not bool(physics_use_ste):
        raise ValueError(
            f"loss.lambda_phys={lambda_phys} > 0 with "
            "staging.physics_use_ste=false: the soft-occupancy path is a silent "
            "dead end — the physics loss would log a healthy value while "
            "contributing ZERO gradient to the model (measured against the "
            "released surrogate: 0 student params with grad, soft field 96% out "
            "of distribution). Set physics_use_ste: true, or keep lambda_phys=0 "
            "and rely on L_occ for decoder supervision.")

    # Audit B27: the evaluation batch IS the gate's sample size. The shuffled
    # control needs >= 2 for a valid derangement, and the historical value of 2
    # (inherited from train.batch_size) was too few to distinguish a real effect
    # from a single swap.
    n_eval = cfg.get("eval", {}).get("n_samples", 32)
    if int(n_eval) < 2:
        raise ValueError(
            f"eval.n_samples={n_eval} < 2: the real-vs-shuffled gate needs at "
            "least two samples for a valid derangement (audit B27)")

    thr = cfg.get("eval", {}).get("gate_beats_fraction_min", 0.5)
    if not 0.0 <= float(thr) <= 1.0:
        raise ValueError(
            f"eval.gate_beats_fraction_min={thr} must be in [0, 1] — it is a "
            "win-rate threshold for the primary gate statistic")

    for w in warnings:
        print(f"[config] WARNING: {w}")
    return warnings


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
        # Review C6: NEVER write the dummy to the configured released-weights
        # path — real mode later accepts that path on a bare os.path.exists, so
        # a leftover dummy would silently stand in for the frozen encoder. The
        # dummy goes to a distinct sibling name that real mode never reads.
        dummy_path = path + ".smoke_dummy"
        os.makedirs(os.path.dirname(dummy_path), exist_ok=True)
        try:
            from model.spec_encoder import VanillaSpectrumEncoder
        except ImportError:
            raise RuntimeError(
                f"spectrum checkpoint {path} missing and VanillaSpectrumEncoder "
                "unavailable — cannot create smoke dummy") from None
        enc = VanillaSpectrumEncoder()
        torch.save(enc.state_dict(), dummy_path)
        print(f"[smoke] Created DUMMY spectrum encoder at {dummy_path} "
              f"(released path {path} left untouched)")
        return dummy_path
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


def _mask_probs_at(cfg, step):
    """Curriculum mask-ratio distribution at `step`, optionally ramped (Phase 2b).

    Target = train_mask_ratio_probs. If `curriculum.mask_schedule` sets
    ramp_steps > 0 and start_mask_ratio_probs, the distribution is linearly
    interpolated from start -> target over ramp_steps, so the representation
    forms on easier masking before full masking becomes common. Returns
    (ratios, probs) with 0.0 excluded.
    """
    cur = cfg["curriculum"]
    ratios = cur.get("train_mask_ratios", cur.get("mask_ratios"))
    target = cur.get("train_mask_ratio_probs", cur.get("mask_ratio_probs"))
    pairs = [(r, p) for r, p in zip(ratios, target) if r > 0.0]
    rs, ps = zip(*pairs) if pairs else ((1.0,), (1.0,))
    ps = torch.tensor(ps, dtype=torch.float32)
    sched = cur.get("mask_schedule") or {}
    ramp = int(sched.get("ramp_steps", 0) or 0)
    start = sched.get("start_mask_ratio_probs")
    if ramp > 0 and start is not None:
        sp = torch.tensor([p for r, p in zip(ratios, start) if r > 0.0],
                          dtype=torch.float32)
        frac = min(1.0, max(0.0, float(step) / ramp))
        ps = sp + (ps - sp) * frac
    ps = ps / ps.sum()
    return list(rs), ps


def sample_mask_ratios(cfg, rng, b, step):
    """PER-SAMPLE occupancy mask ratios from the (optionally ramped) curriculum
    (Phase 2b). Returns a list of B floats; 0.0 is excluded."""
    rs, ps = _mask_probs_at(cfg, step)
    idx = torch.multinomial(ps, b, replacement=True, generator=rng)
    return [float(rs[i]) for i in idx.tolist()]


def _physics_lambda_at(step, lambda_phys, start_step=0, ramp_steps=0):
    """lambda_phys at a given optimizer step (Phase 2c).

    0 before `start_step` (so the representation forms before physics is
    heavy), then a linear ramp to `lambda_phys` over `ramp_steps`. With
    start_step=0 and ramp_steps=0 this is the constant lambda_phys (previous
    behaviour).
    """
    if step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return float(lambda_phys)
    frac = min(1.0, (step - start_step + 1) / ramp_steps)
    return float(lambda_phys) * frac


def _ramp_frac(step, start_step=0, ramp_steps=0):
    """Linear 0->1 ramp factor for a weight that turns on at `start_step`
    (Phase 4 multi-target warmup)."""
    if step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return min(1.0, (step - start_step + 1) / ramp_steps)


def resolve_staging_steps(cfg, total_steps):
    """Length-proportional staging (operator retune 2026-10-05).

    The Phase-2/3/4 schedule was tuned at 10k steps and shipped as ABSOLUTE
    steps (physics off 2000 / ramp 3000; mask ramp 3000; multi-target
    1000+2000). Reused unchanged in a 70k run those windows [accidentally]
    compress to the first ~7 % of training. When a ``*_frac`` key is present
    it OVERRIDES the absolute key with ``round(total_steps * frac)``; absent
    fractions preserve the old absolute behaviour exactly (at 10k the shipped
    fractions reproduce it bit-identically: 0.20->2000, 0.30->3000, 0.10->1000,
    0.20->2000).
    """
    staging = cfg.setdefault("staging", {})
    sched = cfg.setdefault("curriculum", {}).setdefault("mask_schedule", {})

    def _res(container, frac_key, step_key):
        frac = container.get(frac_key)
        if frac is not None:
            container[step_key] = max(1, int(round(total_steps * float(frac))))

    _res(staging, "lambda_phys_start_frac", "lambda_phys_start_step")
    _res(staging, "lambda_phys_ramp_frac", "lambda_phys_ramp_steps")
    _res(staging, "multi_target_start_frac", "multi_target_start_step")
    _res(staging, "multi_target_ramp_frac", "multi_target_ramp_steps")
    _res(sched, "ramp_frac", "ramp_steps")
    return cfg


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
        # Audit B18: the REQUESTED ratio is nominal — block-mask coverage
        # differs from it (min-side clamps + independent block overlap). Track
        # the achieved masked fraction per requested bucket so reports never
        # conflate requested with achieved.
        self.mask_achieved = {r: [] for r in self.mask_ratios}
        self._total = 0          # batches (regime_freq denominator)
        self._mask_total = 0     # samples (mask_freq denominator; Phase 2b)

    def _nearest_ratio(self, ratio):
        return min(self.mask_counts, key=lambda r: abs(float(r) - float(ratio)))

    def record(self, ratios, regime, achieved_masked_fraction=None):
        # Phase 2b: `ratios` may be a scalar (legacy) or a per-sample list.
        # Count each SAMPLE toward its nearest bucket, so mask_freq estimates
        # the per-sample distribution rather than the batch mean's bucket.
        ratios = list(ratios) if isinstance(ratios, (list, tuple)) else [ratios]
        for r in ratios:
            key = r if r in self.mask_counts else self._nearest_ratio(r)
            self.mask_counts[key] += 1
            if achieved_masked_fraction is not None:
                self.mask_achieved.setdefault(key, []).append(
                    float(achieved_masked_fraction))
        self._mask_total += len(ratios)
        self.regime_counts[regime] += 1
        self._total += 1

    def report(self):
        nb = max(1, self._total)          # batches
        ns = max(1, self._mask_total)     # samples
        achieved = {
            r: (sum(v) / len(v) if v else None)
            for r, v in self.mask_achieved.items()
        }
        return {
            "mask_freq": {r: c / ns for r, c in self.mask_counts.items()},
            "regime_freq": {r: c / nb for r, c in self.regime_counts.items()},
            "mask_fraction_achieved_mean": achieved,
        }


# ---------------------------------------------------------------------------
# per-step EMA-frozen guard (Phase 3 MD §6)
# ---------------------------------------------------------------------------

def _assert_no_ema_gradients(model, step, objective=None):
    """Per-step guard: EMA targets, the released encoder, and the objective's
    registered frozen surrogate must receive no gradient (spec §6; audit B18:
    the surrogate was previously outside this per-step guard)."""
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
    surrogate = getattr(objective, "surrogate", None) if objective is not None else None
    if surrogate is not None:
        for name, p in surrogate.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                leaked.append(f"surrogate.{name}")
    if leaked:
        raise RuntimeError(
            f"Step {step}: EMA/released/surrogate params received gradient: "
            f"{leaked}")


# ---------------------------------------------------------------------------
# cosine warmup scheduler (schedule inherited from the retired Milestone-B
# trainer; kept identical)
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

    # Sample PER-SAMPLE occupancy mask ratios from the curriculum (Phase 2b)
    ratios = sample_mask_ratios(cfg, rng, B, step)

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
        M = masker.sample_per_sample(geo_true, ratios, surrogate).to(device)
    else:
        M = masker.sample_per_sample(occ, ratios, surrogate).to(device)

    # Forward + loss
    # Phase 4 MD §3.5.1: goal dropout — replace A_goal with null token ~10%
    gd_p = cfg.get("train", {}).get("guidance_dropout", 0.0)
    goal_mode = goal_dropout("real", gd_p, rng)
    result = objective(model, occ, sv, sk, spec, M, goal_mode=goal_mode)
    loss = result["total_loss"]

    regime_logger.record(
        ratios, regime,
        achieved_masked_fraction=float((M < 0.5).float().mean().item()))

    return result, M, sk


def per_term_grad_share(result, model, objective, eps=1e-12):
    """Per-term share of the gradient budget on the CURRENT graph (Phase 2a).

    Differentiates each WEIGHTED term alone over the trainable student +
    objective parameters (the `protocol_v1/step3_4` approach, now live during
    training instead of post-hoc) and returns the L2-norm share per term.

    This is a PROBE, not a training step — it never updates weights, and it
    leaves every parameter's `.grad` cleared, so it is safe to call immediately
    before the real `loss.backward()` (the graph is traversed with
    `retain_graph=True`).
    """
    params = [p for p in model.parameters() if p.requires_grad] + \
             [p for p in objective.parameters() if p.requires_grad]
    terms = result.get("term_losses") or {}
    norms = {}
    for name, t in terms.items():
        for p in params:
            p.grad = None
        if t.requires_grad:
            t.backward(retain_graph=True)
        sq = 0.0
        for p in params:
            if p.grad is not None:
                g = p.grad.detach()
                sq += float((g * g).sum().item())
        norms[name] = sq ** 0.5
    for p in params:
        p.grad = None
    total = sum(norms.values())
    if total <= eps:
        return {k: 0.0 for k in norms}
    return {k: v / total for k, v in norms.items()}


def _config_differences(saved, live, prefix=""):
    """Key paths where a checkpoint's recorded config differs from the live one.

    `ckpt["cfg"]` was written into every checkpoint and never read, so editing the
    YAML between save and resume silently diverged the run from its own record
    (audit B28-adjacent). Keys starting with "_" are runtime annotations added by
    the trainer, not user config.
    """
    out = []
    for k in sorted(set(saved or {}) | set(live or {}), key=str):
        if isinstance(k, str) and k.startswith("_"):
            continue
        a, b = (saved or {}).get(k), (live or {}).get(k)
        path = f"{prefix}{k}"
        if isinstance(a, dict) and isinstance(b, dict):
            out.extend(_config_differences(a, b, f"{path}."))
        elif a != b:
            out.append((path, a, b))
    return out


@contextlib.contextmanager
def projector_train_mode(objective):
    """Compute the projected terms with the projector in TRAIN mode, without
    contaminating its BatchNorm running statistics.

    Audit B28. `validate()` runs the objective under `eval()`, where the
    projector's two `BatchNorm1d` layers use accumulated RUNNING statistics
    instead of batch statistics. One shared projector serves TWO different
    distributions (the predictor's output and the EMA target's output), so its
    running statistics fit neither — measured at the full-epoch checkpoint, the
    eval-mode values were **759×** (`L_inv`) and **364×** (`L_cov`) larger than
    the values training actually optimises. The projected diagnostics therefore
    measured a different function from the training objective.

    The running statistics are snapshotted and restored, because BatchNorm
    updates them in train mode even under `torch.no_grad()`.
    """
    proj = getattr(objective, "projector", None)
    if proj is None:
        yield
        return
    saved = {k: v.detach().clone() for k, v in proj.state_dict().items()
             if k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    was_training = proj.training
    proj.train()
    try:
        yield
    finally:
        if saved:
            proj.load_state_dict(saved, strict=False)
        proj.train(was_training)


def validate(model, objective, val_batches, cfg, device, strata=None):
    """Run validation on a list of pre-built (occ, sv, spec) batches.

    Unified model signature: model(occupancy, scalar_values, scalar_known,
    spectrum, mask). Reports BOTH raw-latent and projected-latent diagnostics
    on the exact masked-token population used by the training objective, so
    a raw-vs-projected discrepancy (projector/statistics problem) can be
    distinguished from a genuine representation failure.

    Stratified reporting (audit B6, architecture_v5.md §8.3): metrics are
    returned PER STRATUM, never pooled —
      easy : low occupancy mask + all scalars known
      hard : full occupancy mask + all scalars unknown (pure inverse design)
    The hard stratum is where the design's spectrum/scalar-dependence gates
    apply; a pooled number can look healthy while genuine dependence is absent
    exactly there. Stratum mask ratios must be > 0: with zero masked tokens
    the masked-token objective is undefined and VICReg raises by contract
    (no silent zero substitution).

    Field-name convention (item 7):
      - raw_*      : computed on z_hat / z_y_raw (encoder output space)
      - proj_*     : computed on p_hat / p_y (objective projector space,
                     the space in which L_inv/L_var/L_cov are trained)
      - L_*        : the training objective's loss components
      - L_total    : the FULL training objective (all weighted terms) — NOT a
                     reconstruction/physics metric; do not read it as one.
      - mask_ratio / scalars: the stratum metadata (never averaged away).
    """
    model.eval()
    objective.eval()
    cur = cfg.get("curriculum", {})
    if strata is None:
        strata = [
            ("easy", float(cur.get("easy_mask_ratio", 0.5)), True),
            ("hard", float(cur.get("hard_mask_ratio", 1.0)), False),
        ]
    out = {}
    try:
        # Audit B28: the projected diagnostics must be computed with the
        # projector in train mode, or they measure a different function than the
        # objective trains (see projector_train_mode).
        with torch.no_grad(), projector_train_mode(objective):
            for name, mask_ratio, scalars_known in strata:
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
                achieved_fracs = []
                for occ, sv, spec in val_batches:
                    B = occ.shape[0]
                    if scalars_known:
                        sk = torch.ones(B, 3, dtype=torch.bool, device=device)
                    else:
                        sk = torch.zeros(B, 3, dtype=torch.bool, device=device)
                    # Validation masks are deterministic (fixed seed) and
                    # explicitly transferred to the model device.
                    M = val_masker.sample(occ, mask_ratio).to(device)
                    assert M.device == occ.device, (
                        "validation mask must be on the model device")
                    # Audit B18: report achieved coverage next to the requested
                    # ratio (block masking does not hit the nominal fraction).
                    achieved_fracs.append(float((M < 0.5).float().mean().item()))
                    result = objective(model, occ, sv, sk, spec, M, goal_mode="real")
                    out_m = result["out"]
                    mask_bool = out_m["mask"]
                    z_y = out_m.get("z_y_raw")
                    # JEPA/VICReg latent diagnostics are N/A in the conventional
                    # arm (no EMA target): skip them instead of crashing.
                    if z_y is not None:
                        z_hat = out_m["z_hat"]
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
                    se = (out_m["scalar_pred"] - sv).abs().mean()
                    metrics["scalar_err"].append(float(se))

                stratum = {k: float(np.mean(v)) for k, v in metrics.items() if v}
                stratum["mask_ratio"] = float(mask_ratio)
                stratum["mask_ratio_achieved_mean"] = (
                    float(np.mean(achieved_fracs)) if achieved_fracs else None)
                stratum["scalars"] = ("all_known" if scalars_known
                                      else "all_unknown")
                # Audit B28: the physics term is NOT evaluated here. The objective
                # gates it on `model.training` (its STE path is training-only),
                # and validation runs under eval(), so `L_phys` is structurally
                # zero — reporting 0.0 reads as "physics contributes nothing"
                # when it means "not measured".
                stratum["L_phys"] = None
                stratum["L_phys_weighted"] = None
                stratum["L_phys_evaluated"] = False
                out[name] = stratum
    finally:
        model.train()
        objective.train()

    # Phase 4 MD §20.3: guidance-gap diagnostic on the HARD stratum (the
    # stratum where spectrum-dependence is the gate), never pooled.
    try:
        from diagnostics.guidance_gap import compute_guidance_gap
        hard_ratio = float(cur.get("hard_mask_ratio", 1.0))
        occ_v, sv_v, spec_v = val_batches[0]
        B = occ_v.shape[0]
        sk_v = torch.zeros(B, 3, dtype=torch.bool, device=device)
        gap_masker = BlockMasker(
            placement="random", grid=16, min_side=3, k_range=(1, 4),
            seed=12345)
        M = gap_masker.sample(occ_v, hard_ratio).to(device)
        gap_info = compute_guidance_gap(
            model, occ_v, sv_v, sk_v, spec_v, M, device=device)
        out["guidance_gap"] = gap_info["guidance_gap"]
        out["normalized_guidance_gap"] = gap_info["normalized_guidance_gap"]
        out["guidance_gap_stratum"] = "hard"
    except Exception as e:
        out["guidance_gap_error"] = str(e)

    return out


# ---------------------------------------------------------------------------
# main training loop
# ---------------------------------------------------------------------------

def train(cfg, resume_path=None, no_train=False, device=None,
          use_synthetic_smoke=False, max_steps=None):
    """Main entry point. Returns a summary dict.

    Args:
        use_synthetic_smoke: explicit smoke mode — synthetic data and dummy
            spectrum weights allowed. Real mode (default) requires the real
            dataset and released weights and fails loudly if they are missing.
        max_steps: override the schedule length for a verification-scale run
            (a short 5-10% run on real data, to prove the full pipeline —
            data loading, forward/backward, EMA updates, physics path,
            checkpointing — before committing to the full schedule).
    """
    from train.engine import collect_ema_state

    _validate_config(cfg)  # audit B17: loud on invalid/incoherent configs
    set_seed(cfg["train"].get("seed", 42))
    device = device or resolve_device(cfg["train"].get("device", "cpu"))
    total_steps = cfg["train"].get("total_steps", 1500)
    if max_steps is not None:
        total_steps = max(1, int(max_steps))
    # Length-proportional staging (operator retune 2026-10-05): *_frac keys
    # override the 10k-era absolute steps so a 70k run keeps the designed
    # proportions (off 20 % / ramp 30 %), not a 7 %-of-run accident.
    resolve_staging_steps(cfg, total_steps)

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
    # Audit B17: data.num_workers was silently ignored (hardcoded 0).
    num_workers = int(cfg.get("data", {}).get("num_workers", 0))

    # --- model (Fix 6: released spectrum weights required in real mode) ---
    spec_weights = _ensure_spectrum_weights(
        spec_path, device, allow_dummy=use_synthetic_smoke)
    model = build_unified_model(cfg, spec_weights, device=device)
    cfg.setdefault("_architecture_id", model.architecture_id)
    # Audit B2: schedule the EMA momentum ramp from the run's total steps.
    # EMAEncoder defaults to total_steps=1, which saturates current_momentum()
    # at its final value (0.999) from step 1 instead of ramping 0.996 -> 0.999.
    model.set_total_steps(total_steps)

    # --- objective ---
    loss_cfg = cfg.get("loss", {})
    # Phase 4 MD §4: load frozen surrogate when physics loss is active
    lambda_phys = loss_cfg.get("lambda_phys", 0.0)
    ramp_steps = cfg.get("staging", {}).get("lambda_phys_ramp_steps", 0)
    lambda_cond = loss_cfg.get("lambda_cond", 0.0)
    lambda_scal_t = loss_cfg.get("lambda_scal_t", 0.0)
    mt_start = int(cfg.get("staging", {}).get("multi_target_start_step", 0) or 0)
    mt_ramp = int(cfg.get("staging", {}).get("multi_target_ramp_steps", 0) or 0)
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
        # Door (a) of the scalar investigation (operator decision 2026-09-13):
        # weight of the summary-token read-out, which gives the scalar encoder a
        # gradient path that does not run through the predictor.
        lambda_summary=loss_cfg.get("lambda_summary", 0.0),
        gamma=loss_cfg.get("gamma", 1.0),
        eps=loss_cfg.get("eps", 1e-4),
        # Audit B17: the Huber branch was unreachable — the config key was
        # never read.
        scalar_loss_type=loss_cfg.get("scalar_loss_type", "l1"),
        # Operator decision 2026-09-13: §4.1 occupancy BCE (decoder supervision
        # independent of the physics path).
        lambda_occ=loss_cfg.get("lambda_occ", 0.0),
        # Phase 3b: projector ablation (none | linear | mlp | mlp_bn).
        projector_type=loss_cfg.get("projector_type", "mlp_bn"),
        lambda_cond=loss_cfg.get("lambda_cond", 0.0),
        lambda_scal_t=loss_cfg.get("lambda_scal_t", 0.0),
        objective=cfg.get("objective", "jepa"),
        surrogate=surrogate,
        physics_use_ste=cfg.get("staging", {}).get("physics_use_ste", True),
    ).to(device)

    # Objective audit banner (Step 6): unambiguous, so a wrong objective can never
    # be silently trained (the Step-3 scalar experiment exposed that failure mode).
    _obj = cfg.get("objective", "jepa")
    if _obj == "conventional":
        print("=" * 64, flush=True)
        print("OBJECTIVE MODE: CONVENTIONAL", flush=True)
        print("  JEPA latent loss (L_inv/L_var/L_cov): OFF", flush=True)
        print("  VICReg: OFF", flush=True)
        print("  EMA target objective: OFF", flush=True)
        print("  Masked-token prediction: OFF", flush=True)
        print("  Occupancy BCE (FULL target): ON", flush=True)
        print("  Scalar supervision: ON", flush=True)
        print("  Frozen-surrogate spectrum supervision: ON", flush=True)
        print("  Physics ramp / mask curriculum / architecture: SAME AS BASELINE",
              flush=True)
        print("=" * 64, flush=True)
    else:
        print("OBJECTIVE MODE: JEPA (unchanged baseline)", flush=True)

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
                            shuffle=True, num_workers=num_workers,
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
                             shuffle=False, num_workers=num_workers,
                             collate_fn=collate_batch)
        for G, S in vloader:
            occ, sv = factorize_geometry(G)
            val_batches.append((occ.to(device), sv.to(device), S.to(device)))

    # --- resume ---
    start_step = 0
    curriculum_rng_state = None
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
        # Audit B3: restore the saved EMA state (momentum counters + targets)
        # from the checkpoint rather than relying on incidental state-dict
        # coverage. This run's schedule length is then re-applied (B2) so the
        # EMA ramp matches the LR cosine schedule rebuilt from config; a change
        # in schedule length is reported loudly rather than silently applied.
        restore_ema_state(model, ckpt.get("ema_state"))
        ckpt_total = (ckpt.get("ema_state") or {}).get("total_steps")
        model.set_total_steps(total_steps)
        if ckpt_total is not None and int(ckpt_total) != int(total_steps):
            print(f"[resume] NOTE: EMA schedule length changed: checkpoint "
                  f"{int(ckpt_total)} -> this run {int(total_steps)} steps; "
                  "the EMA ramp follows THIS run's schedule (matching the LR "
                  "cosine schedule rebuilt from config).")
        # Audit B4: the private curriculum generator drives mask ratio, scalar
        # regime, and goal dropout; its state must round-trip or a resumed run
        # silently restarts the sampling stream from the seed.
        curriculum_rng_state = ckpt.get("curriculum_rng_state")
        if curriculum_rng_state is None:
            print("[resume] WARNING: checkpoint has no curriculum RNG state "
                  "(legacy checkpoint) — mask-ratio/scalar-regime/goal-dropout "
                  "sampling restarts from the seed and will not match an "
                  "uninterrupted run.")
        # Report any divergence between the config the checkpoint was trained
        # with and this run's config. The live config wins (it is what the user
        # asked for) but never silently.
        saved_cfg = ckpt.get("cfg")
        if saved_cfg:
            diffs = _config_differences(saved_cfg, cfg)
            if diffs:
                print(f"[resume] NOTE: the checkpoint was trained with a different "
                      f"config — {len(diffs)} value(s) differ; THIS run's config wins:")
                for path, old, new in diffs[:20]:
                    print(f"[resume]   {path}: checkpoint={old!r} -> this run={new!r}")
                if len(diffs) > 20:
                    print(f"[resume]   ... and {len(diffs) - 20} more")
        else:
            print("[resume] WARNING: checkpoint carries no recorded config "
                  "(legacy) — cannot check that this run matches the one it resumes.")
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
        _assert_no_ema_gradients(model, 0, objective)
        components = result["components"]
        print(f"[smoke] step=0  loss={float(loss.detach()):.4f}  "
              f"L_inv={components['L_inv']:.4f} "
              f"L_var={components['L_var']:.4f} "
              f"L_cov={components['L_cov']:.4f} "
              f"L_scalar={components['L_scalar']:.4f}")
        assert torch.isfinite(loss), "smoke loss must be finite"
        return {"final_step": 0, "final_loss": float(loss.detach()),
                "components": components, "regime_report": regime_logger.report(),
                "total_steps": total_steps,
                "ema_total_steps": int(model.ema.total_steps)}

    # --- training loop ---
    model.train()
    objective.train()
    rng = torch.Generator().manual_seed(cfg["train"].get("seed", 42))
    if curriculum_rng_state is not None:
        rng.set_state(curriculum_rng_state)
        print("[resume] curriculum RNG state restored")
    regime_logger = RegimeLogger(cfg)
    batch_size = train_cfg.get("batch_size", 2)
    grad_accum = train_cfg.get("grad_accum", 1)
    log_every = train_cfg.get("log_every_steps", 10)
    grad_share_every = train_cfg.get("log_grad_share_every_steps", 0)
    val_every = train_cfg.get("val_every_steps", 50)
    ckpt_every = train_cfg.get("ckpt_every_steps", 100)
    clip_norm = train_cfg.get("clip_grad_norm", 1.0)

    data_iter = iter(train_data) if use_synthetic or not isinstance(train_data, DataLoader) \
        else iter(loader)

    phys_start = int(cfg.get("staging", {}).get("lambda_phys_start_step", 0) or 0)
    last_loss = None
    for step in range(start_step, total_steps):
        # Phase 2c: hold lambda_phys at 0 until phys_start (representation
        # forms first), then ramp to the target over ramp_steps.
        if ramp_steps > 0 or phys_start > 0:
            objective.lambda_phys = _physics_lambda_at(
                step, lambda_phys, phys_start, ramp_steps)
        # Phase 4: ramp the multi-target weights in after a warmup so they never
        # hijack the gradient budget early (the failure mode of the first attempt).
        if lambda_cond > 0 or lambda_scal_t > 0:
            f = _ramp_frac(step, mt_start, mt_ramp)
            objective.lambda_cond = lambda_cond * f
            objective.lambda_scal_t = lambda_scal_t * f

        # Explicitly reset gradients at the START of each optimizer step
        # (Fix 1, spec §3): gradients must accumulate only across the
        # grad_accum microbatches of THIS optimizer step, never leak from the
        # previous step. Not inside the microbatch loop; not left implicit.
        optimizer.zero_grad(set_to_none=True)

        micro_losses = []
        want_share = (grad_share_every > 0 and step % grad_share_every == 0)
        for micro in range(grad_accum):
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
            if want_share and micro == 0:
                shares = per_term_grad_share(result, model, objective)
                print(f"  [grad-share] step {step}: {json.dumps(shares)}")
            (loss / grad_accum).backward()
            micro_losses.append(float(loss.detach()))

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] +
            [p for p in objective.parameters() if p.requires_grad],
            clip_norm)

        # Guard: EMA must not receive gradients
        _assert_no_ema_gradients(model, step, objective)

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
                  # Door (a): the summary-token read-out. Without this on the log
                  # line the mechanism is invisible during training — the first
                  # door-(a) run had to be diagnosed from the checkpoint because
                  # the term never appeared in the step log.
                  f"L_summary={c['L_summary']:.4f} "
                  f"L_summary_w={c['L_summary_weighted']:.4f} "
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
                    scalar_masker_bank),
                    "curriculum_rng_state": rng.get_state()},
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
            scalar_masker_bank),
            "curriculum_rng_state": rng.get_state()},
        device=device, artifact_type="final")

    report = {
        "final_step": total_steps - 1,
        "final_loss": last_loss if last_loss else 0.0,
        "regime_report": regime_logger.report(),
        "total_steps": total_steps,
        "ema_total_steps": int(model.ema.total_steps),
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

def assert_physics_reaches_student(model, surrogate, occ, sv, sk, spec, mask, cfg):
    """Assert that the physics term ALONE carries gradient to the student.

    Audit B24. The preflight's ownership counts come from the FULL objective, so
    `L_inv/L_var/L_cov/L_occ` satisfy them too — a physics path with a dead
    Jacobian passes them unnoticed. Measured against the released surrogate
    (2026-09-13 probe): the soft-occupancy path computes a perfectly healthy
    L_phys (18.58) while ZERO student parameters receive gradient, because the
    soft field is ~96% out of distribution for the surrogate
    (`spectrum_rel_diff = 0.9599`). The STE path gives 360 student params.

    Runs its OWN grad-enabled forward. It must not reuse the caller's `out`: the
    preflight's main `loss.backward()` has already consumed that graph, so
    building on it raises "Trying to backward through the graph a second time"
    (observed on the cloud run) — a fresh forward keeps the check independent of
    the caller's graph state.

    Module-level (not inlined in `preflight`) so the check is unit testable: the
    preflight needs the real splits and released weights, which the dev machine
    does not stage, so an inlined version gets no local coverage — which is how a
    `NameError` and a missing `.backward()` reached the cloud run.

    Returns the number of student parameters that received gradient.
    """
    from physics.physics_loop import physics_loss_from_out

    physics_use_ste = bool(cfg.get("staging", {}).get("physics_use_ste", True))
    was_training = model.training
    model.train()  # the STE branch is training-only
    model.zero_grad(set_to_none=True)
    try:
        out = model(occ, sv, sk, spec, mask, goal_mode="real", with_target=False)
        L_phys_only, _, _ = physics_loss_from_out(
            model, out, surrogate, occ, sv, sk, spec, mask,
            loss_type="smooth_l1", use_ste=physics_use_ste, normalize=True)
        L_phys_only.backward()
        n_with_grad = sum(
            1 for p in model.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    if n_with_grad == 0:
        raise RuntimeError(
            "physics term produced NO student gradient "
            f"(L_phys={float(L_phys_only.detach()):.4f}, "
            f"physics_use_ste={physics_use_ste}): the loss is healthy-looking "
            "while the frozen surrogate contributes nothing to training "
            "(audit B24). With physics_use_ste=false the soft occupancy field "
            "is ~96% out of distribution for the surrogate and every gradient "
            "is zero.")
    return n_with_grad


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
        # The preflight must mirror the TRAINING objective, or it validates a
        # different function than the one that will run. lambda_occ was missing
        # here (fixed alongside door (a)).
        lambda_occ=cfg.get("loss", {}).get("lambda_occ", 0.0),
        lambda_summary=cfg.get("loss", {}).get("lambda_summary", 0.0),
        lambda_phys=max(cfg.get("loss", {}).get("lambda_phys", 0.0), 1.0),
        projector_type=cfg.get("loss", {}).get("projector_type", "mlp_bn"),
        lambda_cond=cfg.get("loss", {}).get("lambda_cond", 0.0),
        lambda_scal_t=cfg.get("loss", {}).get("lambda_scal_t", 0.0),
        objective=cfg.get("objective", "jepa"),
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

    # Everything below is diagnostic — shapes, broadcast invariants and scalar
    # precedence are read off the assembled geometry and nothing here is
    # backwarded (the training forward/backward is already done above). It runs
    # under no_grad so the hard_forward decodes are legal in train mode: the B18
    # guard refuses a hard-thresholded forward only while gradients are actually
    # being tracked (audit B21).
    out = result["out"]
    with torch.no_grad():
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
    precedence_notes = []
    with torch.no_grad():
        wrong_pred = torch.full_like(out["scalar_pred"], 999.0)

        # KNOWN case: all scalars known → assembly must use scalar_values.
        sk_known = torch.ones(b, 3, dtype=torch.bool, device=device)
        geom_known, occ_known = model.decode_geometry(
            out["z_hat"], wrong_pred, occ_input=occ, mask=M,
            scalar_known=sk_known, scalar_values=sv, hard_forward=True)
        # l via channel 2 (dense).
        l_used = geom_known[:, 2, 0, 0]
        if not torch.allclose(l_used, sv[:, 0] / 3.0, atol=1e-5):
            raise RuntimeError(
                "preflight: known-scalar precedence violated for l — assembly "
                "did not use scalar_values for known scalars")

        # h/r must be read at a pixel that is occupied in the ASSEMBLED
        # geometry. Review B2: the old code took the pixel index from the TRUE
        # occupancy, so on a masked pixel the model predicted empty the read
        # value was 0 — a SPURIOUS violation. The delivered occupancy does not
        # depend on the scalars, so one support serves both loops.
        occ_support = occ_known[:, 0] > 0.5          # (B, 64, 64)

        def _first_occupied(i):
            idx = occ_support[i].nonzero()
            return None if idx.numel() == 0 else (int(idx[0, 0]), int(idx[0, 1]))

        for i in range(b):
            pos = _first_occupied(i)
            if pos is None:
                precedence_notes.append(
                    f"sample {i}: assembled geometry has no occupied pixel — "
                    "h/r known-precedence not verifiable")
                continue
            py, px = pos
            h_used = geom_known[i, 1, py, px].item()
            if abs(h_used - sv[i, 1].item()) > 1e-5:
                raise RuntimeError(
                    f"preflight: known-scalar precedence violated for h "
                    f"(sample {i}: got {h_used}, expected {sv[i,1].item()})")
            r_used = geom_known[i, 0, py, px].item()
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
        # h/r via the SAME delivered-support pixel: must be 999 / 999/5.
        for i in range(b):
            pos = _first_occupied(i)
            if pos is None:
                precedence_notes.append(
                    f"sample {i}: assembled geometry has no occupied pixel — "
                    "h/r unknown-precedence not verifiable")
                continue
            py, px = pos
            h_used = geom_unknown[i, 1, py, px].item()
            if abs(h_used - 999.0) > 1e-3:
                raise RuntimeError(
                    f"preflight: unknown-scalar precedence violated for h "
                    f"(sample {i}: got {h_used}, expected 999.0)")
            r_used = geom_unknown[i, 0, py, px].item()
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
        "precedence_notes": precedence_notes,
    }

    # Gradient ownership.
    student_grads = sum(
        1 for p in model.parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    decoder_grads = sum(
        1 for p in model.occupancy_decoder.parameters()
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

    # Physics-alive check (audit B24). The ownership counts above come from the
    # FULL objective, so L_inv/L_var/L_cov/L_occ also satisfy them — a physics
    # path with a dead Jacobian would pass unnoticed. The ownership counts are
    # already in locals, so resetting grads inside the check cannot corrupt them.
    checks["physics_term_student_params_with_grad"] = assert_physics_reaches_student(
        model, surrogate, occ, sv, sk, S, M, cfg)

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
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override train.total_steps for a short "
                             "verification-scale run (e.g. 5-10%% of the "
                             "schedule) on real data before the full run.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or cfg["train"].get("device", "cpu")

    if args.preflight:
        result = preflight(cfg, device=device)
        print(json.dumps(result, indent=2))
        return

    report = train(cfg, resume_path=args.resume, no_train=args.no_train,
                   device=device, use_synthetic_smoke=args.use_synthetic_smoke,
                   max_steps=args.max_steps)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
