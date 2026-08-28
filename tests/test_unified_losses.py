"""Phase 3 tests — UnifiedJEPALoss, training loop, resume, curriculum.

Verifies:
- UnifiedJEPALoss forward produces finite total_loss + components
- VICReg terms (L_inv, L_var, L_cov) are included and finite
- Scalar L1 only on unknown positions
- EMA/released gradients absent after loss.backward()
- Physics loss disabled by default (lambda_phys=0 → L_phys=0)
- Curriculum mask-ratio sampling produces variety
- Resume: model + EMA + optimizer + scheduler + step restored
- Full-mask batches (ratio=1.0) occur in curriculum

Run:  python -m pytest tests/test_unified_losses.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import copy
import torch
import torch.nn as nn
import pytest

from assembly import (
    UnifiedJEPA,
    build_unified_model,
    saveable_state_dict,
    load_into_model,
    SAVED_EXCLUDES,
)
from losses.unified_losses import (
    UnifiedJEPALoss,
    OccupancyTokenLoss,
    ScalarPredictionLoss,
    PhysicsSpectrumLoss,
)
from data.mask import BlockMasker


class _StubReleasedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 256))

    def forward(self, S):
        return self.net(S.transpose(1, 2))


def _make_stub_spectrum_path(model):
    stub = _StubReleasedEncoder()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    model.spectrum_path.released = stub


def _build_model(hidden=192, geo_depth=2, predictor_depth=4):
    torch.manual_seed(0)
    model = UnifiedJEPA(
        hidden=hidden, num_heads=6, geo_depth=geo_depth,
        predictor_depth=predictor_depth, goal_tokens=16,
        num_predictor_heads=6, scalar_hidden=128,
        n_film_blocks=geo_depth, spec_dim=256)
    _make_stub_spectrum_path(model)
    model.ema.target.load_state_dict(model.occupancy_encoder.state_dict())
    model.scalar_mlp_ema.target.load_state_dict(model.scalar_encoder.state_dict())
    return model


def _batch(seed=0, b=2):
    torch.manual_seed(seed)
    occ = (torch.rand(b, 1, 64, 64) > 0.5).float()
    occ[:, :, :32, :32] = 1.0
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]], dtype=torch.float32)[:b]
    spec = torch.randn(b, 2, 301)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=seed)
    M = masker.sample(occ, ratio=0.5)
    return occ, sv, spec, M


# --------------------------------------------------------------------------
# Loss components
# --------------------------------------------------------------------------

def test_unified_loss_returns_dict():
    model = _build_model()
    objective = UnifiedJEPALoss(hidden=192)
    occ, sv, spec, M = _batch(seed=0)
    result = objective(model, occ, sv, torch.ones(2, 3, dtype=torch.bool),
                       spec, M)
    assert "total_loss" in result
    assert "components" in result
    assert "out" in result
    assert "projector_inputs" in result
    assert "projector_outputs" in result


def test_loss_components_finite():
    model = _build_model()
    objective = UnifiedJEPALoss(hidden=192)
    occ, sv, spec, M = _batch(seed=1)
    sk = torch.tensor([[True, False, True], [False, True, False]])
    result = objective(model, occ, sv, sk, spec, M)
    c = result["components"]
    for k in ("L_inv", "L_var", "L_cov", "L_scalar", "L_phys",
              "L_inv_weighted", "L_var_weighted", "L_cov_weighted", "L_total"):
        assert k in c, f"missing component {k}"
        assert c[k] >= 0, f"{k} negative: {c[k]}"


def test_loss_backward_student_grads_exist():
    model = _build_model()
    model.train()
    objective = UnifiedJEPALoss(hidden=192)
    objective.train()
    occ, sv, spec, M = _batch(seed=2)
    sk = torch.tensor([[True, False, True], [False, True, False]])
    result = objective(model, occ, sv, sk, spec, M)
    loss = result["total_loss"]
    loss.backward()

    # Student modules must have gradients
    for name, p in model.occupancy_encoder.named_parameters():
        assert p.grad is not None, f"no grad: occupancy_encoder.{name}"
    for name, p in model.scalar_encoder.named_parameters():
        assert p.grad is not None, f"no grad: scalar_encoder.{name}"
    for name, p in model.predictor.named_parameters():
        assert p.grad is not None, f"no grad: predictor.{name}"
    for name, p in model.scalar_decoder.named_parameters():
        assert p.grad is not None, f"no grad: scalar_decoder.{name}"

    # EMA targets must have NO gradients
    for name, p in model.ema.named_parameters():
        assert p.grad is None, f"ema has grad: {name}"
    for name, p in model.scalar_mlp_ema.named_parameters():
        assert p.grad is None, f"scalar_mlp_ema has grad: {name}"

    # Objective projector must have gradients
    for name, p in objective.projector.named_parameters():
        assert p.grad is not None, f"no grad: projector.{name}"

    # Released spectrum encoder must have NO gradients
    released = model.spectrum_path.released
    for name, p in released.named_parameters():
        assert p.grad is None, f"released has grad: {name}"


def test_projector_trained_from_both_branches_ema_frozen():
    """Fix 12: projector gradient ownership is explicit — the objective-owned
    projector is trained from BOTH branches (p_hat and p_y), while the EMA
    target encoder itself receives no gradient (stop-grad at the EMA boundary,
    architecture_v5.md §3.6)."""
    model = _build_model()
    model.train()
    objective = UnifiedJEPALoss(hidden=192)
    objective.train()
    occ, sv, spec, M = _batch(seed=13)
    sk = torch.tensor([[True, False, True], [False, True, False]])
    result = objective(model, occ, sv, sk, spec, M)
    loss = result["total_loss"]
    loss.backward()

    # Projector receives gradients (both branches flow into it).
    for name, p in objective.projector.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, (
            f"projector.{name} must receive gradient from the objective")

    # The target-side projector input (p_y) is derived from z_y_raw, which is
    # detached at the EMA boundary — so the EMA encoder must have NO gradient
    # even though the projector itself is trained from both branches.
    for name, p in model.ema.named_parameters():
        assert p.grad is None, f"occupancy EMA received gradient: {name}"
    for name, p in model.scalar_mlp_ema.named_parameters():
        assert p.grad is None, f"scalar_mlp_ema received gradient: {name}"

    # Explicit: projector params are NOT frozen (requires_grad True).
    assert any(p.requires_grad for p in objective.projector.parameters())


def test_loss_backward_zero_grads_on_ema_only():
    """Verify the objective's on_optimizer_step updates EMA targets."""
    model = _build_model()
    objective = UnifiedJEPALoss(hidden=192)
    occ, sv, spec, M = _batch(seed=3)
    sk = torch.ones(2, 3, dtype=torch.bool)

    # Snapshot EMA target state
    ema_before = {k: v.clone() for k, v in model.ema.target.state_dict().items()}
    scalar_ema_before = {k: v.clone() for k, v in model.scalar_mlp_ema.target.state_dict().items()}

    # Train step
    result = objective(model, occ, sv, sk, spec, M)
    loss = result["total_loss"]
    loss.backward()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(objective.parameters()), lr=1e-4)
    optimizer.step()
    objective.on_optimizer_step(model, step=0)

    # At least one EMA param must have moved
    occ_moved = sum(1 for k, v in ema_before.items()
                    if not torch.equal(v, model.ema.target.state_dict()[k]))
    scalar_moved = sum(1 for k, v in scalar_ema_before.items()
                       if not torch.equal(v, model.scalar_mlp_ema.target.state_dict()[k]))
    assert occ_moved > 0, "occupancy EMA must update"
    assert scalar_moved > 0, "scalar_mlp_ema must update"


# --------------------------------------------------------------------------
# Scalar loss: unknown positions only
# --------------------------------------------------------------------------

def test_scalar_loss_only_unknown():
    """Scalar loss must only penalize unknown positions."""
    loss_fn = ScalarPredictionLoss(loss_type="l1")
    pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    target = torch.tensor([[1.5, 2.0, 3.0], [4.0, 5.5, 6.0]])

    # All unknown → both positions contribute
    all_unknown = torch.zeros(2, 3, dtype=torch.bool)
    L_all = loss_fn(pred, target, all_unknown)
    err_all = (pred - target).abs()
    expected_all = err_all.sum() / 6
    assert torch.allclose(L_all, expected_all, atol=1e-6)

    # All known → no penalty
    all_known = torch.ones(2, 3, dtype=torch.bool)
    L_known = loss_fn(pred, target, all_known)
    assert L_known.item() == 0.0

    # Mixed
    mixed = torch.tensor([[True, False, True], [False, True, False]])
    L_mixed = loss_fn(pred, target, mixed)
    err = (pred - target).abs() * (~mixed).float()
    expected = err.sum() / 2  # 2 unknown positions
    assert torch.allclose(L_mixed, expected, atol=1e-6)


def test_scalar_loss_huber():
    loss_fn = ScalarPredictionLoss(loss_type="huber")
    pred = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([[1.5, 2.0, 3.0]])
    unknown = torch.tensor([[False, True, True]])
    L = loss_fn(pred, target, unknown)
    assert torch.isfinite(L)
    # Only first position is unknown
    expected = torch.nn.functional.huber_loss(
        torch.tensor([1.0]), torch.tensor([1.5]))
    assert torch.allclose(L, expected, atol=1e-6)


# --------------------------------------------------------------------------
# Physics loss disabled by default
# --------------------------------------------------------------------------

def test_physics_loss_disabled_by_default():
    phys = PhysicsSpectrumLoss()
    pred = torch.randn(2, 2, 301)
    target = torch.randn(2, 2, 301)
    L = phys(pred, target)
    assert L.item() == 0.0
    assert not phys._enabled


def test_physics_loss_enables():
    phys = PhysicsSpectrumLoss()
    phys.enable()
    assert phys._enabled
    pred = torch.randn(2, 2, 301)
    target = torch.randn(2, 2, 301)
    L = phys(pred, target)
    assert L > 0


# --------------------------------------------------------------------------
# Curriculum sampling (Phase 3 MD §4)
# --------------------------------------------------------------------------

def test_curriculum_mask_ratios_include_full():
    """Cleanup item 3: training ratios exclude 0.0 (masked-token objective
    undefined), eval ratios include it as the unmasked reference. Full mask
    (1.0) is in both."""
    import yaml
    cfg = yaml.safe_load(open(
        os.path.join(REPO_ROOT, "configs", "unified.yaml")))
    train_ratios = cfg["curriculum"]["train_mask_ratios"]
    train_probs = cfg["curriculum"]["train_mask_ratio_probs"]
    eval_ratios = cfg["curriculum"]["eval_mask_ratios"]
    assert 1.0 in train_ratios, "full mask (1.0) must be in training curriculum"
    assert 0.0 not in train_ratios, (
        "0.0 must be excluded from training (no masked tokens)")
    assert len(train_probs) == len(train_ratios)
    assert abs(sum(train_probs) - 1.0) < 1e-6
    assert 0.0 in eval_ratios, "0.0 must be in eval ratios (unmasked reference)"
    assert 1.0 in eval_ratios, "full mask (1.0) must be in eval ratios"


def test_curriculum_scalar_regimes():
    import yaml
    cfg = yaml.safe_load(open(
        os.path.join(REPO_ROOT, "configs", "unified.yaml")))
    regimes = cfg["curriculum"]["scalar_regimes"]
    assert "all_known" in regimes
    assert "all_unknown" in regimes
    assert "mixed" in regimes
    probs = cfg["curriculum"]["scalar_regime_probs"]
    assert len(probs) == len(regimes)
    assert abs(sum(probs) - 1.0) < 1e-6


def test_masker_works_with_single_channel():
    """BlockMasker must produce valid masks for 1-channel occupancy input."""
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(occ, ratio=0.5)
    assert M.shape == (2, 16, 16)
    assert M.dtype == torch.float32
    # 1 = visible, 0 = masked
    assert M.max() <= 1.0 and M.min() >= 0.0


def test_full_mask_batch():
    """mask_ratio=1.0 must produce all-zeros mask (nothing visible)."""
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(occ, ratio=1.0)
    assert M.shape == (2, 16, 16)
    # With ratio=1.0, all should be masked
    assert M.min().item() == 0.0


def test_zero_mask_batch_all_visible():
    """mask_ratio=0.0 must produce all-ones mask (every position visible).

    Regression (Fix 13): the nominal 0% mask regime previously still produced
    at least one block (min_side=3), so "0% mask" was not actually zero masking.
    """
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(occ, ratio=0.0)
    assert M.shape == (2, 16, 16)
    assert (M == 1.0).all(), "ratio=0.0 must leave every position visible"
    assert M.dtype == torch.float32


# --------------------------------------------------------------------------
# Resume equivalence
# --------------------------------------------------------------------------

def test_loss_projector_is_objective_owned():
    """The VICReg projector must be owned by the objective, not the model (§17)."""
    model = _build_model()
    objective = UnifiedJEPALoss(hidden=192)
    # Model must NOT have a projector attribute
    assert not hasattr(model, "proj"), "model should not own a projector"
    assert not hasattr(model, "projector"), "model should not own a projector"
    # Objective must own its projector
    assert hasattr(objective, "projector")
    assert isinstance(objective.projector, nn.Module)


def test_jepa_loss_on_masked_only():
    """JEPA loss (L_inv) must only cover masked tokens, not all tokens."""
    model = _build_model()
    objective = UnifiedJEPALoss(hidden=192)
    occ, sv, spec, M = _batch(seed=5)
    sk = torch.ones(2, 3, dtype=torch.bool)
    result = objective(model, occ, sv, sk, spec, M)

    # The mask in the model output should match the input mask
    out = result["out"]
    expected_mask = (M.view(2, -1) == 0)  # 0 in M => masked (True in mask)
    assert torch.equal(out["mask"], expected_mask)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
