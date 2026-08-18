"""Ladder-extension regression tests (Milestone B final anti-collapse screening,
directive §17): registry, Barlow loss/objective, and the §17-required checks.

The full dual-VICReg boundary suite (L_J stop-grad inside jepa_vicreg2 + target
regularization path feeding the projector) lives in
tests/test_jepa_vicreg2_objective.py; this file adds the Barlow rung and the
cumulative six-rung registry, plus lightweight presence checks for the already-
tested dual VICReg components.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
import torch.nn as nn

from losses.barlow import barlow_twins_loss
from losses.jepa_loss import ProjectionMLP, jepa_loss
from losses.objectives import (
    OBJECTIVES,
    JEPABarlowObjective,
    JEPAVICRegDualObjective,
    LeJEPAObjective,
)


class _EmaStub:
    def update(self, encoder, step):
        pass


class _ParamModel(nn.Module):
    """Deterministic objective-level stand-in (same pattern as the vicreg2
    tests): z_hat/z_y are learnable Parameters so gradient flow from any path
    is observable and reproducible across objective calls."""

    def __init__(self, hidden=8, B=2, T=256):
        super().__init__()
        self.proj = ProjectionMLP(hidden=hidden)
        self.ema = _EmaStub()
        self.geometry_encoder = None
        self.z_hat = nn.Parameter(torch.randn(B, T, hidden))
        self.z_y = nn.Parameter(torch.randn(B, T, hidden))

    def forward(self, G, S, M):
        B = G.shape[0]
        mask = (M.view(B, -1) == 0)
        return {"z_hat": self.z_hat, "z_y": self.z_y, "mask": mask}


class _FrozenEmaModel(nn.Module):
    """z_y comes from a FROZEN submodule (mirroring the real EMA target
    encoder): backwarding the total loss must never produce gradients on its
    parameters — the milestone's "EMA target encoder stays frozen" boundary."""

    def __init__(self, hidden=8, T=256, grid=64):
        super().__init__()
        self.proj = ProjectionMLP(hidden=hidden)
        self.ema = _EmaStub()
        self.geometry_encoder = None
        self.z_hat = nn.Parameter(torch.randn(2, T, hidden))
        self.target = nn.Linear(3 * grid * grid, T * hidden)
        for p in self.target.parameters():
            p.requires_grad_(False)

    def forward(self, G, S, M):
        B = G.shape[0]
        mask = (M.view(B, -1) == 0)
        T, H = self.z_hat.shape[1], self.z_hat.shape[2]
        z_y = self.target(G.flatten(1)).view(B, T, H)
        return {"z_hat": self.z_hat, "z_y": z_y, "mask": mask}


@pytest.fixture
def param_model():
    torch.manual_seed(0)
    return _ParamModel(hidden=8)


@pytest.fixture
def tiny_batch():
    torch.manual_seed(1)
    G = torch.randn(2, 3, 64, 64)
    S = torch.randn(2, 301)
    M = torch.ones(2, 256)
    M[:, :2] = 0  # 2 masked tokens per sample -> N=4 features per branch
    return G, S, M


# --------------------------------------------------------------------------
# Registry (§10: six rungs, no more)
# --------------------------------------------------------------------------

def test_ladder_registry_six_rungs():
    expected = ("jepa", "jepa_var", "jepa_vicreg",
                "jepa_vicreg2", "jepa_barlow", "lejepa")
    assert set(OBJECTIVES.keys()) == set(expected), (
        f"registry must be exactly the six screening rungs, got {sorted(OBJECTIVES)}")
    assert OBJECTIVES["jepa_vicreg2"] is JEPAVICRegDualObjective
    assert OBJECTIVES["jepa_barlow"] is JEPABarlowObjective
    assert OBJECTIVES["lejepa"] is LeJEPAObjective


# --------------------------------------------------------------------------
# Barlow loss (§17: finite gradients, n<2 safe, diag/off-diag behavior)
# --------------------------------------------------------------------------

def test_barlow_finite_gradients():
    torch.manual_seed(0)
    z_p = torch.randn(64, 8, requires_grad=True)
    z_t = torch.randn(64, 8, requires_grad=True)
    loss, info = barlow_twins_loss(z_p, z_t)
    loss.backward()
    assert torch.isfinite(z_p.grad).all()
    assert torch.isfinite(z_t.grad).all()
    assert torch.isfinite(loss)
    assert info["diag_term"] >= 0.0 and info["off_diag_term"] >= 0.0


def test_barlow_n_lt_2_safe():
    z_p = torch.randn(1, 8)
    z_t = torch.randn(1, 8)
    loss, info = barlow_twins_loss(z_p, z_t)
    assert loss.item() == 0.0
    assert info["diag_term"] == 0.0 and info["off_diag_term"] == 0.0
    # and the degenerate n=0 row-count case does not crash
    loss0, info0 = barlow_twins_loss(z_p[:0], z_t[:0])
    assert loss0.item() == 0.0 and info0["off_diag_term"] == 0.0


def test_barlow_identical_branches_low_diag_error():
    # z_p == z_t -> standardized columns correlate at ~1 on the diagonal,
    # so the diagonal term must be small (matching dimensions agree).
    torch.manual_seed(0)
    n, d = 64, 8
    z = torch.randn(n, d)
    loss, info = barlow_twins_loss(z, z)
    assert info["diag_term"] < 1e-2, info["diag_term"]


def test_barlow_off_diag_detects_correlated_columns():
    # Redundant (perfectly dependent) columns must push the off-diagonal term
    # up: this is the redundancy-reduction signal the ladder is screening for.
    torch.manual_seed(0)
    n, d = 128, 4
    z = torch.randn(n, d)
    z = z - z.mean(dim=0)
    z[:, 1] = z[:, 0]
    z[:, 3] = z[:, 2]
    loss, info = barlow_twins_loss(z, z)
    assert info["off_diag_term"] > 1e-3, info["off_diag_term"]
    assert info["diag_term"] < 1e-2, info["diag_term"]  # correlation is on off-diag


def test_barlow_alpha_zero_removes_off_diag_contribution():
    torch.manual_seed(0)
    z_p = torch.randn(64, 8)
    z_t = torch.randn(64, 8)
    loss, info = barlow_twins_loss(z_p, z_t, alpha=0.0)
    assert loss.item() == pytest.approx(info["diag_term"])


# --------------------------------------------------------------------------
# Barlow scaling fix (final pre-training pass): mean-form components must
# not scale with D — old raw-sum form gave L_BT ≈ 200 at D=384 vs L_J ≈ 0.05
# --------------------------------------------------------------------------

def test_barlow_normalization_formula():
    """The diagonal term must be mean((1 - diag)^2) and the off-diagonal term
    must be sum(off^2) / (d * (d - 1)) — the exact FIX A formulas, verified by
    recomputing both means from the raw cross-correlation matrix."""
    torch.manual_seed(0)
    n, d = 64, 8
    z_p = torch.randn(n, d, requires_grad=True)
    z_t = torch.randn(n, d)
    loss, info = barlow_twins_loss(z_p, z_t)

    zp = (z_p - z_p.mean(dim=0)) / z_p.std(dim=0, unbiased=True).clamp_min(1e-6)
    zt = (z_t - z_t.mean(dim=0)) / z_t.std(dim=0, unbiased=True).clamp_min(1e-6)
    C = (zp.T @ zt) / n
    diag = torch.diagonal(C)
    # mirror the implementation's exact op order (float32 division, not .item()
    # float64 division) so the comparison is bit-level
    exp_diag = ((1.0 - diag) ** 2).mean().item()
    exp_off = (((C - torch.diag_embed(diag)) ** 2).sum() / (d * (d - 1))).item()
    assert info["diag_term"] == pytest.approx(exp_diag, abs=1e-12)
    assert info["off_diag_term"] == pytest.approx(exp_off, abs=1e-12)
    # float32 tensor math vs float64 recomposition: allow float32 ulp differences
    assert loss.item() == pytest.approx(info["diag_term"] + info["alpha"] * info["off_diag_term"],
                                        abs=1e-5)


def test_barlow_scale_independent_of_dimension():
    """No linear blow-up with D: D=384 must NOT cost ~12x D=32. Both terms are
    per-entry means, so both are O(1/n) regardless of D (old raw-sum form:
    D=384 ~150x D=32 on this synthetic)."""
    torch.manual_seed(0)
    n = 512
    z = torch.randn(n, 384)
    loss_384, _ = barlow_twins_loss(z, z)
    loss_32, _ = barlow_twins_loss(z[:, :32], z[:, :32])
    # identical branches -> diag term ~0 in both; off-diag mean ~1/n in both.
    assert loss_384.item() < 0.05, (
        f"D=384 loss must stay O(1/n), got {loss_384.item():.4f} "
        "(raw-sum form gave ~1.5 on this synthetic)")
    assert loss_384.item() < 5.0 * loss_32.item() + 1e-3, (
        f"no linear blow-up with D: {loss_384.item():.4f} vs {loss_32.item():.4f}")


def test_barlow_off_diag_d1_guard():
    """d=1 has no off-diagonal entries: the term must be a 0 tensor, and the
    loss must equal the diagonal term alone."""
    torch.manual_seed(0)
    z_p = torch.randn(64, 1)
    z_t = torch.randn(64, 1)
    loss, info = barlow_twins_loss(z_p, z_t)
    assert info["off_diag_term"] == 0.0
    assert loss.item() == pytest.approx(info["diag_term"])


# --------------------------------------------------------------------------
# Barlow objective (§17: components present, finite backward, boundary)
# --------------------------------------------------------------------------

def test_jepa_barlow_components_present(param_model, tiny_batch):
    obj = JEPABarlowObjective()
    result = obj(param_model, *tiny_batch)
    c = result["components"]
    for k in ("L_J", "L_BT", "lambda_bt", "bt_diag", "bt_off_diag"):
        assert k in c, f"missing component {k}"
    assert torch.isfinite(result["total_loss"])
    assert torch.isfinite(c["L_J"]) and torch.isfinite(c["L_BT"])


def test_jepa_barlow_total_backward_finite(param_model, tiny_batch):
    obj = JEPABarlowObjective(lambda_bt=1.0, alpha=0.005)
    result = obj(param_model, *tiny_batch)
    result["total_loss"].backward()
    for p in param_model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_jepa_barlow_cosine_path_keeps_bug1_stopgrad_boundary(param_model, tiny_batch):
    """Boundary 1 (original JEPA cosine path): the L_J term inside jepa_barlow
    must produce exactly the projection-head gradient a detached-target
    reference produces — the Bug #1 contract is not weakened by the new
    objective."""
    G, S, M = tiny_batch
    obj = JEPABarlowObjective()
    result = obj(param_model, G, S, M)
    mask = result["out"]["mask"]

    param_model.proj.zero_grad()
    result["components"]["L_J"].backward()
    g_inside = param_model.proj.net[0].weight.grad.clone()
    param_model.proj.zero_grad()

    L_ref, _ = jepa_loss(
        param_model.z_hat, param_model.z_y.detach(), mask, proj=param_model.proj)
    L_ref.backward()
    g_ref = param_model.proj.net[0].weight.grad.clone()

    assert torch.allclose(g_inside, g_ref, atol=1e-6), (
        "L_J term inside jepa_barlow must match a detached-target reference: "
        f"max diff {((g_inside - g_ref).abs().max().item()):.2e}")


def test_jepa_barlow_target_path_feeds_projector(param_model, tiny_batch):
    """Boundary 2 (separate Barlow regularization path): backwarding the Barlow
    term alone must push a nonzero gradient into the shared projection head and
    through the EMA-target side (z_y graph). The target encoder is frozen in the
    real model — this is exactly the projector-only training path the design
    intends."""
    G, S, M = tiny_batch
    obj = JEPABarlowObjective(lambda_bt=1.0)
    comps = obj(param_model, G, S, M)["components"]

    param_model.proj.zero_grad()
    param_model.z_y.grad = None
    (comps["lambda_bt"] * comps["L_BT"]).backward()
    g = param_model.proj.net[0].weight.grad
    assert g is not None and g.abs().sum() > 0, (
        "Barlow path must contribute a nonzero gradient to the projector")
    assert torch.isfinite(g).all()
    assert param_model.z_y.grad is not None, (
        "target side must be attached to the Barlow graph (frozen in the real "
        "model; projector is the only trainable recipient)")


# --------------------------------------------------------------------------
# Dual VICReg (§17 presence checks; full boundary suite in
# tests/test_jepa_vicreg2_objective.py)
# --------------------------------------------------------------------------

def test_jepa_barlow_ema_target_encoder_stays_frozen(tiny_batch):
    """Boundary 3 (final pre-training pass): the EMA target ENCODER must stay
    frozen — no gradient may ever reach its parameters through the total loss,
    even though the separate Barlow path trains the shared projector."""
    model = _FrozenEmaModel()
    obj = JEPABarlowObjective(lambda_bt=1.0)
    result = obj(model, *tiny_batch)
    result["total_loss"].backward()
    for p in model.target.parameters():
        assert p.grad is None, "frozen EMA target encoder must receive no gradient"
    g = model.proj.net[0].weight.grad
    assert g is not None and g.abs().sum() > 0, (
        "projector must still be trained through the separate Barlow path")


def test_dual_vicreg_components_present_and_finite(param_model, tiny_batch):
    obj = JEPAVICRegDualObjective()
    result = obj(param_model, *tiny_batch)
    c = result["components"]
    for k in ("L_J", "L_var_pred", "L_var_target", "L_cov_pred",
              "L_cov_target", "lambda_var", "lambda_cov"):
        assert k in c, f"missing component {k}"
    assert torch.isfinite(result["total_loss"])