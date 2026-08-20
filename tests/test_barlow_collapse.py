"""Synthetic collapse / redundancy tests for the Barlow objective (§21).

Barlow targets dimensional/redundancy collapse directly: the off-diagonal
cross-correlation penalty decorrelates features and the diagonal matching keeps
matching prediction/target dimensions aligned. These tests verify the math on
hand-constructed inputs (constant, identical-branch, redundant-feature,
projector-collapse) without any model code.

Run:  python tests/test_barlow_collapse.py   (pytest-collectable)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
import torch.nn as nn

from diagnostics.representation_health import eff_ranks
from losses.barlow import barlow_twins_loss
from losses.objective_modules import BarlowProjector


def _corr(Z):
    Zc = Z - Z.mean(0, keepdim=True)
    return (Zc[:, 0] * Zc[:, 1]).mean() / (
        Zc[:, 0].std() * Zc[:, 1].std()).clamp_min(1e-8)


def test_constant_input_is_penalized():
    """Constant (delta) features: standardized rows are zero -> C = 0 -> the
    diagonal term (1 - C_ii)^2 = 1 per dimension -> L_BT ~ 1, i.e. collapse
    cannot hide from Barlow."""
    Z = torch.ones(64, 8)
    loss, info = barlow_twins_loss(Z, Z, alpha=0.005)
    assert loss.item() > 0.9, f"collapsed input must be penalized, got {loss.item()}"
    assert info["diag_term"] > 0.9


def test_identical_branches_give_near_zero_loss():
    """z_p == z_t with healthy spread: standardized branches match -> C ~ I ->
    diag ~ 0, off-diag ~ 0 -> L_BT ~ 0 (this is the desired non-collapsed state)."""
    torch.manual_seed(3)
    Z = torch.randn(512, 16)
    loss, info = barlow_twins_loss(Z, Z.clone(), alpha=0.005)
    assert loss.item() < 0.05, f"healthy identical branches must be ~0, got {loss.item()}"
    assert abs(info["off_diag_term"]) < 0.05


def test_off_diagonal_responds_to_redundancy():
    """Perfectly redundant features -> nonzero off-diagonal C; independent
    features -> off-diagonal near zero."""
    torch.manual_seed(4)
    N = 64
    x = torch.randn(N, 1)
    redundant = torch.cat([x, 2 * x], dim=1)
    independent = torch.cat([torch.randn(N, 1), torch.randn(N, 1)], dim=1)
    _, red = barlow_twins_loss(redundant, redundant.clone(), alpha=0.005)
    _, ind = barlow_twins_loss(independent, independent.clone(), alpha=0.005)
    assert red["off_diag_term"] > 10 * ind["off_diag_term"] + 1e-6, (
        "redundant features must produce a much larger off-diagonal term")


def test_barlow_loss_decorrelates_when_minimized():
    """Minimizing only the off-diagonal term on a learnable matrix must actually
    decorrelate the redundant columns."""
    torch.manual_seed(5)
    z = nn.Parameter(torch.randn(64, 2))
    with torch.no_grad():
        z.data[:, 1] = 2.0 * z.data[:, 0]
    opt = torch.optim.Adam([z], lr=1e-2)
    corr0 = _corr(z).item()
    _, info0 = barlow_twins_loss(z.detach(), z.detach(), alpha=1.0)
    for _ in range(400):
        opt.zero_grad()
        loss, _ = barlow_twins_loss(z, z, alpha=1.0)
        loss.backward()
        opt.step()
    assert corr0 > 0.9, f"sanity: initial correlation must be high, got {corr0:.3f}"
    assert abs(_corr(z).item()) < 0.5, "columns must decorrelate"
    _, info1 = barlow_twins_loss(z.detach(), z.detach(), alpha=1.0)
    assert info1["off_diag_term"] < info0["off_diag_term"] / 4, \
        "off-diagonal penalty must drop as columns decorrelate"


def test_projector_collapse_penalized_and_recoverable():
    """Barlow targets REDUNDANCY collapse (its off-diagonal mechanism): a rank-1
    projector output layer makes every output column the same linear function of
    the input, so the standardized cross-correlation off-diagonal is ~1 — the
    term Barlow exists to penalize. Minimizing Barlow on the learnable projector
    must decorrelate the output dimensions (anti-collapse mechanism works).

    NOTE: identical-branch absolute-spread collapse (tiny but uncorrelated
    output) is NOT Barlow's job — that is VICReg's variance term — so this test
    uses the redundancy form the off-diagonal is designed to catch."""
    torch.manual_seed(6)
    N, D = 64, 8
    z = torch.randn(N, D)
    raw_rank = eff_ranks(z)["eff_rank_frac"]
    assert raw_rank > 0.5, "sanity: raw input must be full rank"

    proj = BarlowProjector(input_dim=D, hidden_dim=16, output_dim=D)
    with torch.no_grad():
        # Rank-1 output layer: every column is the same projection of the hidden
        # activation -> output columns are identical -> cross-correlation ~ 1.
        proj.net[-1].weight.copy_(torch.randn(1, 16).repeat(D, 1))
    p_collapsed = proj(z)
    proj_rank_collapsed = eff_ranks(p_collapsed)["eff_rank_frac"]
    assert proj_rank_collapsed < 0.2, (
        f"rank-1 output layer must collapse the projector, got {proj_rank_collapsed:.4f}")
    # Redundancy is detected by the off-diagonal (alpha scales it to the loss).
    loss_hi, info_hi = barlow_twins_loss(p_collapsed.detach(), p_collapsed.detach(),
                                         alpha=1.0)
    assert info_hi["off_diag_term"] > 0.5, (
        f"redundant output must have ~1 off-diagonal, got {info_hi['off_diag_term']}")
    assert loss_hi.item() > 0.5, (
        f"redundant output must be penalized at alpha=1, got {loss_hi.item()}")

    opt = torch.optim.Adam(proj.parameters(), lr=1e-2)
    for _ in range(600):
        opt.zero_grad()
        p = proj(z)
        loss, _ = barlow_twins_loss(p, p, alpha=1.0)
        loss.backward()
        opt.step()
    p_fixed = proj(z)
    loss_lo, info_lo = barlow_twins_loss(p_fixed.detach(), p_fixed.detach(), alpha=1.0)
    assert info_lo["off_diag_term"] < info_hi["off_diag_term"] / 2, (
        "Barlow must decorrelate the output dimensions")
    assert loss_lo.item() < 0.6 * loss_hi.item(), (
        "Barlow loss must drop as the output decorrelates")


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