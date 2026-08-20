"""Synthetic collapse / redundancy tests for the VICReg terms and the
`jepa_vicreg` objective's anti-collapse mechanism, per the Milestone B CODEX
spec (§22: the gates must fire on observable statistics, and the variance /
covariance terms must demonstrably push against collapse and redundancy).

Covers:
  - constant input: variance penalty at its max, covariance zero, total > 0
  - healthy Gaussian: variance penalty near zero, lower than the collapsed case
  - redundant features: covariance penalty responds, and minimizing it on a
    learnable matrix actually decorrelates the columns
  - projector collapse: full-rank raw input, near-zero-rank projected output,
    then the variance term recovers projected spread (rank restoration)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
import torch.nn as nn

from diagnostics.representation_health import eff_ranks
from losses.vicreg import (
    covariance_loss,
    invariance_loss,
    variance_loss,
    vicreg_branch_terms,
)


def _corr(Z):
    Zc = Z - Z.mean(0, keepdim=True)
    return (Zc[:, 0] * Zc[:, 1]).mean() / (
        Zc[:, 0].std() * Zc[:, 1].std()).clamp_min(1e-8)


def test_constant_input_variance_penalty_at_max():
    """All-identical rows -> std 0 per feature -> penalty = relu(1 - eps)^2
    ~ 1 (gamma=1, eps=1e-4 inside the sqrt); the covariance of centered
    constants is exactly zero."""
    Z = torch.ones(8, 4)
    assert variance_loss(Z) > 0.9
    assert abs(covariance_loss(Z).item()) < 1e-6


def test_identical_branches_give_zero_invariance():
    Z = torch.randn(8, 4)
    assert abs(invariance_loss(Z, Z).item()) < 1e-6


def test_collapsed_branch_keeps_total_positive():
    """A collapsed branch (constant) with an identical partner branch yields
    L_inv = 0 but L_var = 1 -> total > 0: the objective cannot go to zero on
    a collapsed representation."""
    Z = torch.ones(8, 4)
    L_inv, L_var, L_cov = vicreg_branch_terms(Z, Z)
    total = 25.0 * L_inv + 25.0 * L_var + 1.0 * L_cov
    assert L_var.item() > 0.5
    assert total.item() > 0.5


def test_healthy_gaussian_variance_penalty_near_zero():
    """std ~ 1 per feature -> relu(1 - std)^2 ~ 0 (small-sample noise only)."""
    torch.manual_seed(3)
    Z = torch.randn(512, 16)
    assert variance_loss(Z) < 0.05


def test_collapsed_penalty_dominates_healthy():
    torch.manual_seed(3)
    healthy = variance_loss(torch.randn(512, 16))
    collapsed = variance_loss(torch.ones(512, 16))
    assert collapsed > 10 * healthy


def test_covariance_penalty_responds_to_redundancy():
    """Perfectly redundant features -> nonzero off-diagonal covariance;
    independent features -> covariance penalty near zero."""
    torch.manual_seed(4)
    N = 64
    x = torch.randn(N, 1)
    redundant = torch.cat([x, 2 * x], dim=1)
    independent = torch.cat([torch.randn(N, 1), torch.randn(N, 1)], dim=1)
    assert covariance_loss(redundant) > 10 * covariance_loss(independent) + 1e-6


def test_covariance_loss_decorrelates_when_minimized():
    """Minimizing only the covariance penalty on a learnable matrix must
    actually reduce the correlation between the redundant columns."""
    torch.manual_seed(5)
    z = nn.Parameter(torch.randn(64, 2))
    with torch.no_grad():
        z.data[:, 1] = 2.0 * z.data[:, 0]   # construct redundancy
    opt = torch.optim.Adam([z], lr=1e-2)
    corr0 = _corr(z).item()
    c0 = covariance_loss(z).item()
    for _ in range(300):
        opt.zero_grad()
        covariance_loss(z).backward()
        opt.step()
    assert corr0 > 0.9, f"sanity: initial correlation must be high, got {corr0:.3f}"
    assert abs(_corr(z).item()) < 0.5, "columns must decorrelate"
    assert covariance_loss(z).item() < c0 / 4, "covariance penalty must drop"


def test_projector_collapse_detected_and_recovered():
    """Full-rank raw input through a collapsed projector output layer:
    eff_rank(P(z)) << eff_rank(z) — the diagnostic detects it — then
    minimizing the objective's variance term on the projected output restores
    projected spread (the anti-collapse mechanism actually works).

    NOTE: the construct starts from NEAR-collapse (output scaled by 1e-3), not
    exact zero: at exactly-zero output the centered values are all zero and the
    variance penalty gradient degenerates to zero by construction, while any
    real (near-)collapse has nonzero centered values and a nonzero gradient.
    """
    torch.manual_seed(6)
    from losses.objective_modules import VICRegProjector
    N, D = 64, 8
    z = torch.randn(N, D)
    raw_rank = eff_ranks(z)["eff_rank_frac"]
    assert raw_rank > 0.5, "sanity: raw input must be full rank"

    proj = VICRegProjector(input_dim=D, hidden_dim=16, output_dim=D)
    with torch.no_grad():
        # rank-1 output layer (every output column the same projection, scaled
        # to 1e-2): output is NEAR-collapsed (rank 1 in D=8 dims, tiny but
        # nonzero variance) — the physically relevant collapse, not the
        # exact-zero point where the variance gradient degenerates.
        proj.net[-1].weight.copy_(
            torch.randn(1, 16).repeat(D, 1) * 1e-2)
    p_collapsed = proj(z)
    proj_rank_collapsed = eff_ranks(p_collapsed)["eff_rank_frac"]
    assert proj_rank_collapsed < 0.2, (
        "collapsed projector must drop to ~rank-1 (1/D), got "
        f"{proj_rank_collapsed:.4f} vs raw {raw_rank:.4f}")
    v_before = variance_loss(p_collapsed).item()
    assert v_before > 0.9, "collapsed projected output must have ~0 variance"

    opt = torch.optim.Adam(proj.parameters(), lr=1e-2)
    for _ in range(600):
        opt.zero_grad()
        p = proj(z)
        (variance_loss(p) + covariance_loss(p)).backward()
        opt.step()
    p_fixed = proj(z)
    assert variance_loss(p_fixed).item() < 0.6 * v_before, (
        "variance term must restore projected spread")
    assert eff_ranks(p_fixed)["eff_rank_frac"] > 0.2, (
        "variance+covariance must lift the space away from rank-1 collapse: "
        f"got {eff_ranks(p_fixed)['eff_rank_frac']:.4f}")
