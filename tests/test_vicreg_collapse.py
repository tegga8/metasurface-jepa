"""Synthetic collapse / redundancy tests for the VICReg terms and the
`jepa_vicreg` objective's anti-collapse mechanism, per the Milestone B CODEX
spec (§22: the gates must fire on observable statistics, and the variance /
covariance terms must demonstrably push against collapse and redundancy).

Covers:
  - exact-math contract (V1/V2): variance_loss and covariance_loss must equal
    the canonical hand-computed formulas (hinge-mean variance, NOT squared;
    off-diagonal squared-sum / D covariance)
  - branch aggregation (V3): L_var averages the two branches (0.5 each),
    L_cov SUMS them (no accidental 0.5 factor)
  - total objective (V4): total = 25*L_inv + 25*L_var + 1*L_cov, no hidden
    normalization
  - constant input: variance penalty at its max, covariance zero, total > 0
  - collapsed target branch (V5): p_y constant cannot zero the objective
  - healthy Gaussian: variance penalty near zero, lower than the collapsed case
  - near-collapse gradient (V6): the variance hinge has a live nonzero gradient
    for small nonzero near-collapsed representations (degenerate at exact zero
    by construction — intentionally not required there)
  - redundant features: covariance penalty responds, and minimizing it on a
    learnable matrix actually decorrelates the columns
  - projector collapse (V7): full-rank raw input, near-rank-1 projected output,
    then variance+covariance recover projected spread AND decorrelate it
    (adversarial: raw full rank, final layer initialized near rank-1, only
    variance+covariance optimized; projected effective rank + variance +
    covariance before/after reported)
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
    """All-identical rows -> std = sqrt(0 + eps) = 0.01 per feature -> hinge
    penalty = relu(1 - 0.01) = 0.99 ~ 1 (gamma=1, eps=1e-4 inside the sqrt);
    the covariance of centered constants is exactly zero."""
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
    """std ~ 1 per feature -> hinge relu(1 - std) ~ 0 (small-sample noise
    only)."""
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


def test_v1_variance_matches_exact_hinge_formula():
    """Exact-math contract (V1): variance_loss must equal the canonical
    hinge mean_d relu(gamma - std_d) — NOT a squared hinge. Direct regression
    guard for the historical squared-hinge bug."""
    torch.manual_seed(11)
    Z = torch.randn(64, 12) * 0.5        # per-feature std ~0.5 -> hinge active
    gamma, eps = 1.0, 1e-4
    Zc = Z - Z.mean(0, keepdim=True)
    std = torch.sqrt(Zc.var(dim=0, unbiased=True) + eps)
    expected = torch.relu(gamma - std).mean()
    assert torch.allclose(variance_loss(Z, gamma, eps), expected, atol=1e-6)
    assert torch.allclose(
        variance_loss(Z, 0.7, eps), torch.relu(0.7 - std).mean(), atol=1e-6)
    squared = torch.relu(gamma - std).pow(2).mean()
    assert not torch.allclose(variance_loss(Z, gamma, eps), squared, atol=1e-3), (
        "squared hinge must NOT be the active form")


def test_v2_covariance_matches_exact_offdiag_formula():
    """Exact-math contract (V2): covariance_loss must equal
    sum_{i != j} C_ij^2 / D with C the unbiased per-feature covariance."""
    torch.manual_seed(12)
    Z = torch.randn(64, 10)
    Zc = Z - Z.mean(0, keepdim=True)
    C = Zc.T @ Zc / (Z.shape[0] - 1)
    off = C - torch.diag_embed(torch.diagonal(C))
    expected = (off ** 2).sum() / Z.shape[1]
    assert torch.allclose(covariance_loss(Z), expected, atol=1e-6)


def test_v3_branch_aggregation_var_averaged_cov_summed():
    """Branch aggregation (V3): L_var = 0.5 * (var_hat + var_y) and
    L_cov = cov_hat + cov_y (no accidental 0.5 on covariance)."""
    torch.manual_seed(13)
    p_hat = torch.randn(64, 8)
    p_y = torch.randn(64, 8) * 0.3
    L_inv, L_var, L_cov = vicreg_branch_terms(p_hat, p_y)
    assert torch.allclose(
        L_var, 0.5 * (variance_loss(p_hat) + variance_loss(p_y)), atol=1e-6)
    assert torch.allclose(
        L_cov, covariance_loss(p_hat) + covariance_loss(p_y), atol=1e-6)
    assert not torch.allclose(
        L_cov, 0.5 * (covariance_loss(p_hat) + covariance_loss(p_y)), atol=1e-3), (
        "an accidental 0.5 factor on the covariance aggregation must not be "
        "reintroduced")


def test_v4_total_objective_25_25_1_with_defaults():
    """Total objective (V4): with default coefficients the total must be
    exactly 25*L_inv + 25*L_var + 1*L_cov — no hidden normalization."""
    torch.manual_seed(14)
    from losses.objectives import VICRegObjective

    class _Stub(nn.Module):
        def __init__(self, B=2, T=256, D=8):
            super().__init__()
            self.ema = nn.Module()
            self.geometry_encoder = None
            self.z_hat = nn.Parameter(torch.randn(B, T, D))
            self.z_y = nn.Parameter(torch.randn(B, T, D))

        def forward(self, G, S, M):
            B = G.shape[0]
            mask = (M.view(B, -1) == 0)
            return {"z_hat": self.z_hat, "z_y_raw": self.z_y, "mask": mask}

    G = torch.randn(2, 3, 64, 64)
    S = torch.randn(2, 301)
    M = torch.ones(2, 256)
    M[:, :8] = 0                       # N = 16 masked tokens
    obj = VICRegObjective(projector_input_dim=8, projector_hidden_dim=16,
                          projector_output_dim=8)
    res = obj(_Stub(), G, S, M)
    c = res["components"]
    expected = 25.0 * c["L_inv"] + 25.0 * c["L_var"] + 1.0 * c["L_cov"]
    assert torch.allclose(res["total_loss"], expected, atol=1e-6)
    assert c["lambda_inv"] == 25.0 and c["lambda_var"] == 25.0 \
        and c["lambda_cov"] == 1.0


def test_v5_collapsed_target_branch_py_constant_keeps_objective_positive():
    """Collapsed target branch (V5): p_hat healthy, p_y a constant — the
    objective cannot go to zero on a collapsed branch (L_var > 0, total > 0)."""
    torch.manual_seed(15)
    p_hat = torch.randn(64, 8)
    p_y = torch.ones(64, 8)
    L_inv, L_var, L_cov = vicreg_branch_terms(p_hat, p_y)
    assert L_var.item() > 0.4, (
        f"collapsed branch must pay the variance hinge, got {L_var.item():.4f}")
    total = 25.0 * L_inv + 25.0 * L_var + 1.0 * L_cov
    assert total.item() > 0.4


def test_v6_near_collapse_variance_has_nonzero_gradient():
    """One-dimensional variance behavior (V6): the variance hinge must have a
    nonzero gradient for a small nonzero near-collapsed representation
    (std << gamma -> hinge active). The gradient at the mathematically exact
    zero-centered representation is degenerate by construction and is
    intentionally NOT required (documented limitation)."""
    torch.manual_seed(16)
    Z = nn.Parameter(1e-3 * torch.randn(64, 8))
    variance_loss(Z).backward()
    assert Z.grad is not None
    assert Z.grad.norm().item() > 0, (
        "near-collapsed (small nonzero) representations must have a live "
        "variance gradient")


def test_projector_collapse_detected_and_recovered():
    """Full-rank raw input through a collapsed projector output layer:
    eff_rank(P(z)) << eff_rank(z) — the diagnostic detects it — then
    minimizing the objective's variance term on the projected output restores
    projected spread (the anti-collapse mechanism actually works).

    NOTE: the construct starts from NEAR-collapse (output scaled by 1e-3), not
    exact zero: at exactly-zero output the centered values are all zero and the
    variance penalty gradient degenerates to zero by construction, while any
    real (near-)collapse has nonzero centered values and a nonzero gradient.

    PORTABILITY FIX (spec §10): the original initialization was EXACTLY
    rank-1 — every output row was the identical base vector scaled by 1e-2.
    Under that exact symmetry every row receives the identical gradient, the
    rank-1 manifold is invariant under the dynamics, and the recovery lands on
    a degenerate fixed point whose value is backend-dependent (ratio ~0.578 on
    one CPU backend, ~0.685 on another — same seed, same v_before). The fix
    breaks the artificial exact symmetry deliberately with a tiny,
    seed-controlled per-row perturbation while preserving the near-collapse
    condition: collapse detection is unchanged (rank still ~1/8, variance
    still ~1.0), but the variance+covariance objective can now genuinely
    escape the rank-1 manifold, so recovery is complete (ratio -> ~0) and
    backend-robust.
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
        # Plus a tiny deterministic per-row perturbation (1e-6) that breaks
        # the artificial exact row symmetry WITHOUT measurably weakening the
        # collapse (rank stays ~1/8, variance stays ~1.0).
        base = torch.randn(1, 16)
        row_noise = torch.randn(D, 16)  # deterministic under manual_seed(6)
        proj.net[-1].weight.copy_(
            base.repeat(D, 1) * 1e-2 + 1e-6 * row_noise)
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


def test_projector_rank1_redundant_recovery_v7():
    """Adversarial projector-collapse recovery (V7, spec §18): full-rank raw
    input through a final projection layer manually initialized NEAR rank-1
    (all output rows the same unit vector scaled to ~0.5 — variance hinge
    active AND off-diagonal covariance large). Optimizing ONLY variance +
    covariance must decrease BOTH penalties: the objective actively recovers
    from the exact projector pathology that damaged the old pipeline.

    NOTE: the initialization is NEAR rank-1 (small row asymmetry), not exact
    rank-1: with exactly identical rows every row receives the identical
    gradient, the rank-1 manifold is invariant under the dynamics, and the
    joint var+cov objective has an interior minimum along that manifold
    (dL/ds = -1 + (D-1)/2 * s^3 = 0 -> s ~ 0.66) — exact rank-1 can never be
    recovered from, which is why the physically relevant test is near-rank-1.

    Reported record (V7): projected effective rank, variance before/after,
    covariance before/after."""
    torch.manual_seed(7)
    N, D = 64, 8
    z = torch.randn(N, D)                              # full-rank raw input
    assert eff_ranks(z)["eff_rank_frac"] > 0.5, "sanity: raw input full rank"

    W = nn.Parameter(torch.randn(D, D))
    with torch.no_grad():
        w = torch.randn(1, D)
        w = w / w.norm()                               # unit row
        W.copy_(w.repeat(D, 1) * 0.5 + 1e-3 * torch.randn(D, D))  # NEAR rank-1

    def out():
        return z @ W.T

    p0 = out()
    rank_before = eff_ranks(p0)["eff_rank_frac"]
    var_before = variance_loss(p0).item()
    cov_before = covariance_loss(p0).item()
    assert rank_before < 0.2, "rank-1 rows must collapse the projected rank"
    assert var_before > 0.3, "std ~0.5 must keep the variance hinge active"
    assert cov_before > 1e-3, "rank-1 rows must make the covariance penalty large"

    opt = torch.optim.Adam([W], lr=1e-2)
    for _ in range(800):
        opt.zero_grad()
        (variance_loss(out()) + covariance_loss(out())).backward()
        opt.step()

    p1 = out()
    var_after = variance_loss(p1).item()
    cov_after = covariance_loss(p1).item()
    rank_after = eff_ranks(p1)["eff_rank_frac"]
    assert var_after < 0.6 * var_before, (
        "variance penalty must decrease under variance+covariance optimization")
    assert cov_after < 0.6 * cov_before, (
        "covariance penalty must decrease under variance+covariance "
        "optimization")
    assert rank_after > 0.5, "projected space must recover from rank-1 collapse"
