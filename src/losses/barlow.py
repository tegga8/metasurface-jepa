"""Barlow Twins-style redundancy reduction (adaptive-ladder Phase 3 rung `jepa_barlow`).

Barlow Twins (Zbinden et al., 2021) cross-correlation loss between the two
projected branches of a joint-embedding pair:

    C_ij = sum_b z_p[b,i] * z_t[b,j] / n          (both branches standardized)
    L    = mean_i (1 - C_ii)^2  +  alpha * mean_{i != j} C_ij^2

    diagonal    : makes matching prediction/target dimensions agree
    off-diagonal: makes representation dimensions less redundant

Scaling fix (final pre-training pass, 2026-08-18): the components are MEANS over
the diagonal / off-diagonal entries, not raw sums — the raw-sum form scaled ~D
(D=384 produced L_BT ≈ 200 vs L_J ≈ 0.05 under lambda_bt=1.0, i.e. a scale that
depends on the projection width rather than the actual redundancy). Mean-form
terms are O(1) per entry, so L_BT no longer blows up with D.

This is specifically relevant to the observed Milestone-B pathology: cross-sample
cosine ≈ 1, effective rank very low, dominant eigenvalue large — i.e.
dimensional/redundancy collapse. Barlow targets the redundancy directly through
the cross-correlation off-diagonal.

Gradient boundary (tested): the JEPA cosine path inside jepa_loss() stays
detached (Bug #1 contract); the Barlow term operates on a SEPARATE projection
path where the projector may be trained from the (frozen) EMA target's output.
"""

import torch


def barlow_twins_loss(z_p, z_t, alpha=0.005, eps=1e-6):
    n, d = z_p.shape

    if n < 2:
        return (
            z_p.new_zeros(()),
            {
                "diag_term": 0.0,
                "off_diag_term": 0.0,
                "alpha": alpha,
            },
        )

    zp = (
        z_p - z_p.mean(dim=0)
    ) / z_p.std(dim=0, unbiased=True).clamp_min(eps)

    zt = (
        z_t - z_t.mean(dim=0)
    ) / z_t.std(dim=0, unbiased=True).clamp_min(eps)

    C = (zp.T @ zt) / n

    diag = torch.diagonal(C)
    # Mean-form diagonal: (1 - C_ii)^2 averaged over the D dimensions, so the
    # term is O(1) per dimension and independent of D (was sum -> ~D scale).
    diag_term = ((1.0 - diag) ** 2).mean()

    off = C - torch.diag_embed(diag)
    # Mean-form off-diagonal: sum of squared off-diagonal entries divided by the
    # number of off-diagonal entries D*(D-1). D=1 has no off-diagonal entries.
    if d > 1:
        off_diag_term = (off ** 2).sum() / (d * (d - 1))
    else:
        off_diag_term = z_p.new_zeros(())

    loss = diag_term + alpha * off_diag_term

    return (
        loss,
        {
            "diag_term": diag_term.item(),
            "off_diag_term": off_diag_term.item(),
            "alpha": alpha,
        },
    )
