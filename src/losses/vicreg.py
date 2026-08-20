"""VICReg loss terms (Bardes et al. 2021) for the `jepa_vicreg` objective.

Two layers of API:

1. Canonical term functions used by `VICRegObjective` (src/losses/objectives.py):

       L_inv = MSE(p_hat, p_y)                                          (invariance)
       L_var = 0.5 * (var_penalty(p_hat) + var_penalty(p_y))            (variance)
       L_cov = 0.5 * (cov_penalty(p_hat) + cov_penalty(p_y))            (covariance)

   with, per branch (official VICReg forms):

       Zc   = Z - Z.mean(dim=0, keepdim=True)
       std  = sqrt(Zc.var(dim=0, unbiased=True) + eps)
       var_penalty = relu(gamma - std)^2 .mean()

       C    = (Zc^T @ Zc) / (N - 1)
       cov_penalty = off_diagonal(C)^2 .sum() / D

2. The historical single-branch helper `vicreg_loss` used by the older ladder
   rungs (jepa_var / jepa_vicreg / jepa_vicreg2) — kept for backward
   compatibility, but its silent zero-fallback for n < 2 is removed: undefined
   variance/covariance statistics now raise, per the CODEX spec's numerical
   rules ("Do not silently replace undefined statistics with zero during real
   training").

Axes: samples are ROWS (N = masked tokens across the batch), features are
COLUMNS (D). Variance/covariance are computed per feature over the samples —
the canonical VICReg sample axis, adapted to masked geometry tokens (§6 of the
CODEX spec: "VICReg statistics are computed over masked geometry tokens").
"""

import torch
import torch.nn.functional as F


def _require_n2(Z, name):
    if Z.shape[0] < 2:
        raise ValueError(
            "VICReg requires at least two valid samples for "
            f"variance/covariance statistics (got {Z.shape[0]} rows for "
            f"{name}); refusing to substitute undefined statistics with zero."
        )


def invariance_loss(p_hat, p_y):
    """L_inv = MSE(p_hat, p_y) — canonical VICReg invariance in projector space.

    p_hat/p_y: (N, D) projected masked tokens of the prediction and EMA-target
    branches. The target encoder stays frozen; the target branch's projector
    path is a separate learnable head (objective-owned), so gradient reaches
    the projector but never the EMA encoder.
    """
    _require_n2(p_hat, "p_hat")
    _require_n2(p_y, "p_y")
    return F.mse_loss(p_hat, p_y)


def variance_loss(Z, gamma=1.0, eps=1e-4):
    """Variance penalty for one branch (official VICReg form):

        std_d = sqrt(var_d + eps) over N samples, per feature d
        penalty = mean_d relu(gamma - std_d)^2
    """
    _require_n2(Z, "branch")
    Zc = Z - Z.mean(dim=0, keepdim=True)
    std = torch.sqrt(Zc.var(dim=0, unbiased=True) + eps)
    return torch.relu(gamma - std).pow(2).mean()


def covariance_loss(Z, eps=1e-4):
    """Covariance penalty for one branch (official VICReg form):

        C = (Zc^T @ Zc) / (N - 1), Zc centered per feature
        penalty = sum_{i != j} C_ij^2 / D
    """
    _require_n2(Z, "branch")
    n, d = Z.shape
    Zc = Z - Z.mean(dim=0, keepdim=True)
    C = (Zc.T @ Zc) / (n - 1)                          # (D, D) unbiased
    off = C - torch.diag_embed(torch.diagonal(C))      # exclude diagonal
    return (off ** 2).sum() / d


def vicreg_branch_terms(p_hat, p_y, gamma=1.0, eps=1e-4):
    """(L_inv, L_var, L_cov) with variance/covariance averaged over both branches
    (the official implementation applies variance pressure to both projected
    branches)."""
    L_inv = invariance_loss(p_hat, p_y)
    L_var = 0.5 * (variance_loss(p_hat, gamma, eps)
                   + variance_loss(p_y, gamma, eps))
    L_cov = 0.5 * (covariance_loss(p_hat, eps)
                   + covariance_loss(p_y, eps))
    return L_inv, L_var, L_cov


def vicreg_loss(z, gamma=1.0, cov_on=True, eps=1e-4):
    """Historical single-branch helper (prediction-only ladder rungs).

    z: (N, D) features (N = masked tokens across the batch). Returns
    (L_var, L_cov) with the same math as the canonical terms. Kept for the
    jepa_var / jepa_vicreg / jepa_vicreg2 ladder rungs; the n < 2 behavior
    changed from silent zeros to a hard error (spec numerical rules).
    """
    _require_n2(z, "features")
    n, d = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0, unbiased=True) + eps)
    var_loss = (F.relu(gamma - std) ** 2).mean()
    if not cov_on:
        cov_loss = z.new_zeros(())
    else:
        C = (zc.T @ zc) / (n - 1)                      # (D, D) unbiased
        off = C - torch.diag_embed(torch.diagonal(C))
        cov_loss = (off ** 2).sum() / d
    return var_loss, cov_loss