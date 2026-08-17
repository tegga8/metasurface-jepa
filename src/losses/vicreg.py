"""VICReg-style variance/covariance regularization (adaptive-ladder Phase 1).

Ladder spec: L = L_J + lambda_var * L_var + lambda_cov * L_cov with

    L_var = (1/D) * sum_d max(0, gamma - std(z_d))^2      (std over samples, per feature)
    L_cov = (1/D) * sum_{i != j} C_ij^2                    (C centered, unbiased)

Applied to the PROJECTED prediction embeddings of masked tokens (B2-native space:
the JEPA loss itself lives in projection space, so the regularizer acts on the same
features). The EMA target is never touched — target embeddings remain stop-gradient
(the target side of `jepa_loss` is the frozen EMA copy).

The design doc does not specify VICReg (it specifies L_SIGReg for LeJEPA); this is
the standard VICReg form (Bardes et al. 2021) with gamma configurable (default 1.0)
and the covariance term toggleable. Reported hyperparameters: gamma, lambda_var,
lambda_cov, cov_on, features space (projected masked predictions).
"""

import torch
import torch.nn.functional as F


def vicreg_loss(z, gamma=1.0, cov_on=True, eps=1e-4):
    """z: (N, D) features (N = masked tokens across the batch). Returns (L_var, L_cov).

    Axes follow the ladder spec: features are COLUMNS (d), samples are ROWS (N);
    variance per feature over the N samples; covariance of the centered features
    with diagonal excluded and normalized by D.
    """
    n, d = z.shape
    if n < 2:
        return z.new_zeros(()), z.new_zeros(())
    zc = z - z.mean(dim=0, keepdim=True)
    std = zc.std(dim=0, unbiased=True)                    # (D,)
    var_loss = (F.relu(gamma - std) ** 2).mean()
    if not cov_on:
        cov_loss = z.new_zeros(())
    else:
        C = (zc.T @ zc) / (n - 1)                         # (D, D) unbiased covariance
        off = C - torch.diag_embed(torch.diagonal(C))     # exclude diagonal
        cov_loss = (off ** 2).sum() / d
    return var_loss, cov_loss
