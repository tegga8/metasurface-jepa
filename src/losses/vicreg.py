"""VICReg loss terms (Bardes et al. 2021) for the `jepa_vicreg` objective.

Canonical term functions used by `VICRegObjective` (src/losses/objectives.py):

    L_inv = MSE(p_hat, p_y)                                          (invariance)
    L_var = 0.5 * (var_penalty(p_hat) + var_penalty(p_y))            (variance)
    L_cov = cov_penalty(p_hat) + cov_penalty(p_y)                    (covariance)

with, per branch (official VICReg forms):

    Zc   = Z - Z.mean(dim=0, keepdim=True)
    std  = sqrt(Zc.var(dim=0, unbiased=True) + eps)
    var_penalty = relu(gamma - std).mean()          (hinge — NOT squared; matches
                                                    the official implementation and
                                                    the paper's variance criterion
                                                    v(Z) = mean_j max(0, gamma -
                                                    sqrt(Var(z_j) + eps)))

    C    = (Zc^T @ Zc) / (N - 1)
    cov_penalty = off_diagonal(C)^2 .sum() / D

Variance is AVERAGED over the two branches (0.5 each), exactly as the official
implementation averages the two branches (`std_loss = mean(relu(1 - std_x)) / 2
+ mean(relu(1 - std_y)) / 2`). Covariance is SUMMED over the two branches (no
0.5), matching the official `cov_loss = cov_x + cov_y` and the paper's
L = lambda_inv*s + lambda_var*[v(Z)+v(Z')] + lambda_cov*[c(Z)+c(Z')].

Axes: samples are ROWS (N = masked geometry tokens across the batch), features
are COLUMNS (D). Variance/covariance are computed per feature over the samples.
This is the deliberate JEPA token-level adaptation of canonical (image-level)
VICReg: the pair construction (predictor output vs. frozen EMA target output)
and the sample unit (masked tokens) differ from the original training topology;
the mathematics of the objective terms themselves are canonical.

Numerical contract: plain MSE is defined for any N >= 1, so invariance has no
N >= 2 requirement. Variance and covariance statistics are undefined for N < 2
and RAISE (ValueError) rather than silently substituting zero, per the CODEX
spec's numerical rules ("do not silently replace undefined statistics with zero
during real training").
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

    Plain mean-squared Euclidean distance between paired projected embeddings:
    no L2 normalization, no temperature, no stop-gradient, no detach. The
    target branch flows through the shared objective-owned projector (trainable
    on both branches); the frozen EMA encoder receives no gradient.

    Valid for any N >= 1 (MSE needs no two-sample statistics); the N >= 2
    requirement applies only to variance/covariance.
    """
    return F.mse_loss(p_hat, p_y)


def variance_loss(Z, gamma=1.0, eps=1e-4):
    """Variance penalty for one branch (official VICReg form):

        std_d = sqrt(var_d + eps) over N samples, per feature d
        penalty = mean_d relu(gamma - std_d)     (hinge, NOT squared)

    Undefined for N < 2 -> raises (no silent zero substitution).
    """
    _require_n2(Z, "branch")
    Zc = Z - Z.mean(dim=0, keepdim=True)
    std = torch.sqrt(Zc.var(dim=0, unbiased=True) + eps)
    return torch.relu(gamma - std).mean()


def covariance_loss(Z, eps=1e-4):
    """Covariance penalty for one branch (official VICReg form):

        C = (Zc^T @ Zc) / (N - 1), Zc centered per feature
        penalty = sum_{i != j} C_ij^2 / D

    Undefined for N < 2 -> raises (no silent zero substitution).
    """
    _require_n2(Z, "branch")
    n, d = Z.shape
    Zc = Z - Z.mean(dim=0, keepdim=True)
    C = (Zc.T @ Zc) / (n - 1)                          # (D, D) unbiased
    off = C - torch.diag_embed(torch.diagonal(C))      # exclude diagonal
    return (off ** 2).sum() / d


def vicreg_branch_terms(p_hat, p_y, gamma=1.0, eps=1e-4):
    """(L_inv, L_var, L_cov) with canonical branch aggregation:

    - invariance: single MSE across both branches
    - variance:   0.5 * (var_penalty(p_hat) + var_penalty(p_y))  (official /2)
    - covariance: cov_penalty(p_hat) + cov_penalty(p_y)          (official sum,
                                                                  NO accidental
                                                                  0.5 factor)
    """
    L_inv = invariance_loss(p_hat, p_y)
    L_var = 0.5 * (variance_loss(p_hat, gamma, eps)
                   + variance_loss(p_y, gamma, eps))
    L_cov = (covariance_loss(p_hat, eps)
             + covariance_loss(p_y, eps))
    return L_inv, L_var, L_cov