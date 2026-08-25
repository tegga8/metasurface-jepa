"""Geometry linear probes on frozen representations (B3, calibration spec).

Measures how much per-geometry physical parameter information (MetaDiT's three
continuous channels: l_lattice, h_atom, r_atom — the dataset `parameter` column
order) is LINEARLY decodable from mean-pooled token embeddings x_geo = X.mean(dim=1).

Design constraints:
- Deterministic: closed-form ridge regression (torch.linalg.solve); the
  train/validation split is a fixed np.random.RandomState(seed) permutation.
  No torch RNG use, no optimizers, no training loops.
- No thresholds, no verdicts: returns R^2 per channel plus provenance; the
  human operator interprets them (calibration spec B3/B10).
- Same-split guarantee: callers comparing encoders (trained / released /
  random) must pass embeddings computed over the SAME geometry set in the SAME
  order; with equal N the internal split is then identical by construction.

R^2 convention: computed on the VALIDATION split against the validation-split
target mean (standard R^2). A constant prediction scores <= 0.
"""

import numpy as np
import torch


def _ridge_fit_solve(Xtr, ytr, lam):
    """Closed-form ridge: W = (X^T X + lam*I)^-1 X^T y  (X standardized columns).

    lam is interpreted relative to the number of training samples so the
    regularization strength is comparable across split sizes."""
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + (lam * Xtr.shape[0]) * torch.eye(d, dtype=Xtr.dtype)
    return torch.linalg.solve(A, Xtr.T @ ytr)


def _r2(yval, ypred):
    """Per-column R^2 on the validation split; NaN when the target variance is 0."""
    ss_res = ((yval - ypred) ** 2).sum(dim=0)
    ss_tot = ((yval - yval.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    out = []
    for res, tot in zip(ss_res.tolist(), ss_tot.tolist()):
        out.append(float("nan") if tot <= 0.0 else 1.0 - res / tot)
    return out


def geometry_linear_probes(X, params, ridge_lambda=1e-2, val_fraction=0.25,
                           seed=0, channel_names=("l_lattice", "h_atom", "r_atom")):
    """Linear-probe R^2 of physical parameters from pooled representation X.

    Inputs:
      X       : (N, T, D) token embeddings (mean-pooled internally over T) or
                (N, D) already-pooled features.
      params  : (N, 3) array/tensor of MetaDiT parameters in dataset column
                order [l_lattice, h_atom, r_atom].
      ridge_lambda : relative ridge strength (multiplied by N_train inside fit).
      val_fraction : fraction of samples held out for R^2 evaluation.
      seed    : split seed; identical across encoder comparisons by contract.

    Returns dict with per-channel R^2 ("{name}_r2"), "mean_r2" (nan-aware mean),
    and provenance ("n_train", "n_val", "ridge_lambda", "seed").
    """
    if X.ndim == 3:
        X = X.mean(dim=1)                       # x_geo = mean over token positions
    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(np.asarray(params), dtype=torch.float64)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but params has {y.shape[0]}")
    n = X.shape[0]
    n_val = max(1, int(round(val_fraction * n)))
    n_train = n - n_val
    if n_train < 2:
        raise ValueError(
            f"not enough samples for probes: n={n} (need >= 3 with "
            f"val_fraction={val_fraction})")

    rs = np.random.RandomState(seed)
    perm = rs.permutation(n)
    val_idx = torch.from_numpy(perm[:n_val])
    train_idx = torch.from_numpy(perm[n_val:])

    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]

    # standardize features with TRAIN statistics only
    mu = Xtr.mean(dim=0, keepdim=True)
    sd = Xtr.std(dim=0, unbiased=False).clamp_min(1e-12)
    Xtr_s = (Xtr - mu) / sd
    Xva_s = (Xva - mu) / sd

    W = _ridge_fit_solve(Xtr_s, ytr, ridge_lambda)
    r2s = _r2(yva, Xva_s @ W)

    result = {f"{name}_r2": r2s[i] for i, name in enumerate(channel_names)}
    finite = [v for v in r2s if v == v]
    result["mean_r2"] = sum(finite) / len(finite) if finite else float("nan")
    result.update({"n_train": int(n_train), "n_val": int(n_val),
                   "ridge_lambda": float(ridge_lambda), "seed": int(seed)})
    return result


def compare_encoders(embeddings_by_name, params, **probe_kwargs):
    """Run identical probes across named embedding sets (same geometries/order).

    embeddings_by_name: {"trained_ema": X1, "released_vit": X2, "random_init": X3}
    Returns {name: probe_result}."""
    return {name: geometry_linear_probes(X, params, **probe_kwargs)
            for name, X in embeddings_by_name.items()}
