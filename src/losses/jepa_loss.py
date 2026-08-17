"""JEPA latent prediction loss (§4, Phase-2 slice): masked-position cosine distance.

L_J = (1/|M|) Σ_{i∈M} d(ẑ_i, z_i), d = cosine distance, masked positions only
(§3.5: "for the strict masked-region JEPA loss, only masked target tokens contribute").
"""

import torch
import torch.nn.functional as F


def cosine_distance(pred, target):
    """(B, 256, D) -> (B, 256) per-token cosine distance in [0, 2]."""
    p = F.normalize(pred, dim=-1)
    t = F.normalize(target, dim=-1)
    return (1.0 - (p * t).sum(dim=-1)).clamp(min=0.0)


def jepa_loss(pred, target, mask):
    """mask: (B, 256) float/bool, 1 = masked (target). Mean cosine distance over masked."""
    d = cosine_distance(pred, target)
    mask = mask.bool()
    if mask.sum() == 0:
        return d.mean(), d.mean(dim=1)
    return d[mask].mean(), d.mean(dim=1)