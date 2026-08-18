"""JEPA latent prediction loss (§4, Phase-2 slice): masked-position cosine distance.

L_J = (1/|M|) Σ_{i∈M} d(ẑ_i, z_i), d = cosine distance after projection, masked
positions only (§3.5: "for the strict masked-region JEPA loss, only masked target
tokens contribute"; §4: "cosine distance or normalized Huber/MSE after projection").
"""

import torch
import torch.nn.functional as F
from torch import nn


class ProjectionMLP(nn.Module):
    """Small learned projection head applied to both ẑ and z_y before cosine (§4's
    "after projection"). Shared between predict and target sides; the target side is
    stop-gradient (EMA output), so gradients flow only through ẑ.
    """

    def __init__(self, hidden=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        return self.net(x)


def cosine_distance(pred, target):
    """(B, 256, D) -> (B, 256) per-token cosine distance in [0, 2]."""
    p = F.normalize(pred, dim=-1)
    t = F.normalize(target, dim=-1)
    return (1.0 - (p * t).sum(dim=-1)).clamp(min=0.0)


def jepa_loss(pred, target, mask, proj=None, stop_grad_target=True):
    """mask: (B, 256) float/bool, 1 = masked (target). Mean cosine distance over masked.

    proj: optional ProjectionMLP applied to both pred and target before cosine (§4).
    stop_grad_target=True (default): the target-side projection runs under torch.no_grad()
    so gradients flow ONLY through ẑ (soft-JEPA target contract). The EMA objectives rely
    on this; LeJEPA (student-as-target, no stop-grad by design) passes False explicitly.
    """
    if proj is not None:
        pred = proj(pred)
        if stop_grad_target:
            with torch.no_grad():
                target = proj(target)
        else:
            target = proj(target)
    d = cosine_distance(pred, target)
    mask = mask.bool()
    if mask.sum() == 0:
        return d.mean(), d.mean(dim=1)
    return d[mask].mean(), d.mean(dim=1)
