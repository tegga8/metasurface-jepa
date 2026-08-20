"""Modules owned by the VICReg objective (candidate `jepa_vicreg`).

The VICReg projector lives here because the original VICReg implementation
(Bardes et al. 2021) has a dedicated learned projector, and in this project the
projector belongs to the objective — never to `GoalConditionedJEPA` (there is
no `model.proj` in the corrected architecture, and the spec forbids creating
one). The projector maps the raw 384-D latent into the VICReg objective space;
the raw 384-D latent remains the representation the project ultimately cares
about and stays directly measurable.

Architecture (exact per CODEX spec §5):

    raw 384-D latent
         |
         v
    VICReg objective projector   (Linear + BN + ReLU) x2 + Linear
         |
         v
    VICReg objective space (p_hat / p_y)

Forbidden anywhere in this phase: a latent bottleneck before the projector,
a shared global `model.proj`, a projector in the raw representation path, or
using the projected representation as the scientific representation.
"""

import torch.nn as nn


class VICRegProjector(nn.Module):
    """Learned projection head of the VICReg objective (official VICReg layout:
    MLP with BatchNorm between hidden layers; final layer has no bias).

    Operates on the last dim of any leading shape (B, T, D) or (N, D):
    BatchNorm statistics are computed over the flattened token/sample axis —
    the same convention the objective's masked-token statistics use.
    """

    def __init__(self, input_dim=384, hidden_dim=384, output_dim=384):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, z):
        original_shape = z.shape
        z = z.reshape(-1, original_shape[-1])
        z = self.net(z)
        return z.reshape(*original_shape[:-1], -1)