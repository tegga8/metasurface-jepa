"""Fusion / context encoder (architecture_v5.md §3.4, Phase 1 MD §6).

Small 2-layer 192-D transformer that lets occupancy tokens, spectrum goal tokens,
and the scalar-summary token attend to each other before the predictor stage.

Inputs:
    occupancy tokens   [B, 256, 192]
    goal tokens        [B, 16, 384]  (from SpectrumPath, projected to 192)
    scalar summary     [B,  1, 192]

Token layout (exact, row-major): 256 occupancy | 16 goal | 1 scalar-summary.

Depth is pinned at 2 layers per architecture_v5.md §3.4 (was open in earlier
drafts). Uses the same TransformerBlock / Attention classes from geometry_encoder.
"""

import torch
from torch import nn

from encoders.geometry_encoder import TransformerBlock

HIDDEN = 192
N_OCC_TOKENS = 256
N_GOAL_TOKENS = 16
N_SCALAR_TOKENS = 1
GOAL_DIM_IN = 384  # SpectrumPath outputs 384-D a_goal


class FusionEncoder(nn.Module):
    """192-D fusion transformer combining occupancy, goal, and scalar-summary tokens.

    Args:
        hidden:       token dimension (192).
        num_heads:    attention heads (6 → 32 dims/head).
        depth:        transformer layers (2, per §3.4).
        goal_dim_in:  input dimension of spectrum goal tokens (384, from SpectrumPath).
    """

    def __init__(self, hidden=HIDDEN, num_heads=6, depth=2,
                 goal_dim_in=GOAL_DIM_IN):
        super().__init__()
        self.hidden = hidden
        self.depth = depth
        self.n_occ = N_OCC_TOKENS
        self.n_goal = N_GOAL_TOKENS
        self.n_scalar = N_SCALAR_TOKENS
        self.total_tokens = self.n_occ + self.n_goal + self.n_scalar

        self.goal_proj = nn.Linear(goal_dim_in, hidden)

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden, num_heads) for _ in range(depth)]
        )

    def forward(self, occ_tokens, goal_tokens, scalar_summary):
        """
        Args:
            occ_tokens:     (B, 256, 192)
            goal_tokens:    (B, 16, 384)  — projected internally to (B, 16, 192)
            scalar_summary:  (B, 1, 192)

        Returns:
            (B, total_tokens, hidden) = (B, 273, 192)
        """
        goal_proj = self.goal_proj(goal_tokens)  # (B, 16, 192)
        x = torch.cat([occ_tokens, goal_proj, scalar_summary], dim=1)  # (B, 273, 192)

        for block in self.blocks:
            x = block(x)

        return x
