"""Occupancy encoder (architecture_v5.md §3.1, Phase 1 MD §4).

Single-channel patch-4 ViT over occupancy M[64,64]:
    Conv2d(1, 192, k=4, s=4) -> 16x16 grid -> 256 tokens of dim 192.

Reuses TransformerBlock / get_2d_sincos_pos_embed from geometry_encoder verbatim
(grid size unchanged; only hidden dim and patch-embed input channels differ).

Scalar FiLM injection (§3.1): after each TransformerBlock, apply
    x = gamma * x + beta
where (gamma, beta) come from the ScalarEncoder's FiLM heads. FiLM heads are
zero-initialized to identity: gamma outputs 1, beta outputs 0, so the encoder
behaves as if unmodified at step 0 — matching this repo's AdaLN-zero convention.

Weight transfer: the released MetaDiT ViT blocks are 384-dim; the patch embedder
is 3-input-channel/384-output. Neither shape matches the 192-dim single-channel
encoder, so NO weight transfer is performed — initialized from scratch. The old
GeometryEncoder is a structural template only, not a compatible state dict.
"""

import numpy as np
import torch
from torch import nn

from encoders.geometry_encoder import (
    TransformerBlock,
    get_2d_sincos_pos_embed,
)

PATCH_SIZE = 4
TOKEN_GRID = 16
N_TOKENS = TOKEN_GRID * TOKEN_GRID  # 256


class OccupancyEncoder(nn.Module):
    """Patch-4 ViT over single-channel occupancy, with per-block scalar FiLM.

    Args:
        hidden:    token embedding dimension (192).
        num_heads: attention heads (6 → 32 dims/head).
        depth:     TransformerBlock count (6, matching the existing GeometryEncoder).
    """

    def __init__(self, hidden=192, num_heads=6, depth=6):
        super().__init__()
        self.patch_size = PATCH_SIZE
        self.token_grid = TOKEN_GRID
        self.hidden = hidden
        self.n_blocks = depth

        self.patch_embed = nn.Conv2d(
            1, hidden, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, bias=True
        )

        pos = get_2d_sincos_pos_embed(hidden, TOKEN_GRID)
        self.register_buffer(
            "pos_embed", torch.from_numpy(pos).float().unsqueeze(0)
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden, num_heads) for _ in range(depth)]
        )

    def forward(self, occupancy, film_params=None, mask=None, mask_token=None):
        """Encode single-channel occupancy into 256 latent tokens.

        Args:
            occupation: (B, 1, 64, 64) binary float occupancy.
            film_params: Optional list of (gamma, beta) tuples, one per block.
                Each is (B, hidden). If None, no FiLM is applied (identity).
            mask: Optional (B, 16, 16) mask (1=visible, 0=masked). If provided
                with mask_token, masked patch positions are replaced with the
                learned placeholder before pos embedding (§2 / architecture_v5.md).
            mask_token: (1, 1, hidden) learned placeholder for masked patches.

        Returns:
            (B, 256, hidden) occupancy latent tokens.
        """
        x = self.patch_embed(occupancy).flatten(2).transpose(1, 2)  # (B, 256, hidden)
        if mask is not None and mask_token is not None:
            m = (mask.view(x.shape[0], -1) == 0).unsqueeze(-1)
            x = torch.where(m, mask_token, x)
        x = x + self.pos_embed

        for i, block in enumerate(self.blocks):
            x = block(x)
            if film_params is not None:
                gamma, beta = film_params[i]
                x = gamma[:, None, :] * x + beta[:, None, :]

        return x
