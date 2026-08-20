"""Context encoder (§3.2): masked geometry -> Z_x (256 tokens), plus the Perceiver-IO
bottleneck that compresses Z_x to 64 tokens before the predictor.

Known patches receive their normal patch embedding; masked locations receive a learned
MASK token plus positional embedding (never zero-fill: "this location is unknown" is
semantically distinct from "this location is physically zero"). The geometry encoder
weights are shared with the student geometry encoder; only the mask token is new.

The Perceiver bottleneck (§3.2) is a single cross-attention from 64 learned latent
queries into the full 256-token Z_x, followed by an MLP — a pure efficiency measure;
full-resolution Z_x remains available for the decoder's masked-replacement step (Milestone C).
"""

import torch
from torch import nn


class ContextEncoder(nn.Module):
    def __init__(self, geometry_encoder, hidden=384):
        super().__init__()
        self.geo = geometry_encoder
        self.hidden = hidden
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, G_c, M):
        """G_c: (B, 3, 64, 64) pixel-masked geometry; M: (B, 16, 16), 1 = visible."""
        b = G_c.shape[0]
        x = self.geo.patch_embed(G_c).flatten(2).transpose(1, 2)   # (B, 256, 384)
        mask = (M.view(b, -1) == 0).unsqueeze(-1)                  # (B, 256, 1)
        x = torch.where(mask, self.mask_token, x)
        x = x + self.geo.pos_embed
        for block in self.geo.blocks:
            x = block(x)
        return x


