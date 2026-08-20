"""Context encoder (§3.2): masked geometry -> Z_x, full-resolution 256 tokens.

Known patches receive their normal patch embedding; masked locations receive a learned
MASK token plus positional embedding (never zero-fill: "this location is unknown" is
semantically distinct from "this location is physically zero"). The geometry encoder
weights are shared with the student geometry encoder; only the mask token is new.

There is NO Perceiver bottleneck in the active path: the predictor receives all 256
full-resolution context tokens (see `src/predictor/gclct.py`). The 64-token
bottleneck described in design doc §3.2 was deliberately not built — the active
architecture keeps 256 context tokens + 16 structured physics/goal tokens + a global
physics condition, per the verified base architecture.
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


