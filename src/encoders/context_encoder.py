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


class PerceiverBottleneck(nn.Module):
    """64 learned queries cross-attend once into Z_x (256 tokens) -> Z_x' (64 tokens)."""

    def __init__(self, num_latents=64, hidden=384, num_heads=6):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.zeros(1, num_latents, hidden))
        nn.init.normal_(self.latents, std=0.02)
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.cross = _PerceiverCross(hidden, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.GELU(approximate="tanh"),
            nn.Linear(hidden * 4, hidden))

    def forward(self, z_x):
        b = z_x.shape[0]
        x = self.latents.expand(b, -1, -1)
        x = x + self.cross(self.norm(x), z_x)
        x = x + self.mlp(self.norm(x))
        return x


class _PerceiverCross(nn.Module):
    """Cross-attention with our Attention/CrossAttention conventions (dim 384, 6 heads)."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, kv):
        b, nq, _ = x.shape
        nk = kv.shape[1]
        q = self.q(x)
        k, v = torch.chunk(self.kv(kv), 2, dim=-1)
        q = q.reshape(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, nq, self.head_dim * self.num_heads)
        return self.proj(out)
