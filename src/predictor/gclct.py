"""Goal-Conditioned Latent Completion Transformer (§3.5) — Milestone B dense-only slice.

Per predictor block (dense attention; sparse top-k routing and classifier-free guidance are
Milestones E/D respectively):
    LayerNorm -> AdaLN-Zero(c_physics) -> self-attention among latent structural queries
    -> cross-attention: q = queries, kv = [Z_x' (64 bottlenecked), A_goal (16 dense)]
    -> MLP -> residual

Queries: all 256 spatial positions — masked locations start from e_mask + pos (missing-state
queries), visible locations from their context features. Residual future-state prediction:
ẑ_i = base_i + Δz_i with base = z^context_i (visible) or e_mask (masked), mirroring §3.5.

Two output heads: 'latent' (the JEPA prediction, Linear 384->384) and 'pixel'
(Linear 384 -> 3*4*4 = 48, for the Baseline 2 direct masked generator, §10.1).
"""

import torch
import torch.nn.functional as F
from torch import nn

from encoders.geometry_encoder import CrossAttention, Attention


class GCLCTBlock(nn.Module):
    def __init__(self, hidden=384, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.hidden = hidden
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden, bias=True))
        self.self_attn = Attention(hidden, num_heads)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden, num_heads)
        self.norm3 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        hidden_mlp = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden_mlp), nn.GELU(approximate="tanh"),
            nn.Linear(hidden_mlp, hidden))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, kv, c, need_weights=False):
        """x: queries (B, 256, 384); kv: (B, 80, 384); c: (B, 384). Returns (x, attn_w)."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(c).chunk(6, dim=1)
        b, n, d = x.shape
        x_n = self.norm1(x)
        x_n = x_n * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)   # AdaLN-Zero
        x = x + gate_msa.unsqueeze(1) * self.self_attn(x_n)
        x_n = self.norm2(x)
        x_n = x_n * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        cross_out, w = self.cross_attn(x_n, kv, need_weights=need_weights)
        x = x + gate_mlp.unsqueeze(1) * cross_out
        x = x + self.mlp(self.norm3(x))
        return x, w


class GCLCT(nn.Module):
    def __init__(self, depth=8, hidden=384, num_heads=6, head_type="latent"):
        super().__init__()
        assert head_type in ("latent", "pixel")
        self.depth = depth
        self.hidden = hidden
        self.head_type = head_type
        self.blocks = nn.ModuleList([GCLCTBlock(hidden, num_heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        out_dim = hidden if head_type == "latent" else 3 * 4 * 4
        self.head = nn.Linear(hidden, out_dim, bias=True)

    def forward(self, queries, kv, c_physics, need_weights=False):
        """queries: (B, 256, 384); kv: (B, 64+16, 384); c_physics: (B, 384).

        Returns delta-predictions (B, 256, out_dim) and per-block cross-attention weights
        (list of (B, H, 256, 80) tensors) when need_weights=True.
        """
        x = queries
        weights = []
        for block in self.blocks:
            x, w = block(x, kv, c_physics, need_weights=need_weights)
            if need_weights:
                weights.append(w)
        return self.head(self.final_norm(x)), weights