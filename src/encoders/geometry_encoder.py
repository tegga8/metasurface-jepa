"""Geometry encoder (§3.1): patch-4 ViT over 3x64x64 geometry, 16x16 = 256 tokens.

Initialized from the released MetaDiT ViT (`metadit-small.bin`): the released DiT operates
on the same 16x16 token grid (patch 2 over the 32x32 quadrant) with hidden dim 384 and
timm-style Attention/Mlp blocks, so its transformer-block weights transfer directly.
The patch embedder is center-initialized from the released 2x2 kernel (a 4x4 kernel with
the 2x2 weights at its center); the released pos_embed is copied verbatim (same grid).
No MetaDiT component is modified — this is an initialization source only.

Block structure mirrors the released DiTBlock minus the adaLN path (plain pre-norm
self-attention + MLP), with affine-less LayerNorms (elementwise_affine=False, eps=1e-6)
to match the released weights exactly.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """(grid_size*grid_size, embed_dim) 2-D sincos positional embedding (MAE-style)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = np.arange(grid_size, dtype=np.float64)
    out_h = np.einsum("m,d->md", pos, omega)
    out_w = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate(
        [np.repeat(out_h, grid_size, axis=0), np.tile(out_w, (grid_size, 1))], axis=1)
    return emb


class Attention(nn.Module):
    """timm-style multi-head self-attention with qk-norm (matches released DiTBlock.attn)."""

    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, n, self.head_dim * self.num_heads))


class CrossAttention(nn.Module):
    """Standard cross-attention: Q from query tokens, K/V from context tokens."""

    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=True):
        super().__init__()

        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Q comes ONLY from query tokens.
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        # K and V come ONLY from context tokens.
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, kv, need_weights=False):
        """
        x  : [B, Nq, D]   query tokens
        kv : [B, Nk, D]   context tokens

        returns:
            out     : [B, Nq, D]
            weights : [B, H, Nq, Nk] if requested, else None
        """
        B, Nq, D = x.shape
        Nk = kv.shape[1]

        # --------------------------------------------------
        # Separate Q / K / V projections
        # --------------------------------------------------
        q = self.q(x)       # [B, Nq, D]
        k = self.k(kv)      # [B, Nk, D]
        v = self.v(kv)      # [B, Nk, D]

        # --------------------------------------------------
        # Split heads
        # --------------------------------------------------
        q = q.reshape(
            B, Nq, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [B,H,Nq,d]

        k = k.reshape(
            B, Nk, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [B,H,Nk,d]

        v = v.reshape(
            B, Nk, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [B,H,Nk,d]

        # --------------------------------------------------
        # Q/K normalization
        # --------------------------------------------------
        q = self.q_norm(q)
        k = self.k_norm(k)

        # --------------------------------------------------
        # Attention
        # --------------------------------------------------
        if need_weights:
            scores = (
                q @ k.transpose(-2, -1)
            ) / math.sqrt(self.head_dim)

            weights = torch.softmax(scores, dim=-1)
            out = weights @ v
        else:
            weights = None
            out = F.scaled_dot_product_attention(
                q, k, v
            )

        # --------------------------------------------------
        # Merge heads
        # --------------------------------------------------
        out = out.permute(0, 2, 1, 3).reshape(
            B, Nq, D
        )

        out = self.proj(out)

        return out, weights


class TransformerBlock(nn.Module):
    """Plain pre-norm block; parameter layout matches released DiTBlock (no adaLN)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GeometryEncoder(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, token_grid=16, hidden=384,
                 num_heads=6, depth=6):
        super().__init__()
        self.patch_size = patch_size
        self.token_grid = token_grid
        self.hidden = hidden
        self.patch_embed = nn.Conv2d(in_channels, hidden, kernel_size=patch_size,
                                     stride=patch_size, bias=True)
        pos = get_2d_sincos_pos_embed(hidden, token_grid)
        self.register_buffer("pos_embed", torch.from_numpy(pos).float().unsqueeze(0))
        self.blocks = nn.ModuleList([TransformerBlock(hidden, num_heads) for _ in range(depth)])

    def forward(self, G):
        x = self.patch_embed(G).flatten(2).transpose(1, 2)   # (B, 256, 384)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return x

    def init_from_metadit(self, metadit_state_dict, blocks_to_take=6):
        """Copy released metadit-small weights: pos_embed verbatim, blocks 0..k-1, and the
        patch embed with the released 2x2 kernel centered inside our 4x4 kernel."""
        sd = metadit_state_dict
        missing = [k for k in sd if k.startswith("blocks.") and not k.startswith(
            "blocks.0.adaLN") and k.split(".")[1].isdigit()]
        n_blocks = max(int(k.split(".")[1]) for k in missing) + 1
        assert blocks_to_take <= n_blocks, f"released ViT has {n_blocks} blocks"

        pe = sd["pos_embed"]
        assert pe.shape == self.pos_embed.shape, f"pos_embed {pe.shape} != {self.pos_embed.shape}"
        self.pos_embed.copy_(pe)

        rel = sd["x_embedder.proj.weight"]                    # (384, 3, 2, 2)
        assert rel.shape[2] == 2 and self.patch_embed.weight.shape[2] == 4
        new_w = self.patch_embed.weight.detach().clone()
        new_w[:, :, 1:3, 1:3] = rel
        self.patch_embed.weight.data.copy_(new_w)
        self.patch_embed.bias.data.copy_(sd["x_embedder.proj.bias"])

        for i in range(blocks_to_take):
            renamed = _rename_block(sd, f"blocks.{i}")
            self.blocks[i].load_state_dict(renamed, strict=True)
        return self


def _rename_block(sd, prefix):
    """Map released block keys (attn.*, mlp.fc1/fc2) onto our block layout (mlp.0/mlp.2)."""
    out = {}
    for k, v in sd.items():
        if not k.startswith(prefix + "."):
            continue
        kk = k[len(prefix) + 1:]
        if kk.startswith("mlp."):
            if kk == "mlp.fc1.weight":
                kk = "mlp.0.weight"
            elif kk == "mlp.fc1.bias":
                kk = "mlp.0.bias"
            elif kk == "mlp.fc2.weight":
                kk = "mlp.2.weight"
            elif kk == "mlp.fc2.bias":
                kk = "mlp.2.bias"
        if kk.startswith("adaLN"):
            continue
        out[kk] = v
    return out
