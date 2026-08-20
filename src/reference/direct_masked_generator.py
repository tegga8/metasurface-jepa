"""REFERENCE-ONLY — direct masked generator (Baseline 2, design doc §10.1).

This is the historical direct pixel-generation baseline: masked context + spectrum
-> predicted pixels via a pixel-headed GCLCT, masked-pixel L1, with NO JEPA latent
objective anywhere. It exists for provenance and for the Milestone-B baseline
comparison / ablation table only.

It is explicitly NOT part of the active research pipeline:

  - `build_model("jepa")` never constructs it (the active assembly raises on any
    non-"jepa" variant).
  - No active training/eval script imports this module.
  - Its predictor is a self-contained copy (pixel head) so the active GCLCT carries
    no dead constructor parameters.

Per design doc §10.1, "direct masked generator" is Baseline 2 (no JEPA latent
objective), kept only because the Milestone-B comparison and Ablation B reference it.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
from torch import nn

from assembly import _JEPAForwardMixin
from data.mask import apply_mask_to_pixels
from encoders.context_encoder import ContextEncoder
from encoders.geometry_encoder import CrossAttention, Attention
from encoders.geometry_encoder import GeometryEncoder
from encoders.spectrum_encoder import ReleasedSpectrumEncoder, SpectrumPath

PIXEL_GRID = 16  # 64 / patch_size 4


def unpatchify(tokens, patch_size=4, channels=3):
    """(B, 256, channels*patch^2) -> (B, channels, 64, 64)."""
    b, n, _ = tokens.shape
    grid = int(n ** 0.5)
    x = tokens.reshape(b, grid, grid, patch_size, patch_size, channels)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(b, channels, grid * patch_size, grid * patch_size)


class _ReferencePixelGCLCTBlock(nn.Module):
    """Self-contained block for the reference generator (no FiLM conditioning:
    the baseline is unconditioned on physics by construction)."""

    def __init__(self, hidden=384, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, eps=1e-6)
        self.self_attn = Attention(hidden, num_heads)
        self.norm2 = nn.LayerNorm(hidden, eps=1e-6)
        self.cross_attn = CrossAttention(hidden, num_heads)
        self.norm3 = nn.LayerNorm(hidden, eps=1e-6)
        hidden_mlp = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden_mlp),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_mlp, hidden),
        )

    def forward(self, x, kv, need_weights=False):
        x = x + self.self_attn(self.norm1(x))
        cross_out, weights = self.cross_attn(
            self.norm2(x), kv, need_weights=need_weights)
        x = x + cross_out
        x = x + self.mlp(self.norm3(x))
        return x, weights


class _ReferencePixelGCLCT(nn.Module):
    def __init__(self, depth=8, hidden=384, num_heads=6):
        super().__init__()
        self.depth = depth
        self.hidden = hidden
        self.blocks = nn.ModuleList(
            [_ReferencePixelGCLCTBlock(hidden, num_heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(hidden, 3 * 4 * 4, bias=True)

    def forward(self, queries, kv, c_physics, need_weights=False):
        """c_physics accepted for API parity with the active GCLCT but NOT used
        (the baseline is unconditioned on physics by design)."""
        x = queries
        weights = []
        for block in self.blocks:
            x, w = block(x, kv, need_weights=need_weights)
            if need_weights:
                weights.append(w)
        return self.head(self.final_norm(x)), weights


class DirectMaskedGenerator(_JEPAForwardMixin, nn.Module):
    """Baseline 2 (§10.1): G_c + S -> Ĝ pixels, masked-pixel L1, no JEPA objective."""

    def __init__(self, hidden=384, num_heads=6, geo_depth=6, predictor_depth=8,
                 goal_tokens=16, num_predictor_heads=6):
        super().__init__()
        self.hidden = hidden
        geo = GeometryEncoder(hidden=hidden, num_heads=num_heads, depth=geo_depth)
        self.context_encoder = ContextEncoder(geo, hidden=hidden)
        self.spectrum_path = SpectrumPath(None, hidden=hidden, goal_tokens=goal_tokens)
        self.predictor = _ReferencePixelGCLCT(depth=predictor_depth, hidden=hidden,
                                              num_heads=num_predictor_heads)
        self.geometry_encoder = geo

    def forward(self, G, S, M, goal_mode="real", need_attn=False):
        _, delta, z_x, mask, weights = self._encode(G, S, M, goal_mode, need_attn)
        g_hat = unpatchify(delta)
        return dict(g_hat=g_hat, z_latent=delta, mask=mask, attn_weights=weights)

    def loss(self, G, S, M, goal_mode="real"):
        out = self.forward(G, S, M, goal_mode=goal_mode)
        pmask = M.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2)
        ub = (pmask == 0).unsqueeze(1)
        ub = ub.expand_as(out["g_hat"])
        diff = (out["g_hat"] - G).abs()
        masked = diff[ub]
        full = diff.mean()
        return (masked.mean() if masked.numel() else full), out


def build_reference_direct_generator(cfg, spec_weights, device="cpu",
                                    init_from_metadit=True, metadit_weights=None):
    """Construct the reference baseline. Mirrors the old assembly.build_model 'direct'
    path. Kept out of the active module so the reference cannot be imported by
    production training by accident."""
    model = DirectMaskedGenerator(
        hidden=cfg.get("hidden", 384),
        num_heads=cfg.get("num_heads", 6),
        geo_depth=cfg.get("geo_depth", 6),
        predictor_depth=cfg.get("predictor_depth", 8),
        goal_tokens=cfg.get("goal_tokens", 16),
        num_predictor_heads=cfg.get("num_predictor_heads", 6),
    )
    from assembly import init_geometry_from_metadit, set_spectrum_path
    set_spectrum_path(model, spec_weights, device)
    if init_from_metadit:
        assert metadit_weights is not None
        init_geometry_from_metadit(model, metadit_weights,
                                   blocks_to_take=cfg.get("geo_depth", 6))
    model.to(device)
    return model