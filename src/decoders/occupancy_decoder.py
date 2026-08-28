"""Occupancy decoder (architecture_v5.md §4.1).

Decodes predicted occupancy tokens into occupancy LOGITS only:

    z [B,256,192]
      → reshape [B,192,16,16]   (row-major, matches patch_embed flatten order)
      → conv/upsample blocks, FiLM-conditioned by effective (l,h,r) at every
        conditioned layer
      → occupancy logits [B,1,64,64]

The three physical parameters (l_lattice, h_atom, r_atom) are represented by
ScalarDecoder, NOT by this decoder — the only spatial quantity decoded here is
occupancy. The [B,3,64,64] MetaDiT broadcast tensor is assembled exactly once,
at the physics-surrogate boundary (assemble_metadit_geometry), never here.

Decode-time FiLM rule (architecture_v5.md §4.1, fixed): conditioning uses
exactly whatever the step's scalar-masking curriculum determined — the true
value when the scalar is known, the predicted value when unknown. Training and
inference use the identical rule. Callers compute the effective scalars
(torch.where(scalar_known, scalar_values, scalar_pred)) and pass them here.
"""

import torch
from torch import nn


class _FilmBlock(nn.Module):
    """Conv + GroupNorm + GELU + upsample, with scalar FiLM after normalization.

    FiLM heads are zero-initialized so the block starts as an identity
    modulation: gamma outputs 1, beta outputs 0 (AdaLN-zero convention).
    """

    def __init__(self, in_ch, out_ch, scalar_hidden, scale_factor=2,
                 groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_ch)
        self.act = nn.GELU()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.film = nn.Linear(scalar_hidden, 2 * out_ch)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        with torch.no_grad():
            self.film.bias[:out_ch].fill_(1.0)  # gamma portion → 1

    def forward(self, x, s):
        h = self.conv(x)
        h = self.norm(h)
        gamma, beta = self.film(s).chunk(2, dim=-1)
        gamma = gamma[..., None, None]
        beta = beta[..., None, None]
        h = gamma * h + beta
        h = self.act(h)
        return self.up(h)


class OccupancyDecoder(nn.Module):
    """FiLM-conditioned occupancy decoder (architecture_v5.md §4.1).

    Args:
        hidden:          token embedding dimension (192).
        base_dim:        first conv feature count (hidden // 2 by default).
        scalar_hidden:   hidden width of the shared scalar-conditioning MLP.
        in_groups:       GroupNorm group count for the first block.

    No 3-channel geometry head: the output is occupancy logits [B,1,64,64].
    """

    def __init__(self, hidden=192, base_dim=96, scalar_hidden=128,
                 in_groups=8):
        super().__init__()
        self.hidden = hidden
        self.base_dim = base_dim

        # Shared small scalar-conditioning MLP feeding per-block FiLM heads.
        self.scalar_mlp = nn.Sequential(
            nn.Linear(3, scalar_hidden),
            nn.GELU(),
            nn.Linear(scalar_hidden, scalar_hidden),
        )

        # 16x16 → 32x32 → 64x64.
        self.block1 = _FilmBlock(hidden, base_dim, scalar_hidden)
        self.block2 = _FilmBlock(base_dim, base_dim // 2, scalar_hidden)
        self.head = nn.Conv2d(base_dim // 2, 1, kernel_size=3, padding=1)

    def forward(self, z, scalars):
        """Decode occupancy logits from predicted tokens.

        Args:
            z:        [B, 256, hidden] predicted occupancy tokens.
            scalars:  [B, 3] effective (l_lattice, h_atom, r_atom) — known
                      values substituted for known scalars, predictions for
                      unknown ones.

        Returns:
            occ_logits: [B, 1, 64, 64].
        """
        B, T, D = z.shape
        assert T == 256, f"expected 256 tokens, got {T}"
        assert D == self.hidden, f"expected dim {self.hidden}, got {D}"

        x = z.view(B, 16, 16, D).permute(0, 3, 1, 2).contiguous()

        s = self.scalar_mlp(scalars)  # (B, scalar_hidden)

        x = self.block1(x, s)
        x = self.block2(x, s)
        return self.head(x)


# Backward-compatible alias: the unified path's active decoder.
__all__ = ["OccupancyDecoder"]
