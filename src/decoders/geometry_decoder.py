"""Geometry decoder: frozen EMA latent → pixel-space geometry (Phase 1).

Converts a (B, 256, 384) token sequence into:
  - geometry output (B, 3, 64, 64): r_atom/5, h_atom, l_lattice/3
  - occupancy logits (B, 1, 64, 64): auxiliary binary mask prediction

Token order is row-major and maps directly to a 16×16 spatial grid
via view + permute — no learned remapping, no mean-pooling.
"""

import torch
import torch.nn as nn


class GeometryDecoder(nn.Module):
    """Lightweight convolutional decoder over the 16×16 spatial token grid.

    Args:
        hidden_dim:  Token embedding dimension (384).
        base_dim:    First conv feature count (192).
        num_channels: Geometry output channels (3).
        occupancy_head: Whether to produce auxiliary occupancy logits.
    """

    def __init__(self, hidden_dim=384, base_dim=192, num_channels=3,
                 occupancy_head=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.occupancy_head = occupancy_head

        self.backbone = nn.Sequential(
            nn.Conv2d(hidden_dim, base_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dim),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_dim, base_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dim // 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_dim // 2, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.geometry_head = nn.Conv2d(32, num_channels, kernel_size=3, padding=1)
        if occupancy_head:
            self.occ_head = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        else:
            self.occ_head = None

    def forward(self, z):
        """Decode latent tokens to pixel-space geometry.

        Args:
            z: (B, 256, 384) — token sequence from frozen EMA encoder.

        Returns:
            geometry:       (B, 3, 64, 64)
            occupancy_logits: (B, 1, 64, 64) if occupancy_head else None
        """
        B, T, D = z.shape
        assert T == 256, f"Expected 256 tokens, got {T}"
        assert D == self.hidden_dim, (
            f"Expected dim {self.hidden_dim}, got {D}"
        )

        x = z.view(B, 16, 16, D).permute(0, 3, 1, 2).contiguous()

        feat = self.backbone(x)
        geometry = self.geometry_head(feat)
        occ_logits = self.occ_head(feat) if self.occ_head is not None else None

        return geometry, occ_logits
