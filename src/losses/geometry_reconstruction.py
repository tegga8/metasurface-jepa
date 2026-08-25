"""Reconstruction loss for Phase 1 geometry decoder.

Separate channel handling for the three geometry channels:
  channel 0: r_atom/5 on occupied pixels only
  channel 1: h_atom   on occupied pixels only
  channel 2: l_lattice/3 everywhere (dense)

Channel-scale normalization:
  L_r = mean(|pred - target|) / scale_r
  L_h = mean(|pred - target|) / scale_h
where scale_r/scale_h are computed from training-set absolute means.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryReconstructionLoss(nn.Module):
    """Weighted sum of occupancy + per-channel geometry losses.

    Args:
        lambda_occ:     Weight for occupancy BCE loss.
        lambda_value:   Weight for occupied-pixel value losses (L_r + L_h).
        lambda_lattice: Weight for dense lattice-channel L1 loss.
        lambda_r:       Relative weight for channel 0 (r_atom/5) within L_value.
        lambda_h:       Relative weight for channel 1 (h_atom) within L_value.
        scale_r:        Normalization divisor for channel 0 L1 loss (from training set).
        scale_h:        Normalization divisor for channel 1 L1 loss (from training set).
    """

    def __init__(self, lambda_occ=1.0, lambda_value=1.0, lambda_lattice=0.25,
                 lambda_r=1.0, lambda_h=1.0, scale_r=1.0, scale_h=1.0):
        super().__init__()
        self.lambda_occ = lambda_occ
        self.lambda_value = lambda_value
        self.lambda_lattice = lambda_lattice
        self.lambda_r = lambda_r
        self.lambda_h = lambda_h
        self.scale_r = scale_r
        self.scale_h = scale_h

    @staticmethod
    def occupancy_target(geometry):
        """Derive binary occupancy from ground-truth geometry.

        A pixel is occupied if either channel 0 or channel 1 is non-zero.

        Args:
            geometry: (B, 3, 64, 64)

        Returns:
            (B, 1, 64, 64) float32 in {0, 1}
        """
        return (
            (geometry[:, 0:1] != 0) | (geometry[:, 1:2] != 0)
        ).float()

    def forward(self, geometry_pred, occ_logits, geometry_target):
        """Compute reconstruction loss.

        Args:
            geometry_pred:   (B, 3, 64, 64) — decoder geometry output.
            occ_logits:      (B, 1, 64, 64) — decoder occupancy logits (may be None).
            geometry_target: (B, 3, 64, 64) — ground truth.

        Returns:
            L_total:         scalar
            components:      dict of individual loss terms
        """
        occ_target = self.occupancy_target(geometry_target)
        mask = occ_target.squeeze(1).bool()  # (B, 64, 64)

        # Occupancy loss
        if occ_logits is not None:
            L_occ = F.binary_cross_entropy_with_logits(occ_logits, occ_target)
        else:
            L_occ = geometry_pred.new_zeros(())

        # Channel 0: r_atom/5 on occupied pixels, normalized by scale_r
        if mask.any():
            L_r = F.l1_loss(geometry_pred[:, 0][mask],
                            geometry_target[:, 0][mask]) / self.scale_r
        else:
            L_r = geometry_pred.new_zeros(())

        # Channel 1: h_atom on occupied pixels, normalized by scale_h
        if mask.any():
            L_h = F.l1_loss(geometry_pred[:, 1][mask],
                            geometry_target[:, 1][mask]) / self.scale_h
        else:
            L_h = geometry_pred.new_zeros(())

        # Channel 2: l_lattice/3 everywhere (dense, native scale)
        L_lattice = F.l1_loss(geometry_pred[:, 2], geometry_target[:, 2])

        # Combined
        L_value = self.lambda_r * L_r + self.lambda_h * L_h
        L_total = (
            self.lambda_occ * L_occ
            + self.lambda_value * L_value
            + self.lambda_lattice * L_lattice
        )

        components = {
            "L_occ": L_occ.detach(),
            "L_r": L_r.detach(),
            "L_h": L_h.detach(),
            "L_lattice": L_lattice.detach(),
            "L_value": L_value.detach(),
        }
        return L_total, components
