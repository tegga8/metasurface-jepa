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
        variant:         "base"     — 16→32→64 upsample chain (shipped base v1;
                                      the default, byte-identical state dict).
                         "quadrant" — emit a 32×32 quadrant and build the 64×64
                                      field by mirroring it, i.e. the corpus's own
                                      output parameterisation (MetaDiT's
                                      restore_structure mirrors a 32×32 quadrant,
                                      which is why 100% of the released corpus is
                                      bit-identical under fliplr/flipud and why
                                      MetaDiT's bodies are clean: holes 0.21 /
                                      mirror 1.000 against our 1.63 / 0.916).
                                      Disclosed trade-off: outputs cannot be
                                      asymmetric, so asymmetric OOD probes are
                                      capped by construction (the released
                                      dataset has no asymmetric designs).
        depth:           number of FiLM blocks in the quadrant variant (1
                         upsample + depth-1 refinement blocks at 32×32).

    No 3-channel geometry head: the output is occupancy logits [B,1,64,64].
    """

    def __init__(self, hidden=192, base_dim=96, scalar_hidden=128,
                 variant="base", depth=2):
        super().__init__()
        self.hidden = hidden
        self.base_dim = base_dim
        self.variant = variant

        # Shared small scalar-conditioning MLP feeding per-block FiLM heads.
        # INPUT CONVENTION (architecture_v5.md §3.2, audit B10): 6 numbers —
        # [value, known-flag] for each of (l, h, r), the same convention as
        # ScalarEncoder — so a known value and a predicted value of the same
        # magnitude are distinguishable by the FiLM conditioning.
        self.scalar_mlp = nn.Sequential(
            nn.Linear(6, scalar_hidden),
            nn.GELU(),
            nn.Linear(scalar_hidden, scalar_hidden),
        )

        if variant == "base":
            # 16x16 → 32x32 → 64x64.
            self.block1 = _FilmBlock(hidden, base_dim, scalar_hidden)
            self.block2 = _FilmBlock(base_dim, base_dim // 2, scalar_hidden)
            self.head = nn.Conv2d(base_dim // 2, 1, kernel_size=3, padding=1)
        elif variant == "quadrant":
            # 16x16 → 32x32, then refine at 32x32; the head emits the quadrant.
            self.block1 = _FilmBlock(hidden, base_dim, scalar_hidden)
            self.extra = nn.ModuleList([
                _FilmBlock(base_dim, base_dim, scalar_hidden, scale_factor=1)
                for _ in range(max(0, int(depth) - 1))])
            self.head = nn.Conv2d(base_dim, 1, kernel_size=3, padding=1)
        else:
            raise ValueError(f"unknown decoder variant: {variant!r}")

    def forward(self, z, scalars):
        """Decode occupancy logits from predicted tokens.

        Args:
            z:        [B, 256, hidden] predicted occupancy tokens.
            scalars:  [B, 6] effective scalar conditioning, §3.2 convention:
                      [l_val, l_known, h_val, h_known, r_val, r_known] — the
                      value is the true one where the scalar is known and the
                      prediction where unknown; the flag is 1.0 (known) or 0.0
                      (unknown), so equal magnitudes with different provenance
                      remain distinguishable (audit B10).

        Returns:
            occ_logits: [B, 1, 64, 64].
        """
        B, T, D = z.shape
        assert T == 256, f"expected 256 tokens, got {T}"
        assert D == self.hidden, f"expected dim {self.hidden}, got {D}"
        assert scalars.shape == (B, 6), (
            "decoder conditioning must be [B,6] [value, known-flag] x 3 "
            f"(§3.2), got {tuple(scalars.shape)}")

        x = z.view(B, 16, 16, D).permute(0, 3, 1, 2).contiguous()

        s = self.scalar_mlp(scalars)  # (B, scalar_hidden)

        if self.variant == "base":
            x = self.block1(x, s)
            x = self.block2(x, s)
            return self.head(x)

        x = self.block1(x, s)          # (B, base_dim, 32, 32)
        for blk in self.extra:
            x = blk(x, s)
        return mirror_quadrant(self.head(x))


def mirror_quadrant(q):
    """Build a 64×64 field from its top-left 32×32 quadrant by mirroring.

    [ q        fliplr(q) ]
    [ flipud(q) rot180(q)]  — exactly the corpus's own construction
    (external/metadit/metric.py::restore_structure), so outputs live on the
    same manifold as the released dataset by construction.
    """
    top = torch.cat([q, torch.flip(q, dims=[3])], dim=3)
    bottom = torch.cat([torch.flip(q, dims=[2]), torch.flip(q, dims=[2, 3])],
                       dim=3)
    return torch.cat([top, bottom], dim=2)


# Backward-compatible alias: the unified path's active decoder.
__all__ = ["OccupancyDecoder"]
