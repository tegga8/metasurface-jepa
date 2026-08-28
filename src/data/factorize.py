"""Factorized semantic data contract for the unified occupancy-parameter-spectrum JEPA.

architecture_v5.md §0, Phase 1 MD §2.

The legacy [B, 3, 64, 64] broadcast tensor (channel 0 = r_atom/5 on occupied,
channel 1 = h_atom on occupied, channel 2 = l_lattice/3 everywhere) is factored
into its true semantic components: a single-channel occupancy mask, and the three
global scalar parameters (l_lattice, h_atom, r_atom) as raw physical values.

Round-trip: assemble(factorize(G)) == G exactly (verified by test on real and
synthetic samples). The broadcast convention lives only at the MetaDiT surrogate
boundary (§4.3) and is not used for internal masking/representation learning.
"""

import torch


def factorize_geometry(geometry):
    """Extract (occupancy, scalars) from a legacy [B, 3, 64, 64] broadcast tensor.

    Convention (verified from src/data/dataset.py):
        ch0 = occupancy * (r_atom / 5)   — nonzero only on occupied pixels
        ch1 = occupancy * h_atom         — nonzero only on occupied pixels
        ch2 = l_lattice / 3              — constant everywhere per sample

    Args:
        geometry: (B, 3, 64, 64) float tensor in the broadcast convention.

    Returns:
        occupancy: (B, 1, 64, 64) float32 in {0.0, 1.0}
        scalars:   (B, 3) float32 = [l_lattice, h_atom, r_atom] (raw physical values)
    """
    assert geometry.dim() == 4 and geometry.shape[1] == 3, (
        f"expected [B,3,64,64], got {tuple(geometry.shape)}")

    # Occupancy: a pixel is occupied if channel 0 or channel 1 is nonzero.
    occ = ((geometry[:, 0:1] != 0) | (geometry[:, 1:2] != 0)).to(
        geometry.dtype
    )  # (B, 1, 64, 64)

    # l_lattice: channel 2 is l_lattice/3 everywhere (constant per sample), so
    # any pixel recovers it: l_lattice = grid[2, *] * 3
    l = geometry[:, 2, 0, 0] * 3.0  # (B,)

    # h_atom: channel 1 is h_atom on occupied pixels (raw, positive). Unoccupied
    # pixels are 0. Use spatial max over occupied regions.
    h = geometry[:, 1].amax(dim=(1, 2))  # (B,)

    # r_atom: channel 0 is r_atom/5 on occupied pixels. Recover raw r_atom.
    r_scaled = geometry[:, 0].amax(dim=(1, 2))  # (B,) = r_atom/5
    r = r_scaled * 5.0  # (B,)

    # Occupancy recovery ((ch0 != 0) | (ch1 != 0)) and scalar recovery via amax
    # both rely on h_atom and r_atom being strictly positive: if either were
    # exactly zero for a sample, an occupied pixel on that channel would be
    # indistinguishable from an unoccupied one and occupancy recovery would
    # silently misclassify pixels for that sample. Physically these are positive
    # dimensions, so this is an asserted invariant rather than a silent one.
    assert (h > 0).all() and (r > 0).all(), (
        "factorize_geometry assumes h_atom and r_atom are strictly positive "
        "(occupancy recovery relies on nonzero channel values on occupied "
        "pixels) — got a sample with a zero or negative scalar; occupancy "
        "recovery would be unreliable for this sample."
    )

    scalars = torch.stack([l, h, r], dim=-1)  # (B, 3)
    return occ, scalars


def assemble_geometry(occupancy, scalars):
    """Reconstruct [B, 3, 64, 64] broadcast tensor from occupancy + scalars.

    Exact inverse of factorize_geometry, using the same normalization constants
    (l/3, h raw, r/5) as src/data/dataset.py.

    Args:
        occupancy: (B, 1, 64, 64) binary float tensor.
        scalars:   (B, 3) = [l_lattice, h_atom, r_atom] (raw physical values).

    Returns:
        (B, 3, 64, 64) broadcast tensor matching the dataset convention.
    """
    assert occupancy.dim() == 4 and occupancy.shape[1] == 1, (
        f"expected occupancy [B,1,64,64], got {tuple(occupancy.shape)}")
    assert scalars.dim() == 2 and scalars.shape[1] == 3, (
        f"expected scalars [B,3], got {tuple(scalars.shape)}")

    b = occupancy.shape[0]
    device = occupancy.device
    dtype = occupancy.dtype

    l = scalars[:, 0:1].view(b, 1, 1, 1)  # (B,1,1,1)
    h = scalars[:, 1:2].view(b, 1, 1, 1)
    r = scalars[:, 2:3].view(b, 1, 1, 1)

    ch0 = occupancy * (r / 5.0)             # occ * r/5
    ch1 = occupancy * h                      # occ * h
    ch2 = (l / 3.0).expand(b, 1, 64, 64)     # l/3 everywhere

    return torch.cat([ch0, ch1, ch2], dim=1)  # (B, 3, 64, 64)


SCALAR_CONVENTION = {
    "parameter_order": ["l_lattice", "h_atom", "r_atom"],
    "parameter_index_in_dataset": [0, 1, 2],
    "broadcast_channel": {
        "l_lattice": {"channel": 2, "divisor": 3.0, "dense": True},
        "h_atom": {"channel": 1, "divisor": 1.0, "dense": False},
        "r_atom": {"channel": 0, "divisor": 5.0, "dense": False},
    },
    "factorized_scalar_representation": "raw physical values (not rescaled)",
}


def assemble_metadit_geometry(occupancy, l_lattice, h_atom, r_atom):
    """Assemble [B,3,64,64] MetaDiT geometry from occupancy + individual scalars.

    Thin wrapper around assemble_geometry for the physics-loop interface
    (Phase 4 MD §1). Convention verified from src/data/dataset.py:

        G0 = occupancy * r_atom / 5
        G1 = occupancy * h_atom
        G2 = l_lattice / 3  (everywhere)

    Args:
        occupancy:  (B, 1, 64, 64) float — may be soft (sigmoid) for
                   differentiability through the surrogate (Phase 4 MD §3).
        l_lattice:  (B,) scalar — lattice constant.
        h_atom:     (B,) scalar — atom height.
        r_atom:     (B,) scalar — atom radius.

    Returns:
        (B, 3, 64, 64) broadcast tensor matching the surrogate's expected input.
    """
    scalars = torch.stack([l_lattice, h_atom, r_atom], dim=-1)
    return assemble_geometry(occupancy, scalars)
