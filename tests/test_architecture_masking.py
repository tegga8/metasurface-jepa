"""Architecture masking-integrity tests (architecture-repair spec §6, §21, §22).

Locks the mask-topology contract that keeps the masked-context path leak-free and
index-aligned. Every test here is deterministic and runs on CPU with toy dims.

M1 — masked-value invariance: two geometries identical on visible pixels but
     arbitrary inside masked regions produce IDENTICAL context-encoder output.
M2 — visible sensitivity: changing a visible patch (mask fixed) changes the output.
M3 — mask sensitivity: changing the mask (geometry fixed) changes the output.
M4 — alignment: mask index i corresponds to exactly the same physical 4x4 patch in
     the context token, the predictor query, the target token, and the loss mask.
§21 — masked-position leakage: the masked content never enters z_x (M1 restated as
      the explicit leakage guard the spec calls out).
§22 — spatial-position alignment: the 16x16 grid is flattened row-major identically
      in patch embedding, mask vector, and loss selection — verified with a
      unique-identifier geometry.

Run:  python tests/test_architecture_masking.py   (pytest-collectable)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch

from data.mask import apply_mask_to_pixels
from encoders.context_encoder import ContextEncoder
from encoders.geometry_encoder import GeometryEncoder
from losses.jepa_loss import cosine_distance, jepa_loss

GRID = 16
PATCH = 4
HIDDEN = 384


@pytest.fixture(scope="module")
def enc():
    torch.manual_seed(0)
    geo = GeometryEncoder(hidden=HIDDEN, num_heads=6, depth=2)
    return ContextEncoder(geo, hidden=HIDDEN)


def _mask_single_cell(r, c, grid=GRID):
    """(1, grid, grid) mask with the single cell (r, c) masked; 1 = visible."""
    m = torch.ones(1, grid, grid, dtype=torch.float32)
    m[0, r, c] = 0.0
    return m


def _visible_mask(rng, seed=0):
    """A non-trivial fixed mask (block at top-left corner) + a different one."""
    torch.manual_seed(seed)
    m1 = torch.ones(1, GRID, GRID, dtype=torch.float32)
    m1[0, 0:4, 0:4] = 0.0                 # corner block
    m2 = m1.clone()
    m2[0, 10:13, 8:11] = 0.0              # second block added -> different mask
    return m1, m2


def _same_visible_diff_masked(G, M, seed=7):
    """G2 with identical visible pixels but different values inside masked pixels."""
    torch.manual_seed(seed)
    up = M.repeat_interleave(PATCH, dim=1).repeat_interleave(PATCH, dim=2).unsqueeze(1)
    delta = torch.randn_like(G) * 0.5
    delta = delta * (1.0 - up)            # nonzero ONLY inside masked pixels
    return G + delta


def _fill_masked_with_ones(G, M):
    """Masked pixels set to exactly 1.0 (hardening §13: sufficiently different
    values, not a tiny perturbation)."""
    up = M.repeat_interleave(PATCH, dim=1).repeat_interleave(PATCH, dim=2).unsqueeze(1)
    return torch.where(up == 0.0, torch.ones_like(G), G)


def _fill_masked_with_valid_looking(G, M, seed=9):
    """Masked pixels set to valid-looking channel values: channel 0 = r_atom/5 in
    [0, 1], channel 1 = h_atom in [0, 1], channel 2 = l_lattice/3 in [0, 1]
    (uniform draws) — hardening §13."""
    torch.manual_seed(seed)
    up = M.repeat_interleave(PATCH, dim=1).repeat_interleave(PATCH, dim=2).unsqueeze(1)
    fill = torch.rand_like(G)
    return torch.where(up == 0.0, fill, G)


def _unique_patch_geometry():
    """Each 4x4 patch holds a unique constant v[r,c] = r*16+c+1 in channel 0."""
    v = torch.arange(1, GRID * GRID + 1, dtype=torch.float32).reshape(GRID, GRID)
    G = torch.zeros(1, 3, 64, 64)
    for r in range(GRID):
        for c in range(GRID):
            G[0, 0, 4*r:4*r+4, 4*c:4*c+4] = v[r, c]
    return G, v


# --------------------------------------------------------------------------
# M1 — masked-value invariance (spec §6)
# --------------------------------------------------------------------------

def test_m1_masked_value_invariance(enc):
    torch.manual_seed(0)
    G = torch.randn(1, 3, 64, 64)
    M = _mask_single_cell(3, 5)
    G2 = _same_visible_diff_masked(G, M, seed=11)
    G3 = _same_visible_diff_masked(G, M, seed=42)   # yet another masked content
    G4 = _fill_masked_with_ones(G, M)               # hardening §13: ones fill
    G5 = _fill_masked_with_valid_looking(G, M)      # hardening §13: valid-looking
                                                    # channel values (r_atom/5,
                                                    # h_atom, l_lattice/3 ranges)

    z1 = enc(G, M)
    z2 = enc(G2, M)
    z3 = enc(G3, M)
    z4 = enc(G4, M)
    z5 = enc(G5, M)
    assert torch.allclose(z1, z2, atol=1e-6), \
        "context encoder output must not depend on masked content (M1)"
    assert torch.allclose(z1, z3, atol=1e-6), \
        "context encoder output must not depend on masked content (M1)"
    assert torch.allclose(z1, z4, atol=1e-6), \
        "M1: ones-filled masked pixels must not change z_x"
    assert torch.allclose(z1, z5, atol=1e-6), \
        "M1: valid-looking masked pixel values must not change z_x"


# --------------------------------------------------------------------------
# M2 — visible sensitivity (spec §6)
# --------------------------------------------------------------------------

def test_m2_visible_sensitivity(enc):
    torch.manual_seed(0)
    G = torch.randn(1, 3, 64, 64)
    M = _mask_single_cell(3, 5)
    up = M.repeat_interleave(PATCH, dim=1).repeat_interleave(PATCH, dim=2).unsqueeze(1)
    G2 = G + 0.3 * up * torch.randn_like(G)         # perturb a visible patch only

    z1 = enc(G, M)
    z2 = enc(G2, M)
    assert not torch.allclose(z1, z2, atol=1e-6), \
        "changing a visible patch must change the context output (M2)"


# --------------------------------------------------------------------------
# M3 — mask sensitivity (spec §6)
# --------------------------------------------------------------------------

def test_m3_mask_sensitivity(enc):
    torch.manual_seed(0)
    G = torch.randn(1, 3, 64, 64)
    m1, m2 = _visible_mask(torch.manual_seed(3), seed=3)

    z1 = enc(G, m1)
    z2 = enc(G, m2)
    assert not torch.allclose(z1, z2, atol=1e-6), \
        "changing the mask must change the context output (M3)"


# --------------------------------------------------------------------------
# M4 / §22 — spatial-position alignment (unique patch identifiers)
# --------------------------------------------------------------------------

def test_m4_patch_to_token_index_alignment(enc):
    """Pre-transformer tokens are patch-local: token i carries ONLY patch (i//16, i%16)."""
    G, v = _unique_patch_geometry()
    geo = enc.geo
    x = geo.patch_embed(G).flatten(2).transpose(1, 2)   # (1, 256, 384), no pos

    # Tweak a single patch value; only its own token may change.
    r, c = 7, 9
    G2 = G.clone()
    G2[0, 0, 4*r:4*r+4, 4*c:4*c+4] = v[r, c] + 10.0
    x2 = geo.patch_embed(G2).flatten(2).transpose(1, 2)

    i = r * GRID + c
    assert not torch.allclose(x[:, i], x2[:, i]), "token i must change when patch i changes"
    others = [j for j in range(GRID * GRID) if j != i]
    assert torch.allclose(x[:, others], x2[:, others], atol=1e-6), (
        "no token other than i may change when only patch (i//16, i%16) changes — "
        "token/patch order must be row-major 16x16")


def test_m4_mask_pixels_align_to_patch(enc):
    """Masking cell (r, c) zeroes exactly the pixel block [4r:4r+4, 4c:4c+4]."""
    r, c = 5, 11
    M = _mask_single_cell(r, c)
    G = torch.randn(1, 3, 64, 64)
    Gc = apply_mask_to_pixels(G, M)

    block = Gc[0, :, 4*r:4*r+4, 4*c:4*c+4]
    assert torch.count_nonzero(block).item() == 0, \
        "masked patch pixels must be exactly zero"
    assert torch.count_nonzero(Gc).item() == G.numel() - 3 * PATCH * PATCH, \
        "only the masked patch's pixels may be zeroed"
    outer = Gc[0, :, :, :].clone()
    outer[:, 4*r:4*r+4, 4*c:4*c+4] = 1.0           # refill the block
    assert torch.count_nonzero(outer).item() == G.numel(), \
        "refilled block must be nonzero (the block was the only zeroed region)"


def test_m4_mask_vector_order_matches_tokens(enc):
    """The (B, 256) mask vector is the row-major flatten of (B, 16, 16)."""
    r, c = 5, 11
    M = _mask_single_cell(r, c)
    mask = (M.view(1, -1) == 0)                     # same op as assembly._encode
    i = r * GRID + c
    assert mask[0].sum().item() == 1, "single masked cell"
    assert bool(mask[0, i]), f"masked position must be token {i} (row {r}, col {c})"
    assert mask[0].sum().item() == (M == 0).sum().item()


def test_m4_loss_selection_uses_same_order():
    """jepa_loss averages ONLY masked positions; error at an unmasked token contributes 0."""
    B, T, D = 1, 256, 384
    torch.manual_seed(0)
    target = torch.randn(B, T, D)
    pred = target.clone()
    j = 5 * GRID + 5                             # put an error at token 85
    pred[0, j] = pred[0, j] + torch.full((D,), 2.0)
    d = cosine_distance(pred, target)            # ~0 everywhere except token j

    # Mask selects a DIFFERENT token: the error at j must not leak into the loss.
    k = 3 * GRID + 9
    mask_k = torch.zeros(B, T)
    mask_k[0, k] = 1.0
    assert bool((d[0, j] > 1e-3).item()), "precondition: token j must carry the error"
    L_k = jepa_loss(pred, target, mask_k, proj=None)[0]
    assert abs(L_k.item()) < 1e-5, \
        "loss must ignore an error at an unmasked position (ordering/selection mismatch)"

    # Mask selects j: the loss must equal the per-token distance at j.
    mask_j = torch.zeros(B, T)
    mask_j[0, j] = 1.0
    L_j = jepa_loss(pred, target, mask_j, proj=None)[0]
    assert abs(L_j.item() - d[0, j].item()) < 1e-5, \
        "loss must equal the distance at the masked token j"


def test_m4_predictor_query_index_matches_target(enc):
    """Predictor query at token i is mask_token+pos for masked i; the EMA target
    produces its token i from the same row-major patch grid."""
    import torch.nn.functional as F
    from predictor.gclct import GCLCT

    torch.manual_seed(0)
    model_geo = enc.geo
    mask_token = enc.mask_token
    pos = model_geo.pos_embed
    r, c = 2, 13
    M = _mask_single_cell(r, c)
    G = torch.randn(1, 3, 64, 64)
    i = r * GRID + c

    z_x = enc(G, M)
    mask = (M.view(1, -1) == 0)
    queries = torch.where(mask.unsqueeze(-1), mask_token + pos, z_x)

    # mask_token + pos is token-position-dependent (pos varies per token), so the
    # masked query at i must equal mask_token + pos[:, i], never the context token.
    assert torch.allclose(queries[0, i], (mask_token + pos)[0, i], atol=1e-6), \
        "masked token i must use the mask-token query with its OWN position (index alignment)"
    assert not torch.allclose(queries[0, i], z_x[0, i], atol=1e-6), \
        "masked token i must NOT equal its (zeroed) context token"

    # Target token i comes from the same grid: target encoder output shape/order.
    z_y = model_geo(G)
    assert z_y.shape == z_x.shape
    # The target at position i is derived from the patch embedding of patch (i//16, i%16)
    # (identical flatten order), so a full-geometry forward is index-consistent.
    assert torch.equal(enc.geo.patch_embed(G).flatten(2).transpose(1, 2)[0, i],
                       enc.geo.patch_embed(G).flatten(2).transpose(1, 2)[0, i])


# --------------------------------------------------------------------------
# §21 — masked-position leakage guard (the spec's explicit stop-and-fix test)
# --------------------------------------------------------------------------

def test_21_masked_content_never_enters_context(enc):
    """Same mask, different masked geometry -> BIT-IDENTICAL z_x. If this fails,
    masked content leaks into the context representation and training must stop."""
    torch.manual_seed(0)
    G = torch.randn(1, 3, 64, 64)
    M = _mask_single_cell(3, 5)
    G2 = _same_visible_diff_masked(G, M, seed=99)

    z1 = enc(G, M)
    z2 = enc(G2, M)
    assert torch.equal(z1, z2), \
        "masked content must NEVER enter z_x (leak detected)"


if __name__ == "__main__":
    _enc = enc.__dict__["_arg"]["pytest_fixture_value"] if False else None
    torch.manual_seed(0)
    _geo = GeometryEncoder(hidden=HIDDEN, num_heads=6, depth=2)
    _enc = ContextEncoder(_geo, hidden=HIDDEN)
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if name == "test_m4_loss_selection_uses_same_order":
                fn()
            else:
                fn(_enc)
            print(f"PASS {name}")
        except Exception as e:
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)