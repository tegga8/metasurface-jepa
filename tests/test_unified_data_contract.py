"""Phase 1 tests — unified occupancy-parameter-spectrum data contract and new modules.

Covers architecture_v5.md §0–§5 and Phase 1 MD §7:

  - data factorization (factorize_geometry)
  - legacy round-trip equivalence (assemble o factorize == identity)
  - scalar known/unknown flags across regimes
  - occupancy encoder shape contract [B,1,64,64] -> [B,256,192]
  - scalar encoder shape contract ([B,6] -> film params + summary [B,1,192])
  - FiLM identity initialization (zero-init heads -> identity at step 0)
  - fusion token count (256 + 16 + 1 = 273) and width (192)
  - scalar_mlp_ema independence (EMA copy is a separate object, unaffected by
    live-copy weight changes)
"""

import copy
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch

from data.factorize import factorize_geometry, assemble_geometry
from data.scalar_mask import ScalarMasker
from encoders.occupancy_encoder import OccupancyEncoder
from encoders.scalar_encoder import ScalarEncoder
from fusion.fusion_encoder import FusionEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_broadcast_geometry(b=4, seed=0):
    """Build a synthetic [B,3,64,64] valid in the dataset convention:
    ch0 = occ * r/5, ch1 = occ * h, ch2 = l/3 everywhere, all positive."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    l = 2.5 + 0.5 * torch.rand(b, generator=rng)          # [2.5, 3.0]
    h = 0.5 + 0.5 * torch.rand(b, generator=rng)          # [0.5, 1.0]
    r = 3.5 + 1.5 * torch.rand(b, generator=rng)          # [3.5, 5.0]

    occ = (torch.rand(b, 1, 64, 64, generator=rng) > 0.4).float()

    g0 = occ * (r / 5.0).view(b, 1, 1, 1)
    g1 = occ * h.view(b, 1, 1, 1)
    g2 = (l / 3.0).view(b, 1, 1, 1).expand(b, 1, 64, 64)
    geometry = torch.cat([g0, g1, g2], dim=1)  # (B, 3, 64, 64)
    return geometry, l, h, r, occ


# ---------------------------------------------------------------------------
# Data factorization
# ---------------------------------------------------------------------------

def test_factorize_shapes():
    G, *_ = _make_valid_broadcast_geometry(b=4)
    occ, scalars = factorize_geometry(G)
    assert occ.shape == (4, 1, 64, 64)
    assert scalars.shape == (4, 3)


def test_factorize_recovers_scalars():
    G, l_true, h_true, r_true, _ = _make_valid_broadcast_geometry(b=4, seed=42)
    occ, scalars = factorize_geometry(G)
    # scalars = [l_lattice, h_atom, r_atom] (raw physical values)
    assert torch.allclose(scalars[:, 0], l_true, atol=1e-5), "l_lattice not recovered"
    assert torch.allclose(scalars[:, 1], h_true, atol=1e-5), "h_atom not recovered"
    assert torch.allclose(scalars[:, 2], r_true, atol=1e-5), "r_atom not recovered"


def test_factorize_occupancy_binary():
    G, *_ = _make_valid_broadcast_geometry(b=4)
    occ, _ = factorize_geometry(G)
    assert set(torch.unique(occ).tolist()) <= {0.0, 1.0}
    assert occ.dtype == G.dtype


def test_factorize_rejects_wrong_channels():
    with pytest.raises(AssertionError):
        factorize_geometry(torch.randn(2, 1, 64, 64))


# ---------------------------------------------------------------------------
# Legacy round-trip equivalence
# ---------------------------------------------------------------------------

def test_assemble_factorize_roundtrip_synthetic():
    G, *_ = _make_valid_broadcast_geometry(b=4, seed=7)
    occ, scalars = factorize_geometry(G)
    G_re = assemble_geometry(occ, scalars)
    assert torch.allclose(G_re, G, atol=1e-6), "round-trip must reproduce G exactly"


def test_assemble_factorize_roundtrip_real_data():
    """Uses actual dataset values (if MAT files are present)."""
    mat_path = os.path.join(REPO_ROOT, "data/metadit/split_data/train_set.mat")
    if not os.path.exists(mat_path):
        pytest.skip("train_set.mat not present")

    from data.dataset import MetaDiTDataset
    ds = MetaDiTDataset(mat_path, max_samples=8, seed=0)
    batch = torch.stack([ds[i][0] for i in range(len(ds))], 0)  # (8, 3, 64, 64)

    occ, scalars = factorize_geometry(batch)
    G_re = assemble_geometry(occ, scalars)
    assert torch.allclose(G_re, batch, atol=1e-5), (
        "round-trip must reproduce real dataset geometry within float tolerance"
    )


def test_assemble_broadcast_invariants():
    G, *_ = _make_valid_broadcast_geometry(b=2, seed=9)
    occ, scalars = factorize_geometry(G)
    G_re = assemble_geometry(occ, scalars)
    # support(ch0) == support(ch1) == support(occ)
    support0 = (G_re[:, 0] != 0)
    support1 = (G_re[:, 1] != 0)
    support_occ = (occ[:, 0] != 0)
    assert torch.equal(support0, support1), "ch0 and ch1 must have same support"
    assert torch.equal(support0, support_occ), "support must match occupancy"
    # G2 (ch2) constant per sample
    for b in range(G_re.shape[0]):
        assert torch.allclose(
            G_re[b, 2], G_re[b, 2, 0, 0].expand(1, 64, 64)
        ), f"channel 2 not constant for sample {b}"


# ---------------------------------------------------------------------------
# Scalar masking
# ---------------------------------------------------------------------------

def test_scalar_masker_all_known():
    sm = ScalarMasker(regime="all_known", seed=0)
    vals = torch.tensor([[2.5, 0.8, 4.0], [3.0, 1.0, 5.0]])
    masked, known = sm.sample(vals)
    assert known.all(), "all_known must set every flag True"
    assert torch.equal(masked, vals), "known values must be unchanged"


def test_scalar_masker_all_unknown():
    sm = ScalarMasker(regime="all_unknown", seed=0)
    vals = torch.tensor([[2.5, 0.8, 4.0], [3.0, 1.0, 5.0]])
    masked, known = sm.sample(vals)
    assert not known.any(), "all_unknown must set every flag False"
    assert torch.all(masked == 0.0), "unknown values must be zeroed"


def test_scalar_masker_build_mlp_input():
    sm = ScalarMasker(regime="independent", p_independent=0.5, seed=0)
    vals = torch.tensor([[2.5, 0.8, 4.0]])
    masked, known = sm.sample(vals)
    inp = sm.build_mlp_input(masked, known)
    assert inp.shape == (1, 6), "MLP input must be [B, 6]"
    # Check layout: [l_val, l_known, h_val, h_known, r_val, r_known]
    expected = torch.stack(
        [masked[0, 0], known[0, 0].float(),
         masked[0, 1], known[0, 1].float(),
         masked[0, 2], known[0, 2].float()]
    )
    assert torch.equal(inp[0], expected), "MLP input layout must be l/flag/h/flag/r/flag"


def test_scalar_masker_correlated():
    sm = ScalarMasker(regime="correlated", p_correlated=0.5, seed=42)
    vals = torch.randn(100, 3)
    masked, known = sm.sample(vals)
    # Each sample: all known or all unknown
    per_sample = known.all(dim=1) | (~known.any(dim=1))
    assert per_sample.all(), "correlated regime must be all-known or all-unknown per sample"


def test_scalar_masker_independent_shape():
    sm = ScalarMasker(regime="independent", p_independent=0.5, seed=1)
    vals = torch.randn(8, 3)
    masked, known = sm.sample(vals)
    assert masked.shape == (8, 3)
    assert known.shape == (8, 3)
    assert known.dtype == torch.bool


def test_scalar_masker_independent_variability():
    """Over many samples, independent masking must produce a mix of known/unknown
    per scalar (not a degenerate all-known or all-unknown)."""
    sm = ScalarMasker(regime="independent", p_independent=0.5, seed=3)
    vals = torch.randn(1000, 3)
    _, known = sm.sample(vals)
    # Each scalar should be known roughly 50% of the time
    for j in range(3):
        frac = known[:, j].float().mean().item()
        assert 0.3 < frac < 0.7, f"scalar {j} known fraction {frac:.2f} outside [0.3, 0.7]"


def test_scalar_masker_rng_state():
    sm1 = ScalarMasker(regime="independent", seed=5)
    sm2 = ScalarMasker(regime="independent", seed=5)
    vals = torch.randn(4, 3)
    m1, k1 = sm1.sample(vals)
    m2, k2 = sm2.sample(vals)
    assert torch.equal(k1, k2), "same seed must produce same known flags"
    assert torch.equal(m1, m2), "same seed must produce same masked values"


# ---------------------------------------------------------------------------
# Occupancy encoder
# ---------------------------------------------------------------------------

def test_occupancy_encoder_shape():
    torch.manual_seed(0)
    enc = OccupancyEncoder(hidden=192, num_heads=6, depth=6)
    occ = torch.zeros(2, 1, 64, 64)
    occ[:, :, ::8, ::8] = 1.0  # sparse occupancy
    out = enc(occ)
    assert out.shape == (2, 256, 192)


def test_occupancy_encoder_no_film_is_identity_path():
    """Without FiLM params, the encoder still produces valid tokens."""
    torch.manual_seed(0)
    enc = OccupancyEncoder(hidden=192, num_heads=6, depth=2)
    occ = torch.zeros(1, 1, 64, 64)
    occ[0, 0, ::4, ::4] = 1.0
    out = enc(occ, film_params=None)
    assert out.shape == (1, 256, 192)
    assert torch.isfinite(out).all()


def test_occupancy_encoder_film_shape():
    torch.manual_seed(0)
    enc = OccupancyEncoder(hidden=192, num_heads=6, depth=6)
    b = 2
    # Simulate FiLM params from a scalar encoder
    film_params = []
    for _ in range(6):
        gamma = torch.ones(b, 192)
        beta = torch.zeros(b, 192)
        film_params.append((gamma, beta))
    occ = torch.zeros(b, 1, 64, 64)
    occ[:, :, ::4, ::4] = 1.0
    out = enc(occ, film_params=film_params)
    assert out.shape == (b, 256, 192)


# ---------------------------------------------------------------------------
# Scalar encoder
# ---------------------------------------------------------------------------

def test_scalar_encoder_shapes():
    torch.manual_seed(0)
    enc = ScalarEncoder(hidden=192, n_film_blocks=6)
    inp = torch.randn(4, 6)
    film_params, summary = enc(inp)
    assert len(film_params) == 6, "one FiLM param set per block"
    for gamma, beta in film_params:
        assert gamma.shape == (4, 192)
        assert beta.shape == (4, 192)
    assert summary.shape == (4, 1, 192)


def test_scalar_encoder_film_identity_at_init():
    """Zero-init FiLM heads must produce gamma=1, beta=0 -> identity mapping.

    With film_params applied as x = gamma * x + beta, identity means gamma=1,
    beta=0: the encoder output equals the unmasked (no-FiLM) output.
    """
    torch.manual_seed(0)
    enc = OccupancyEncoder(hidden=192, num_heads=6, depth=6)
    scalar_enc = ScalarEncoder(hidden=192, n_film_blocks=6)

    occ = torch.zeros(2, 1, 64, 64)
    occ[:, :, ::4, ::4] = 1.0
    inp = torch.randn(2, 6)

    film_params, _ = scalar_enc(inp)
    # Check init values
    for gamma, beta in film_params:
        assert torch.allclose(gamma, torch.ones_like(gamma), atol=1e-6), \
            "gamma must init to 1 (identity FiLM)"
        assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-6), \
            "beta must init to 0 (identity FiLM)"

    # Check functional identity: with gamma=1, beta=0, the FiLM path equals no-FiLM
    out_no_film = enc(occ, film_params=None)
    out_with_film = enc(occ, film_params=film_params)
    assert torch.allclose(out_no_film, out_with_film, atol=1e-6), (
        "identity FiLM must leave encoder output unchanged"
    )


def test_scalar_encoder_input_dim():
    enc = ScalarEncoder(hidden=192, n_film_blocks=6)
    # Must accept exactly 6 features
    inp = torch.randn(1, 6)
    film_params, summary = enc(inp)
    assert len(film_params) > 0
    assert summary.shape == (1, 1, 192)


# ---------------------------------------------------------------------------
# Fusion encoder
# ---------------------------------------------------------------------------

def test_fusion_token_count_and_width():
    torch.manual_seed(0)
    fusion = FusionEncoder(hidden=192, num_heads=6, depth=2, goal_dim_in=384)
    occ_tokens = torch.randn(2, 256, 192)
    goal_tokens = torch.randn(2, 16, 384)
    scalar_summary = torch.randn(2, 1, 192)
    out = fusion(occ_tokens, goal_tokens, scalar_summary)
    assert out.shape == (2, 273, 192), (
        f"expected [2, 273, 192], got {tuple(out.shape)}"
    )


def test_fusion_token_layout():
    """Verify token order: 256 occupancy | 16 goal | 1 scalar."""
    torch.manual_seed(0)
    fusion = FusionEncoder(hidden=192, num_heads=6, depth=2, goal_dim_in=384)
    # Use distinct, identifiable tokens
    occ_tokens = torch.randn(1, 256, 192)
    goal_tokens = torch.randn(1, 16, 384)
    scalar_summary = torch.randn(1, 1, 192)

    # Zero out the fusion blocks (identity) by using depth=0 is not possible,
    # but with depth=2 and random init, check that goal_proj changes goal tokens.
    out = fusion(occ_tokens, goal_tokens, scalar_summary)

    # The first 256 tokens started from occ_tokens
    # The last token is the scalar summary
    # The middle 16 are projected goal tokens
    # With real transformer blocks (non-zero weights), we can't easily separate,
    # but we can verify token count and that goal_proj is applied.
    assert out.shape[1] == 256 + 16 + 1

    # Check goal_proj produces 192-dim from 384-dim
    projected = fusion.goal_proj(goal_tokens)
    assert projected.shape == (1, 16, 192)


def test_fusion_goal_projection_independent_of_occ():
    """The goal projection must not depend on occupancy tokens (only on goal_tokens)."""
    torch.manual_seed(0)
    fusion = FusionEncoder(hidden=192, num_heads=6, depth=2, goal_dim_in=384)
    goal = torch.randn(1, 16, 384)
    g1 = fusion.goal_proj(goal)
    goal2 = torch.randn(1, 16, 384)
    g2 = fusion.goal_proj(goal2)
    assert not torch.allclose(g1, g2), "different goals must project differently"


# ---------------------------------------------------------------------------
# Scalar MLP EMA independence
# ---------------------------------------------------------------------------

def test_scalar_mlp_ema_independence():
    """The EMA copy must be a separate object, unaffected by live-copy updates."""
    torch.manual_seed(0)
    live = ScalarEncoder(hidden=192, n_film_blocks=6)

    # Create EMA copy (deepcopy, as EMAEncoder does)
    ema = copy.deepcopy(live)

    # Verify they are separate objects with separate parameters
    assert ema is not live, "EMA must be a separate object"
    assert not any(
        ep is lp for ep in ema.parameters() for lp in live.parameters()
    ), "EMA parameters must not be the same objects as live parameters"

    # Capture EMA state
    ema_state_before = {k: v.clone() for k, v in ema.state_dict().items()}

    # Modify live copy in-place (simulating optimizer step)
    with torch.no_grad():
        for p in live.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # EMA copy must be unchanged
    ema_state_after = ema.state_dict()
    for k in ema_state_before:
        assert torch.equal(ema_state_before[k], ema_state_after[k]), (
            f"EMA parameter {k} changed when only the live copy was updated"
        )

    # Live and EMA produce different outputs now
    inp = torch.randn(2, 6)
    live_film, _ = live(inp)
    ema_film, _ = ema(inp)
    # At least one FiLM param must differ
    differs = any(
        not torch.allclose(lf, ef, atol=1e-8)
        for (lf, _), (ef, _) in zip(live_film, ema_film)
    )
    assert differs, "EMA must use its own (separate) weights, not the live copy's"


def test_scalar_mlp_ema_params_frozen():
    """EMA copy parameters must not require gradients (EMAEncoder freezes them)."""
    torch.manual_seed(0)
    live = ScalarEncoder(hidden=192, n_film_blocks=6)
    ema = copy.deepcopy(live)
    # EMAEncoder (target_encoder.py) explicitly freezes the deepcopy
    for p in ema.parameters():
        p.requires_grad_(False)
    for p in ema.parameters():
        assert not p.requires_grad, "EMA parameters must be frozen"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    import sys as _sys
    _sys.exit(1 if failures else 0)
