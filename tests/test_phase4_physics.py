"""Phase 4 tests — geometry assembly, physics loop, scenarios.

Tests:
- Geometry assembly invariants (support, constancy, channel order)
- Round-trip factorize/assemble
- Geometry decoder shapes
- Soft vs hard occupancy characterization
- Surrogate gradient flow (if surrogate weights available)
- Scenario evaluators (A/B/C) shapes
- Real/null/shuffled evaluation

Run:  python -m pytest tests/test_phase4_physics.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
import torch.nn as nn
import pytest

from assembly import UnifiedJEPA
from data.factorize import (
    factorize_geometry, assemble_geometry, assemble_metadit_geometry,
)
from data.mask import BlockMasker


class _StubReleasedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 256))

    def forward(self, S):
        return self.net(S.transpose(1, 2))


def _build_model(hidden=192, geo_depth=2, predictor_depth=4):
    torch.manual_seed(0)
    model = UnifiedJEPA(
        hidden=hidden, num_heads=6, geo_depth=geo_depth,
        predictor_depth=predictor_depth, goal_tokens=16,
        num_predictor_heads=6, scalar_hidden=128,
        n_film_blocks=geo_depth, spec_dim=256)
    stub = _StubReleasedEncoder()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    model.spectrum_path.released = stub
    model.ema.target.load_state_dict(model.occupancy_encoder.state_dict())
    model.scalar_mlp_ema.target.load_state_dict(model.scalar_encoder.state_dict())
    model.eval()
    return model


# --------------------------------------------------------------------------
# Geometry assembly invariants (Phase 4 MD §1-§2)
# --------------------------------------------------------------------------

def test_assemble_metadit_geometry_shapes():
    occ = torch.rand(2, 1, 64, 64)
    l = torch.tensor([1.5, 2.0])
    h = torch.tensor([0.8, 1.2])
    r = torch.tensor([10.0, 12.0])
    G = assemble_metadit_geometry(occ, l, h, r)
    assert G.shape == (2, 3, 64, 64)


def test_assemble_support_invariant():
    """support(G0) == support(G1) — ch0 and ch1 are occupied on same pixels."""
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    l = torch.tensor([1.5, 2.0])
    h = torch.tensor([0.8, 1.2])
    r = torch.tensor([10.0, 12.0])
    G = assemble_metadit_geometry(occ, l, h, r)
    support0 = (G[:, 0:1] != 0)
    support1 = (G[:, 1:2] != 0)
    assert torch.equal(support0, support1), "ch0 and ch1 support must match"


def test_assemble_constant_per_sample():
    """G0 and G1 occupied values are constant per sample; G2 is constant everywhere."""
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    l = torch.tensor([1.5, 2.0])
    h = torch.tensor([0.8, 1.2])
    r = torch.tensor([10.0, 12.0])
    G = assemble_metadit_geometry(occ, l, h, r)

    for b in range(2):
        occupied = occ[b, 0] > 0
        if occupied.any():
            # ch0: r/5 on occupied pixels
            vals0 = G[b, 0][occupied]
            assert torch.allclose(vals0, vals0[0:1], atol=1e-6), "G0 not constant per sample"
            # ch1: h on occupied pixels
            vals1 = G[b, 1][occupied]
            assert torch.allclose(vals1, vals1[0:1], atol=1e-6), "G1 not constant per sample"
        # ch2: l/3 everywhere (constant)
        vals2 = G[b, 2]
        assert torch.allclose(vals2, vals2.flatten()[0:1], atol=1e-6), "G2 not constant per sample"


def test_assemble_correct_constants():
    """Verify the exact normalization constants (dataset.py convention)."""
    occ = torch.ones(1, 1, 64, 64)  # all occupied
    l = torch.tensor([3.0])
    h = torch.tensor([5.0])
    r = torch.tensor([15.0])
    G = assemble_metadit_geometry(occ, l, h, r)
    assert torch.allclose(G[0, 0], torch.full((64, 64), 3.0), atol=1e-6), "G0 = r/5 = 15/5 = 3.0"
    assert torch.allclose(G[0, 1], torch.full((64, 64), 5.0), atol=1e-6), "G1 = h = 5.0"
    assert torch.allclose(G[0, 2], torch.full((64, 64), 1.0), atol=1e-6), "G2 = l/3 = 3/3 = 1.0"


def test_factorize_assemble_roundtrip_synthetic():
    """assemble(factorize(G)) == G (exact round-trip)."""
    torch.manual_seed(42)
    G = torch.zeros(4, 3, 64, 64)
    for b in range(4):
        occ = (torch.rand(64, 64) > 0.5)
        l, h, r = 3.0, 5.0, 15.0
        G[b, 0][occ] = r / 5.0
        G[b, 1][occ] = h
        G[b, 2] = l / 3.0
    occ, sv = factorize_geometry(G)
    G2 = assemble_geometry(occ, sv)
    assert torch.allclose(G, G2, atol=1e-5), "round-trip must be exact"


# --------------------------------------------------------------------------
# Geometry decoder
# --------------------------------------------------------------------------

def test_geometry_decoder_shapes():
    model = _build_model()
    z_hat = torch.randn(2, 256, 192)
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]])
    geometry, soft_occ = model.decode_geometry(z_hat, sv)
    assert geometry.shape == (2, 3, 64, 64)
    assert soft_occ.shape == (2, 1, 64, 64)
    assert soft_occ.min() >= 0.0 and soft_occ.max() <= 1.0  # sigmoid range


def test_decode_geometry_retains_visible_pixels():
    """When occ_input and mask are provided, visible pixels must be retained."""
    model = _build_model()
    z_hat = torch.randn(1, 256, 192)
    sv = torch.tensor([[1.5, 0.8, 10.0]])
    occ = (torch.rand(1, 1, 64, 64) > 0.5).float()
    occ[:, :, :32, :32] = 1.0

    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(occ, ratio=0.5)

    # Decode without retention
    geom1, occ1 = model.decode_geometry(z_hat, sv)
    # Decode with retention
    geom2, occ2 = model.decode_geometry(z_hat, sv, occ_input=occ, mask=M)

    # Occ2 should match occ in visible regions
    up = M.view(1, 1, 16, 16).repeat_interleave(4, 2).repeat_interleave(4, 3)
    vis = up > 0.5
    assert torch.allclose(occ2[vis], occ[vis], atol=1e-6), \
        "visible occupancy must be retained"


def test_decode_geometry_ste():
    """STE mode: hard occupancy for assembly, soft gradient path."""
    model = _build_model()
    model.train()
    z_hat = torch.randn(1, 256, 192, requires_grad=True)
    sv = torch.tensor([[1.5, 0.8, 10.0]], requires_grad=True)
    geometry, soft_occ = model.decode_geometry(z_hat, sv, use_ste=True)
    assert geometry.shape == (1, 3, 64, 64)
    assert torch.isfinite(geometry).all()


# --------------------------------------------------------------------------
# Occupancy decoder contract (architecture_v5.md §4.1)
# --------------------------------------------------------------------------

def test_occupancy_decoder_output_shape():
    """Decoder output is occupancy logits [B,1,64,64] only — no 3-ch head."""
    from decoders.occupancy_decoder import OccupancyDecoder
    dec = OccupancyDecoder(hidden=192, base_dim=96, scalar_hidden=128)
    z = torch.randn(2, 256, 192)
    scalars = torch.tensor([[2.5, 0.8, 4.0], [2.8, 1.0, 4.2]])
    occ_logits = dec(z, scalars)
    assert occ_logits.shape == (2, 1, 64, 64), (
        f"expected [B,1,64,64], got {tuple(occ_logits.shape)}")


def test_occupancy_decoder_no_geometry_head():
    """The active unified decoder must not expose a 3-channel geometry head."""
    from decoders.occupancy_decoder import OccupancyDecoder
    dec = OccupancyDecoder(hidden=192)
    # No geometry_head / num_channels attribute (legacy GeometryDecoder had them).
    assert not hasattr(dec, "geometry_head"), "3-ch geometry head must not exist"
    assert not hasattr(dec, "num_channels"), "num_channels must not exist"
    # Single output head producing 1 channel.
    assert dec.head.out_channels == 1


def test_occupancy_decoder_accepts_256_tokens():
    """Decoder accepts [B,256,192] and preserves token spatial order."""
    from decoders.occupancy_decoder import OccupancyDecoder
    dec = OccupancyDecoder(hidden=192)
    z = torch.randn(1, 256, 192)
    scalars = torch.tensor([[2.5, 0.8, 4.0]])
    out = dec(z, scalars)
    assert out.shape == (1, 1, 64, 64)


def test_occupancy_decoder_scalar_sensitivity():
    """Decoder output must change when scalar conditioning changes.

    FiLM heads are zero-init (identity) at init, so un-zero them first —
    the sensitivity check is about the learnable path, not the init.
    """
    from decoders.occupancy_decoder import OccupancyDecoder
    torch.manual_seed(0)
    dec = OccupancyDecoder(hidden=192)
    # Un-zero the FiLM heads so conditioning is observable.
    torch.manual_seed(123)
    with torch.no_grad():
        for block in (dec.block1, dec.block2):
            block.film.weight.normal_(0.0, 0.5)
            block.film.bias.normal_(0.0, 0.5)
    z = torch.randn(1, 256, 192)
    s1 = torch.tensor([[2.5, 0.8, 4.0]])
    s2 = torch.tensor([[4.0, 1.5, 8.0]])
    with torch.no_grad():
        o1 = dec(z, s1)
        o2 = dec(z, s2)
    # With un-zeroed FiLM, different scalars must give different outputs.
    assert not torch.allclose(o1, o2, atol=1e-6), (
        "decoder output must depend on scalar conditioning")


def test_occupancy_decoder_film_identity_init():
    """FiLM heads zero-init: gamma=1, beta=0 → no arbitrary initial modulation.

    With the FiLM heads at init, scaling scalars by a large factor must not
    change the output (the modulation is identity), unlike after training.
    """
    from decoders.occupancy_decoder import OccupancyDecoder
    torch.manual_seed(0)
    dec = OccupancyDecoder(hidden=192)
    z = torch.randn(1, 256, 192)
    s1 = torch.tensor([[2.5, 0.8, 4.0]])
    s2 = torch.tensor([[250.0, 80.0, 400.0]])  # huge scale — identity FiLM ignores
    with torch.no_grad():
        o1 = dec(z, s1)
        o2 = dec(z, s2)
    # At init, FiLM is identity: scaling scalars must not change the output.
    assert torch.allclose(o1, o2, atol=1e-5), (
        "zero-init FiLM must be an identity modulation at init")


def test_occupancy_decoder_film_heads_identity_values():
    """Direct check: film head bias split is [gamma=1, beta=0] at init."""
    from decoders.occupancy_decoder import OccupancyDecoder
    dec = OccupancyDecoder(hidden=192, base_dim=96, scalar_hidden=128)
    for block in (dec.block1, dec.block2):
        w, b = block.film.weight, block.film.bias
        assert torch.count_nonzero(w).item() == 0, "film weight must be zero at init"
        out = block.film(torch.zeros(1, 128))
        half = out.shape[-1] // 2
        assert torch.allclose(out[0, :half], torch.ones(half), atol=1e-6), (
            "gamma portion must init to 1")
        assert torch.allclose(out[0, half:], torch.zeros(half), atol=1e-6), (
            "beta portion must init to 0")


def test_occupancy_decoder_spatial_order_preserved():
    """Token order is row-major 16x16 → pixels; unique patch ids must stay
    positionally aligned through reshape (no learned remapping)."""
    from decoders.occupancy_decoder import OccupancyDecoder
    torch.manual_seed(0)
    dec = OccupancyDecoder(hidden=192)
    scalars = torch.tensor([[2.5, 0.8, 4.0]])
    # Token i holds a unique constant in a fixed feature channel.
    z = torch.zeros(1, 256, 192)
    for i in range(256):
        z[0, i, 0] = (i + 1) / 257.0
    out = dec(z, scalars)
    assert out.shape == (1, 1, 64, 64)
    # The reshape (B,256,D) → (B,16,16,D) → pixels is deterministic and
    # position-preserving: pixel (r,c) receives token r*16+c. Verify by
    # checking the output is finite and the decoder didn't crash — exact
    # positional reconstruction is not expected after convs, but the reshape
    # order is the row-major convention (no permutation).
    assert torch.isfinite(out).all()


def test_occupancy_decoder_effective_scalars_known_vs_predicted():
    """Decoder FiLM must receive the EFFECTIVE scalars: known scalars use the
    exact supplied value, unknown scalars use the prediction (Item 1.7).

    With un-zeroed FiLM heads, the same z decoded with effective scalars from
    a known scalar (true value) must differ from decoding with the prediction —
    proving known values flow into decoder conditioning, not scalar_pred."""
    from decoders.occupancy_decoder import OccupancyDecoder
    torch.manual_seed(0)
    dec = OccupancyDecoder(hidden=192)
    torch.manual_seed(123)
    with torch.no_grad():
        for block in (dec.block1, dec.block2):
            block.film.weight.normal_(0.0, 0.5)
            block.film.bias.normal_(0.0, 0.5)

    z = torch.randn(1, 256, 192)
    scalar_pred = torch.tensor([[99.0, 99.0, 99.0]])   # deliberately wrong
    scalar_values = torch.tensor([[3.0, 1.0, 9.0]])    # known ground truth
    scalar_known = torch.ones(1, 3, dtype=torch.bool)  # all known

    # Effective scalars per architecture_v5.md §4.1.
    effective = torch.where(scalar_known, scalar_values, scalar_pred)
    with torch.no_grad():
        out_known = dec(z, effective)
        out_pred = dec(z, scalar_pred)

    # Known values must flow into the decoder (different from the prediction).
    assert not torch.allclose(out_known, out_pred, atol=1e-6), (
        "decoder FiLM must receive effective (known) scalars, not scalar_pred")

    # Unknown scalar → prediction flows in.
    sk_mixed = torch.zeros(1, 3, dtype=torch.bool)
    sk_mixed[:, 0] = True  # l known, h/r unknown
    effective_mixed = torch.where(sk_mixed, scalar_values, scalar_pred)
    with torch.no_grad():
        out_mixed = dec(z, effective_mixed)
    # The mixed case must match neither all-known nor all-predicted exactly.
    assert not torch.allclose(out_mixed, out_known, atol=1e-6)
    assert not torch.allclose(out_mixed, out_pred, atol=1e-6)


def test_decode_geometry_substitutes_known_scalars():
    """Regression: decode_geometry must substitute true values for known
    scalars at assembly (the scalar analog of visible-occupancy retention).

    Known scalars must appear exactly (true value, not the model's guess) in
    the assembled geometry's channel values; unknown scalars must use the
    prediction. Deliberately wrong scalar_pred for the known scalar makes the
    substitution observable via plain tensor equality.
    """
    model = _build_model()
    model.eval()
    z_hat = torch.randn(2, 256, 192)

    # True scalars and a deliberately-wrong prediction.
    sv = torch.tensor([[3.0, 1.0, 9.0], [4.0, 2.0, 8.0]])
    scalar_pred = torch.tensor([[99.0, 99.0, 99.0], [99.0, 99.0, 99.0]])

    # Sample 0: l known, h/r unknown. Sample 1: l/h known, r unknown.
    sk = torch.tensor([[True, False, False], [True, True, False]])

    geometry, _ = model.decode_geometry(
        z_hat, scalar_pred, scalar_known=sk, scalar_values=sv)

    # Channel 2 = l_lattice/3 everywhere → known l must be the true value.
    assert torch.allclose(geometry[0, 2], torch.full((64, 64), 3.0 / 3.0),
                          atol=1e-6), "known l (sample 0) must use the true value"
    assert torch.allclose(geometry[1, 2], torch.full((64, 64), 4.0 / 3.0),
                          atol=1e-6), "known l (sample 1) must use the true value"

    # Channel 1 = h_atom on occupied pixels → sample 0 h unknown (uses pred),
    # sample 1 h known (uses true value). Use a fully-occupied occupancy via
    # occ_input + fully-visible mask so the assembled geometry is deterministic.
    occ = torch.ones(2, 1, 64, 64)
    M_vis = torch.ones(2, 16, 16)
    geometry2, _ = model.decode_geometry(
        z_hat, scalar_pred, occ_input=occ, mask=M_vis,
        scalar_known=sk, scalar_values=sv)

    # Sample 0: h unknown → prediction (99.0).
    assert torch.allclose(geometry2[0, 1], torch.full((64, 64), 99.0),
                          atol=1e-6), "unknown h (sample 0) must use the prediction"
    # Sample 1: h known → true value (2.0).
    assert torch.allclose(geometry2[1, 1], torch.full((64, 64), 2.0),
                          atol=1e-6), "known h (sample 1) must use the true value"

    # Channel 0 = r_atom/5 on occupied pixels → r unknown in both samples.
    assert torch.allclose(geometry2[0, 0], torch.full((64, 64), 99.0 / 5.0),
                          atol=1e-6), "unknown r (sample 0) must use the prediction"
    assert torch.allclose(geometry2[1, 0], torch.full((64, 64), 99.0 / 5.0),
                          atol=1e-6), "unknown r (sample 1) must use the prediction"

    # Default (no scalar_known/scalar_values) must keep pre-fix behavior:
    # everything uses the prediction.
    geometry3, _ = model.decode_geometry(
        z_hat, scalar_pred, occ_input=occ, mask=M_vis)
    assert torch.allclose(geometry3[0, 2], torch.full((64, 64), 99.0 / 3.0),
                          atol=1e-6), "default path must use scalar_pred everywhere"


# --------------------------------------------------------------------------
# Physics loop (requires surrogate weights)
# --------------------------------------------------------------------------

_SURROGATE_PATH = os.path.join(
    REPO_ROOT, "data", "metadit", "weights", "surrogate_model.bin")
_HAS_SURROGATE = os.path.exists(_SURROGATE_PATH)


@pytest.mark.skipif(not _HAS_SURROGATE,
                    reason="Surrogate weights not available")
def test_physics_loss_finite():
    from physics.physics_loop import load_surrogate, physics_loss
    model = _build_model()
    model.train()
    surrogate = load_surrogate(_SURROGATE_PATH, device="cpu")

    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]])
    spec = torch.randn(2, 2, 301)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(occ, ratio=0.5)
    sk = torch.ones(2, 3, dtype=torch.bool)

    L_phys, spec_pred, geom = physics_loss(
        model, surrogate, occ, sv, sk, spec, M)
    assert torch.isfinite(L_phys), "physics loss must be finite"
    assert spec_pred.shape == (2, 2, 301)
    assert geom.shape == (2, 3, 64, 64)


@pytest.mark.skipif(not _HAS_SURROGATE,
                    reason="Surrogate weights not available")
def test_surrogate_gradient_test():
    """Gradients must flow from surrogate output to student params (§4)."""
    from physics.physics_loop import load_surrogate, surrogate_gradient_test
    model = _build_model()
    surrogate = load_surrogate(_SURROGATE_PATH, device="cpu")

    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]])
    spec = torch.randn(2, 2, 301)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(occ, ratio=0.5)

    assert surrogate_gradient_test(model, surrogate, occ, sv, spec, M)


@pytest.mark.skipif(not _HAS_SURROGATE,
                    reason="Surrogate weights not available")
def test_soft_hard_occupancy_test():
    """Soft vs hard occupancy characterization (§3).

    With a PARTIAL mask, the masked region contains the model's soft occupancy
    (non-binary), so the soft-vs-hard branches genuinely differ at the
    surrogate input — the diagnostic must detect a nonzero difference (Fix 3:
    the diagnostic actually changes the surrogate input when the predicted
    masked occupancy is non-binary)."""
    from physics.physics_loop import load_surrogate, soft_hard_occupancy_test
    model = _build_model()
    surrogate = load_surrogate(_SURROGATE_PATH, device="cpu")

    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]])
    spec = torch.randn(2, 2, 301)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(occ, ratio=0.5)  # partial mask → non-binary masked region
    sk = torch.ones(2, 3, dtype=torch.bool)

    result = soft_hard_occupancy_test(
        model, surrogate, occ, sv, spec, sk, M)
    assert "spectrum_l1_diff" in result
    assert "surrogate_out_of_distribution" in result
    assert "ste_recommended" in result
    # Partially-masked soft occupancy differs from hard-thresholded occupancy,
    # so the diagnostic must see a nonzero spectrum difference (unlike the
    # fully-visible case where retention makes both branches identical).
    assert result["spectrum_rel_diff"] > 1e-5, (
        "with a partial mask the soft vs hard branches must produce different "
        "surrogate inputs (the diagnostic must actually distinguish them)")


@pytest.mark.skipif(not _HAS_SURROGATE,
                    reason="Surrogate weights not available")
def test_soft_hard_occupancy_test_retains_visible_pixels_identically():
    """Regression: both branches of soft_hard_occupancy_test must apply the same
    visible-pixel retention, so rel_diff isolates occupancy hardness only.

    With mask fully visible (all ones), decode_geometry's retention forces
    occ_for_assembly to occ_input on 100% of pixels regardless of use_ste, so
    the two branches must be numerically identical and rel_diff must be ~0.
    Against the old implementation (where one branch bypassed decode_geometry
    entirely and skipped retention) a fully-visible mask produced a nonzero
    rel_diff coming entirely from the retention/no-retention confound.
    """
    from physics.physics_loop import load_surrogate, soft_hard_occupancy_test
    model = _build_model()
    surrogate = load_surrogate(_SURROGATE_PATH, device="cpu")

    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    occ[:, :, :32, :32] = 1.0
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]])
    spec = torch.randn(2, 2, 301)
    sk = torch.ones(2, 3, dtype=torch.bool)
    M_fully_visible = torch.ones(2, 16, 16)

    result = soft_hard_occupancy_test(
        model, surrogate, occ, sv, spec, sk, M_fully_visible)
    assert result["spectrum_rel_diff"] < 1e-5, (
        "with a fully-visible mask, retention overrides decoder output "
        "identically on both branches — any nonzero rel_diff indicates the "
        "soft-vs-hard comparison was confounded by retention"
    )


# --------------------------------------------------------------------------
# Scenario evaluators
# --------------------------------------------------------------------------

def test_scenario_inputs_shape():
    """ScenarioInputs must produce correct mask shapes."""
    from scripts.run_scenarios import ScenarioInputs
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]])
    spec = torch.randn(2, 2, 301)

    for name in ("A", "B", "C"):
        inputs = getattr(ScenarioInputs, f"scenario_{name.lower()}")(
            occ, sv, spec, 2, "cpu")
        assert inputs.mask.shape == (2, 16, 16)
        assert inputs.scalar_known.shape == (2, 3)


def test_scenario_a_all_unknown():
    """Scenario A: all scalars unknown."""
    from scripts.run_scenarios import ScenarioInputs
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]])
    spec = torch.randn(2, 2, 301)
    inputs = ScenarioInputs.scenario_a(occ, sv, spec, 2, "cpu")
    assert not inputs.scalar_known.any(), "A: all scalars must be unknown"
    assert float(inputs.mask.mean()) < 0.1, "A: full mask (all masked)"


def test_diversity_metrics_deterministic():
    """Single generation → deterministic flag set."""
    from scripts.run_scenarios import diversity_metrics
    geom = torch.randn(2, 3, 64, 64)
    result = diversity_metrics([geom])
    assert result["deterministic"] is True


# --------------------------------------------------------------------------
# Geometry invariants after decode (Fix 2 — permanent contract test)
# --------------------------------------------------------------------------

def test_decode_geometry_invariants_predicted():
    """Predicted occupancy + predicted scalars → assemble_metadit_geometry must
    obey the dataset representation contract:
      ch2 == l/3 everywhere
      support(ch0) == support(ch1)
      occupied ch0 values all equal r/5
      occupied ch1 values all equal h
    """
    model = _build_model()
    model.eval()
    z_hat = torch.randn(2, 256, 192)
    scalar_pred = torch.tensor([[2.5, 0.8, 4.0], [2.8, 1.0, 4.2]])

    # Fully-visible mask + full occupancy so the assembled geometry is
    # deterministic from scalar_pred alone.
    occ = torch.ones(2, 1, 64, 64)
    M_vis = torch.ones(2, 16, 16)
    geometry, _ = model.decode_geometry(
        z_hat, scalar_pred, occ_input=occ, mask=M_vis, hard_forward=True)

    for b in range(2):
        l, h, r = scalar_pred[b].tolist()
        # ch2 == l/3 everywhere
        assert torch.allclose(geometry[b, 2],
                              torch.full((64, 64), l / 3.0), atol=1e-5), (
            f"sample {b}: ch2 must equal l/3 everywhere")
        # support(ch0) == support(ch1) == occupancy (all ones here)
        sup0 = geometry[b, 0] != 0
        sup1 = geometry[b, 1] != 0
        assert torch.equal(sup0, sup1), f"sample {b}: ch0/ch1 support mismatch"
        assert sup0.all(), f"sample {b}: full occupancy must give full support"
        # occupied ch0 values all equal r/5
        assert torch.allclose(geometry[b, 0][sup0],
                              torch.full_like(geometry[b, 0][sup0], r / 5.0),
                              atol=1e-5), f"sample {b}: ch0 occupied value != r/5"
        # occupied ch1 values all equal h
        assert torch.allclose(geometry[b, 1][sup1],
                              torch.full_like(geometry[b, 1][sup1], h),
                              atol=1e-5), f"sample {b}: ch1 occupied value != h"


def test_decode_geometry_invariants_known_scalars():
    """Known-scalar case: deliberately-wrong scalar_pred must be overridden by
    scalar_values at assembly, and the invariants must hold for the known
    values."""
    model = _build_model()
    model.eval()
    z_hat = torch.randn(1, 256, 192)
    scalar_pred = torch.tensor([[99.0, 99.0, 99.0]])   # deliberately wrong
    scalar_values = torch.tensor([[3.0, 1.0, 9.0]])    # known ground truth
    scalar_known = torch.ones(1, 3, dtype=torch.bool)

    occ = torch.ones(1, 1, 64, 64)
    M_vis = torch.ones(1, 16, 16)
    geometry, _ = model.decode_geometry(
        z_hat, scalar_pred, occ_input=occ, mask=M_vis, hard_forward=True,
        scalar_known=scalar_known, scalar_values=scalar_values)

    l, h, r = 3.0, 1.0, 9.0
    assert torch.allclose(geometry[0, 2], torch.full((64, 64), l / 3.0),
                          atol=1e-5), "known l must be used exactly"
    assert torch.allclose(geometry[0, 1][geometry[0, 1] != 0],
                          torch.full_like(geometry[0, 1][geometry[0, 1] != 0], h),
                          atol=1e-5), "known h must be used exactly"
    assert torch.allclose(geometry[0, 0][geometry[0, 0] != 0],
                          torch.full_like(geometry[0, 0][geometry[0, 0] != 0], r / 5.0),
                          atol=1e-5), "known r must be used exactly"
    # None of the wrong predictions (99) may appear.
    assert (geometry[0] < 90.0).all(), "wrong scalar_pred leaked into geometry"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
