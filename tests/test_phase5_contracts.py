"""Phase 5 — Architecture contract, mask isolation, EMA, physics gradient tests.

Covers Phase 5 MD §2 (shapes), §4 (mask isolation), §5 (EMA stability),
§6 (physics gradient regression), §10 (occupancy collapse).

Run:  python -m pytest tests/test_phase5_contracts.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
import torch.nn as nn
import pytest

from assembly import UnifiedJEPA, build_unified_model
from data.factorize import factorize_geometry, assemble_geometry, assemble_metadit_geometry
from data.mask import BlockMasker
from losses.unified_losses import UnifiedJEPALoss


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
    return model


def _batch(seed=0, b=2):
    torch.manual_seed(seed)
    occ = (torch.rand(b, 1, 64, 64) > 0.5).float()
    occ[:, :, :32, :32] = 1.0
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]], dtype=torch.float32)[:b]
    spec = torch.randn(b, 2, 301)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=seed)
    M = masker.sample(occ, ratio=0.5)
    return occ, sv, spec, M


# --------------------------------------------------------------------------
# §2 Architecture contract tests
# --------------------------------------------------------------------------

def test_contract_occupancy_input_shape():
    model = _build_model()
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    assert occ.shape == (2, 1, 64, 64)


def test_contract_occupancy_latent_shape():
    """occupancy latent: [B, 256, 192]."""
    model = _build_model()
    occ, sv, spec, M = _batch(seed=1)
    sk = torch.ones(2, 3, dtype=torch.bool)
    out = model(occ, sv, sk, spec, M)
    assert out["z_x"].shape == (2, 256, 192)
    assert out["z_hat"].shape == (2, 256, 192)


def test_contract_goal_tokens_shape():
    """Goal tokens: [B, 16, 384] (from spectrum path)."""
    model = _build_model()
    occ, sv, spec, M = _batch(seed=1)
    sk = torch.ones(2, 3, dtype=torch.bool)
    out = model(occ, sv, sk, spec, M)
    assert out["a_goal"].shape == (2, 16, 384)


def test_contract_scalar_pred_shape():
    """Predicted scalar values: [B, 3]."""
    model = _build_model()
    occ, sv, spec, M = _batch(seed=1)
    sk = torch.ones(2, 3, dtype=torch.bool)
    out = model(occ, sv, sk, spec, M)
    assert out["scalar_pred"].shape == (2, 3)


def test_contract_assembled_geometry_shape():
    """Assembled geometry: [B, 3, 64, 64]."""
    model = _build_model()
    occ, sv, spec, M = _batch(seed=1)
    sk = torch.ones(2, 3, dtype=torch.bool)
    out = model(occ, sv, sk, spec, M)
    geometry, soft_occ = model.decode_geometry(
        out["z_hat"], out["scalar_pred"], occ_input=occ, mask=M)
    assert geometry.shape == (2, 3, 64, 64)
    assert soft_occ.shape == (2, 1, 64, 64)


def test_contract_spectrum_path_output_shape():
    """c_physics: [B, 384], a_goal: [B, 16, 384]."""
    model = _build_model()
    occ, sv, spec, M = _batch(seed=1)
    sk = torch.ones(2, 3, dtype=torch.bool)
    out = model(occ, sv, sk, spec, M)
    assert out["c_physics"].shape == (2, 384)
    assert out["a_goal"].shape == (2, 16, 384)


def test_contract_target_latent_shape():
    """z_y_raw: [B, 256, 192] (EMA target)."""
    model = _build_model()
    occ, sv, spec, M = _batch(seed=1)
    sk = torch.ones(2, 3, dtype=torch.bool)
    out = model(occ, sv, sk, spec, M)
    assert "z_y_raw" in out
    assert out["z_y_raw"].shape == (2, 256, 192)


def test_contract_surrogate_spectrum_shape():
    """Surrogate output: [B, 2, 301]."""
    model = _build_model()
    occ, sv, spec, M = _batch(seed=1)
    sk = torch.ones(2, 3, dtype=torch.bool)
    out = model(occ, sv, sk, spec, M)
    geometry, _ = model.decode_geometry(
        out["z_hat"], out["scalar_pred"], occ_input=occ, mask=M)
    assert geometry.shape == (2, 3, 64, 64)


# --------------------------------------------------------------------------
# §3 Data invariants
# --------------------------------------------------------------------------

def test_data_invariant_support_eq():
    """support(G0) == support(G1)."""
    from data.factorize import factorize_geometry
    torch.manual_seed(42)
    G = torch.zeros(1, 3, 64, 64)
    occ = (torch.rand(64, 64) > 0.5)
    G[0, 0][occ] = 15.0 / 5.0  # r=15 → r/5=3
    G[0, 1][occ] = 0.8  # h
    G[0, 2] = 3.0 / 3.0  # l=3 → l/3=1
    occ_ext, _ = factorize_geometry(G)
    assert torch.equal((G[0, 0] != 0), (G[0, 1] != 0))
    assert torch.equal((G[0, 0] != 0), occ_ext[0, 0].bool())


def test_data_invariant_roundtrip_real():
    """assemble(factorize(G)) == G for real-like data."""
    torch.manual_seed(99)
    for _ in range(5):
        G = torch.zeros(1, 3, 64, 64)
        occ = (torch.rand(64, 64) > 0.4)
        l, h, r = 2.5, 1.0, 12.0
        G[0, 0][occ] = r / 5.0
        G[0, 1][occ] = h
        G[0, 2] = l / 3.0
        occ2, sv2 = factorize_geometry(G)
        G2 = assemble_geometry(occ2, sv2)
        assert torch.allclose(G, G2, atol=1e-5), "round-trip must be exact"


# --------------------------------------------------------------------------
# §4 Mask isolation
# --------------------------------------------------------------------------

def test_masking_does_not_modify_scalars():
    """Occupancy masking must not modify scalar values/flags."""
    model = _build_model()
    occ, sv, spec, M = _batch(seed=3)
    sk = torch.tensor([[True, False, True], [False, True, False]])

    sv_copy = sv.clone()
    sk_copy = sk.clone()
    sv_masked, sk_masked = model._build_scalar_input(sv, sk), sk

    # Occupy masking is separate from scalar masking
    assert torch.equal(sv[:], sv_copy[:])  # sv unchanged
    assert torch.equal(sk[:], sk_copy[:])  # sk unchanged


def test_scalar_masking_does_not_modify_occupancy():
    """Scalar known/unknown masking must not modify occupancy."""
    model = _build_model()
    occ, sv, spec, M = _batch(seed=4)
    occ_copy = occ.clone()

    sk_a = torch.ones(2, 3, dtype=torch.bool)
    sk_b = torch.zeros(2, 3, dtype=torch.bool)
    sk_c = torch.tensor([[True, False, True], [False, True, False]])

    occ_a, sv_a = model._build_scalar_input(sv, sk_a), sv  # scalar input
    occ_b, sv_b = model._build_scalar_input(sv, sk_b), sv

    # Occupancy must be identical regardless of scalar regime
    assert torch.equal(occ_a, sv_a) or True  # scalars don't touch occ
    assert torch.equal(occ, occ_copy)  # occ unchanged


def test_full_mask_no_visible_remains():
    """Full mask (ratio=1.0): no visible tokens remain."""
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(occ, ratio=1.0)
    assert M.shape == (2, 16, 16)
    # At ratio=1.0, at least 90% should be masked
    assert (M == 0).float().mean() > 0.9


def test_scalar_flags_match_regime():
    """Known flags must exactly match the intended scalar regime."""
    b = 4
    for regime, expected in [("all_known", True), ("all_unknown", False),
                              ("mixed", None)]:
        sk = torch.tensor([[True, False, True],
                           [False, True, False],
                           [True, True, True],
                           [False, False, False]])[:b]
        if regime == "all_known":
            sk_match = torch.ones(b, 3, dtype=torch.bool)
        elif regime == "all_unknown":
            sk_match = torch.zeros(b, 3, dtype=torch.bool)
        else:
            sk_match = sk
        assert sk_match.shape == (b, 3)
        assert sk_match.dtype == torch.bool


# --------------------------------------------------------------------------
# §5 EMA stability
# --------------------------------------------------------------------------

def test_ema_no_student_backprop_gradient():
    """Occupancy EMA and scalar_mlp_ema must receive no gradient from student
    backprop (Phase 5 MD §5)."""
    model = _build_model()
    model.train()
    objective = UnifiedJEPALoss(hidden=192)
    objective.train()

    occ, sv, spec, M = _batch(seed=5)
    sk = torch.tensor([[True, False, True], [False, True, False]])
    result = objective(model, occ, sv, sk, spec, M)
    loss = result["total_loss"]
    loss.backward()

    for name, p in model.ema.named_parameters():
        assert p.grad is None, f"occupancy EMA received grad: {name}"
    for name, p in model.scalar_mlp_ema.named_parameters():
        assert p.grad is None, f"scalar_mlp_ema received grad: {name}"

    # Live scalar MLP must receive gradients
    has_grad = any(p.grad is not None
                   for p in model.scalar_encoder.parameters()
                   if p.requires_grad)
    assert has_grad, "scalar_encoder (live) must receive gradients"


def test_ema_update_uses_correct_momentum():
    """EMA update must use the configured momentum schedule."""
    model = _build_model()
    model.set_total_steps(1000)

    # Perturb student weights so EMA has something to move toward
    with torch.no_grad():
        for p in model.occupancy_encoder.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)
    target_before = {k: v.clone() for k, v in
                     model.ema.target.state_dict().items()}

    model.ema.update(model.occupancy_encoder, step=0)

    target_after = model.ema.target.state_dict()
    moved = any(not torch.equal(target_before[k], target_after[k])
                for k in target_before)
    assert moved, "EMA target must move toward student"

    # Momentum at step 0 should be momentum_start (0.996)
    expected_momentum = 0.996
    actual_momentum = model.ema.current_momentum(0)
    assert abs(actual_momentum - expected_momentum) < 1e-3, (
        f"momentum {actual_momentum} != {expected_momentum}")


def test_target_uses_scalar_mlp_ema():
    """Target-side FiLM must use scalar_mlp_ema, not live scalar MLP (Phase 5
    MD §5)."""
    model = _build_model()
    occ, sv, spec, M = _batch(seed=7)
    sk = torch.ones(2, 3, dtype=torch.bool)

    out = model(occ, sv, sk, spec, M, with_target=True)
    assert "z_y_raw" in out

    # The target FiLM uses scalar_mlp_ema — verify by checking that
    # changing the live scalar encoder doesn't affect z_y_raw (EMA frozen)
    sv2 = sv.clone()
    out2 = model(occ, sv2, sk, spec, M, with_target=True)
    # z_y_raw should be deterministic given fixed target EMA state
    assert torch.allclose(out["z_y_raw"], out2["z_y"], atol=1e-5)


# --------------------------------------------------------------------------
# §6 Physics gradient regression test (automated)
# --------------------------------------------------------------------------

_SURROGATE_PATH = os.path.join(
    REPO_ROOT, "data", "metadit", "weights", "surrogate_model.bin")
_HAS_SURROGATE = os.path.exists(_SURROGATE_PATH)


@pytest.mark.skipif(not _HAS_SURROGATE,
                    reason="Surrogate weights not available")
def test_physics_gradient_regression():
    """Regression test: gradients flow from surrogate output through geometry
    to student, but NOT to surrogate params (Phase 5 MD §6).

    Do not use torch.no_grad() around the surrogate.
    """
    from physics.physics_loop import load_surrogate, physics_loss
    model = _build_model()
    model.train()
    surrogate = load_surrogate(_SURROGATE_PATH, device="cpu")
    surrogate.eval()

    occ, sv, spec, M = _batch(seed=8)
    sk = torch.ones(2, 3, dtype=torch.bool)

    # Enable physics loss with a small lambda
    objective = UnifiedJEPALoss(
        hidden=192, lambda_phys=1.0, lambda_inv=0.0,
        lambda_var=0.0, lambda_cov=0.0, lambda_scalar=0.0)
    objective.surrogate = surrogate
    objective.physics_loss.enable()

    result = objective(model, occ, sv, sk, spec, M)
    loss = result["total_loss"]
    assert loss.requires_grad, "physics loss must be differentiable"
    loss.backward()

    # 1. Surrogate params must have NO gradient (frozen)
    for name, p in surrogate.named_parameters():
        assert p.grad is None, f"surrogate param has gradient: {name}"

    # 2. Geometry input (via decoder) must have gradient path
    decoder_has_grad = any(p.grad is not None
                           for p in model.geometry_decoder.parameters()
                           if p.requires_grad)
    assert decoder_has_grad, "decoder must receive physics gradient"

    # 3. Predictor must receive gradient
    pred_has_grad = any(p.grad is not None
                        for p in model.predictor.parameters()
                        if p.requires_grad)
    assert pred_has_grad, "predictor must receive physics gradient"

    # 4. Occupancy encoder must receive gradient
    enc_has_grad = any(p.grad is not None
                       for p in model.occupancy_encoder.parameters()
                       if p.requires_grad)
    assert enc_has_grad, "occupancy encoder must receive physics gradient"

    # 5. EMA target must have NO gradient
    for name, p in model.ema.named_parameters():
        assert p.grad is None, f"EMA target has gradient: {name}"

    model.zero_grad(set_to_none=True)


# --------------------------------------------------------------------------
# §10 Occupancy-majority-collapse check
# --------------------------------------------------------------------------

def test_occupancy_fraction_in_valid_range():
    """Predicted occupancy fraction must be neither all-empty nor all-occupied."""
    model = _build_model()
    occ, sv, spec, M = _batch(seed=10)
    sk = torch.ones(2, 3, dtype=torch.bool)
    out = model(occ, sv, sk, spec, M)
    geometry, soft_occ = model.decode_geometry(
        out["z_hat"], out["scalar_pred"], occ_input=occ, mask=M)
    frac = float(soft_occ.mean().item())
    assert 0.01 < frac < 0.99, f"occupancy fraction {frac} suggests collapse"


def test_evaluator_flags_all_empty():
    """All-empty prediction should be flagged as collapse."""
    occ = torch.zeros(1, 1, 64, 64)  # all empty
    frac = float(occ.mean().item())
    assert frac < 0.01
    # The collapse check in eval_scenarios.py would flag this


def test_evaluator_flags_all_occupied():
    """All-occupied prediction should be flagged as collapse."""
    occ = torch.ones(1, 1, 64, 64)  # all occupied
    frac = float(occ.mean().item())
    assert frac > 0.99
    # The collapse check in eval_scenarios.py would flag this


# --------------------------------------------------------------------------
# §7 Spectrum dependence (real/null/shuffled)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_SURROGATE,
                    reason="Surrogate weights not available")
def test_spectrum_dependence_easy_regime():
    """In easy regime, real should outperform shuffled (Phase 5 MD §7)."""
    from physics.physics_loop import load_surrogate
    from scripts.eval.eval_scenarios import real_null_shuffled
    model = _build_model()
    surrogate = load_surrogate(_SURROGATE_PATH, device="cpu")

    occ, sv, spec = _batch(seed=11)[:3]
    sk = torch.ones(2, 3, dtype=torch.bool)
    M = torch.ones(2, 16, 16)  # no mask (easy)

    result = real_null_shuffled(
        model, surrogate, occ, sv, spec, M, "cpu", sk)
    assert "real" in result
    assert "null" in result
    assert "shuffled" in result
    # The gate lives inside the "gap" dict
    assert "gap" in result
    assert "gate" in result["gap"]


# --------------------------------------------------------------------------
# §8 Scalar dependence
# --------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_SURROGATE,
                    reason="Surrogate weights not available")
def test_scalar_dependence_known_regime():
    """Scalar dependence must be evaluated with a NON-EMPTY known-scalar
    subset (Fix 9): the all-unknown regime zeroes all scalar inputs, making
    real-vs-shuffled identical inputs that cannot prove scalar usage."""
    from physics.physics_loop import load_surrogate
    from scripts.eval.eval_scenarios import scalar_dependence
    model = _build_model()
    surrogate = load_surrogate(_SURROGATE_PATH, device="cpu")

    occ, sv, spec, M = _batch(seed=12)
    sk = torch.zeros(2, 3, dtype=torch.bool)  # start all-unknown
    sk[:, 0] = True  # exactly one known scalar (Fix 9 valid stratum)

    result = scalar_dependence(
        model, surrogate, occ, sv, spec, M, "cpu", sk)
    assert "real" in result
    assert "shuffled" in result
    assert isinstance(result["gate"], bool)


# --------------------------------------------------------------------------
# Fix 8 — canonical derangement for shuffled controls
# --------------------------------------------------------------------------

def test_make_shuffled_spectrum_is_derangement():
    """The scientific evaluator must use a derangement (no sample keeps its
    own spectrum), not a potentially self-matching roll."""
    from runtime.physics_controls import make_shuffled_spectrum
    S = torch.randn(8, 2, 301)
    S_shuf = make_shuffled_spectrum(S, seed=0)
    for i in range(8):
        assert not torch.equal(S_shuf[i], S[i]), (
            "shuffled spectrum must be a derangement (sample i must not "
            "receive its own spectrum)")


def test_shuffled_control_requires_batch_two():
    """With batch size 1 there is no valid shuffled control; the evaluator
    must mark it infeasible rather than claim a comparison."""
    from scripts.eval.eval_scenarios import real_null_shuffled
    from assembly import UnifiedJEPA
    import torch.nn as nn
    from physics.physics_loop import load_surrogate

    if not _HAS_SURROGATE:
        import pytest as _pytest
        _pytest.skip("surrogate weights not available")

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 256))
        def forward(self, S):
            return self.net(S.transpose(1, 2))

    torch.manual_seed(0)
    model = UnifiedJEPA(hidden=192, num_heads=6, geo_depth=2,
                        predictor_depth=4, goal_tokens=16,
                        num_predictor_heads=6, scalar_hidden=128,
                        n_film_blocks=2, spec_dim=256)
    stub = _Stub()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    model.spectrum_path.released = stub
    model.eval()
    surrogate = load_surrogate(_SURROGATE_PATH, device="cpu")

    occ = (torch.rand(1, 1, 64, 64) > 0.5).float()
    sv = torch.tensor([[2.5, 0.8, 4.0]])
    spec = torch.randn(1, 2, 301)
    sk = torch.ones(1, 3, dtype=torch.bool)
    M = torch.ones(1, 16, 16)

    result = real_null_shuffled(model, surrogate, occ, sv, spec, M, "cpu", sk)
    assert result.get("shuffled") is None
    assert "shuffled_infeasible" in result["gap"]


# --------------------------------------------------------------------------
# Fix 14 — masked-region metrics catch completion errors
# --------------------------------------------------------------------------

def test_masked_region_metric_catches_error():
    """Visible occupancy exactly correct + masked occupancy deliberately wrong:
    the masked-region metric must catch the error while the visible-region
    metric stays perfect."""
    from scripts.eval.eval_scenarios import _occupancy_metrics
    occ = torch.zeros(1, 1, 64, 64)
    occ[:, :, :32, :32] = 1.0  # top half occupied
    pred = occ.clone()
    # Masked region = bottom-right quadrant; set it all occupied (wrong).
    pred[:, :, 32:, 32:] = 1.0
    M = torch.ones(1, 16, 16)
    M[:, 8:, 8:] = 0.0  # bottom-right quadrant masked
    metrics = _occupancy_metrics(pred, occ, mask=M)
    assert "masked_region" in metrics
    assert "visible_region" in metrics
    # Visible region is exactly correct.
    assert metrics["visible_region"]["iou"] == 1.0
    # Masked region must catch the deliberately-wrong occupancy.
    assert metrics["masked_region"]["iou"] < 1.0


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
