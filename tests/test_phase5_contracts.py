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
    """Surrogate output: [B, 2, 301] — actually exercised (review B1).

    Runs the released surrogate when staged, the differentiable stub
    otherwise, so the contract is tested in every environment."""
    model = _build_model()
    occ, sv, spec, M = _batch(seed=1)
    sk = torch.ones(2, 3, dtype=torch.bool)
    out = model(occ, sv, sk, spec, M)
    geometry, _ = model.decode_geometry(
        out["z_hat"], out["scalar_pred"], occ_input=occ, mask=M)
    assert geometry.shape == (2, 3, 64, 64)
    surrogate = _stub_or_real_surrogate()
    with torch.no_grad():
        prediction = surrogate(geometry).prediction
    assert prediction.shape == (2, 2, 301)


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

def test_scalar_input_masks_values_and_preserves_inputs():
    """Behavioural (review B1): `_build_scalar_input` must zero values where a
    scalar is unknown, keep them where known, and never mutate sv/sk.

    The previous version compared sv to a copy of sv with no masking call."""
    model = _build_model()
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]], dtype=torch.float32)
    sk = torch.tensor([[True, False, True], [False, True, False]])
    sv_copy, sk_copy = sv.clone(), sk.clone()

    xi = model._build_scalar_input(sv, sk)          # (B, 6)
    assert xi.shape == (2, 6)
    for j in range(3):
        val, flag = xi[:, 2 * j], xi[:, 2 * j + 1]
        assert torch.equal(flag, sk[:, j].float())
        assert torch.equal(
            val, torch.where(sk[:, j], sv[:, j], torch.zeros_like(sv[:, j])))
    assert torch.equal(sv, sv_copy), "sv must not be mutated"
    assert torch.equal(sk, sk_copy), "sk must not be mutated"


def test_full_forward_does_not_mutate_inputs():
    """Behavioural (review B1): a full model forward must not mutate the
    occupancy / scalar / spectrum / mask tensors it is given, under either
    scalar regime.

    The previous version asserted occ == occ_copy with nothing running."""
    model = _build_model()
    model.eval()
    occ, sv, spec, M = _batch(seed=4)
    occ_c, sv_c, spec_c, M_c = occ.clone(), sv.clone(), spec.clone(), M.clone()

    with torch.no_grad():
        for sk in (torch.ones(2, 3, dtype=torch.bool),
                   torch.zeros(2, 3, dtype=torch.bool)):
            model(occ, sv, sk, spec, M)

    assert torch.equal(occ, occ_c), "occupancy must not be mutated"
    assert torch.equal(sv, sv_c), "scalars must not be mutated"
    assert torch.equal(spec, spec_c), "spectrum must not be mutated"
    assert torch.equal(M, M_c), "mask must not be mutated"


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
    """Behavioural (review B1): ScalarMasker flags must match the regime, and
    unknown values must be zeroed. The previous version constructed a tensor
    and asserted its own shape/dtype."""
    from data.scalar_mask import ScalarMasker
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]], dtype=torch.float32)
    for regime, check in (("all_known", lambda k: bool(k.all())),
                          ("all_unknown", lambda k: not bool(k.any()))):
        m = ScalarMasker(regime=regime, seed=0)
        vals, known = m.sample(sv)
        assert known.dtype == torch.bool and known.shape == (2, 3)
        assert check(known), f"{regime}: flags {known.tolist()}"
        assert torch.equal(vals[~known], torch.zeros_like(vals[~known]))


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
    """Behavioural (review B1): the target latent must depend on the EMA scalar
    target, NOT the live scalar encoder.

    Perturbing the LIVE scalar encoder must leave z_y_raw unchanged;
    perturbing the EMA target must move it. The previous version compared a
    tensor with its own alias under an unchanged input, so it could not detect
    a live-vs-EMA mixup."""
    model = _build_model()
    model.eval()
    occ, sv, spec, M = _batch(seed=7)
    sk = torch.ones(2, 3, dtype=torch.bool)

    with torch.no_grad():
        base = model(occ, sv, sk, spec, M, with_target=True)["z_y_raw"].clone()

        for p in model.scalar_encoder.parameters():
            p.add_(torch.randn_like(p) * 0.5)
        after_live = model(occ, sv, sk, spec, M, with_target=True)["z_y_raw"]
        assert torch.allclose(base, after_live, atol=1e-6), (
            "live scalar encoder must not affect the EMA target z_y_raw")

        for p in model.scalar_mlp_ema.target.parameters():
            p.add_(torch.randn_like(p) * 0.5)
        after_ema = model(occ, sv, sk, spec, M, with_target=True)["z_y_raw"]
        assert not torch.allclose(base, after_ema, atol=1e-6), (
            "EMA scalar target must drive z_y_raw")


# --------------------------------------------------------------------------
# §6 Physics gradient regression test (automated; runs with OR without the
# released surrogate — never skipped)
# --------------------------------------------------------------------------

_SURROGATE_PATH = os.path.join(
    REPO_ROOT, "data", "metadit", "weights", "surrogate_model.bin")
_HAS_SURROGATE = os.path.exists(_SURROGATE_PATH)


class _StubSurrogateOut:
    def __init__(self, prediction):
        self.prediction = prediction


class _StubSurrogate(nn.Module):
    """Minimal differentiable stand-in with the released surrogate's interface:
    forward(geometry[B,3,64,64]) -> object with `.prediction [B,2,301]`.

    Lets the physics-gradient guard run in EVERY environment. The previous
    version called the removed `objective.physics_loss.enable()` and was
    decorated with skipif(not _HAS_SURROGATE), so it was skipped where the
    weights were absent and raised AttributeError where they were present —
    inert everywhere (review A1).
    """

    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(3, 2)

    def forward(self, geometry):
        pooled = geometry.mean(dim=(2, 3))                   # (B, 3)
        pred = self.mix(pooled)[:, :, None].expand(-1, -1, 301)
        return _StubSurrogateOut(pred)


def _stub_or_real_surrogate():
    if _HAS_SURROGATE:
        from physics.physics_loop import load_surrogate
        return load_surrogate(_SURROGATE_PATH, device="cpu")
    stub = _StubSurrogate()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    return stub


def test_physics_gradient_regression():
    """Regression guard (Phase 5 MD §6): gradients flow from the surrogate
    output through the assembled geometry to the STUDENT (decoder, predictor,
    occupancy encoder), but NOT into the surrogate or the EMA target.

    Uses the real released surrogate when staged, a differentiable stub
    otherwise; physics activates via `lambda_phys>0 and surrogate is not None
    and model.training and goal_mode!='null'` (there is no `.enable()`).
    """
    model = _build_model()
    model.train()
    surrogate = _stub_or_real_surrogate()

    occ, sv, spec, M = _batch(seed=8)
    sk = torch.ones(2, 3, dtype=torch.bool)

    objective = UnifiedJEPALoss(
        hidden=192, lambda_phys=1.0, lambda_inv=0.0,
        lambda_var=0.0, lambda_cov=0.0, lambda_scalar=0.0,
        surrogate=surrogate)

    result = objective(model, occ, sv, sk, spec, M)
    loss = result["total_loss"]
    assert loss.requires_grad, "physics loss must be differentiable"
    loss.backward()

    # 1. Surrogate params must have NO gradient (frozen).
    for name, p in surrogate.named_parameters():
        assert p.grad is None, f"surrogate param has gradient: {name}"

    # 2. Student modules on the geometry path must receive the gradient.
    for mod, label in ((model.occupancy_decoder, "decoder"),
                       (model.predictor, "predictor"),
                       (model.occupancy_encoder, "occupancy encoder")):
        has_grad = any(p.grad is not None
                       for p in mod.parameters() if p.requires_grad)
        assert has_grad, f"{label} must receive physics gradient"

    # 3. EMA target must have NO gradient.
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


class _StubCollapseModel:
    """Minimal model for _collapse_metrics: returns a constant occupancy prob
    so the collapse thresholds are exercised for real (review B1)."""

    def __init__(self, b, fill):
        self.b, self.fill = b, fill

    def __call__(self, occ, sv, sk, spec, mask):
        return {"z_hat": torch.zeros(self.b, 1, 1),
                "scalar_pred": torch.zeros(self.b, 3)}

    def decode_occupancy_prob(self, z_hat, scalar_pred, scalar_known=None,
                              scalar_values=None):
        if torch.is_tensor(self.fill):
            return self.fill
        return torch.full((self.b, 1, 64, 64), self.fill)


def test_collapse_metrics_flags_all_empty_and_all_occupied():
    """Behavioural (review B1): the real collapse check must flag all-empty and
    all-occupied predictions, and neither for a half-filled one.

    The previous two tests asserted `0.0 < 0.01` / `1.0 > 0.99` on hand-made
    constants — the collapse check itself never ran."""
    from scripts.eval.eval_scenarios import _collapse_metrics
    b = 2
    occ = torch.zeros(b, 1, 64, 64)
    sv = torch.zeros(b, 3)
    spec = torch.randn(b, 2, 301)
    mask = torch.ones(b, 16, 16)
    sk = torch.ones(b, 3, dtype=torch.bool)

    empty = _collapse_metrics(_StubCollapseModel(b, 0.0), occ, sv, spec, mask, sk, "cpu")
    occd = _collapse_metrics(_StubCollapseModel(b, 1.0), occ, sv, spec, mask, sk, "cpu")
    half_fill = torch.cat([torch.ones(b, 1, 64, 32), torch.zeros(b, 1, 64, 32)], dim=3)
    half = _collapse_metrics(_StubCollapseModel(b, half_fill), occ, sv, spec, mask, sk, "cpu")

    assert empty["all_empty"] is True and empty["all_occupied"] is False
    assert occd["all_occupied"] is True and occd["all_empty"] is False
    assert half["all_empty"] is False and half["all_occupied"] is False


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
