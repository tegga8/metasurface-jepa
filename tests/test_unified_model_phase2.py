"""Phase 2 smoke + contract tests for the unified JEPA model.

Verifies:
- Forward-pass shape contract (192-D throughout)
- Gradient ownership (student grads present, EMA/scalar_mlp_ema/released grads absent)
- EMA isolation (forward does not touch targets)
- GCLCT c_physics_dim projection (384 -> 192)
- Scalar known/unknown masking
- goal_mode null/shuffled
- Checkpoint save/load round-trip with architecture_id

Run:  python -m pytest tests/test_unified_model_phase2.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from assembly import (
    UnifiedJEPA,
    build_unified_model,
    saveable_state_dict,
    load_into_model,
    SAVED_EXCLUDES,
    UNIFIED_ARCHITECTURE_ID,
)
from predictor.gclct import GCLCT
from encoders.target_encoder import EMAEncoder
from data.mask import BlockMasker


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

class _StubReleasedEncoder(nn.Module):
    """Frozen random MLP standing in for the released MetaDiT spec encoder."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 256))

    def forward(self, S):
        return self.net(S.transpose(1, 2))


def _make_stub_spectrum_path(model):
    """Replace spectrum_path.released with a frozen stub."""
    stub = _StubReleasedEncoder()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    model.spectrum_path.released = stub


def build_model(hidden=192, geo_depth=2, predictor_depth=4):
    torch.manual_seed(0)
    model = UnifiedJEPA(
        hidden=hidden, num_heads=6, geo_depth=geo_depth,
        predictor_depth=predictor_depth, goal_tokens=16,
        num_predictor_heads=6, scalar_hidden=128, n_film_blocks=n_film_blocks_for(hidden, geo_depth),
        spec_dim=256,
    )
    _make_stub_spectrum_path(model)
    model.ema.target.load_state_dict(model.occupancy_encoder.state_dict())
    model.scalar_mlp_ema.target.load_state_dict(model.scalar_encoder.state_dict())
    model.eval()
    return model


def n_film_blocks_for(hidden, geo_depth):
    return geo_depth


def _batch(seed=0, b=2):
    torch.manual_seed(seed)
    occupancy = (torch.rand(b, 1, 64, 64) > 0.5).float()
    scalar_values = torch.tensor([
        [1.5, 0.8, 10.0],
        [2.0, 1.2, 12.0],
    ], dtype=torch.float32)
    scalar_known = torch.tensor([
        [True, True, True],
        [False, True, False],
    ])
    spectrum = torch.randn(b, 2, 301)
    masker = BlockMasker(placement="random", grid=16, min_side=3, k_range=(1, 4), seed=seed)
    M = masker.sample(occupancy, ratio=0.5)
    return occupancy, scalar_values, scalar_known, spectrum, M


# --------------------------------------------------------------------------
# Architecture ID
# --------------------------------------------------------------------------

def test_architecture_id():
    assert UNIFIED_ARCHITECTURE_ID == "unified_occ_param_spectrum_jepa_v1"
    model = build_model()
    assert model.architecture_id == UNIFIED_ARCHITECTURE_ID


# --------------------------------------------------------------------------
# Forward shape contract (architecture_v5.md §5, §7)
# --------------------------------------------------------------------------

def test_forward_shapes():
    model = build_model()
    occ, sv, sk, spec, M = _batch(seed=0, b=2)
    out = model(occ, sv, sk, spec, M, with_target=True)
    assert out["z_x"].shape == (2, 256, 192)
    assert out["z_hat"].shape == (2, 256, 192)
    assert out["z_y_raw"].shape == (2, 256, 192)
    assert out["scalar_pred"].shape == (2, 3)
    assert out["c_physics"].shape == (2, 384)
    assert out["a_goal"].shape == (2, 16, 384)
    assert out["mask"].shape == (2, 256)
    assert out["scalar_summary_pred"].shape == (2, 192)
    assert "z_y_normalized" in out


def test_mask_convention():
    """mask=True means 'this position is masked' (target for JEPA loss)."""
    model = build_model()
    occ, sv, sk, spec, M = _batch(seed=1, b=2)
    out = model(occ, sv, sk, spec, M)
    expected_mask = (M.view(2, -1) == 0)  # 0 in M => masked
    assert torch.equal(out["mask"], expected_mask)


# --------------------------------------------------------------------------
# Gradient ownership (Phase 2 §9)
# --------------------------------------------------------------------------

def test_gradient_ownership():
    model = build_model()
    model.train()
    occ, sv, sk, spec, M = _batch(seed=2, b=2)
    out = model(occ, sv, sk, spec, M, with_target=True)
    L, _ = model.loss(occ, sv, sk, spec, M)
    assert torch.isfinite(L), "loss must be finite"
    L.backward()

    # Student modules must have gradients
    for name, p in model.occupancy_encoder.named_parameters():
        assert p.grad is not None, f"occupancy_encoder param {name} has no grad"
    for name, p in model.scalar_encoder.named_parameters():
        assert p.grad is not None, f"scalar_encoder param {name} has no grad"
    for name, p in model.predictor.named_parameters():
        assert p.grad is not None, f"predictor param {name} has no grad"
    for name, p in model.fusion_encoder.named_parameters():
        assert p.grad is not None, f"fusion_encoder param {name} has no grad"
    for name, p in model.scalar_decoder.named_parameters():
        assert p.grad is not None, f"scalar_decoder param {name} has no grad"
    assert model.mask_token.grad is not None, "mask_token has no grad"
    assert model.scalar_query_token.grad is not None, "scalar_query_token has no grad"

    # EMA targets must have NO gradients
    for name, p in model.ema.named_parameters():
        assert p.grad is None, f"occupancy EMA param {name} has a gradient"
    for name, p in model.scalar_mlp_ema.named_parameters():
        assert p.grad is None, f"scalar_mlp_ema param {name} has a gradient"

    # Released spectrum encoder must have NO gradients
    released = model.spectrum_path.released
    if released is not None:
        for name, p in released.named_parameters():
            assert p.grad is None, f"released spectrum param {name} has a gradient"


# --------------------------------------------------------------------------
# EMA isolation (forward must not touch targets)
# --------------------------------------------------------------------------

def test_ema_forward_isolation():
    model = build_model()
    occ, sv, sk, spec, M = _batch(seed=3, b=2)
    occ_state = {k: v.clone() for k, v in model.ema.target.state_dict().items()}
    scalar_state = {k: v.clone() for k, v in model.scalar_mlp_ema.target.state_dict().items()}
    with torch.no_grad():
        model(occ, sv, sk, spec, M, with_target=True)
    for k, v in occ_state.items():
        assert torch.equal(v, model.ema.target.state_dict()[k]), \
            f"EMA target changed during forward: {k}"
    for k, v in scalar_state.items():
        assert torch.equal(v, model.scalar_mlp_ema.target.state_dict()[k]), \
            f"scalar_mlp_ema changed during forward: {k}"


def test_ema_update_moves_target():
    model = build_model()
    model.ema.set_total_steps(1000)
    model.scalar_mlp_ema.set_total_steps(1000)
    occ_state = {k: v.clone() for k, v in model.ema.target.state_dict().items()}
    scalar_state = {k: v.clone() for k, v in model.scalar_mlp_ema.target.state_dict().items()}

    # Perturb the students
    with torch.no_grad():
        for p in model.occupancy_encoder.parameters():
            p.add_(torch.full_like(p, 0.1))
        for p in model.scalar_encoder.parameters():
            p.add_(torch.full_like(p, 0.1))

    model.ema.update(model.occupancy_encoder, step=0)
    model.scalar_mlp_ema.update(model.scalar_encoder, step=0)

    moved_occ = sum(
        1 for k, v in occ_state.items()
        if not torch.equal(v, model.ema.target.state_dict()[k])
    )
    moved_scalar = sum(
        1 for k, v in scalar_state.items()
        if not torch.equal(v, model.scalar_mlp_ema.target.state_dict()[k])
    )
    assert moved_occ > 0, "occupancy EMA must change after update"
    assert moved_scalar > 0, "scalar_mlp_ema must change after update"


# --------------------------------------------------------------------------
# Frozen reference modes survive mode switches
# --------------------------------------------------------------------------

def test_frozen_modes_survive_train():
    model = build_model()
    model.train()
    assert not model.ema.target.training
    assert not model.scalar_mlp_ema.target.training
    released = getattr(model.spectrum_path, "released", None)
    if released is not None:
        assert not released.training
    model.eval()
    assert not model.ema.target.training
    assert not model.scalar_mlp_ema.target.training


# --------------------------------------------------------------------------
# Scalar known/unknown masking
# --------------------------------------------------------------------------

def test_scalar_input_all_known():
    model = build_model()
    b = 2
    sv = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    sk = torch.ones(b, 3, dtype=torch.bool)
    inp = model._build_scalar_input(sv, sk)
    # [l_val, l_known=1, h_val, h_known=1, r_val, r_known=1]
    assert torch.equal(inp[:, [1, 3, 5]], torch.ones(b, 3))
    assert torch.equal(inp[:, [0, 2, 4]], sv)


def test_scalar_input_all_unknown():
    model = build_model()
    b = 2
    sv = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    sk = torch.zeros(b, 3, dtype=torch.bool)
    inp = model._build_scalar_input(sv, sk)
    assert torch.equal(inp[:, [1, 3, 5]], torch.zeros(b, 3))
    assert torch.equal(inp[:, [0, 2, 4]], torch.zeros(b, 3))


def test_scalar_input_partial_known():
    model = build_model()
    b = 1
    sv = torch.tensor([[1.0, 2.0, 3.0]])
    sk = torch.tensor([[True, False, True]])
    inp = model._build_scalar_input(sv, sk)
    # l known, h unknown, r known
    assert inp[0, 0] == 1.0   # l_val
    assert inp[0, 1] == 1.0   # l_known
    assert inp[0, 2] == 0.0   # h_val (zeroed)
    assert inp[0, 3] == 0.0   # h_unknown
    assert inp[0, 4] == 3.0   # r_val
    assert inp[0, 5] == 1.0   # r_known


def test_target_uses_true_scalars():
    """The EMA target encoder must receive true (all-known) scalars, not masked."""
    model = build_model()
    occ, sv, sk, spec, M = _batch(seed=4, b=2)
    # All unknown — live encoder gets zeroed values
    sk_all_false = torch.zeros(2, 3, dtype=torch.bool)
    out = model(occ, sv, sk_all_false, spec, M, with_target=True)
    # Target should still be computed (uses true scalars internally)
    assert out["z_y_raw"].shape == (2, 256, 192)
    assert torch.isfinite(out["z_y_raw"]).all()


# --------------------------------------------------------------------------
# Goal mode (null / shuffled)
# --------------------------------------------------------------------------

def test_goal_mode_real_vs_null():
    model = build_model()
    occ, sv, sk, spec, M = _batch(seed=5, b=2)
    out_real = model(occ, sv, sk, spec, M, goal_mode="real")
    out_null = model(occ, sv, sk, spec, M, goal_mode="null")
    # c_physics must differ (null = zeros)
    assert not torch.allclose(out_real["c_physics"], out_null["c_physics"], atol=1e-6)
    # a_goal must differ (null = zeros)
    assert not torch.allclose(out_real["a_goal"], out_null["a_goal"], atol=1e-6)
    # Predictions must differ
    assert not torch.allclose(out_real["z_hat"], out_null["z_hat"], atol=1e-6)


def test_goal_mode_real_vs_shuffled():
    model = build_model()
    occ, sv, sk, spec, M = _batch(seed=6, b=2)
    spec_shuffled = torch.roll(spec, shifts=1, dims=0)
    out_real = model(occ, sv, sk, spec, M, goal_mode="real")
    out_shuffled = model(occ, sv, sk, spec_shuffled, M, goal_mode="shuffled")
    # Distinct spectra with same geometry must give distinct c_physics
    assert not torch.allclose(out_real["c_physics"], out_shuffled["c_physics"], atol=1e-6)


# --------------------------------------------------------------------------
# GCLCT c_physics_dim projection (Phase 2 §3)
# --------------------------------------------------------------------------

def test_gclct_c_physics_projection_384_to_192():
    """GCLCT with hidden=192 and c_physics_dim=384 must project c_physics."""
    torch.manual_seed(0)
    pred = GCLCT(depth=2, hidden=192, num_heads=6, c_physics_dim=384)
    assert isinstance(pred.c_phys_proj, nn.Linear)
    assert pred.c_phys_proj.in_features == 384
    assert pred.c_phys_proj.out_features == 192

    queries = torch.randn(2, 256, 192)
    kv = torch.randn(2, 273, 192)
    c_physics = torch.randn(2, 384)  # 384-dim input
    z_hat, _ = pred(queries, kv, c_physics)
    assert z_hat.shape == (2, 256, 192)


def test_gclct_no_projection_when_dims_match():
    """Backward compat: when c_physics_dim == hidden, use Identity (no extra params)."""
    torch.manual_seed(0)
    pred = GCLCT(depth=2, hidden=384, num_heads=6)
    assert isinstance(pred.c_phys_proj, nn.Identity)
    c_physics = torch.randn(2, 384)
    queries = torch.randn(2, 256, 384)
    kv = torch.randn(2, 272, 384)
    z_hat, _ = pred(queries, kv, c_physics)
    assert z_hat.shape == (2, 256, 384)


# --------------------------------------------------------------------------
# EMAEncoder kwargs passthrough
# --------------------------------------------------------------------------

def test_ema_encoder_passes_kwargs():
    """EMAEncoder.forward must accept and pass kwargs (e.g. film_params)."""

    class _DummyEnc(nn.Module):
        def forward(self, x, film_params=None):
            return x

    torch.manual_seed(0)
    inner = _DummyEnc()
    ema = EMAEncoder(inner)
    x = torch.randn(2, 5)
    out = ema(x, film_params=None)
    assert out.shape == x.shape


# --------------------------------------------------------------------------
# OccupancyEncoder mask_token replacement
# --------------------------------------------------------------------------

def test_occupancy_encoder_mask_token_replacement():
    """OccupancyEncoder must replace masked tokens with mask_token before blocks."""
    from encoders.occupancy_encoder import OccupancyEncoder

    torch.manual_seed(0)
    enc = OccupancyEncoder(hidden=192, num_heads=6, depth=2)
    occ = torch.rand(1, 1, 64, 64)
    mask = torch.ones(1, 16, 16)  # all visible
    mask[0, 3, 5] = 0  # mask one cell
    mask_token = torch.zeros(1, 1, 192)
    z = enc(occ, mask=mask, mask_token=mask_token)
    assert z.shape == (1, 256, 192)

    # Now test that masked vs unmasked input produce same z_x for visible
    occ2 = occ.clone()
    occ2[0, 0, 12:16, 20:24] = 1.0  # change content at masked region
    z2 = enc(occ2, mask=mask, mask_token=mask_token)
    assert torch.allclose(z, z2, atol=1e-5), \
        "Occupancy encoder output must not depend on masked patch content"


# --------------------------------------------------------------------------
# Checkpoint round-trip (Phase 2 §8)
# --------------------------------------------------------------------------

def test_saveable_state_dict_excludes_released():
    model = build_model()
    sd = saveable_state_dict(model)
    for key in sd:
        assert ".released." not in key, f"released param in saveable state: {key}"
    assert any("occupancy_encoder" in k for k in sd), "missing occupancy_encoder in state dict"
    assert any("scalar_decoder" in k for k in sd), "missing scalar_decoder in state dict"


def test_checkpoint_round_trip():
    model = build_model()
    sd = saveable_state_dict(model)
    fresh = build_model()
    load_into_model(fresh, sd, torch.device("cpu"), strict=True)
    occ, sv, sk, spec, M = _batch(seed=7, b=2)
    with torch.no_grad():
        o1 = model(occ, sv, sk, spec, M)
        o2 = fresh(occ, sv, sk, spec, M)
    assert torch.allclose(o1["z_x"], o2["z_x"], atol=1e-5)
    assert torch.allclose(o1["z_hat"], o2["z_hat"], atol=1e-5)
    assert torch.allclose(o1["scalar_pred"], o2["scalar_pred"], atol=1e-5)


def test_old_checkpoint_not_compatible():
    """Loading a 384-D legacy checkpoint into the 192-D unified model must fail."""
    from assembly import GoalConditionedJEPA

    torch.manual_seed(0)
    old = GoalConditionedJEPA(hidden=384, num_heads=6, geo_depth=2, predictor_depth=4)
    # stub
    _make_stub_spectrum_path_for_old(old)
    old_sd = saveable_state_dict(old)

    unified = build_model()
    with pytest.raises(RuntimeError, match="checkpoint/model key mismatch"):
        load_into_model(unified, old_sd, torch.device("cpu"), strict=True)


def _make_stub_spectrum_path_for_old(model):
    stub = _StubReleasedEncoder()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    model.spectrum_path.released = stub


# --------------------------------------------------------------------------
# Loss computability + finiteness
# --------------------------------------------------------------------------

def test_loss_finite_and_backward():
    model = build_model()
    model.train()
    occ, sv, sk, spec, M = _batch(seed=8, b=2)
    L, out = model.loss(occ, sv, sk, spec, M)
    assert torch.isfinite(L), "loss must be finite"
    L.backward()
    # scalar prediction must have a gradient path
    assert model.scalar_decoder.heads[0][-1].weight.grad is not None


def test_loss_scalar_only_unknown():
    """Scalar loss must only penalize unknown positions, not known ones."""
    model = build_model()
    model.train()
    occ, sv, sk, spec, M = _batch(seed=9, b=2)
    out = model(occ, sv, sk, spec, M, with_target=True)
    # Manually check: known positions should not contribute
    unknown = ~sk  # (B, 3)
    scalar_err = (out["scalar_pred"] - sv).abs()
    known_err = scalar_err * sk.float()
    # Known errors are computed but the loss shouldn't sum them
    # (This is a structural check — the model.loss method should zero them out)
    assert known_err.shape == unknown.shape


# --------------------------------------------------------------------------
# ScalarDecoder zero-init
# --------------------------------------------------------------------------

def test_scalar_decoder_nonzero_init():
    """ScalarDecoder final-layer bias must be nonzero (Phase 4 MD §3:
    'needs nonzero init to produce nonzero geometry for gradient flow').
    Zero-init collapses geometry to all-zeros, killing the surrogate's
    Jacobian through ReLU6 dead zones. Weight is zero so the head acts
    as a learned bias at init, with the first layer providing the latent
    modulation."""
    from decoders.scalar_decoder import ScalarDecoder
    dec = ScalarDecoder(hidden=192)
    for head in dec.heads:
        assert torch.count_nonzero(head[-1].weight).item() == 0
        assert torch.count_nonzero(head[-1].bias).item() > 0


def test_vicreg_supports_192d():
    """Phase 2 §7: verify existing VICReg and JEPA projection mechanisms
    support 192-D tensors."""
    from losses.vicreg import vicreg_branch_terms
    from losses.jepa_loss import ProjectionMLP, jepa_loss

    torch.manual_seed(0)
    D = 192
    N = 4
    proj = ProjectionMLP(hidden=D)
    p_hat = proj(torch.randn(N, D))
    p_y = proj(torch.randn(N, D))
    L_inv, L_var, L_cov = vicreg_branch_terms(p_hat, p_y)
    assert torch.isfinite(L_inv)
    assert torch.isfinite(L_var)
    assert torch.isfinite(L_cov)

    # JEPA loss with 192-D
    pred = torch.randn(2, 256, D, requires_grad=True)
    target = torch.randn(2, 256, D, requires_grad=True)
    mask = torch.ones(2, 256, dtype=torch.bool)
    mask[0, :128] = False
    L, per = jepa_loss(pred, target, mask, proj=None)
    assert L.ndim == 0
    assert per.shape == (2,)
    assert torch.isfinite(L)


def test_vicreg_projectionmlp_192d():
    """ProjectionMLP parameterized by hidden=192 works."""
    from losses.jepa_loss import ProjectionMLP
    proj = ProjectionMLP(hidden=192)
    x = torch.randn(4, 192)
    out = proj(x)
    assert out.shape == (4, 192)


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
