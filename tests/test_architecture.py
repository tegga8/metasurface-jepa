"""Consolidated architecture-integrity tests (architecture-repair spec §35, A2–A20).

Locks the shared architecture's structural invariants: one geometry encoder, one
frozen EMA target outside the optimizer, no Perceiver/bottleneck/model.proj/
base+delta, physics reaching every predictor block and never the target, explicit
raw+normalized target boundary, deterministic finite init, and a non-pathological
representation scale. Objective mathematics (VICReg/Barlow/LeJEPA/SIGReg) is
explicitly out of scope for this file.

Run:  python tests/test_architecture.py   (pytest-collectable)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from assembly import GoalConditionedJEPA
from data.mask import BlockMasker, apply_mask_to_pixels
from diagnostics.representation_health import eff_ranks, goal_token_stats
from encoders.context_encoder import ContextEncoder
from encoders.geometry_encoder import GeometryEncoder
from predictor.gclct import GCLCT

from test_architecture_masking import (
    test_m1_masked_value_invariance as _m1,
    test_21_masked_content_never_enters_context as _leak,
    test_m4_mask_pixels_align_to_patch as _m4_pixels,
    test_m4_loss_selection_uses_same_order as _m4_loss,
)

GRID, PATCH, HIDDEN = 16, 4, 384


class _StubReleasedEncoder(nn.Module):
    """Frozen random MLP mapping (B, 2, 301) -> (B, 301, 256), standing in for the
    released MetaDiT spectrum encoder in tests that must not touch the dataset."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 256))

    def forward(self, S):
        return self.net(S.transpose(1, 2))


def build_model():
    torch.manual_seed(0)
    model = GoalConditionedJEPA(
        hidden=HIDDEN, num_heads=6, geo_depth=1, predictor_depth=2,
        goal_tokens=16, num_predictor_heads=6,
    )
    stub = _StubReleasedEncoder()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    model.spectrum_path.released = stub
    return model


def _batch(seed=0, n=4, mask_ratio=0.5):
    torch.manual_seed(seed)
    G = torch.randn(n, 3, 64, 64)
    S = torch.randn(n, 2, 301)
    M = BlockMasker(seed=seed, placement="random").sample(G, ratio=mask_ratio)
    return G, S, M


@pytest.fixture(scope="module")
def model():
    return build_model()


# --------------------------------------------------------------------------
# A2/A3/A4/A5 — forbidden structures absent
# --------------------------------------------------------------------------

def test_a2_no_perceiver(model):
    for m in model.modules():
        assert "perceiver" not in type(m).__name__.lower(), \
            f"Perceiver module present: {type(m).__name__}"


def test_a3_no_geometry_bottleneck(model):
    # Context path preserves all 256 tokens at full 384 dims; no pooling anywhere.
    assert isinstance(model.context_encoder, ContextEncoder)
    for m in model.modules():
        t = type(m).__name__
        assert not any(pool in t for pool in ("Pool", "Perceiver")), (
            f"pooling/bottleneck module present: {t}")
    G, S, M = _batch()
    out = model(G, S, M)
    assert tuple(out["z_x"].shape) == (4, 256, 384)
    assert tuple(out["z_hat"].shape) == (4, 256, 384)
    assert not hasattr(model.context_encoder, "bottleneck")


def test_a4_no_model_proj(model):
    assert not hasattr(model, "proj"), "shared model.proj must not exist"
    assert not hasattr(model, "projector"), "shared model projector must not exist"
    for name, sub in model.named_children():
        assert name not in ("proj", "projector"), f"shared projection module: {name}"


def test_a5_no_base_delta(model):
    keys = list(model.state_dict())
    assert not any(("base" in k or "delta" in k) for k in keys), (
        f"base+delta parameters present: {[k for k in keys if 'base' in k or 'delta' in k]}")


# --------------------------------------------------------------------------
# A6/A7/A8/A9 — encoder ownership, EMA isolation, init equality
# --------------------------------------------------------------------------

def test_a6_shared_student_geometry_object(model):
    assert model.context_encoder.geo is model.geometry_encoder, \
        "context encoder must share the single student geometry encoder"
    assert model.ema.target is not model.geometry_encoder, \
        "EMA target must be a SEPARATE (deep-copied) object"


def test_a7_ema_frozen(model):
    for name, p in model.ema.named_parameters():
        assert not p.requires_grad, f"EMA target parameter is trainable: {name}"


def test_a8_ema_excluded_from_optimizer(model):
    opt_params = [id(p) for p in model.parameters() if p.requires_grad]
    ema_ids = {id(p) for p in model.ema.parameters()}
    assert not any(i in ema_ids for i in opt_params), \
        "EMA target parameters must be absent from the optimizer"
    # A parameter may only be in the optimizer if it requires grad; EMA params
    # never do, so a filter(p.requires_grad) optimizer cannot contain them.
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    assert not any(any(p is ep for p in model.ema.parameters())
                   for g in opt.param_groups for ep in g["params"])


def test_a9_init_student_target_equal(model):
    s = model.geometry_encoder.state_dict()
    t = model.ema.target.state_dict()
    assert set(s) == set(t)
    for k in s:
        assert torch.equal(s[k], t[k]), f"student/target differ at init: {k}"


def test_a9_init_finite(model):
    for k, v in model.state_dict().items():
        assert torch.isfinite(v).all(), f"non-finite parameter at init: {k}"


def test_a9_film_output_zero_at_init():
    torch.manual_seed(0)
    pred = GCLCT(depth=2, hidden=HIDDEN, num_heads=6)
    for blk in pred.blocks:
        assert torch.count_nonzero(blk.cond[-1].weight).item() == 0
        assert torch.count_nonzero(blk.cond[-1].bias).item() == 0
        c = torch.randn(3, HIDDEN)
        cond_out = blk.cond(c)
        assert cond_out.shape == (3, 6 * HIDDEN), f"cond output shape {tuple(cond_out.shape)}"
        assert torch.allclose(cond_out, torch.zeros_like(cond_out)), \
            "FiLM cond must output exactly 0 at init (identity modulation)"
    assert any(torch.any(p != 0) for p in pred.head.parameters()), \
        "the predictor head must keep its NORMAL init (only FiLM may be zero-initialized)"


# --------------------------------------------------------------------------
# A10 — EMA update ordering behavior
# --------------------------------------------------------------------------

def test_a10_forward_does_not_touch_target(model):
    before = {k: v.clone() for k, v in model.ema.target.state_dict().items()}
    G, S, M = _batch()
    with torch.no_grad():
        model(G, S, M)
    for k, v in before.items():
        assert torch.equal(v, model.ema.target.state_dict()[k]), \
            "forward() must not modify the EMA target"


def test_a10_update_moves_target_toward_student(model):
    # Student changes first (simulating optimizer.step), then EMA update.
    with torch.no_grad():
        for p in model.geometry_encoder.parameters():
            p.add_(torch.full_like(p, 0.1))
    before = {k: v.clone() for k, v in model.ema.target.state_dict().items()}
    model.ema.update(model.geometry_encoder, step=0)     # m = momentum_start = 0.996
    moved = 0
    for k, v in before.items():
        if not torch.equal(v, model.ema.target.state_dict()[k]):
            moved += 1
    assert moved > 0, "EMA update must change the target"
    # Target must remain frozen after update.
    assert all(not p.requires_grad for p in model.ema.parameters())


def test_a10_momentum_schedule():
    from encoders.target_encoder import EMAEncoder
    enc = EMAEncoder(GeometryEncoder(hidden=16, num_heads=2, depth=1),
                     momentum_start=0.9, momentum_end=0.99)
    enc.set_total_steps(1000)
    assert abs(enc.current_momentum(0) - 0.9) < 1e-9
    assert abs(enc.current_momentum(1000) - 0.99) < 1e-9
    assert abs(enc.current_momentum(500) - 0.945) < 1e-9


# --------------------------------------------------------------------------
# A11 — target receives no gradient
# --------------------------------------------------------------------------

def test_a11_target_no_gradient(model):
    model.train()
    G, S, M = _batch()
    L, _ = model.loss(G, S, M)
    L.backward()
    for name, p in model.ema.named_parameters():
        assert p.grad is None, f"EMA target received gradient: {name}"
    for p in model.ema.buffers():
        assert p.grad is None, "EMA target buffer received gradient"


# --------------------------------------------------------------------------
# A12/A13 — masking invariants (delegated to the masking test module)
# --------------------------------------------------------------------------

def test_a12_masked_value_invariance():
    enc = ContextEncoder(GeometryEncoder(hidden=HIDDEN, num_heads=6, depth=2), hidden=HIDDEN)
    _m1(enc)


def test_a13_mask_index_alignment():
    enc = ContextEncoder(GeometryEncoder(hidden=HIDDEN, num_heads=6, depth=2), hidden=HIDDEN)
    _m4_pixels(enc)
    _m4_loss()


def test_21_masked_content_leakage_guard():
    enc = ContextEncoder(GeometryEncoder(hidden=HIDDEN, num_heads=6, depth=2), hidden=HIDDEN)
    _leak(enc)


# --------------------------------------------------------------------------
# A14/A15/A16 — physics conditioning is real
# --------------------------------------------------------------------------

def test_a14_every_predictor_block_consumes_c_physics(model):
    pred = model.predictor
    assert len(pred.blocks) == 2
    for i, blk in enumerate(pred.blocks):
        assert isinstance(blk.cond[0], nn.SiLU), f"block {i}: cond must start with SiLU"
        assert isinstance(blk.cond[1], nn.Linear), f"block {i}: cond must end with Linear"
        assert blk.cond[1].out_features == 6 * HIDDEN, f"block {i}: wrong FiLM width"
    # After one backward, every block's conditioner must be on the learning path.
    model.train()
    G, S, M = _batch(seed=3)
    L, _ = model.loss(G, S, M)
    L.backward()
    for i, blk in enumerate(pred.blocks):
        assert blk.cond[-1].weight.grad is not None, f"block {i}: no cond weight grad"
        assert blk.cond[-1].bias.grad is not None, f"block {i}: no cond bias grad"
        assert blk.cond[-1].weight.grad.abs().sum().item() > 0, \
            f"block {i}: cond weight grad is zero"
        assert blk.cond[-1].bias.grad.abs().sum().item() > 0, \
            f"block {i}: cond bias grad is zero"


def test_a15_condition_sensitivity_real_vs_null(model):
    model.eval()
    G, S, M = _batch(seed=5)
    with torch.no_grad():
        out_real = model(G, S, M, goal_mode="real")
        out_null = model(G, S, M, goal_mode="null")
    mask = out_real["mask"].bool()
    diff = (out_real["z_hat"] - out_null["z_hat"]).norm(dim=-1)
    assert diff[mask].mean().item() > 1e-6, \
        "real vs null goal must change masked predictions (conditioning dead?)"
    assert not torch.allclose(out_real["z_hat"], out_null["z_hat"], atol=1e-6)


def test_a15_condition_sensitivity_real_vs_shuffled(model):
    # Same geometries, same mask, different spectra -> predictions must change.
    model.eval()
    G, S, M = _batch(seed=5)
    S_alt = torch.roll(S, shifts=1, dims=0)
    with torch.no_grad():
        out_a = model(G, S, M, goal_mode="real")
        out_b = model(G, S_alt, M, goal_mode="real")
    assert not torch.allclose(out_a["c_physics"], out_b["c_physics"], atol=1e-6), \
        "distinct spectra must give distinct c_physics"
    assert not torch.allclose(out_a["z_hat"], out_b["z_hat"], atol=1e-6), \
        "distinct spectra must change the predictions (goal ignored?)"


def test_a16_condition_gradient_after_activation():
    """§15 sequence: identity at init -> condition weights change after one step ->
    gradient through the condition path > 0 after activation."""
    torch.manual_seed(0)
    pred = GCLCT(depth=1, hidden=HIDDEN, num_heads=6)
    B, T, D, K = 2, 256, HIDDEN, 272
    x = torch.randn(B, T, D)
    kv = torch.randn(B, K, D)
    c = torch.nn.Parameter(torch.randn(B, D))
    opt = torch.optim.AdamW([c] + [p for p in pred.parameters()], lr=1e-3)

    # Step 0: identity modulation.
    with torch.no_grad():
        y0 = pred(x, kv, c)[0]
    assert torch.allclose(y0, pred(x, kv, c.detach())[0], atol=0.0, rtol=0.0)

    # One optimizer step: condition weights must change from zero.
    loss = pred(x, kv, c)[0].mean()
    loss.backward()
    opt.step()
    assert any(torch.count_nonzero(blk.cond[-1].weight).item() > 0
               for blk in pred.blocks), "condition weights must change after one step"

    # After activation: gradient through the condition path > 0.
    opt.zero_grad()
    pred.train()
    loss2 = pred(x, kv, c)[0].mean()
    loss2.backward()
    assert c.grad is not None and c.grad.abs().sum().item() > 0, \
        "gradient through the condition path must be > 0 after activation"


def test_a16_condition_gradient_reaches_spectrum_path(model):
    """Gradient must flow into the trainable spectrum pooling path (proj_g / proj_goal).

    The FiLM cond is zero-initialized, so the very first backward has zero gradient
    w.r.t. c_physics BY DESIGN (spec §15: do not classify that as a dead path).
    Run a few optimizer steps first to thaw the conditioner, then verify gradients."""
    model.train()
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    G, S, M = _batch(seed=7)
    for _ in range(3):
        opt.zero_grad()
        L, _ = model.loss(G, S, M)
        L.backward()
        opt.step()
    assert any(torch.count_nonzero(blk.cond[-1].weight).item() > 0
               for blk in model.predictor.blocks), \
        "cond weights must be nonzero after 3 steps (precondition for the grad check)"
    opt.zero_grad()
    L, _ = model.loss(G, S, M)
    L.backward()
    for name, p in model.spectrum_path.named_parameters():
        if name.startswith("released"):
            continue
        assert p.grad is not None, f"spectrum path param has no grad: {name}"
        assert p.grad.abs().sum().item() > 0, f"spectrum path param grad is zero: {name}"


# --------------------------------------------------------------------------
# A17 — physics embedding diversity (trainable path is not collapsed)
# --------------------------------------------------------------------------

def test_a17_physics_embedding_diversity(model):
    model.eval()
    G, S, M = _batch(seed=11, n=8)
    with torch.no_grad():
        out = model(G, S, M)
    cp = out["c_physics"]
    ag = out["a_goal"]

    # Distinct spectra -> distinct global conditions and distinct goal tokens.
    # c_physics is a mean-pooled summary over 301 spectrum locations, so random-init
    # embeddings are similar-but-not-identical across samples (mean pooling
    # concentrates); the gate here is "not collapsed to a constant" (functional
    # sensitivity is checked separately in A15 via real/null/shuffled predictions).
    cp_n = F.normalize(cp, dim=-1)
    G_cp = cp_n @ cp_n.T
    idx = torch.triu_indices(8, 8, offset=1, device=cp.device)
    cos = G_cp[idx[0], idx[1]]
    assert cos.mean().item() < 0.999, f"c_physics pairwise cosine collapsed: {cos.mean().item():.4f}"
    assert cos.max().item() < 0.99999, "c_physics nearly identical across samples"

    rank = eff_ranks(cp)["eff_rank_frac"]
    assert rank > 0.005, f"c_physics effective rank collapsed: {rank:.4f}"

    gts = goal_token_stats(ag)
    # Learned goal queries start near-zero (std 0.02), so attention is near-uniform
    # at init and the 16 goal tokens are similar-but-not-identical. The structural
    # gate here is "not collapsed to a single constant token"; real diversity is
    # measured on trained checkpoints by the §32 architecture audit, not hard-gated
    # at random init.
    assert gts["goal_token_pairwise_cosine_mean"] < 0.9999, \
        f"goal tokens collapsed: {gts['goal_token_pairwise_cosine_mean']:.4f}"
    assert gts["goal_token_effective_rank"] > 0.5, \
        f"goal-token effective rank collapsed: {gts['goal_token_effective_rank']:.4f}"

    # a_goal must be sample-specific across distinct spectra.
    ag_pooled = ag.mean(dim=1)
    ag_n = F.normalize(ag_pooled, dim=-1)
    G_ag = ag_n @ ag_n.T
    ag_cos = G_ag[idx[0], idx[1]].mean().item()
    assert ag_cos < 0.999, f"a_goal cross-sample cosine collapsed: {ag_cos:.4f}"


# --------------------------------------------------------------------------
# A18 — output finiteness
# --------------------------------------------------------------------------

def test_a18_output_finiteness(model):
    model.eval()
    G, S, M = _batch(seed=13)
    with torch.no_grad():
        out = model(G, S, M)
    for k in ("z_hat", "z_x", "z_y_raw", "z_y_normalized", "c_physics", "a_goal"):
        assert torch.isfinite(out[k]).all(), f"non-finite {k}"


# --------------------------------------------------------------------------
# A19 — representation scale audit (visible, not assumed)
# --------------------------------------------------------------------------

def test_a19_representation_scale_audit(model):
    model.eval()
    G, S, M = _batch(seed=17, n=4)
    with torch.no_grad():
        out = model(G, S, M)

    scales = {}
    for k in ("z_x", "z_hat", "z_y_raw", "z_y_normalized"):
        t = out[k]
        scales[k + "_std"] = t.std().item()
        scales[k + "_mean_norm"] = t.norm(dim=-1).mean().item()
    scales["c_physics_std"] = out["c_physics"].std().item()

    for k, v in scales.items():
        assert torch.isfinite(torch.tensor(v)), f"non-finite scale metric: {k}"
        assert v > 1e-4, f"scale metric collapsed to zero: {k} = {v}"

    # Feature-wise normalization boundary must actually normalize (std ~ 1).
    assert 0.9 < scales["z_y_normalized_std"] < 1.1, \
        f"z_y_normalized std must be ~1, got {scales['z_y_normalized_std']:.4f}"

    # No orders-of-magnitude pathology among the raw representations.
    raw_stds = [scales["z_x_std"], scales["z_hat_std"], scales["z_y_raw_std"]]
    assert max(raw_stds) / min(raw_stds) < 50.0, (
        f"raw representation scales diverge by >50x: {raw_stds}")


# --------------------------------------------------------------------------
# A20 — explicit target-normalization boundary (§10)
# --------------------------------------------------------------------------

def test_a20_target_normalized_available(model):
    model.eval()
    G, S, M = _batch(seed=19)
    with torch.no_grad():
        out = model(G, S, M)
    raw = out["z_y_raw"]
    assert "z_y_normalized" in out
    assert torch.allclose(out["z_y_normalized"],
                          F.layer_norm(raw, (raw.shape[-1],)), atol=1e-6), \
        "z_y_normalized must be the feature-wise LayerNorm of z_y_raw"
    assert torch.equal(out["z_y"], raw), "z_y alias must equal z_y_raw"

    # The normalization boundary adds NO learnable parameters.
    keys = list(model.state_dict())
    assert not any("normalize" in k or "z_y_norm" in k for k in keys), (
        "target normalization must not add learnable parameters")

    # Raw target remains accessible and distinct from the normalized copy.
    assert not torch.allclose(raw, out["z_y_normalized"], atol=1e-3)


if __name__ == "__main__":
    m = build_model()
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if name in ("test_a9_film_output_zero_at_init",
                        "test_a10_momentum_schedule",
                        "test_a16_condition_gradient_after_activation",
                        "test_a12_masked_value_invariance",
                        "test_a13_mask_index_alignment",
                        "test_21_masked_content_leakage_guard",
                        "test_m4_loss_selection_uses_same_order"):
                fn()
            else:
                fn(m)
            print(f"PASS {name}")
        except Exception as e:
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)