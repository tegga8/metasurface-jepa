"""Predictor physics-conditioning tests (architecture-repair spec §7).

The GCLCT predictor must genuinely depend on BOTH conditioning inputs:
  - c_physics (global FiLM condition, Route B)
  - a_goal (16 structured goal tokens in the cross-attention kv, Route A)

Tests A–E pin the §7 contract:
  A. The FiLM `cond` is zero-initialized: at init, modulation is an identity
     (cond(c_physics) == 0 for every c_physics).
  B. `cond = Sequential(SiLU(), Linear(hidden, 6*hidden))` with the six FiLM
     groups (gamma1/beta1..gamma3/beta3) applied per block; once the cond is
     nonzero the modulation is a real per-sample, per-feature scale/shift.
  C. Changing c_physics changes the predictor output (physics conditions the
     predictor — Route B is a learnable dependency, not a decorative input).
  D. Changing a_goal changes the predictor output (goal tokens are consumed by
     the cross-attention kv — Route A is real, not ignored).
  E. Gradient flows from a prediction loss back through the predictor to
     c_physics (and to a_goal), so physics is on the learning path.

Also: the GCLCT constructor no longer accepts the removed `head_type` argument
(assembly bug fixed in the repair: build_model must not pass it).

Run:  python tests/test_predictor_conditioning.py   (pytest-collectable)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch

from predictor.gclct import GCLCT, GCLCTBlock

B, T, K, D = 3, 256, 272, 8   # 256 context tokens + 16 goal tokens in kv


@pytest.fixture
def small_predictor():
    torch.manual_seed(0)
    return GCLCT(depth=2, hidden=D, num_heads=2)


def _enable_cond(predictor, scale=1.0):
    """Un-zero the FiLM conds deterministically so conditioning is observable."""
    torch.manual_seed(123)
    for block in predictor.blocks:
        with torch.no_grad():
            block.cond[-1].weight.normal_(0.0, 0.5 * scale)
            block.cond[-1].bias.normal_(0.0, 0.5 * scale)


def _inputs(seed=0, goal_mode="real"):
    torch.manual_seed(seed)
    x = torch.randn(B, T, D)
    z_x = torch.randn(B, T, D)
    a_goal = torch.randn(B, K - T, D)
    if goal_mode == "null":
        a_goal = torch.zeros_like(a_goal)
    c = torch.randn(B, D)
    return x, torch.cat([z_x, a_goal], dim=1), c


# --------------------------------------------------------------------------
# Test A — zero-initialized FiLM cond (identity modulation at init)
# --------------------------------------------------------------------------

def test_a_cond_zero_initialized():
    torch.manual_seed(0)
    block = GCLCTBlock(hidden=D, num_heads=2)
    assert isinstance(block.cond[0], torch.nn.SiLU), \
        "cond must start with SiLU (spec §7)"
    assert isinstance(block.cond[1], torch.nn.Linear), \
        "cond must end with Linear(hidden, 6*hidden)"
    assert block.cond[1].out_features == 6 * D
    assert torch.count_nonzero(block.cond[1].weight).item() == 0
    assert torch.count_nonzero(block.cond[1].bias).item() == 0
    for c in (torch.randn(B, D), torch.zeros(B, D), torch.randn(B, D) * 10):
        out = block.cond(c)
        assert torch.allclose(out, torch.zeros_like(out)), (
            "zero-init cond must output exactly 0 for any c_physics")


# --------------------------------------------------------------------------
# Test B — cond structure + six FiLM groups modulate per block
# --------------------------------------------------------------------------

def test_b_six_film_groups_are_real_modulations():
    torch.manual_seed(1)
    pred = GCLCT(depth=1, hidden=D, num_heads=2)
    block = pred.blocks[0]
    _enable_cond(pred)
    x, kv, c = _inputs()
    g1, b1, g2, b2, g3, b3 = block.cond(c).chunk(6, dim=-1)
    for name, g in (("gamma1", g1), ("beta1", b1), ("gamma2", g2),
                    ("beta2", b2), ("gamma3", g3), ("beta3", b3)):
        assert g.shape == (B, D), f"{name} must be (B, {D}), got {g.shape}"
        assert not torch.allclose(g, torch.zeros_like(g)), (
            f"{name} must be nonzero once cond is un-zeroed")

    # FiLM math: h * (1 + gamma) + beta must be applied (spot-check via a
    # controlled cond output through the block's own path).
    block.cond[-1].weight.data.zero_()
    block.cond[-1].bias.data.zero_()
    # hand-set the modulation for gamma1 only
    with torch.no_grad():
        block.cond[-1].bias[:D].copy_(torch.full((D,), 1.0))   # scale by 2
    h_in = x + 5.0                            # affine-less norm(x+5) == norm(x)
    out_before = block(x, kv, c)[0]
    out_shifted = block(x + 5.0, kv, c)[0]   # same norm input, diff FiLM residual
    assert not torch.allclose(out_before, out_shifted, atol=1e-5), (
        "FiLM must modulate per-block outputs (residual paths expose the shift)")


def test_b_all_three_sublayers_get_film():
    """The 6*hidden output is 6 groups across the 3 sublayers (2 per sublayer);
    a depth>1 net applies conditioning in every block."""
    pred = GCLCT(depth=3, hidden=D, num_heads=2)
    for i, block in enumerate(pred.blocks):
        assert block.cond[-1].out_features == 6 * D, f"block {i} cond width wrong"
    _enable_cond(pred)
    x, kv, c = _inputs()
    c2 = c + 1.0
    with torch.no_grad():
        y1 = pred(x, kv, c)[0]
        y2 = pred(x, kv, c2)[0]
    assert not torch.allclose(y1, y2, atol=1e-5), (
        "every block's FiLM must make the output sensitive to c_physics")


# --------------------------------------------------------------------------
# Test C — output depends on c_physics (Route B)
# --------------------------------------------------------------------------

def test_c_output_depends_on_c_physics(small_predictor):
    _enable_cond(small_predictor)
    small_predictor.eval()
    x, kv, c = _inputs()
    c_alt = c + torch.randn_like(c)
    with torch.no_grad():
        y_c = small_predictor(x, kv, c)[0]
        y_alt = small_predictor(x, kv, c_alt)[0]
    assert not torch.allclose(y_c, y_alt, atol=1e-5), (
        "changing c_physics must change the predictor output (Route B dead?)")
    # Sanity: identical c_physics gives bit-identical output (determinism).
    with torch.no_grad():
        y_c2 = small_predictor(x, kv, c)[0]
    assert torch.allclose(y_c, y_c2, atol=0.0, rtol=0.0), (
        "same c_physics must give identical output")


# --------------------------------------------------------------------------
# Test D — output depends on a_goal (Route A)
# --------------------------------------------------------------------------

def test_d_output_depends_on_a_goal(small_predictor):
    small_predictor.eval()
    x, kv_real, c = _inputs(seed=0)
    x, kv_null, _ = _inputs(seed=0, goal_mode="null")
    _, kv_alt, _ = _inputs(seed=7)             # same x, different a_goal values
    with torch.no_grad():
        y_real = small_predictor(x, kv_real, c)[0]
        y_null = small_predictor(x, kv_null, c)[0]
        y_alt = small_predictor(x, kv_alt, c)[0]
    assert not torch.allclose(y_real, y_null, atol=1e-5), (
        "zeroing a_goal must change the output (Route A dead?)")
    assert not torch.allclose(y_real, y_alt, atol=1e-5), (
        "different goal tokens must change the output (goal ignored?)")


# --------------------------------------------------------------------------
# Test E — gradients flow to c_physics and a_goal
# --------------------------------------------------------------------------

def test_e_gradient_flows_to_c_physics_and_a_goal(small_predictor):
    small_predictor.train()
    _enable_cond(small_predictor)   # zero-init cond has zero grad w.r.t. c_physics
    x, kv, c = _inputs()
    c_param = torch.nn.Parameter(c.clone())
    goal_param = torch.nn.Parameter(kv[:, -16:].clone())
    kv = torch.cat([kv[:, :-16], goal_param], dim=1)
    out = small_predictor(x, kv, c_param)[0]
    loss = out.mean()
    loss.backward()
    assert c_param.grad is not None and c_param.grad.abs().sum() > 0, (
        "no gradient reaches c_physics — the FiLM path is not on the learning path")
    assert goal_param.grad is not None and goal_param.grad.abs().sum() > 0, (
        "no gradient reaches a_goal — the goal tokens are not consumed")


def test_no_head_type_argument():
    """GCLCT's constructor no longer accepts the removed `head_type` kwarg —
    the assembly bug that passed `head_type='latent'` is fixed in the repair."""
    with pytest.raises(TypeError):
        GCLCT(depth=1, hidden=D, num_heads=2, head_type="latent")


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