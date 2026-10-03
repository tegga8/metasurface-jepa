"""Step 2 regression tests: scalar-predictor FiLM in the GCLCT.

Covers: flag off == current architecture (no params, scalar_cond ignored);
zero-init means identity at step 0; a non-zero FiLM changes the output; gradient
ownership; and (model-level) the FiLM is driven by the LEARNED scalar summary, not
raw (l,h,r) values.

Run: python -m pytest tests/test_scalar_predictor_film.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
import torch.nn as nn

from predictor.gclct import GCLCT


def _inputs(b=2, tq=5, tkv=7, hidden=16):
    torch.manual_seed(0)
    return (torch.randn(b, tq, hidden), torch.randn(b, tkv, hidden),
            torch.randn(b, hidden), torch.randn(b, hidden))


def test_flag_off_ignores_scalar_cond_and_has_no_params():
    g = GCLCT(depth=2, hidden=16, num_heads=2, c_physics_dim=16, scalar_film=False)
    assert all(b.cond_scalar is None for b in g.blocks)
    assert not any("cond_scalar" in k for k in g.state_dict())
    q, kv, c, sc = _inputs()
    a, _ = g(q, kv, c)
    b, _ = g(q, kv, c, scalar_cond=sc)
    assert torch.equal(a, b), "scalar_cond must be ignored when the flag is off"


def test_zero_init_film_is_identity_at_step0():
    g = GCLCT(depth=2, hidden=16, num_heads=2, c_physics_dim=16, scalar_film=True)
    assert all(b.cond_scalar is not None for b in g.blocks)
    q, kv, c, sc = _inputs()
    base, _ = g(q, kv, c, scalar_cond=None)
    withfilm, _ = g(q, kv, c, scalar_cond=sc)
    assert torch.allclose(base, withfilm, atol=1e-6), \
        "zero-initialized scalar FiLM must not change the output at init"


def test_nonzero_film_changes_output_and_gets_gradient():
    g = GCLCT(depth=2, hidden=16, num_heads=2, c_physics_dim=16, scalar_film=True)
    # break the zero-init so the modulation acts
    with torch.no_grad():
        for b in g.blocks:
            b.cond_scalar[-1].weight.normal_(0.0, 0.1)
            b.cond_scalar[-1].bias.normal_(0.0, 0.1)
    q, kv, c, sc = _inputs()
    base, _ = g(q, kv, c, scalar_cond=None)
    withfilm, _ = g(q, kv, c, scalar_cond=sc)
    assert not torch.allclose(base, withfilm, atol=1e-6)
    withfilm.sum().backward()
    for b in g.blocks:
        assert b.cond_scalar[-1].weight.grad is not None
        assert b.cond_scalar[-1].weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# model level: the FiLM is driven by the LEARNED summary, not raw (l,h,r)
# ---------------------------------------------------------------------------

class _StubReleased(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 256))

    def forward(self, S):
        return self.net(S.transpose(1, 2))


def _model(scalar_predictor_film):
    from assembly import UnifiedJEPA
    torch.manual_seed(0)
    m = UnifiedJEPA(hidden=192, num_heads=6, geo_depth=2, predictor_depth=4,
                    goal_tokens=16, num_predictor_heads=6, scalar_hidden=128,
                    n_film_blocks=2, spec_dim=256,
                    scalar_predictor_film=scalar_predictor_film)
    stub = _StubReleased()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    m.spectrum_path.released = stub
    m.ema.target.load_state_dict(m.occupancy_encoder.state_dict())
    m.scalar_mlp_ema.target.load_state_dict(m.scalar_encoder.state_dict())
    m.eval()
    return m


def _batch(b=2):
    torch.manual_seed(1)
    occ = (torch.rand(b, 1, 64, 64) > 0.5).float()
    sv = torch.tensor([[2.7, 0.8, 4.2], [2.6, 0.7, 3.9]])[:b]
    sk = torch.ones(b, 3, dtype=torch.bool)
    spec = torch.randn(b, 2, 301)
    mask = torch.zeros(b, 16, 16)
    return occ, sv, sk, spec, mask


def test_film_input_is_learned_summary_not_raw_values():
    m = _model(scalar_predictor_film=True)
    occ, sv, sk, spec, mask = _batch()
    captured = {}
    h = m.predictor.blocks[0].cond_scalar.register_forward_pre_hook(
        lambda mod, inp: captured.__setitem__("x", inp[0].detach().clone()))
    with torch.no_grad():
        out = m(occ, sv, sk, spec, mask, with_target=False)
        summary = m.scalar_encoder(m._build_scalar_input(sv, sk))[1]
    h.remove()
    assert "x" in captured
    assert captured["x"].shape[-1] == 192, "FiLM input must be the hidden-dim summary"
    assert captured["x"].shape[-1] != sv.shape[-1], "not raw (l,h,r)"
    assert torch.allclose(captured["x"], summary.reshape(sv.shape[0], -1), atol=1e-6)


def test_flag_off_model_has_no_scalar_film_params():
    m = _model(scalar_predictor_film=False)
    assert not any("cond_scalar" in k for k in m.state_dict())


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
