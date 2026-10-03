"""Phase 4 — multi-target objective (spectrum-conditioned geometry target +
scalar-latent target), keeping the stable spectrum-free target.

Corrected design (see docs/benchmarking/PHASE4_DIAGNOSIS.md): cosine (scale-free)
losses, ramped in; the spectrum film is FROZEN with a small non-zero init so the
target is genuinely spectrum-dependent and cannot be optimized back to identity.

Run:  python -m pytest tests/test_phase4_multitarget.py -v
      python tests/test_phase4_multitarget.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "train"))

import pytest
import torch
import torch.nn as nn

from assembly import UnifiedJEPA
from data.mask import BlockMasker
from losses.unified_losses import UnifiedJEPALoss
from train_unified import _ramp_frac


class _StubReleasedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 256))

    def forward(self, S):
        return self.net(S.transpose(1, 2))


def _model():
    torch.manual_seed(0)
    m = UnifiedJEPA(hidden=192, num_heads=6, geo_depth=2, predictor_depth=4,
                    goal_tokens=16, num_predictor_heads=6, scalar_hidden=128,
                    n_film_blocks=2, spec_dim=256)
    stub = _StubReleasedEncoder()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    m.spectrum_path.released = stub
    m.ema.target.load_state_dict(m.occupancy_encoder.state_dict())
    m.scalar_mlp_ema.target.load_state_dict(m.scalar_encoder.state_dict())
    return m


def _batch(b=2, seed=0):
    torch.manual_seed(seed)
    occ = (torch.rand(b, 1, 64, 64) > 0.5).float()
    occ[:, :, :32, :32] = 1.0
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]], dtype=torch.float32)[:b]
    spec = torch.randn(b, 2, 301)
    M = BlockMasker(placement="random", grid=16, min_side=3, k_range=(1, 4),
                    seed=seed).sample(occ, ratio=0.5)
    return occ, sv, spec, M


def test_multitarget_shapes_and_spectrum_dependence():
    model = _model()
    model.eval()
    occ, sv, spec, M = _batch()
    sk = torch.ones(2, 3, dtype=torch.bool)
    with torch.no_grad():
        out = model(occ, sv, sk, spec, M, with_target=True)
        out2 = model(occ, sv, sk, spec * 1.5, M, with_target=True)

    assert out["z_y_occ_spec"].shape == (2, 256, 192)
    assert out["z_y_scal"].shape == (2, 192)
    assert out["z_hat_occ_spec"].shape == (2, 256, 192)
    assert out["z_hat_scal"].shape == (2, 192)

    # Stable target stays spectrum-free; the spec target must depend on it — and
    # it does from step 0, because the film is non-identity (not zero-init).
    assert torch.allclose(out["z_y_raw"], out2["z_y_raw"], atol=1e-6)
    assert not torch.allclose(out["z_y_occ_spec"], out2["z_y_occ_spec"], atol=1e-6)


def test_spectrum_film_is_frozen_and_spectrum_dependent():
    model = _model()
    assert all(not p.requires_grad for p in model.spectrum_film.parameters()), (
        "the spectrum conditioning must be frozen (non-nullifiable)")
    c1, c2 = torch.randn(2, 384), torch.randn(2, 384)
    g1 = model.spectrum_film(c1)[0][0]
    g2 = model.spectrum_film(c2)[0][0]
    assert not torch.equal(g1, g2), "the film must depend on c_physics"


def test_spec_head_depends_on_goal_no_shortcut():
    """Gate 2 of TARGET_DESIGN.md: at fixed occupancy, the spec-target head must
    change under null vs real goal (i.e. it actually uses the spectrum)."""
    model = _model()
    model.eval()
    occ, sv, spec, M = _batch()
    sk = torch.ones(2, 3, dtype=torch.bool)
    with torch.no_grad():
        zr = model(occ, sv, sk, spec, M, goal_mode="real", with_target=False)["z_hat_occ_spec"]
        zn = model(occ, sv, sk, spec, M, goal_mode="null", with_target=False)["z_hat_occ_spec"]
    assert not torch.allclose(zr, zn, atol=1e-6), (
        "z_hat_occ_spec must depend on the goal (no shortcut)")


def test_multitarget_loss_terms_and_gradient_ownership():
    model = _model()
    model.train()
    obj = UnifiedJEPALoss(
        hidden=192, lambda_inv=0.0, lambda_var=0.0, lambda_cov=0.0,
        lambda_scalar=0.0, lambda_occ=0.0, lambda_phys=0.0, lambda_summary=0.0,
        lambda_cond=0.5, lambda_scal_t=0.5)
    occ, sv, spec, M = _batch(seed=1)
    sk = torch.ones(2, 3, dtype=torch.bool)
    res = obj(model, occ, sv, sk, spec, M)
    assert {"L_cond", "L_scal_t"} <= set(res["term_losses"])

    res["total_loss"].backward()

    # The live spec head receives gradient; the frozen film receives none.
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.spec_proj.parameters())
    assert all(p.grad is None for p in model.spectrum_film.parameters())
    # Target latents carry no gradient (stop-grad at the EMA boundary).
    assert not res["out"]["z_y_occ_spec"].requires_grad
    assert not res["out"]["z_y_scal"].requires_grad


def test_multi_target_ramp():
    assert _ramp_frac(999, 1000, 2000) == 0.0
    assert 0.0 < _ramp_frac(1500, 1000, 2000) < 1.0
    assert _ramp_frac(3000, 1000, 2000) == pytest.approx(1.0)
    assert _ramp_frac(5, 0, 0) == 1.0


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
