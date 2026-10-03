"""Phase 3 — representation hygiene (3a pixel-mask removal, 3b projector ablation).

Run:  python -m pytest tests/test_phase3_hygiene.py -v
      python tests/test_phase3_hygiene.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
import torch.nn as nn

from assembly import UnifiedJEPA
from data.mask import BlockMasker
from losses.unified_losses import UnifiedJEPALoss
from losses.vicreg import build_projector


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
    m.eval()
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


def test_masked_pixel_values_do_not_affect_student_latent():
    """Phase 3a: the student latent must not depend on the raw pixels inside the
    masked region — which is why the pre-patch-embed pixel mask was redundant.
    (Visible pixels still matter; only the masked ones are irrelevant.)"""
    model = _model()
    occ, sv, spec, M = _batch(seed=0)
    sk = torch.ones(2, 3, dtype=torch.bool)
    with torch.no_grad():
        z1 = model(occ, sv, sk, spec, M, with_target=False)["z_x"].clone()

        up = M.view(M.shape[0], 1, 16, 16).repeat_interleave(4, 2).repeat_interleave(4, 3)
        masked = up < 0.5                                   # (B,1,64,64)
        occ2 = occ.clone()
        occ2[masked] = torch.rand(int(masked.sum()))
        z2 = model(occ2, sv, sk, spec, M, with_target=False)["z_x"]
    assert torch.allclose(z1, z2, atol=1e-6), (
        "masked-region pixel values must not change the student latent")

    # And visible pixels DO matter (sanity: the model is not constant).
    with torch.no_grad():
        occ3 = occ.clone()
        vis = ~masked
        occ3[vis] = torch.rand(int(vis.sum()))
        z3 = model(occ3, sv, sk, spec, M, with_target=False)["z_x"]
    assert not torch.allclose(z1, z3, atol=1e-6), (
        "visible pixels must still affect the latent")


def test_build_projector_variants_shape_and_params():
    z = torch.randn(2, 5, 8)
    for kind in ("none", "linear", "mlp", "mlp_bn"):
        p = build_projector(kind, 8)
        out = p(z)
        assert out.shape == z.shape, f"{kind}: {out.shape}"
    assert isinstance(build_projector("none", 8), nn.Identity)
    assert sum(p.numel() for p in build_projector("none", 8).parameters()) == 0
    for kind in ("linear", "mlp", "mlp_bn"):
        assert sum(p.numel() for p in build_projector(kind, 8).parameters()) > 0
    # mlp_bn is the only one with BatchNorm.
    assert any(isinstance(m, nn.BatchNorm1d)
               for m in build_projector("mlp_bn", 8).modules())
    assert not any(isinstance(m, nn.BatchNorm1d)
                   for m in build_projector("mlp", 8).modules())


def test_build_projector_rejects_unknown():
    with pytest.raises(ValueError):
        build_projector("bogus", 8)


def test_objective_accepts_each_projector_type():
    for kind in ("none", "linear", "mlp", "mlp_bn"):
        obj = UnifiedJEPALoss(hidden=8, projector_type=kind)
        assert obj.projector is not None
    # Default remains the shipped MLP+BN.
    assert any(isinstance(m, nn.BatchNorm1d)
               for m in UnifiedJEPALoss(hidden=8).projector.modules())


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
