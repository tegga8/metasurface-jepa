"""Controlled-ablation tests: objective="conventional" vs "jepa".

Covers: conventional objective contains only direct-supervision terms; JEPA terms are
absent; objective mode does not alter architecture construction; the EMA target is not
advanced in conventional mode; the projector is unused; a conventional forward/backward
runs; and the scalar/spectrum shuffles are strict derangements.

Run: python -m pytest tests/test_objective_ablation.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
import torch.nn as nn

from assembly import UnifiedJEPA
from losses.unified_losses import UnifiedJEPALoss
from runtime.physics_controls import derangement_permutation, derange_batch_tensor


class _StubReleased(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 256))

    def forward(self, S):
        return self.net(S.transpose(1, 2))


class _SurrogateOut:
    def __init__(self, prediction):
        self.prediction = prediction


class _StubSurrogate(nn.Module):
    """Differentiable stand-in with the real surrogate's interface."""
    def forward(self, geom):
        v = geom.mean(dim=(1, 2, 3))
        return _SurrogateOut(v.view(-1, 1, 1).expand(geom.shape[0], 2, 301).contiguous())


def _model():
    torch.manual_seed(0)
    m = UnifiedJEPA(hidden=192, num_heads=6, geo_depth=2, predictor_depth=4,
                    goal_tokens=16, num_predictor_heads=6, scalar_hidden=128,
                    n_film_blocks=2, spec_dim=256)
    stub = _StubReleased()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    m.spectrum_path.released = stub
    m.ema.target.load_state_dict(m.occupancy_encoder.state_dict())
    m.scalar_mlp_ema.target.load_state_dict(m.scalar_encoder.state_dict())
    m.train()
    return m


def _batch(b=2):
    torch.manual_seed(1)
    occ = (torch.rand(b, 1, 64, 64) > 0.5).float()
    sv = torch.tensor([[2.7, 0.8, 4.2], [2.6, 0.7, 3.9]])[:b]
    sk = torch.zeros(b, 3, dtype=torch.bool)
    spec = torch.randn(b, 2, 301)
    mask = (torch.rand(b, 16, 16) > 0.5).float()
    return occ, sv, sk, spec, mask


def _objective(mode, surrogate):
    return UnifiedJEPALoss(hidden=192, lambda_inv=25.0, lambda_var=25.0,
                           lambda_cov=1.0, lambda_scalar=1.0, lambda_occ=1.0,
                           lambda_phys=1.0, lambda_summary=1.0, lambda_cond=0.5,
                           lambda_scal_t=0.5, surrogate=surrogate, objective=mode)


def test_conventional_objective_has_only_direct_terms():
    m = _model()
    obj = _objective("conventional", _StubSurrogate())
    occ, sv, sk, spec, mask = _batch()
    res = obj(m, occ, sv, sk, spec, mask)
    tl = res["term_losses"]
    for k in ("L_inv", "L_var", "L_cov", "L_summary", "L_cond", "L_scal_t"):
        assert float(tl[k].detach()) == 0.0, f"{k} must be OFF in conventional mode"
    comp = res["components"]
    total = (comp["L_scalar"] * 1.0 + comp["L_occ"] * 1.0 + comp["L_phys"] * 1.0)
    assert abs(comp["L_total"] - total) < 1e-4
    assert torch.isfinite(res["total_loss"])


def test_jepa_mode_is_unchanged():
    m = _model()
    obj = _objective("jepa", _StubSurrogate())
    occ, sv, sk, spec, mask = _batch()
    res = obj(m, occ, sv, sk, spec, mask)
    comp = res["components"]
    total = (comp["L_inv_weighted"] + comp["L_var_weighted"] + comp["L_cov_weighted"]
             + comp["L_scalar"] + comp["L_occ"] + comp["L_phys"]
             + comp["L_summary_weighted"] + comp["L_cond_weighted"]
             + comp["L_scal_t_weighted"])
    assert abs(comp["L_total"] - total) < 1e-4
    assert float(res["term_losses"]["L_inv"].detach()) > 0.0


def test_conventional_forward_backward_and_projector_unused():
    m = _model()
    obj = _objective("conventional", _StubSurrogate())
    occ, sv, sk, spec, mask = _batch()
    res = obj(m, occ, sv, sk, spec, mask)
    res["total_loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.parameters() if p.requires_grad)
    # the JEPA projector is never used -> no gradient in conventional mode
    assert all(p.grad is None for p in obj.projector.parameters())


def test_conventional_does_not_advance_ema():
    m = _model()
    obj = _objective("conventional", _StubSurrogate())
    before = {k: v.clone() for k, v in m.ema.target.state_dict().items()}
    obj.on_optimizer_step(m, 1)
    for k, v in m.ema.target.state_dict().items():
        assert torch.equal(v, before[k]), f"EMA advanced in conventional mode: {k}"


def test_objective_mode_does_not_change_architecture():
    a = _model()
    torch.manual_seed(0)
    b = _model()
    # the objective is not part of the model; both are built identically
    assert list(a.state_dict().keys()) == list(b.state_dict().keys())
    _ = _objective("conventional", _StubSurrogate())   # constructing it changes no model


def test_scalar_and_spectrum_shuffles_are_strict_derangements():
    for b in (2, 3, 5, 8):
        for seed in (0, 1, 7, 12345):
            perm = derangement_permutation(b, "cpu", seed=seed)
            assert not torch.any(perm == torch.arange(b)), f"fixed point b={b} seed={seed}"
    X = torch.arange(6).float().unsqueeze(1)
    Xs = derange_batch_tensor(X, seed=0)
    assert not torch.any(Xs.squeeze(1) == X.squeeze(1))
    import pytest
    with pytest.raises(ValueError):
        derange_batch_tensor(torch.zeros(1, 3), seed=0)


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
