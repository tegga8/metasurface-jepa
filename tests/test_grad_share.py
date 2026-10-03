"""Phase 2a — live per-term gradient-share probe (per_term_grad_share).

Guards the behaviour-neutral instrumentation added to the trainer: the objective
exposes per-term weighted losses, the probe attributes the gradient budget, and
it must be safe to call immediately before the real backward.

Run:  python -m pytest tests/test_grad_share.py -v
      python tests/test_grad_share.py
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
from train_unified import per_term_grad_share

TERMS = {"L_inv", "L_var", "L_cov", "L_scalar", "L_occ", "L_phys", "L_summary",
         "L_cond", "L_scal_t"}


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
    m.train()
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


def test_term_losses_exposed_and_shares_sum_to_one():
    model = _model()
    obj = UnifiedJEPALoss(hidden=192, lambda_inv=1.0, lambda_var=1.0,
                          lambda_cov=1.0, lambda_scalar=1.0, lambda_occ=1.0,
                          lambda_summary=1.0, lambda_phys=0.0)
    occ, sv, spec, M = _batch()
    sk = torch.ones(2, 3, dtype=torch.bool)
    result = obj(model, occ, sv, sk, spec, M, goal_mode="real")

    assert set(result["term_losses"]) == TERMS
    shares = per_term_grad_share(result, model, obj)
    assert set(shares) == TERMS
    assert abs(sum(shares.values()) - 1.0) < 1e-6
    # lambda_phys = 0 -> the physics term contributes no gradient.
    assert shares["L_phys"] == 0.0
    assert max(shares.values()) > 0.05, "some term must carry the budget"


def test_probe_clears_grads_and_does_not_update_weights():
    model = _model()
    obj = UnifiedJEPALoss(hidden=192, lambda_inv=1.0, lambda_var=1.0,
                          lambda_cov=1.0, lambda_scalar=1.0, lambda_occ=1.0,
                          lambda_summary=0.0, lambda_phys=0.0)
    occ, sv, spec, M = _batch(seed=1)
    sk = torch.ones(2, 3, dtype=torch.bool)
    result = obj(model, occ, sv, sk, spec, M, goal_mode="real")

    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if p.requires_grad}
    per_term_grad_share(result, model, obj)

    # Safe before the real backward: no gradient left behind ...
    assert all(p.grad is None for p in model.parameters() if p.requires_grad)
    # ... and no weight moved.
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p.detach(), before[n]), f"{n} changed"


def test_probe_is_safe_before_real_backward():
    model = _model()
    obj = UnifiedJEPALoss(hidden=192, lambda_inv=1.0, lambda_var=1.0,
                          lambda_cov=1.0, lambda_scalar=1.0, lambda_occ=1.0,
                          lambda_summary=1.0, lambda_phys=0.0)
    occ, sv, spec, M = _batch(seed=2)
    sk = torch.ones(2, 3, dtype=torch.bool)
    result = obj(model, occ, sv, sk, spec, M, goal_mode="real")

    per_term_grad_share(result, model, obj)      # probe first (retains the graph)
    result["total_loss"].backward()              # the real backward
    has = any(p.grad is not None and p.grad.abs().sum() > 0
              for p in model.parameters() if p.requires_grad)
    assert has, "real backward after the probe must populate gradients"


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
