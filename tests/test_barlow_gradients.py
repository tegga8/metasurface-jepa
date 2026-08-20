"""Gradient-boundary tests for the Barlow objective (`jepa_barlow`, §21).

The critical contract (mirrors the VICReg boundary tests): L_BT flows
student -> projector; the target branch reaches the objective-owned Barlow
projector but NEVER the frozen EMA target encoder; the EMA target is updated
only via `objective.on_optimizer_step` -> `model.ema.update`, never by the
optimizer. The objective owns its own BarlowProjector (never shared with VICReg).

Run:  python tests/test_barlow_gradients.py   (pytest-collectable)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
import torch.nn as nn

from losses.barlow import barlow_twins_loss
from losses.objective_modules import BarlowProjector
from losses.objectives import OBJECTIVES, BarlowObjective


class _EmaStub:
    def update(self, encoder, step):
        pass


class _RecordingEma(_EmaStub):
    def __init__(self):
        self.calls = []

    def update(self, encoder, step):
        self.calls.append((encoder, step))


class _FrozenEmaModel(nn.Module):
    """z_hat is a learnable student parameter; z_y comes from a FROZEN
    submodule mirroring the real EMA target encoder. Carries NO projector."""

    def __init__(self, hidden=8, B=2, T=256, grid=64):
        super().__init__()
        self.ema = _RecordingEma()
        self.geometry_encoder = None
        self.z_hat = nn.Parameter(torch.randn(B, T, hidden))
        self.target = nn.Linear(3 * grid * grid, T * hidden)
        for p in self.target.parameters():
            p.requires_grad_(False)

    def forward(self, G, S, M):
        B = G.shape[0]
        mask = (M.view(B, -1) == 0)
        T, H = self.z_hat.shape[1], self.z_hat.shape[2]
        z_y = self.target(G.flatten(1)).view(B, T, H)
        return {"z_hat": self.z_hat, "z_y": z_y, "mask": mask}


@pytest.fixture
def model():
    torch.manual_seed(0)
    return _FrozenEmaModel(hidden=8)


@pytest.fixture
def tiny_batch():
    torch.manual_seed(1)
    G = torch.randn(2, 3, 64, 64)
    S = torch.randn(2, 301)
    M = torch.ones(2, 256)
    M[:, :4] = 0  # 8 masked tokens -> N=8 >= 2
    return G, S, M


def _objective():
    return BarlowObjective(lambda_bt=1.0, alpha=0.005,
                           projector_input_dim=8, projector_hidden_dim=16,
                           projector_output_dim=8)


def _grad_norm(p):
    return p.grad.abs().sum().item() if p.grad is not None else 0.0


def test_barlow_registered_and_owns_projector():
    assert "jepa_barlow" in OBJECTIVES
    assert OBJECTIVES["jepa_barlow"] is BarlowObjective
    obj = _objective()
    assert isinstance(obj.projector, BarlowProjector)
    assert obj.term_names == ("L_BT",)


def test_bt_loss_flows_to_student_and_projector(model, tiny_batch):
    G, S, M = tiny_batch
    obj = _objective()
    c = obj(model, G, S, M)["components"]
    model.zero_grad()
    obj.zero_grad()
    c["L_BT"].backward()
    assert _grad_norm(model.z_hat) > 0, "L_BT must reach the student latent"
    assert _grad_norm(obj.projector.net[0].weight) > 0, "L_BT must reach the projector"
    for p in model.target.parameters():
        assert p.grad is None, "L_BT must never reach the frozen EMA encoder"


def test_target_branch_feeds_projector_never_ema(model, tiny_batch):
    """Backwarding ONLY the target side of the cross-correlation still trains the
    objective-owned projector, never the frozen EMA encoder."""
    G, S, M = tiny_batch
    obj = _objective()
    result = obj(model, G, S, M)
    p_y = result["projector_outputs"]["p_y"][result["out"]["mask"]]
    model.zero_grad()
    obj.zero_grad()
    p_y.pow(2).mean().backward()   # target-branch-only scalar
    assert _grad_norm(obj.projector.net[0].weight) > 0, \
        "target branch must reach the projector"
    for p in model.target.parameters():
        assert p.grad is None, "target branch must never reach the EMA encoder"


def test_total_backward_keeps_ema_encoder_frozen(model, tiny_batch):
    G, S, M = tiny_batch
    obj = _objective()
    result = obj(model, G, S, M)
    result["total_loss"].backward()
    for p in model.target.parameters():
        assert p.grad is None, "EMA target encoder must receive no gradient"
    assert _grad_norm(obj.projector.net[0].weight) > 0
    assert _grad_norm(model.z_hat) > 0


def test_optimizer_moves_student_and_projector_but_not_ema(model, tiny_batch):
    G, S, M = tiny_batch
    obj = _objective()
    params = [p for p in list(model.parameters()) + list(obj.parameters())
              if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-2)

    z0 = model.z_hat.detach().clone()
    t0 = [p.detach().clone() for p in model.target.parameters()]
    p0 = [p.detach().clone() for p in obj.parameters()]

    result = obj(model, G, S, M)
    result["total_loss"].backward()
    opt.step()
    obj.on_optimizer_step(model, step=0)

    assert not torch.allclose(model.z_hat.detach(), z0, atol=1e-9), \
        "student latent must move"
    assert any(not torch.allclose(a, b, atol=1e-9)
               for a, b in zip(p0, obj.parameters())), \
        "Barlow projector must move"
    for a, b in zip(t0, model.target.parameters()):
        assert torch.allclose(a, b, atol=1e-9), \
            "frozen EMA target encoder must not move via the optimizer"


def test_ema_update_is_driver_owned_not_optimizer(model, tiny_batch):
    """The EMA target is updated ONLY through objective.on_optimizer_step."""
    G, S, M = tiny_batch
    obj = _objective()
    opt = torch.optim.AdamW(
        [p for p in list(model.parameters()) + list(obj.parameters())
         if p.requires_grad], lr=1e-2)
    result = obj(model, G, S, M)
    result["total_loss"].backward()
    opt.step()
    assert model.ema.calls == [], \
        "a bare optimizer.step() must not update the EMA"
    obj.on_optimizer_step(model, step=3)
    assert len(model.ema.calls) == 1 and model.ema.calls[0][1] == 3


def test_barlow_components_mean_form():
    """L_BT = diag_term + alpha * off_diag_term, each O(1) per entry (mean-form
    scaling fix): identical standardized branches give C ~ I -> L_BT near 0."""
    torch.manual_seed(3)
    z = torch.randn(64, 8)
    loss, info = barlow_twins_loss(z, z.clone(), alpha=0.005)
    # Off-diagonal sampling noise is O(alpha/D) ~ 1e-3 at N=64; the mean-form
    # terms themselves must be ~0 for identical standardized branches.
    assert abs(loss.item()) < 1e-3, f"identical branches must be ~0, got {loss.item()}"
    assert abs(info["diag_term"]) < 1e-3
    assert abs(info["off_diag_term"]) < 0.05  # ~ (1/sqrt(N))^2 = 0.016 at N=64


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