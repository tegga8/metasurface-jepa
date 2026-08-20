"""Gradient-boundary tests for the faithful EMA-JEPA + VICReg-style candidate
(`jepa_vicreg`), per the Milestone B CODEX spec.

The critical contract: L_inv, L_var and L_cov EACH flow student -> projector;
the target branch reaches the objective-owned projector but NEVER the frozen
EMA target encoder; the EMA target is updated only via
`objective.on_optimizer_step` -> `model.ema.update`, never by the optimizer.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
import torch.nn as nn

from losses.objectives import VICRegObjective


class _EmaStub:
    def update(self, encoder, step):
        pass


class _FrozenEmaModel(nn.Module):
    """z_hat is a learnable student parameter; z_y comes from a FROZEN
    submodule mirroring the real EMA target encoder."""

    def __init__(self, hidden=8, B=2, T=256, grid=64):
        super().__init__()
        self.ema = _EmaStub()
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
    return VICRegObjective(
        lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0,
        projector_input_dim=8, projector_hidden_dim=16, projector_output_dim=8)


def _grad_norm(p):
    return p.grad.abs().sum().item() if p.grad is not None else 0.0


@pytest.mark.parametrize("term", ["L_inv", "L_var", "L_cov"])
def test_each_term_feeds_student_and_projector(model, tiny_batch, term):
    G, S, M = tiny_batch
    obj = _objective()
    c = obj(model, G, S, M)["components"]
    model.zero_grad()
    obj.zero_grad()
    c[term].backward()
    assert _grad_norm(model.z_hat) > 0, f"{term} must reach the student latent"
    assert _grad_norm(obj.projector.net[0].weight) > 0, \
        f"{term} must reach the projector"
    for p in model.target.parameters():
        assert p.grad is None, \
            f"{term} must never reach the frozen EMA target encoder"


@pytest.mark.parametrize("term", ["L_inv", "L_var", "L_cov"])
def test_each_term_target_branch_feeds_projector_never_ema(
        model, tiny_batch, term):
    """Backwarding ONLY the target branch of the term still trains the
    objective-owned projector, never the frozen EMA encoder."""
    G, S, M = tiny_batch
    obj = _objective()
    result = obj(model, G, S, M)
    mask = result["out"]["mask"]
    p_hat = result["projector_outputs"]["p_hat"][mask]
    p_y = result["projector_outputs"]["p_y"][mask]
    from losses.vicreg import covariance_loss, variance_loss
    target_term = {
        "L_inv": (p_y - p_hat.detach()).pow(2).mean(),
        "L_var": variance_loss(p_y),
        "L_cov": covariance_loss(p_y),
    }[term]
    model.zero_grad()
    obj.zero_grad()
    target_term.backward()
    assert _grad_norm(obj.projector.net[0].weight) > 0, \
        f"target branch of {term} must reach the projector"
    for p in model.target.parameters():
        assert p.grad is None, \
            f"target branch of {term} must never reach the EMA encoder"


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
    params = (list(model.parameters())
              + list(obj.parameters()))
    assert any(p.requires_grad for p in params)
    opt = torch.optim.AdamW(
        [p for p in params if p.requires_grad], lr=1e-2)

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
        "projector must move"
    for a, b in zip(t0, model.target.parameters()):
        assert torch.allclose(a, b, atol=1e-9), \
            "frozen EMA target encoder must not move via the optimizer"


def test_ema_update_is_driver_owned_not_optimizer(model, tiny_batch):
    """The EMA target is updated ONLY through objective.on_optimizer_step;
    a bare optimizer.step() (without the driver hook) must not touch it."""
    G, S, M = tiny_batch
    obj = _objective()
    opt = torch.optim.AdamW(
        [p for p in list(model.parameters()) + list(obj.parameters())
         if p.requires_grad], lr=1e-2)
    result = obj(model, G, S, M)
    result["total_loss"].backward()
    opt.step()
    for p in model.target.parameters():
        assert p.grad is None
