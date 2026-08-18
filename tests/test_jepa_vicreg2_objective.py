"""Smoke + gradient-boundary tests for the corrected branch-symmetric VICReg
objective (`jepa_vicreg2`), per the Milestone B final screening directive §6/§17.

Two independent target-gradient boundary tests are required:

1. Original JEPA cosine path  — the Bug #1 stop-grad contract (target detached
   inside jepa_loss()) must hold inside the new objective's L_J term. The exact
   regression assertion already lives in tests/test_jepa_loss.py and is reused
   here at the objective level (proj gradient of the L_J term == the gradient a
   detached-target reference produces). The original test is NOT modified.
2. New dual-VICReg target regularization path — the projector must receive
   gradient from the separate target-branch terms (L_var_target, L_cov_target).
   This is intentional: the EMA target encoder is frozen, so this path trains
   the shared projection head only.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
import torch.nn as nn

from losses.jepa_loss import ProjectionMLP, jepa_loss
from losses.objectives import (
    OBJECTIVES,
    JEPAVICRegDualObjective,
    JEPAVICRegObjective,
)


class _EmaStub:
    def update(self, encoder, step):
        pass


class _RecordingEma:
    def __init__(self, seen):
        self.seen = seen

    def update(self, encoder, step):
        self.seen.append((encoder, step))


class _ParamModel(nn.Module):
    """Deterministic objective-level stand-in: z_hat/z_y are learnable
    Parameters, so gradients from any path (L_J, prediction branch, target
    branch) are observable and reproducible across objective calls on the same
    model. (In the real assembly, z_y is the frozen EMA encoder's output; the
    learnable stand-in only makes target-branch gradients measurable in tests.)
    """

    def __init__(self, hidden=8, B=2, T=256):
        super().__init__()
        self.proj = ProjectionMLP(hidden=hidden)
        self.ema = _EmaStub()
        self.geometry_encoder = None
        self.z_hat = nn.Parameter(torch.randn(B, T, hidden))
        self.z_y = nn.Parameter(torch.randn(B, T, hidden))

    def forward(self, G, S, M):
        B = G.shape[0]
        mask = (M.view(B, -1) == 0)
        return {"z_hat": self.z_hat, "z_y": self.z_y, "mask": mask}


class _FrozenEmaModel(nn.Module):
    """z_y comes from a FROZEN submodule (mirroring the real EMA target
    encoder): backwarding the total loss must never produce gradients on its
    parameters — the milestone's "EMA target encoder stays frozen" boundary."""

    def __init__(self, hidden=8, T=256, grid=64):
        super().__init__()
        self.proj = ProjectionMLP(hidden=hidden)
        self.ema = _EmaStub()
        self.geometry_encoder = None
        self.z_hat = nn.Parameter(torch.randn(2, T, hidden))
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
def param_model():
    torch.manual_seed(0)
    return _ParamModel(hidden=8)


@pytest.fixture
def tiny_batch():
    torch.manual_seed(1)
    G = torch.randn(2, 3, 64, 64)
    S = torch.randn(2, 301)
    M = torch.ones(2, 256)
    M[:, :2] = 0  # 2 masked tokens per sample -> N=4 features per branch
    return G, S, M


def test_jepa_vicreg2_registered():
    assert "jepa_vicreg2" in OBJECTIVES
    assert OBJECTIVES["jepa_vicreg2"] is JEPAVICRegDualObjective


def test_jepa_vicreg2_components_present(param_model, tiny_batch):
    obj = JEPAVICRegDualObjective()
    result = obj(param_model, *tiny_batch)
    c = result["components"]
    for k in ("L_J", "L_var_pred", "L_var_target",
              "L_cov_pred", "L_cov_target",
              "lambda_var", "lambda_cov"):
        assert k in c, f"missing component {k}"
    assert torch.isfinite(result["total_loss"])
    assert torch.isfinite(c["L_var_pred"]) and torch.isfinite(c["L_var_target"])
    assert torch.isfinite(c["L_cov_pred"]) and torch.isfinite(c["L_cov_target"])


def test_jepa_vicreg2_total_backward_finite(param_model, tiny_batch):
    obj = JEPAVICRegDualObjective()
    result = obj(param_model, *tiny_batch)
    result["total_loss"].backward()
    for p in param_model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_jepa_vicreg2_on_optimizer_step_updates_ema(param_model):
    seen = []
    param_model.ema = _RecordingEma(seen)
    obj = JEPAVICRegDualObjective()
    obj.on_optimizer_step(param_model, 3)
    assert seen == [(None, 3)]


def test_jepa_cosine_path_keeps_bug1_stopgrad_boundary(param_model, tiny_batch):
    """Boundary 1 (original JEPA cosine path): the L_J term inside jepa_vicreg2
    must produce exactly the projection-head gradient a detached-target
    reference produces. Bug #1 (tests/test_jepa_loss.py) must not be weakened
    by the new objective — the cosine path stays a pred-only gradient path."""
    G, S, M = tiny_batch
    obj = JEPAVICRegDualObjective()
    result = obj(param_model, G, S, M)
    mask = result["out"]["mask"]

    param_model.proj.zero_grad()
    result["components"]["L_J"].backward()
    g_inside = param_model.proj.net[0].weight.grad.clone()
    param_model.proj.zero_grad()

    L_ref, _ = jepa_loss(
        param_model.z_hat, param_model.z_y.detach(), mask, proj=param_model.proj)
    L_ref.backward()
    g_ref = param_model.proj.net[0].weight.grad.clone()

    assert torch.allclose(g_inside, g_ref, atol=1e-6), (
        "L_J term inside jepa_vicreg2 must match a detached-target reference: "
        f"max diff {((g_inside - g_ref).abs().max().item()):.2e}")


def test_vicreg2_target_regularization_path_feeds_projector(param_model, tiny_batch):
    """Boundary 2 (new dual-VICReg path): backwarding ONLY the target-branch
    regularization terms (L_var_target, L_cov_target) must push a nonzero
    gradient into the shared projection head. Intentional by design — the EMA
    target encoder (frozen in the real model) cannot receive it; the
    projector can, and does."""
    G, S, M = tiny_batch
    obj = JEPAVICRegDualObjective(lambda_var=0.5, lambda_cov=0.2)
    comps = obj(param_model, G, S, M)["components"]

    param_model.proj.zero_grad()
    (comps["lambda_var"] * comps["L_var_target"]
     + comps["lambda_cov"] * comps["L_cov_target"]).backward()
    g = param_model.proj.net[0].weight.grad
    assert g is not None, "target regularization path must reach the projector"
    assert torch.isfinite(g).all()
    assert g.abs().sum() > 0, (
        "target branch must contribute a nonzero gradient to the projector")


def test_vicreg2_ema_target_encoder_stays_frozen(tiny_batch):
    """Boundary 3 (final pre-training pass): the EMA target ENCODER must stay
    frozen — no gradient may ever reach its parameters through the total loss,
    even though the separate regularization path trains the shared projector."""
    model = _FrozenEmaModel()
    obj = JEPAVICRegDualObjective(lambda_var=0.5, lambda_cov=0.2)
    result = obj(model, *tiny_batch)
    result["total_loss"].backward()
    for p in model.target.parameters():
        assert p.grad is None, "frozen EMA target encoder must receive no gradient"
    g = model.proj.net[0].weight.grad
    assert g is not None and g.abs().sum() > 0, (
        "projector must still be trained through the separate target path")


def test_vicreg2_proj_gradient_differs_from_historical_single_branch(
        param_model, tiny_batch):
    """The corrected objective must actually change the projector gradient
    versus the historical prediction-only variant on identical inputs — i.e.
    the target branch is genuinely attached, not a no-op."""
    G, S, M = tiny_batch
    hist = JEPAVICRegObjective(lambda_var=0.1, lambda_cov=0.04, gamma=1.0)
    dual = JEPAVICRegDualObjective(lambda_var=0.1, lambda_cov=0.04, gamma=1.0)

    param_model.proj.zero_grad()
    hist(param_model, G, S, M)["total_loss"].backward()
    g_hist = param_model.proj.net[0].weight.grad.clone()
    param_model.proj.zero_grad()
    dual(param_model, G, S, M)["total_loss"].backward()
    g_dual = param_model.proj.net[0].weight.grad.clone()

    assert torch.isfinite(g_hist).all() and torch.isfinite(g_dual).all()
    assert not torch.allclose(g_hist, g_dual, atol=1e-8), (
        "dual-branch objective's projector gradient must differ from the "
        "prediction-only variant's (target branch must actually contribute)")