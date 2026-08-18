"""Smoke + registry tests for the jepa_var screening objective."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
import torch.nn as nn

from losses.jepa_loss import ProjectionMLP
from losses.objectives import OBJECTIVES, JEPAVarianceObjective


class _EmaStub:
    def update(self, encoder, step):
        pass


class _TinyModel(nn.Module):
    """Minimal model stand-in (same pattern as the repo's other objective-level
    fakes): forward returns z_hat/z_y/mask, .proj is a real projection head,
    .ema/.geometry_encoder exist for the on_optimizer_step contract."""

    def __init__(self, hidden=8):
        super().__init__()
        self.hidden = hidden
        self.proj = ProjectionMLP(hidden=hidden)
        self.ema = _EmaStub()
        self.geometry_encoder = None

    def forward(self, G, S, M):
        B = G.shape[0]
        mask = (M.view(B, -1) == 0)
        return {"z_hat": torch.randn(B, 256, self.hidden),
                "z_y": torch.randn(B, 256, self.hidden),
                "mask": mask}

    def __call__(self, G, S, M):
        return self.forward(G, S, M)


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return _TinyModel(hidden=8)


@pytest.fixture
def tiny_batch():
    torch.manual_seed(1)
    G = torch.randn(2, 3, 64, 64)
    S = torch.randn(2, 301)
    M = torch.ones(2, 256)
    M[:, :2] = 0  # 2 masked tokens per sample
    return G, S, M


def test_jepa_var_registered():
    assert "jepa_var" in OBJECTIVES
    assert OBJECTIVES["jepa_var"] is JEPAVarianceObjective


def test_jepa_var_components_present(tiny_model, tiny_batch):
    # tiny_model / tiny_batch: use whatever existing fixtures the repo's other
    # objective tests already use (JEPAObjective / JEPAVICRegObjective tests)
    # for consistency — do not invent a new fixture pattern here.
    obj = JEPAVarianceObjective(lambda_var=1.0, gamma=1.0)
    G, S, M = tiny_batch
    result = obj(tiny_model, G, S, M)
    assert "L_J" in result["components"]
    assert "L_var" in result["components"]
    assert "L_cov" not in result["components"]  # variance-only, no covariance term
    assert torch.isfinite(result["total_loss"])


def test_jepa_var_backward_finite(tiny_model, tiny_batch):
    obj = JEPAVarianceObjective(lambda_var=1.0, gamma=1.0)
    G, S, M = tiny_batch
    result = obj(tiny_model, G, S, M)
    result["total_loss"].backward()
    for p in tiny_model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()