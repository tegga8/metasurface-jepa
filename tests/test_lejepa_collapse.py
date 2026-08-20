"""Synthetic collapse tests for the teacher-free LeJEPA objective (§22).

With no EMA/teacher copy, collapse prevention comes from SIGReg-style
distributional regularization on BOTH projected branches: a collapsed (delta)
distribution must be heavily penalized while a healthy Gaussian distribution is
not. These tests verify that (a) the objective never touches an EMA, and (b)
the SIGReg terms actually discriminate collapse on hand-constructed features.

Run:  python tests/test_lejepa_collapse.py   (pytest-collectable)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
import torch.nn as nn

from losses.objectives import LeJEPAObjective
from losses.sigreg import sigreg_loss


class _CountingEma:
    def __init__(self):
        self.calls = []

    def update(self, encoder, step):
        self.calls.append((encoder, step))


class _StudentEncoder(nn.Module):
    def __init__(self, hidden=8, T=256, grid=64):
        super().__init__()
        self.linear = nn.Linear(3 * grid * grid, T * hidden)

    def forward(self, G):
        B = G.shape[0]
        return self.linear(G.flatten(1)).view(B, 256, 8)


class _Student(nn.Module):
    def __init__(self, z_hat_init, hidden=8, B=2, T=256, grid=64):
        super().__init__()
        self.geometry_encoder = _StudentEncoder(hidden=hidden, T=T, grid=grid)
        self.ema = _CountingEma()
        self.z_hat = nn.Parameter(z_hat_init.clone())

    def forward(self, G, S, M, goal_mode="real", need_attn=False, with_target=True):
        B = G.shape[0]
        mask = (M.view(B, -1) == 0)
        return {"z_hat": self.z_hat, "mask": mask}


def _obj(**kw):
    defaults = dict(lambda_sigreg=1.0, num_slices=16, num_points=64, seed=0,
                    projector_input_dim=8, projector_hidden_dim=16,
                    projector_output_dim=8)
    defaults.update(kw)
    return LeJEPAObjective(**defaults)


@pytest.fixture
def tiny_batch():
    torch.manual_seed(1)
    G = torch.randn(2, 3, 64, 64)
    S = torch.randn(2, 301)
    M = torch.ones(2, 256)
    M[:, :2] = 0  # 2 masked tokens per sample -> N=4 features
    return G, S, M


def test_objective_never_touches_ema(tiny_batch):
    """The LeJEPA forward path and on_optimizer_step must never call model.ema."""
    G, S, M = tiny_batch
    torch.manual_seed(0)
    model = _Student(torch.randn(2, 256, 8))
    obj = _obj()
    obj(model, G, S, M)
    obj.on_optimizer_step(model, 7)
    assert model.ema.calls == [], "LeJEPA must never touch an EMA (teacher-free)"


def test_collapsed_prediction_is_penalized(tiny_batch):
    """A constant (delta) prediction branch must carry a large SIGReg term —
    collapse cannot hide from the Gaussianity test."""
    G, S, M = tiny_batch
    const = torch.ones(2, 256, 8)
    model = _Student(const)
    obj = _obj()
    c = obj(model, G, S, M)["components"]
    assert c["L_SIGReg_pred"].item() > 0.1, (
        f"collapsed prediction must be penalized, got {c['L_SIGReg_pred'].item()}")


def test_collapsed_target_is_penalized(tiny_batch):
    """The target is the STUDENT's own output — it gets the same distributional
    pressure as the prediction (no teacher to keep it healthy)."""
    G, S, M = tiny_batch
    model = _Student(torch.randn(2, 256, 8))
    obj = _obj()
    with torch.no_grad():
        model.geometry_encoder.linear.weight.zero_()   # force constant student output
        model.geometry_encoder.linear.bias.zero_()
    c = obj(model, G, S, M)["components"]
    assert c["L_SIGReg_target"].item() > 0.1, (
        f"collapsed student target must be penalized, got {c['L_SIGReg_target'].item()}")


def test_sigreg_discriminates_collapse_from_healthy():
    """Direct sigreg_loss check: a delta distribution must score much worse than
    a healthy Gaussian."""
    torch.manual_seed(3)
    healthy = sigreg_loss(torch.randn(256, 8), num_slices=16, num_points=64)[0]
    collapsed = sigreg_loss(torch.ones(256, 8), num_slices=16, num_points=64)[0]
    assert collapsed.item() > 10 * healthy.item() + 1e-6, (
        f"collapsed {collapsed.item():.4f} must dominate healthy {healthy.item():.4f}")


def test_sigreg_positive_definite_semantics():
    """A perfectly Gaussian slice must score near zero: N(0,1) samples against
    the ECF of N(0,1) give phi(t) ~ exp(-t^2/2) -> loss ~ 0 for large n."""
    torch.manual_seed(4)
    loss, info = sigreg_loss(torch.randn(4096, 8), num_slices=8, num_points=512)
    assert loss.item() < 0.05, f"Gaussian samples must score near zero, got {loss.item()}"
    assert info["num_points"] == 512


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