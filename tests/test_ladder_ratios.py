"""Ladder component-logging tests (Milestone B final screening, Batch 4 directive):

Every regularized rung must report, per step:
- the unweighted regularizer terms (L_var / L_cov split per branch for
  jepa_vicreg2, L_BT with diag/off-diag, L_SIGReg),
- the lambda-weighted regularizer (`*_weighted`),
- the regularizer's share of the total loss (`var_ratio` / `cov_ratio` /
  `barlow_ratio` / `sigreg_ratio`),

all as tensors so the training loop's tensor-only accumulator
(comp_sums/comp_counts in train_milestone_b.py) picks them up. The plain `jepa`
rung has no regularizer and must stay clean of ratio keys.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
import torch.nn as nn

from losses.jepa_loss import ProjectionMLP
from losses.objectives import (
    JEPABarlowObjective,
    JEPAVICRegDualObjective,
    JEPAVICRegObjective,
    JEPAVarianceObjective,
    JEPAObjective,
    LeJEPAObjective,
)


class _EmaStub:
    def update(self, encoder, step):
        pass


class _ParamModel(nn.Module):
    """Deterministic objective-level stand-in (same pattern as the repo's other
    objective tests): z_hat/z_y learnable Parameters, proj a real head."""

    def __init__(self, hidden=8, B=2, T=256):
        super().__init__()
        self.proj = ProjectionMLP(hidden=hidden)
        self.ema = _EmaStub()
        self.geometry_encoder = None
        self.z_hat = nn.Parameter(torch.randn(B, T, hidden))
        self.z_y = nn.Parameter(torch.randn(B, T, hidden))

    def forward(self, G, S, M, with_target=True):
        B = G.shape[0]
        mask = (M.view(B, -1) == 0)
        out = {"z_hat": self.z_hat, "z_y": self.z_y, "mask": mask}
        return out


class _StudentEncoder(nn.Module):
    """Real contract: geometry_encoder(G) -> (B, T, hidden) token features."""

    def __init__(self, hidden=8, T=256, grid=64):
        super().__init__()
        self.hidden = hidden
        self.T = T
        self.linear = nn.Linear(3 * grid * grid, T * hidden)

    def forward(self, G):
        B = G.shape[0]
        return self.linear(G.flatten(1)).view(B, self.T, self.hidden)


class _LeJEPAStubModel(nn.Module):
    """LeJEPA stand-in: forward(with_target=False) returns no z_y; the objective
    calls model.geometry_encoder(G) itself (see test_lejepa_objective.py)."""

    def __init__(self, hidden=8, B=2, T=256, grid=64):
        super().__init__()
        self.proj = ProjectionMLP(hidden=hidden)
        self.geometry_encoder = _StudentEncoder(hidden=hidden, T=T, grid=grid)
        self.ema = _EmaStub()
        self.z_hat = nn.Parameter(torch.randn(B, T, hidden))

    def forward(self, G, S, M, goal_mode="real", need_attn=False, with_target=True):
        B = G.shape[0]
        mask = (M.view(B, -1) == 0)
        out = {"z_hat": self.z_hat, "mask": mask}
        if with_target:
            out["z_y"] = self.geometry_encoder(G)
        return out


@pytest.fixture
def param_model():
    torch.manual_seed(0)
    return _ParamModel(hidden=8)


@pytest.fixture
def lejepa_model():
    torch.manual_seed(0)
    return _LeJEPAStubModel(hidden=8)


@pytest.fixture
def tiny_batch():
    torch.manual_seed(1)
    G = torch.randn(2, 3, 64, 64)
    S = torch.randn(2, 301)
    M = torch.ones(2, 256)
    M[:, :2] = 0  # 2 masked tokens per sample -> N=4 features per branch
    return G, S, M


def _assert_ratio(components, weighted_key, ratio_key, total):
    assert weighted_key in components, f"missing {weighted_key}"
    assert ratio_key in components, f"missing {ratio_key}"
    w = components[weighted_key]
    r = components[ratio_key]
    assert torch.isfinite(w), f"{weighted_key} must be finite"
    assert torch.isfinite(r), f"{ratio_key} must be finite"
    assert isinstance(w, torch.Tensor) and isinstance(r, torch.Tensor), (
        "weighted/ratio components must be tensors for the loop accumulator")
    assert torch.allclose(w / total.clamp_min(1e-8), r, atol=1e-4), (
        f"{ratio_key} must equal {weighted_key} / total_loss")


def test_jepa_plain_has_no_regularizer_components(param_model, tiny_batch):
    result = JEPAObjective()(param_model, *tiny_batch)
    for k in result["components"]:
        assert not k.endswith("_weighted"), f"plain jepa must not log {k}"
        assert not k.endswith("_ratio"), f"plain jepa must not log {k}"
    assert set(result["components"].keys()) == {"L_J"}


def test_jepa_var_weighted_and_ratio(param_model, tiny_batch):
    obj = JEPAVarianceObjective(lambda_var=1.0, gamma=1.0)
    result = obj(param_model, *tiny_batch)
    c = result["components"]
    assert torch.allclose(c["L_var_weighted"], c["lambda_var"] * c["L_var"],
                          atol=1e-6)
    _assert_ratio(c, "L_var_weighted", "var_ratio", result["total_loss"])


def test_jepa_vicreg_weighted_and_ratios(param_model, tiny_batch):
    obj = JEPAVICRegObjective(lambda_var=0.1, lambda_cov=0.04, gamma=1.0)
    result = obj(param_model, *tiny_batch)
    c = result["components"]
    for k in ("L_var", "L_cov", "L_var_weighted", "L_cov_weighted",
              "var_ratio", "cov_ratio"):
        assert k in c, f"missing component {k}"
    assert torch.allclose(c["L_var_weighted"], c["lambda_var"] * c["L_var"],
                          atol=1e-6)
    assert torch.allclose(c["L_cov_weighted"], c["lambda_cov"] * c["L_cov"],
                          atol=1e-6)
    _assert_ratio(c, "L_var_weighted", "var_ratio", result["total_loss"])
    _assert_ratio(c, "L_cov_weighted", "cov_ratio", result["total_loss"])


def test_jepa_vicreg2_weighted_and_ratios(param_model, tiny_batch):
    obj = JEPAVICRegDualObjective(lambda_var=0.1, lambda_cov=0.04, gamma=1.0)
    result = obj(param_model, *tiny_batch)
    c = result["components"]
    for k in ("L_var_pred", "L_var_target", "L_cov_pred", "L_cov_target",
              "L_var", "L_cov", "L_var_weighted", "L_cov_weighted",
              "var_ratio", "cov_ratio"):
        assert k in c, f"missing component {k}"
    # combined == sum of the two branch terms, weighted == lambda * combined
    assert torch.allclose(c["L_var"], c["L_var_pred"] + c["L_var_target"],
                          atol=1e-6)
    assert torch.allclose(c["L_cov"], c["L_cov_pred"] + c["L_cov_target"],
                          atol=1e-6)
    assert torch.allclose(c["L_var_weighted"], c["lambda_var"] * c["L_var"],
                          atol=1e-6)
    assert torch.allclose(c["L_cov_weighted"], c["lambda_cov"] * c["L_cov"],
                          atol=1e-6)
    _assert_ratio(c, "L_var_weighted", "var_ratio", result["total_loss"])
    _assert_ratio(c, "L_cov_weighted", "cov_ratio", result["total_loss"])


def test_jepa_barlow_weighted_and_ratio(param_model, tiny_batch):
    obj = JEPABarlowObjective(lambda_bt=1.0, alpha=0.005)
    result = obj(param_model, *tiny_batch)
    c = result["components"]
    for k in ("L_BT", "bt_diag", "bt_off_diag", "L_BT_weighted", "barlow_ratio"):
        assert k in c, f"missing component {k}"
    assert torch.allclose(c["L_BT_weighted"], c["lambda_bt"] * c["L_BT"],
                          atol=1e-6)
    # bt_diag/bt_off_diag must be tensors (loop accumulator) — Batch 4 fix:
    # the raw barlow info returns python floats which the tensor-only
    # comp_sums/comp_counts accumulator silently drops.
    assert isinstance(c["bt_diag"], torch.Tensor) and c["bt_diag"].ndim == 0
    assert isinstance(c["bt_off_diag"], torch.Tensor) and c["bt_off_diag"].ndim == 0
    assert torch.isfinite(c["bt_diag"]) and torch.isfinite(c["bt_off_diag"])
    # L_BT == diag + alpha * off_diag (barlow_twins_loss definition)
    assert torch.allclose(c["L_BT"],
                          c["bt_diag"] + c["bt_diag"].new_tensor(obj.alpha) * c["bt_off_diag"],
                          atol=1e-4)
    _assert_ratio(c, "L_BT_weighted", "barlow_ratio", result["total_loss"])


def test_lejepa_weighted_and_ratio(lejepa_model, tiny_batch):
    obj = LeJEPAObjective(lambda_sigreg=0.1, num_slices=8, num_points=16, seed=0)
    result = obj(lejepa_model, *tiny_batch)
    c = result["components"]
    for k in ("L_SIGReg", "L_SIGReg_weighted", "sigreg_ratio"):
        assert k in c, f"missing component {k}"
    assert torch.allclose(c["L_SIGReg_weighted"],
                          c["lambda_sigreg"] * c["L_SIGReg"], atol=1e-6)
    _assert_ratio(c, "L_SIGReg_weighted", "sigreg_ratio", result["total_loss"])
