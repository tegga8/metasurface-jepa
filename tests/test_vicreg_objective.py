"""Objective-level tests for the faithful EMA-JEPA + VICReg-style candidate
(`jepa_vicreg`), per the Milestone B CODEX spec.

Covers:
  - registry placement (six-rung ladder, `jepa_vicreg` -> VICRegObjective)
  - objective-owned projector; the refactored model has NO `model.proj` and
    must not need one
  - component/weighting contract (weighted terms sum to total, ratios to 1)
  - token-level statistics ARE the loss; geometry-level pooled statistics are
    health components only (reported, never added to the loss)
  - N >= 2 hard guard (no silent zero-substitution of undefined statistics)
  - EMA update ownership: on_optimizer_step -> model.ema.update(...)
  - projector output shapes on the (B, T, D) path
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
import torch.nn as nn

from losses.objective_modules import VICRegProjector
from losses.objectives import OBJECTIVES, VICRegObjective
from losses.vicreg import (
    covariance_loss,
    invariance_loss,
    variance_loss,
    vicreg_branch_terms,
)


class _EmaStub:
    def update(self, encoder, step):
        pass


class _RecordingEma:
    def __init__(self, seen):
        self.seen = seen

    def update(self, encoder, step):
        self.seen.append((encoder, step))


class _NoProjModel(nn.Module):
    """The corrected architecture has NO `model.proj` (that was a leaked
    historical design the refactor removed); the objective must work without
    touching any `.proj` attribute on the model."""

    def __init__(self, hidden=8, B=2, T=256):
        super().__init__()
        self.ema = _EmaStub()
        self.geometry_encoder = None
        self.z_hat = nn.Parameter(torch.randn(B, T, hidden))
        self.z_y = nn.Parameter(torch.randn(B, T, hidden))

    def forward(self, G, S, M):
        B = G.shape[0]
        mask = (M.view(B, -1) == 0)
        return {"z_hat": self.z_hat, "z_y": self.z_y, "mask": mask}


@pytest.fixture
def no_proj_model():
    torch.manual_seed(0)
    return _NoProjModel(hidden=8)


@pytest.fixture
def tiny_batch():
    torch.manual_seed(1)
    G = torch.randn(2, 3, 64, 64)
    S = torch.randn(2, 301)
    M = torch.ones(2, 256)
    M[:, :4] = 0  # 8 masked tokens across the batch -> N=8 >= 2
    return G, S, M


def test_jepa_vicreg_registered_as_vicreg_objective():
    assert "jepa_vicreg" in OBJECTIVES
    assert OBJECTIVES["jepa_vicreg"] is VICRegObjective


def test_six_rung_registry_stable():
    assert set(OBJECTIVES.keys()) == {
        "jepa", "jepa_var", "jepa_vicreg", "jepa_vicreg2",
        "jepa_barlow", "lejepa"}


def test_objective_owns_projector_no_model_proj_needed(no_proj_model, tiny_batch):
    obj = VICRegObjective(
        projector_input_dim=8, projector_hidden_dim=16, projector_output_dim=8)
    assert isinstance(obj.projector, VICRegProjector)
    assert not hasattr(no_proj_model, "proj"), (
        "refactored model must not carry a leaked `proj` head")
    result = obj(no_proj_model, *tiny_batch)
    assert torch.isfinite(result["total_loss"])


def test_components_present_and_finite(no_proj_model, tiny_batch):
    obj = VICRegObjective(
        projector_input_dim=8, projector_hidden_dim=16, projector_output_dim=8)
    c = obj(no_proj_model, *tiny_batch)["components"]
    for k in ("L_inv", "L_var", "L_cov",
              "L_inv_weighted", "L_var_weighted", "L_cov_weighted",
              "inv_ratio", "var_ratio", "cov_ratio",
              "lambda_inv", "lambda_var", "lambda_cov",
              "geo_inv", "geo_var", "geo_cov"):
        assert k in c, f"missing component {k}"
        assert torch.isfinite(torch.as_tensor(c[k])), f"non-finite {k}"
    assert c["lambda_inv"] == 25.0 and c["lambda_var"] == 25.0 \
        and c["lambda_cov"] == 1.0


def test_weighted_terms_sum_to_total(no_proj_model, tiny_batch):
    obj = VICRegObjective(
        lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0,
        projector_input_dim=8, projector_hidden_dim=16, projector_output_dim=8)
    result = obj(no_proj_model, *tiny_batch)
    c = result["components"]
    assert torch.allclose(
        result["total_loss"],
        c["L_inv_weighted"] + c["L_var_weighted"] + c["L_cov_weighted"],
        atol=1e-6)
    assert abs(c["inv_ratio"] + c["var_ratio"] + c["cov_ratio"] - 1.0) < 1e-4


def test_custom_lambdas_change_weighted_terms(no_proj_model, tiny_batch):
    obj = VICRegObjective(
        lambda_inv=1.0, lambda_var=2.0, lambda_cov=3.0,
        projector_input_dim=8, projector_hidden_dim=16, projector_output_dim=8)
    result = obj(no_proj_model, *tiny_batch)
    c = result["components"]
    assert torch.allclose(c["L_inv_weighted"], 1.0 * c["L_inv"])
    assert torch.allclose(c["L_var_weighted"], 2.0 * c["L_var"])
    assert torch.allclose(c["L_cov_weighted"], 3.0 * c["L_cov"])


def test_projector_output_shapes_on_btd_path(no_proj_model, tiny_batch):
    obj = VICRegObjective(
        projector_input_dim=8, projector_hidden_dim=16, projector_output_dim=8)
    result = obj(no_proj_model, *tiny_batch)
    p_hat = result["projector_outputs"]["p_hat"]
    p_y = result["projector_outputs"]["p_y"]
    assert p_hat.shape == (2, 256, 8)
    assert p_y.shape == (2, 256, 8)
    assert result["projector_inputs"]["z_hat"].shape == (2, 256, 8)


def test_token_level_stats_are_the_loss_geometry_level_is_health_only(
        no_proj_model, tiny_batch):
    """Token-level statistics define L_inv/L_var/L_cov. The geometry-level
    pooled statistics (geo_inv/geo_var/geo_cov) must be reported but MUST NOT
    appear in the weighted total."""
    obj = VICRegObjective(
        projector_input_dim=8, projector_hidden_dim=16, projector_output_dim=8)
    result = obj(no_proj_model, *tiny_batch)
    c = result["components"]
    total_terms = (c["L_inv_weighted"] + c["L_var_weighted"] + c["L_cov_weighted"])
    token_L = c["lambda_inv"] * c["L_inv"] + c["lambda_var"] * c["L_var"] \
        + c["lambda_cov"] * c["L_cov"]
    assert torch.allclose(total_terms, token_L, atol=1e-5), (
        "total must be built exclusively from token-level terms")
    assert not torch.allclose(
        c["geo_inv"] + c["geo_var"] + c["geo_cov"],
        torch.zeros_like(c["geo_inv"]), atol=1e-9), (
        "geo-level health stats must actually be computed")


def test_n_below_two_raises_not_silently_zeroed(tiny_batch):
    """Spec numerical rules: undefined variance/covariance statistics raise,
    they are never silently replaced with zero during real training."""
    G, S, M = tiny_batch
    M = torch.ones_like(M)
    M[0, 0] = 0  # exactly ONE masked token across the batch -> N = 1

    obj = VICRegObjective(
        projector_input_dim=8, projector_hidden_dim=16, projector_output_dim=8)
    model = _NoProjModel(hidden=8)
    with pytest.raises(ValueError, match="at least two valid samples"):
        obj(model, G, S, M)

    with pytest.raises(ValueError, match="at least two valid samples"):
        invariance_loss(torch.randn(1, 8), torch.randn(1, 8))
    with pytest.raises(ValueError, match="at least two valid samples"):
        variance_loss(torch.randn(1, 8))
    with pytest.raises(ValueError, match="at least two valid samples"):
        covariance_loss(torch.randn(1, 8))
    with pytest.raises(ValueError, match="at least two valid samples"):
        vicreg_branch_terms(torch.randn(1, 8), torch.randn(1, 8))


def test_on_optimizer_step_updates_ema(no_proj_model):
    seen = []
    no_proj_model.ema = _RecordingEma(seen)
    obj = VICRegObjective(
        projector_input_dim=8, projector_hidden_dim=16, projector_output_dim=8)
    obj.on_optimizer_step(no_proj_model, 3)
    assert seen == [(None, 3)], (
        "EMA must update only via objective.on_optimizer_step")


def test_forward_backward_full_path_finite(no_proj_model, tiny_batch):
    obj = VICRegObjective(
        projector_input_dim=8, projector_hidden_dim=16, projector_output_dim=8)
    result = obj(no_proj_model, *tiny_batch)
    result["total_loss"].backward()
    for p in no_proj_model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()
    for p in obj.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), (
            "projector must always receive gradient from the total loss")
