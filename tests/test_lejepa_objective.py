"""LeJEPA objective smoke tests (architecture-repair spec §22):

Verify the exact LeJEPA variant contract:
- the student geometry encoder supplies z_y (forward runs with_target=False; no
  EMA target appears anywhere in the objective's forward path)
- jepa_loss is called with stop_grad_target=False (target side stays on the
  gradient graph — the regularizer, not a teacher copy, prevents collapse)
- on_optimizer_step performs NO EMA update (no teacher copy in this variant)
- the objective OWNS its projector (LeJEPAProjector); there is no model.proj (§17)
- L = L_J + lambda_sigreg * L_SIGReg, exactly
- finite loss and finite gradients everywhere
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
import torch.nn as nn

from losses.jepa_loss import jepa_loss
from losses.objective_modules import LeJEPAProjector
from losses.objectives import OBJECTIVES, LeJEPAObjective


class _CountingEma:
    def __init__(self):
        self.calls = []

    def update(self, encoder, step):
        self.calls.append((encoder, step))


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
    """Mimics GoalConditionedJEPA's LeJEPA contract: forward(with_target=False)
    returns NO z_y (exactly like the real assembly when the EMA path is skipped),
    and z_y must come from the student geometry_encoder, which the objective
    calls itself. geometry_encoder is a real learnable module so gradient flow
    into the student is observable. The model has NO `.proj` attribute."""

    def __init__(self, hidden=8, B=2, T=256, grid=64):
        super().__init__()
        self.geometry_encoder = _StudentEncoder(hidden=hidden, T=T, grid=grid)
        self.ema = _CountingEma()
        self.z_hat = nn.Parameter(torch.randn(B, T, hidden))
        self.forward_kwargs = []

    def forward(self, G, S, M, goal_mode="real", need_attn=False, with_target=True):
        B = G.shape[0]
        mask = (M.view(B, -1) == 0)
        out = {"z_hat": self.z_hat, "mask": mask}
        if with_target:
            out["z_y"] = self.geometry_encoder(G)
        self.forward_kwargs.append(dict(with_target=with_target,
                                        goal_mode=goal_mode, need_attn=need_attn))
        return out


def _obj(**kwargs):
    defaults = dict(projector_input_dim=8, projector_hidden_dim=16,
                    projector_output_dim=8)
    defaults.update(kwargs)
    return LeJEPAObjective(**defaults)


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
    M[:, :2] = 0  # 2 masked tokens per sample -> N=4 features for SIGReg
    return G, S, M


def test_lejepa_registered():
    assert "lejepa" in OBJECTIVES
    assert OBJECTIVES["lejepa"] is LeJEPAObjective


def test_lejepa_owns_projector_no_model_proj(lejepa_model, tiny_batch):
    obj = _obj()
    assert isinstance(obj.projector, LeJEPAProjector)
    assert not hasattr(lejepa_model, "proj"), (
        "LeJEPA must not need a model-level proj head (§17)")
    result = obj(lejepa_model, *tiny_batch)
    assert torch.isfinite(result["total_loss"])


def test_lejepa_student_encoder_supplies_z_y(lejepa_model, tiny_batch):
    """z_y must come from model.geometry_encoder(G) — the student itself — and
    the model forward must be called with with_target=False (no EMA target)."""
    G, S, M = tiny_batch
    obj = _obj()
    result = obj(lejepa_model, G, S, M)

    assert lejepa_model.forward_kwargs == [dict(with_target=False,
                                                goal_mode="real",
                                                need_attn=False)], (
        "LeJEPA forward must skip the EMA target path")
    expected_z_y = lejepa_model.geometry_encoder(G)
    assert torch.allclose(result["out"]["z_y"], expected_z_y), (
        "objective's z_y must be the student geometry encoder's output")
    assert lejepa_model.ema.calls == [], (
        "no EMA update may occur anywhere in the LeJEPA forward path")


def test_lejepa_no_ema_update(lejepa_model, tiny_batch):
    obj = _obj()
    obj(lejepa_model, *tiny_batch)
    obj.on_optimizer_step(lejepa_model, 7)
    assert lejepa_model.ema.calls == [], (
        "LeJEPA has no teacher copy: on_optimizer_step must never touch the EMA")


def test_lejepa_stop_grad_target_false(lejepa_model, tiny_batch):
    """L_J inside the objective must be computed with stop_grad_target=False:
    the target (student z_y through the objective projector) stays attached to
    the graph, so backward flows into the student encoder — the SIGReg design
    contract. The reference L_J is raw-space (proj=None), matching the
    objective's own jepa_loss call."""
    G, S, M = tiny_batch
    obj = _obj()
    result = obj(lejepa_model, G, S, M)
    mask = result["out"]["mask"]

    # Reference L_J with stop_grad_target=False on the same inputs, raw space.
    z_y = lejepa_model.geometry_encoder(G)
    L_ref, _ = jepa_loss(lejepa_model.z_hat, z_y, mask, proj=None,
                         stop_grad_target=False)
    assert torch.allclose(result["components"]["L_J"], L_ref, atol=1e-6), (
        "L_J inside LeJEPA must equal a stop_grad_target=False raw-space reference")

    # Backward must reach the student target path (geometry_encoder weights).
    lejepa_model.zero_grad()
    result["total_loss"].backward()
    enc_grads = [p.grad for p in lejepa_model.geometry_encoder.parameters()]
    assert all(g is not None for g in enc_grads), (
        "stop_grad_target=False must attach the student encoder to the graph")
    assert all(torch.isfinite(g).all() for g in enc_grads)


def test_lejepa_loss_composition(lejepa_model, tiny_batch):
    """L = L_J + lambda_sigreg * L_SIGReg, exactly, with the weighted regularizer
    and sigreg_ratio components reported."""
    obj = _obj(lambda_sigreg=0.1, num_slices=8, num_points=16, seed=0)
    result = obj(lejepa_model, *tiny_batch)
    c = result["components"]
    for k in ("L_J", "L_SIGReg", "L_SIGReg_weighted", "sigreg_ratio",
              "lambda_sigreg", "sigreg_info"):
        assert k in c, f"missing component {k}"
    assert torch.allclose(
        result["total_loss"],
        c["L_J"] + c["lambda_sigreg"] * c["L_SIGReg"], atol=1e-6)
    assert torch.allclose(c["L_SIGReg_weighted"],
                          c["lambda_sigreg"] * c["L_SIGReg"], atol=1e-6)
    assert torch.allclose(c["sigreg_ratio"],
                          c["L_SIGReg_weighted"] / result["total_loss"],
                          atol=1e-4)
    assert c["sigreg_info"]["pred"]["test"].startswith("sliced ECF"), c["sigreg_info"]


def test_lejepa_total_backward_finite(lejepa_model, tiny_batch):
    obj = _obj()
    result = obj(lejepa_model, *tiny_batch)
    assert torch.isfinite(result["total_loss"])
    result["total_loss"].backward()
    for p in lejepa_model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()
    for p in obj.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), (
            "LeJEPA projector must receive gradient from the total loss")
    assert torch.isfinite(result["components"]["L_J"])
    assert torch.isfinite(result["components"]["L_SIGReg"])


def test_lejepa_sigreg_metadata_complete(lejepa_model, tiny_batch):
    """FIX E: sigreg_info must report num_slices, num_points, t_grid, seed, and
    mean_phi_dev — the runnable metadata of the SIGReg test, recorded per phase
    (dict preserved through the report schema, NOT flattened into scalar-only
    loss components). Both branches are reported: pred and target."""
    import losses.sigreg as sigreg_mod

    obj = _obj(lambda_sigreg=0.1, num_slices=8, num_points=16, seed=3)
    result = obj(lejepa_model, *tiny_batch)
    sigreg_info = result["components"]["sigreg_info"]
    assert isinstance(sigreg_info, dict), "sigreg_info must stay a dict in components"
    assert set(sigreg_info) == {"pred", "target"}, \
        "sigreg_info must carry per-branch metadata for both learnable branches"
    for branch in ("pred", "target"):
        info = sigreg_info[branch]
        for k in ("num_slices", "num_points", "t_grid", "seed", "mean_phi_dev"):
            assert k in info, f"sigreg_info[{branch}] missing {k}"
        assert info["num_slices"] == 8
        # sigreg reports the ACTUAL number of points used: min(masked features, requested)
        assert info["num_points"] == 4  # tiny_batch has 2 masked tokens x 2 samples
        assert len(info["t_grid"]) == len(sigreg_mod.DEFAULT_T_GRID)
        assert all(isinstance(t, float) for t in info["t_grid"])
        assert info["seed"] == 3
        assert isinstance(info["mean_phi_dev"], float) and info["mean_phi_dev"] >= 0.0