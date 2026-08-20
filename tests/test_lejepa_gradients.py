"""Gradient-boundary tests for the teacher-free LeJEPA objective (§22).

The critical contract: with no EMA/teacher copy, BOTH learnable branches
receive gradient — L_J is computed with stop_grad_target=False (the target is
the student geometry encoder's own output, attached to the graph), and L_SIGReg
pushes on the projected prediction AND target branches. The objective owns its
projector; there is no model.proj (§17); on_optimizer_step never touches an EMA.

Run:  python tests/test_lejepa_gradients.py   (pytest-collectable)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
import torch.nn as nn

from losses.objectives import LeJEPAObjective


class _CountingEma:
    def __init__(self):
        self.calls = []

    def update(self, encoder, step):
        self.calls.append((encoder, step))


class _StudentEncoder(nn.Module):
    def __init__(self, hidden=8, T=256, grid=64):
        super().__init__()
        self.hidden = hidden
        self.T = T
        self.linear = nn.Linear(3 * grid * grid, T * hidden)

    def forward(self, G):
        B = G.shape[0]
        return self.linear(G.flatten(1)).view(B, self.T, self.hidden)


class _Student(nn.Module):
    """LeJEPA stub: geometry_encoder is the LEARNABLE student (mirrors the real
    assembly where the student geometry encoder is trainable); z_hat is a
    learnable prediction parameter. Carries an EMA attribute ONLY to prove the
    objective never touches it (teacher-free)."""

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
        self.forward_kwargs.append(dict(with_target=with_target,
                                        goal_mode=goal_mode, need_attn=need_attn))
        return out


def _obj(**kw):
    defaults = dict(lambda_sigreg=0.1, num_slices=8, num_points=16, seed=0,
                    projector_input_dim=8, projector_hidden_dim=16,
                    projector_output_dim=8)
    defaults.update(kw)
    return LeJEPAObjective(**defaults)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return _Student(hidden=8)


@pytest.fixture
def tiny_batch():
    torch.manual_seed(1)
    G = torch.randn(2, 3, 64, 64)
    S = torch.randn(2, 301)
    M = torch.ones(2, 256)
    M[:, :2] = 0  # 2 masked tokens per sample -> N=4 features
    return G, S, M


def _grad_norm(p):
    return p.grad.abs().sum().item() if p.grad is not None else 0.0


def test_lj_reaches_student_encoder_no_stop_grad(model, tiny_batch):
    """stop_grad_target=False: backward from L_J alone must reach the student
    geometry encoder (the target path stays on the graph by design)."""
    G, S, M = tiny_batch
    obj = _obj()
    c = obj(model, G, S, M)["components"]
    model.zero_grad()
    obj.zero_grad()
    c["L_J"].backward()
    enc_grads = [p.grad for p in model.geometry_encoder.parameters()]
    assert all(g is not None for g in enc_grads), (
        "L_J must reach the student encoder (stop_grad_target=False)")
    assert all(torch.isfinite(g).all() for g in enc_grads)
    assert _grad_norm(model.z_hat) > 0, "L_J must reach the student prediction"
    assert model.ema.calls == [], "no EMA update may occur in the forward/backward"


def test_sigreg_reaches_projector_and_student(model, tiny_batch):
    G, S, M = tiny_batch
    obj = _obj()
    c = obj(model, G, S, M)["components"]
    model.zero_grad()
    obj.zero_grad()
    c["L_SIGReg"].backward()
    assert _grad_norm(obj.projector.net[0].weight) > 0, \
        "L_SIGReg must reach the objective projector"
    enc_grads = [p.grad for p in model.geometry_encoder.parameters()]
    assert all(g is not None for g in enc_grads), \
        "L_SIGReg must reach the student encoder (target is a student output)"
    assert _grad_norm(model.z_hat) > 0, "L_SIGReg must reach the prediction"


def test_sigreg_target_branch_pushes_student(model, tiny_batch):
    """The target-side SIGReg term must push the student encoder even when the
    prediction branch is detached (both branches get distributional pressure)."""
    G, S, M = tiny_batch
    obj = _obj()
    result = obj(model, G, S, M)
    p_y = result["projector_outputs"]["p_y"][result["out"]["mask"]]
    model.zero_grad()
    obj.zero_grad()
    from losses.sigreg import sigreg_loss
    term, _ = sigreg_loss(p_y, **obj.sigreg_kwargs)
    term.backward()
    enc_grads = [p.grad for p in model.geometry_encoder.parameters()]
    assert all(g is not None for g in enc_grads), \
        "target-side SIGReg must reach the student encoder"


def test_total_backward_finite_and_ema_untouched(model, tiny_batch):
    G, S, M = tiny_batch
    obj = _obj()
    result = obj(model, G, S, M)
    assert torch.isfinite(result["total_loss"])
    result["total_loss"].backward()
    for p in list(model.parameters()) + list(obj.parameters()):
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()
    assert model.ema.calls == [], "LeJEPA must never update an EMA"
    obj.on_optimizer_step(model, 5)
    assert model.ema.calls == [], "on_optimizer_step must be a no-op (no teacher)"


def test_no_model_proj_needed(model, tiny_batch):
    assert not hasattr(model, "proj"), "LeJEPA must not need a model-level proj (§17)"
    obj = _obj()
    result = obj(model, *tiny_batch)
    assert torch.isfinite(result["total_loss"])


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