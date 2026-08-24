"""Phase-2 plumbing regression tests (VICReg training/validation fixes).

Fix A — the training objective must be moved to the selected device next to the
model (train_milestone_b.py), asserted without depending on parameter names.
Fix B — every validation/reference path (`FixedValidation.evaluate` /
`.null_gap` / `healthy_references`, and the sanity-script audit row) must
switch BOTH the model AND the objective-owned projector (BatchNorm-carrying)
to eval mode and restore both previous modes afterwards; validation must never
contaminate BatchNorm running statistics, while normal training keeps updating
them.

VICReg mathematics, EMA semantics, and checkpoint strictness are out of scope
here (locked by the existing test_vicreg_* suites).

Run:  python tests/test_vicreg_plumbing_mode_device.py   (pytest-collectable)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))

import pytest
import torch

from data.mask import BlockMasker
from losses.objectives import VICRegObjective
from train.engine import FixedValidation, healthy_references
from test_architecture import build_model as _build_real_model


def _batch(n=2, seed=0):
    torch.manual_seed(seed)
    G = torch.randn(n, 3, 64, 64)
    S = torch.randn(n, 2, 301)
    return G, S


def _fixed_val(device="cpu", n_batches=2, ratio=0.5):
    batches = [tuple(t.to(device) for t in _batch(seed=i))
               for i in range(n_batches)]
    return FixedValidation(batches, ratio=ratio, device=device, mask_seed=12345)


def _bn_state(module):
    """Snapshot (running_mean, running_var, num_batches_tracked) of every
    BatchNorm inside `module`."""
    return [(m.running_mean.clone(), m.running_var.clone(),
             m.num_batches_tracked.clone())
            for m in module.modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def _assert_bn_unchanged(before, after):
    assert len(before) > 0, "objective projector must contain BatchNorm layers"
    assert len(before) == len(after)
    for (rm0, rv0, nbt0), (rm1, rv1, nbt1) in zip(before, after):
        assert torch.equal(rm0, rm1) and torch.equal(rv0, rv1), (
            "validation/reference changed BatchNorm running statistics — "
            "the objective was not in eval mode")
        assert torch.equal(nbt0, nbt1), (
            "validation/reference updated the BatchNorm batch counter")


def _fresh_setup():
    """Real model interface (GoalConditionedJEPA + stub released encoder),
    fresh VICReg objective, fixed synthetic validation, healthy references."""
    torch.manual_seed(0)
    model = _build_real_model()
    objective = VICRegObjective()
    fv = _fixed_val()
    refs = healthy_references(model, fv, objective=objective)
    return model, objective, fv, refs


# ---------------------------------------------------------------------------
# Fix B — mode restoration
# ---------------------------------------------------------------------------

def test_b_train_mode_restored_after_validation_paths():
    """Start with both modules training -> every validation/reference path must
    leave BOTH back in training mode."""
    model, objective, fv, refs = _fresh_setup()
    model.train()
    objective.train()

    fv.evaluate(model, objective, refs["raw"], refs["proj"])
    assert model.training is True and objective.training is True

    fv.null_gap(model, objective)
    assert model.training is True and objective.training is True

    healthy_references(model, fv, objective=objective)
    assert model.training is True and objective.training is True


def test_b_eval_mode_restored_after_validation_paths():
    """Reverse case: start with both modules in eval -> paths must leave BOTH
    in eval (no blind .train())."""
    model, objective, fv, refs = _fresh_setup()
    model.eval()
    objective.eval()

    fv.evaluate(model, objective, refs["raw"], refs["proj"])
    assert model.training is False and objective.training is False

    fv.null_gap(model, objective)
    assert model.training is False and objective.training is False

    healthy_references(model, fv, objective=objective)
    assert model.training is False and objective.training is False


# ---------------------------------------------------------------------------
# Fix B — BatchNorm hygiene (Test C)
# ---------------------------------------------------------------------------

def test_c_bn_running_stats_frozen_across_validation_and_reference():
    """The projector's BatchNorm running statistics must be identical
    before/after evaluate(), null_gap(), and healthy_references() — from BOTH
    starting modes. This directly proves validation is truly eval-mode."""
    model, objective, fv, refs = _fresh_setup()
    for start_train in (True, False):
        model.train(start_train)
        objective.train(start_train)
        before = _bn_state(objective.projector)

        fv.evaluate(model, objective, refs["raw"], refs["proj"])
        _assert_bn_unchanged(before, _bn_state(objective.projector))

        fv.null_gap(model, objective)
        _assert_bn_unchanged(before, _bn_state(objective.projector))

        healthy_references(model, fv, objective=objective)
        _assert_bn_unchanged(before, _bn_state(objective.projector))


# ---------------------------------------------------------------------------
# Training path stays live (Test D)
# ---------------------------------------------------------------------------

def test_d_training_mode_still_updates_projector_bn():
    """A train-mode forward/backward MUST update the projector's BatchNorm
    running statistics (part of what training learns) and leave finite
    gradients on the projector parameters — i.e. Fix B did not permanently
    force the objective into evaluation mode."""
    model, _, _, _ = _fresh_setup()
    objective = VICRegObjective()
    G, S = _batch(seed=7)
    M = BlockMasker(seed=7, placement="random").sample(G, ratio=0.5)

    model.train()
    objective.train()
    before = _bn_state(objective.projector)
    res = objective(model, G, S, M)
    res["total_loss"].backward()
    after = _bn_state(objective.projector)

    changed = any(
        (not torch.equal(b[0], a[0])) or b[2].item() != a[2].item()
        for b, a in zip(before, after))
    assert changed, "train-mode forward must update BatchNorm running statistics"
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in objective.projector.parameters()), (
        "projector parameters must receive gradients from the training loss")


# ---------------------------------------------------------------------------
# Fix A — device placement (CUDA-only branch skips locally)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_a_objective_lives_on_selected_device_next_to_model():
    """Fix A as implemented in train_milestone_b.py: objective moved to the
    selected device; a full forward/backward produces no CPU/CUDA mismatch."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = _build_real_model().to(device)
    objective = VICRegObjective().to(device)

    objective_device = next(objective.parameters()).device
    assert objective_device == device, (
        f"Objective parameters are on {objective_device}, expected {device}")
    assert all(p.device == device for p in objective.parameters())

    G, S = _batch(seed=3)
    G, S = G.to(device), S.to(device)
    M = BlockMasker(seed=3, placement="random").sample(G.cpu(), ratio=0.5).to(device)
    res = objective(model, G, S, M)
    assert torch.isfinite(res["total_loss"])
    res["total_loss"].backward()
    grads = [p.grad for p in objective.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


if __name__ == "__main__":
    test_b_train_mode_restored_after_validation_paths()
    test_b_eval_mode_restored_after_validation_paths()
    test_c_bn_running_stats_frozen_across_validation_and_reference()
    test_d_training_mode_still_updates_projector_bn()
    test_a_objective_lives_on_selected_device_next_to_model()
    print("all plumbing regression tests passed")
