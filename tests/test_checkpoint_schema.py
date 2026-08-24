"""Test checkpoint schema validation (hardening spec §11).

Validates required keys, schema version, and fails loudly on mismatch.
"""

import sys
import os
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
from train.engine import (
    save_checkpoint, load_checkpoint, CHECKPOINT_SCHEMA_VERSION, REQUIRED_CHECKPOINT_KEYS,
    _validate_checkpoint_schema, collect_ema_state,
)
from assembly import build_model
from losses.objectives import build_objective
from scripts.train.train_milestone_b import build_scheduler
from data.mask import BlockMasker
from runtime.device import resolve_device


def _make_tiny_setup(device):
    cfg = {
        "model": {"hidden": 64, "num_heads": 2, "geo_depth": 1, "predictor_depth": 1,
                  "goal_tokens": 4, "num_predictor_heads": 2,
                  "ema_momentum_start": 0.99, "ema_momentum_end": 0.999},
        "weights": {"spectrum": os.path.join(REPO_ROOT, "data/metadit/weights/spec_encoder.pth"),
                    "metadit": os.path.join(REPO_ROOT, "data/metadit/weights/metadit-small.bin")},
    }
    model = build_model(cfg["model"], cfg["weights"]["spectrum"], device=device, init_from_metadit=False)
    objective = build_objective("jepa_vicreg", {}, projector_input_dim=64).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad] + \
                [p for p in objective.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    total_steps = 2
    scheduler = build_scheduler(optimizer, 1e-3, 0, total_steps)
    model.ema.set_total_steps(total_steps)
    masker = BlockMasker(placement="random", grid=16, min_side=3, k_range=(1, 2), seed=12345)
    return model, objective, optimizer, scheduler, masker, cfg


def test_checkpoint_schema_validation():
    """Checkpoint missing required keys fails loudly."""
    device = resolve_device("cpu")
    model, objective, optimizer, scheduler, masker, cfg = _make_tiny_setup(device)

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "bad_ckpt.pt")
        # Save incomplete checkpoint
        torch.save({"objective_name": "jepa_vicreg", "step": 0}, ckpt_path)

        model2, objective2, optimizer2, scheduler2, _, _ = _make_tiny_setup(device)
        with pytest.raises(RuntimeError, match="missing required keys"):
            load_checkpoint(ckpt_path, model2, objective2, optimizer2, scheduler2, device)


def test_checkpoint_schema_version_mismatch():
    """Wrong schema version fails loudly."""
    device = resolve_device("cpu")
    model, objective, optimizer, scheduler, masker, cfg = _make_tiny_setup(device)

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "bad_version.pt")
        torch.save({
            "schema_version": 999,
            "objective_name": "jepa_vicreg",
            "step": 0,
            "epoch": 0,
            "micro_step": 0,
            "is_epoch_end": True,
            "cfg": {},
            "best_prediction": {},
            "best_healthy_prediction": {},
            "model": model.state_dict(),
            "objective_state": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_param_shapes": [[tuple(p.shape) for p in g["params"]] for g in optimizer.param_groups],
            "scheduler_state": scheduler.state_dict(),
            "ema_state": collect_ema_state(model),
            "rng_state": {},
            "masker_rng_state": masker.get_rng_state(),
            "git_commit": "test",
            "git_dirty": False,
            "env_versions": {},
            "device_info": {"device_type": "cpu"},
            "artifact_type": "full",
        }, ckpt_path)

        model2, objective2, optimizer2, scheduler2, _, _ = _make_tiny_setup(device)
        with pytest.raises(RuntimeError, match="schema version"):
            load_checkpoint(ckpt_path, model2, objective2, optimizer2, scheduler2, device)


def test_checkpoint_valid_schema_passes():
    """Valid checkpoint passes schema validation."""
    device = resolve_device("cpu")
    model, objective, optimizer, scheduler, masker, cfg = _make_tiny_setup(device)

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "good_ckpt.pt")
        save_checkpoint(
            ckpt_path, model, objective, optimizer, scheduler, cfg,
            global_step=1, epoch=0, micro_step=0, is_epoch_end=True,
            metrics={}, health=None,
            ema_state=collect_ema_state(model),
            best_prediction={}, best_healthy_prediction={},
            masker_rng_state=masker.get_rng_state(),
            device=device, artifact_type="full",
        )

        model2, objective2, optimizer2, scheduler2, _, _ = _make_tiny_setup(device)
        obj = load_checkpoint(ckpt_path, model2, objective2, optimizer2, scheduler2, device)
        assert obj["schema_version"] == CHECKPOINT_SCHEMA_VERSION


def test_checkpoint_required_keys_present():
    """All REQUIRED_CHECKPOINT_KEYS are present in saved checkpoint."""
    device = resolve_device("cpu")
    model, objective, optimizer, scheduler, masker, cfg = _make_tiny_setup(device)

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "ckpt.pt")
        save_checkpoint(
            ckpt_path, model, objective, optimizer, scheduler, cfg,
            global_step=1, epoch=0, micro_step=0, is_epoch_end=True,
            metrics={}, health=None,
            ema_state=collect_ema_state(model),
            best_prediction={}, best_healthy_prediction={},
            masker_rng_state=masker.get_rng_state(),
            device=device, artifact_type="full",
        )

        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for key in REQUIRED_CHECKPOINT_KEYS:
            assert key in obj, f"Missing required key: {key}"


if __name__ == "__main__":
    test_checkpoint_schema_validation()
    test_checkpoint_schema_version_mismatch()
    test_checkpoint_valid_schema_passes()
    test_checkpoint_required_keys_present()
    print("All checkpoint schema tests passed")