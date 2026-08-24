"""Test best checkpoint schema (hardening spec §12).

Verifies best_prediction, best_healthy_prediction, latest_checkpoint, final_checkpoint
are kept distinct.
"""

import sys
import os
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
from train.engine import save_checkpoint, collect_ema_state, load_checkpoint
from assembly import build_model
from losses.objectives import build_objective
from scripts.train.train_milestone_b import build_scheduler
from data.mask import BlockMasker
from runtime.device import resolve_device


def test_best_checkpoint_schema():
    device = resolve_device("cpu")
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

    with tempfile.TemporaryDirectory() as td:
        # Save best_prediction
        best_path = os.path.join(td, "best.pt")
        save_checkpoint(
            best_path, model, objective, optimizer, scheduler, cfg,
            global_step=5, epoch=1, micro_step=0, is_epoch_end=True,
            metrics={"cos_err_r0.5": 0.1}, health={"status": "HEALTHY"},
            ema_state=collect_ema_state(model),
            best_prediction={"primary": 0.1, "metrics": {"cos_err_r0.5": 0.1}, "step": 5, "health": {"status": "HEALTHY"}},
            best_healthy_prediction={"primary": 0.15, "metrics": {"cos_err_r0.5": 0.15}, "step": 3, "health": {"status": "HEALTHY"}},
            masker_rng_state=masker.get_rng_state(),
            device=device, artifact_type="best",
        )

        obj = torch.load(best_path, map_location="cpu", weights_only=False)
        assert obj["artifact_type"] == "best"
        assert "best_prediction" in obj
        assert "best_healthy_prediction" in obj
        assert obj["best_prediction"]["primary"] == 0.1
        assert obj["best_healthy_prediction"]["primary"] == 0.15

        # Save latest
        latest_path = os.path.join(td, "latest.pt")
        save_checkpoint(
            latest_path, model, objective, optimizer, scheduler, cfg,
            global_step=10, epoch=2, micro_step=0, is_epoch_end=True,
            metrics={"cos_err_r0.5": 0.12}, health={"status": "HEALTHY"},
            ema_state=collect_ema_state(model),
            best_prediction={"primary": 0.1, "metrics": {"cos_err_r0.5": 0.1}, "step": 5, "health": {"status": "HEALTHY"}},
            best_healthy_prediction={"primary": 0.15, "metrics": {"cos_err_r0.5": 0.15}, "step": 3, "health": {"status": "HEALTHY"}},
            masker_rng_state=masker.get_rng_state(),
            device=device, artifact_type="latest",
        )

        obj = torch.load(latest_path, map_location="cpu", weights_only=False)
        assert obj["artifact_type"] == "latest"
        assert obj["step"] == 10

        # Save final
        final_path = os.path.join(td, "final.pt")
        save_checkpoint(
            final_path, model, objective, optimizer, scheduler, cfg,
            global_step=10, epoch=2, micro_step=0, is_epoch_end=True,
            metrics={"cos_err_r0.5": 0.12}, health={"status": "HEALTHY"},
            ema_state=collect_ema_state(model),
            best_prediction={"primary": 0.1, "metrics": {"cos_err_r0.5": 0.1}, "step": 5, "health": {"status": "HEALTHY"}},
            best_healthy_prediction={"primary": 0.15, "metrics": {"cos_err_r0.5": 0.15}, "step": 3, "health": {"status": "HEALTHY"}},
            masker_rng_state=masker.get_rng_state(),
            device=device, artifact_type="final",
        )

        obj = torch.load(final_path, map_location="cpu", weights_only=False)
        assert obj["artifact_type"] == "final"
        assert obj["step"] == 10

        # Verify they are all distinct full checkpoints (loadable)
        for path, art_type in [(best_path, "best"), (latest_path, "latest"), (final_path, "final")]:
            model2 = build_model(cfg["model"], cfg["weights"]["spectrum"], device=device, init_from_metadit=False)
            objective2 = build_objective("jepa_vicreg", {}, projector_input_dim=64).to(device)
            trainable2 = [p for p in model2.parameters() if p.requires_grad] + \
                         [p for p in objective2.parameters() if p.requires_grad]
            optimizer2 = torch.optim.AdamW(trainable2, lr=1e-3)
            scheduler2 = build_scheduler(optimizer2, 1e-3, 0, total_steps)
            model2.ema.set_total_steps(total_steps)
            load_checkpoint(path, model2, objective2, optimizer2, scheduler2, device)
            # Should load without error

    print("PASS: test_best_checkpoint_schema")


if __name__ == "__main__":
    test_best_checkpoint_schema()