#!/usr/bin/env python3
"""Checkpoint Integrity Check (Phase 2 §6).

Given a checkpoint, validates:
- schema validation
- reconstruct matching model/objective
- restore it
- run validation forward
- verify finite outputs/loss
- verify objective state
- verify EMA state
- verify optimizer ownership metadata
- verify Git/environment metadata

Exit non-zero on any failure.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_DIR))

from assembly import build_model, load_into_model, saveable_state_dict
from data.dataset import MetaDiTDataset
from data.mask import BlockMasker
from losses.objectives import build_objective
from runtime.device import resolve_device
from train.engine import (
    load_checkpoint, collect_ema_state, restore_ema_state,
    fixed_validation_from_loader, build_deterministic_reference, healthy_references,
    CHECKPOINT_SCHEMA_VERSION, REQUIRED_CHECKPOINT_KEYS, _validate_checkpoint_schema,
)
from scripts.train.train_milestone_b import build_scheduler
from torch.utils.data import DataLoader


class IntegrityError(Exception):
    """Checkpoint integrity failure."""
    pass


def log_step(step: str):
    print(f"\n{'='*60}")
    print(f"INTEGRITY: {step}")
    print(f"{'='*60}")


def log_info(msg: str):
    print(f"  [INFO] {msg}")


def log_pass(msg: str):
    print(f"  [PASS] {msg}")


def log_fail(msg: str):
    print(f"  [FAIL] {msg}")


def main():
    parser = argparse.ArgumentParser(description="Checkpoint Integrity Check")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--config", default="configs/milestone_b.yaml", help="Config file path")
    parser.add_argument("--data-root", default=None, help="Dataset root override")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda, cuda:N)")
    parser.add_argument(
        "--allow-commit-mismatch",
        action="store_true",
        help="Allow checkpoint git commit mismatch (for forensic inspection only)"
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[FAIL] Checkpoint not found: {ckpt_path}")
        return 1

    print(f"Checking checkpoint: {ckpt_path}")

    try:
        # 1. Load and validate schema
        log_step("Schema Validation")
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        _validate_checkpoint_schema(obj, str(ckpt_path))
        log_pass(f"Schema version {obj['schema_version']} validated")

        # 2. Verify required keys present
        log_step("Required Keys Check")
        for k in REQUIRED_CHECKPOINT_KEYS:
            if k not in obj:
                raise IntegrityError(f"Missing required key: {k}")
        log_pass(f"All {len(REQUIRED_CHECKPOINT_KEYS)} required keys present")

        # 3. Print provenance
        log_step("Provenance")
        print(f"  Objective: {obj['objective_name']}")
        print(f"  Step: {obj['step']}")
        print(f"  Epoch: {obj['epoch']}")
        print(f"  Micro-step: {obj.get('micro_step', 'N/A')}")
        print(f"  Is epoch end: {obj.get('is_epoch_end', 'N/A')}")
        print(f"  Git commit: {obj.get('git_commit', 'unknown')}")
        print(f"  Git dirty: {obj.get('git_dirty', 'unknown')}")
        print(f"  Artifact type: {obj.get('artifact_type', 'unknown')}")
        env = obj.get('env_versions', {})
        print(f"  Python: {env.get('python', 'unknown')}")
        print(f"  PyTorch: {env.get('torch', 'unknown')}")
        print(f"  Torchvision: {env.get('torchvision', 'unknown')}")
        print(f"  CUDA: {env.get('cuda', 'unknown')}")
        print(f"  GPU: {env.get('gpu', 'unknown')}")
        dev = obj.get('device_info', {})
        print(f"  Device: {dev.get('device_type', 'unknown')} {dev.get('device_index', '')}")
        log_pass("Provenance recorded")

        # 4. Check git commit match (hard failure per Bug #13)
        log_step("Git Commit Check")
        try:
            import subprocess
            current_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, cwd=REPO_ROOT
            ).strip()
            saved_commit = obj.get('git_commit', '')
            if saved_commit and saved_commit != current_commit:
                if args.allow_commit_mismatch:
                    print(f"  [WARN] Checkpoint commit ({saved_commit[:8]}) != current commit ({current_commit[:8]})")
                    print("  Continuing with --allow-commit-mismatch (forensic inspection only).")
                else:
                    raise IntegrityError(
                        f"Checkpoint git commit {saved_commit} does not match "
                        f"current repository commit {current_commit}. "
                        "Use --allow-commit-mismatch only for deliberate forensic inspection."
                    )
            else:
                log_pass(f"Git commit matches: {current_commit[:8]}")
        except Exception as e:
            raise IntegrityError(f"Could not verify git commit: {e}")

        # 5. Reconstruct model/objective from config
        log_step("Model + Objective Reconstruction")

        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        device = resolve_device(args.device)
        log_info(f"Device: {device}")

        # Data root
        if args.data_root:
            data_root = Path(args.data_root)
        else:
            # Auto-discover
            search_paths = [Path("/kaggle/input"), REPO_ROOT / "data" / "metadit"]
            data_root = None
            for base in search_paths:
                if base.exists():
                    for candidate in base.rglob("*"):
                        if candidate.is_dir():
                            required = [
                                "split_data/train_set.mat", "split_data/val_set.mat",
                                "weights/metadit-small.bin", "weights/spec_encoder.pth",
                                "weights/surrogate_model.bin",
                            ]
                            if all((candidate / r).exists() for r in required):
                                data_root = candidate
                                break
                if data_root:
                    break
            if not data_root:
                raise IntegrityError("Could not auto-discover dataset. Use --data-root.")

        log_info(f"Data root: {data_root}")

        # Build model
        model = build_model(
            cfg["model"],
            str(data_root / "weights" / "spec_encoder.pth"),
            device=device,
            init_from_metadit=True,
            metadit_weights=str(data_root / "weights" / "metadit-small.bin"),
        )
        model.eval()

        # Build objective
        objective_name = obj["objective_name"]
        objective = build_objective(
            objective_name, cfg.get("objective_params", {}).get(objective_name, {}),
            projector_input_dim=cfg["model"].get("hidden", 384),
        ).to(device)
        objective.eval()

        # 6. Load checkpoint into model/objective
        log_step("Checkpoint Load")
        # Need optimizer/scheduler for load_checkpoint
        trainable = [p for p in model.parameters() if p.requires_grad] + \
                    [p for p in objective.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
        scheduler = build_scheduler(optimizer, 1e-3, 0, obj.get('step', 0) + 10)
        model.ema.set_total_steps(obj.get('step', 0) + 10)

        loaded = load_checkpoint(str(ckpt_path), model, objective, optimizer, scheduler, device)
        restore_ema_state(model, loaded.get("ema_state"))
        log_pass("Checkpoint loaded into model/objective/optimizer/scheduler/EMA")

        # 7. Verify model state restored (exact tensor comparison per Bug #14)
        log_step("Model State Verification")
        saved_model_state = obj["model"]
        for name, saved_tensor in saved_model_state.items():
            if name not in model.state_dict():
                raise IntegrityError(f"Checkpoint model key {name!r} missing after reconstruction")
            loaded_tensor = model.state_dict()[name]
            if not torch.equal(
                loaded_tensor.detach().cpu(),
                saved_tensor.detach().cpu(),
            ):
                raise IntegrityError(f"Model tensor mismatch after checkpoint load: {name}")
        log_pass("Model tensors exactly match checkpoint")

        # 8. Verify objective state restored
        log_step("Objective State Verification")
        saved_obj_state = obj["objective_state"]
        for name, saved_tensor in saved_obj_state.items():
            if name not in objective.state_dict():
                raise IntegrityError(f"Checkpoint objective key {name!r} missing after reconstruction")
            loaded_tensor = objective.state_dict()[name]
            if not torch.equal(
                loaded_tensor.detach().cpu(),
                saved_tensor.detach().cpu(),
            ):
                raise IntegrityError(f"Objective tensor mismatch after checkpoint load: {name}")
        log_pass("Objective tensors exactly match checkpoint")

        # 9. Verify optimizer ownership metadata
        log_step("Optimizer Ownership Metadata")
        saved_shapes = obj.get("optimizer_param_shapes")
        if saved_shapes is None:
            raise IntegrityError("Checkpoint missing optimizer_param_shapes")
        cur_shapes = [[tuple(p.shape) for p in g["params"]] for g in optimizer.param_groups]
        if cur_shapes != saved_shapes:
            raise IntegrityError(
                f"Optimizer param shapes mismatch!\n"
                f"  Saved: {saved_shapes}\n"
                f"  Current: {cur_shapes}"
            )
        log_pass("Optimizer param shapes match checkpoint")

        # 10. Verify EMA state
        log_step("EMA State Verification")
        ema_state = obj.get("ema_state", {})
        if ema_state:
            if "total_steps" in ema_state:
                if model.ema.total_steps != ema_state["total_steps"]:
                    raise IntegrityError(
                        f"EMA total_steps mismatch: model={model.ema.total_steps}, "
                        f"checkpoint={ema_state['total_steps']}"
                    )
            log_pass(f"EMA total_steps: {model.ema.total_steps}")
        else:
            print("  [WARN] No EMA state in checkpoint")

        # 11. Run validation forward
        log_step("Validation Forward Pass")
        val_path = data_root / "split_data" / "val_set.mat"
        val_ds = MetaDiTDataset(str(val_path))

        fv = fixed_validation_from_loader(
            val_ds, n_samples=4, batch_size=2, device=device, ratio=0.5, mask_seed=12345
        )

        # Build reference
        ref_model = build_deterministic_reference(
            lambda: build_model(
                cfg["model"],
                str(data_root / "weights" / "spec_encoder.pth"),
                device=device,
                init_from_metadit=True,
                metadit_weights=str(data_root / "weights" / "metadit-small.bin"),
            )
        )
        refs = healthy_references(ref_model, fv, objective=objective)

        model.eval()
        objective.eval()
        metrics, health = fv.evaluate(model, objective, refs["raw"], refs["proj"])

        log_info(f"Validation metrics: {metrics}")
        log_info(f"Health status: {health['status']}")

        # Verify finite outputs
        for k, v in metrics.items():
            if not isinstance(v, float) or not (v == v and abs(v) < float('inf')):
                raise IntegrityError(f"Non-finite metric {k}: {v}")
        log_pass("All metrics finite")

        # 12. Test objective forward produces finite loss
        log_step("Objective Forward Pass")
        G = torch.randn(2, 3, 64, 64, device=device)
        S = torch.randn(2, 2, 301, device=device)
        masker = BlockMasker(placement="random", grid=16, min_side=3, k_range=(1, 4), seed=12345)
        M = masker.sample(G, 0.5).to(device)

        model.train()
        objective.train()
        res = objective(model, G, S, M)
        total = res["total_loss"]
        if not torch.isfinite(total):
            raise IntegrityError(f"Non-finite loss from restored model: {total}")
        log_pass(f"Loss finite: {total.item():.6f}")

        # 13. Check no EMA gradient leak
        total.backward()
        leaked = [n for n, p in model.ema.named_parameters() if p.grad is not None]
        if leaked:
            raise IntegrityError(f"EMA gradient leak after restore: {leaked}")
        log_pass("No EMA gradient leak")

        print("\n" + "="*60)
        print("CHECKPOINT INTEGRITY CHECK PASSED")
        print("="*60)
        return 0

    except IntegrityError as e:
        print(f"\n[INTEGRITY FAILED] {e}")
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n[INTEGRITY ERROR] {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())