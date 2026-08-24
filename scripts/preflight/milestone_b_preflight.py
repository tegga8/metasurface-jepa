#!/usr/bin/env python3
"""Milestone B Preflight Check (Phase 2 §1-2).

Comprehensive preflight verification before starting full Milestone-B training on Kaggle/Colab.
Verifies environment, dataset, model/objective construction, tiny training, validation,
checkpoint save/load, and resume — all without manual source-code edits.

Exit codes:
  0 = PASS (all checks pass)
  1 = FAIL (any check fails, with diagnostic output)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import torch

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_DIR))

from assembly import build_model
from data.dataset import MetaDiTDataset
from data.mask import BlockMasker
from losses.objectives import build_objective
from runtime.device import resolve_device
from runtime.reproducibility import set_seed, collect_rng_state
from runtime.physics_controls import compute_physics_metrics, validate_goal_mode
from scripts.train.train_milestone_b import build_scheduler, validate_config
from train.engine import (
    save_checkpoint, load_checkpoint, collect_ema_state, restore_ema_state,
    fixed_validation_from_loader, build_deterministic_reference, healthy_references,
)
from torch.utils.data import DataLoader


class PreflightError(Exception):
    """Preflight check failure with context."""
    pass


def log_step(step: str):
    print(f"\n{'='*60}")
    print(f"PREFLIGHT: {step}")
    print(f"{'='*60}")


def log_info(msg: str):
    print(f"  [INFO] {msg}")


def log_pass(msg: str):
    print(f"  [PASS] {msg}")


def log_fail(msg: str):
    print(f"  [FAIL] {msg}")


def verify_environment():
    """Verify Python, PyTorch, Torchvision, CUDA, GPU, device contract."""
    log_step("Environment Verification")

    # Python
    py_ver = sys.version.split()[0]
    log_info(f"Python: {py_ver}")

    # PyTorch
    torch_ver = torch.__version__
    log_info(f"PyTorch: {torch_ver}")

    # Torchvision
    try:
        import torchvision
        tv_ver = torchvision.__version__
        log_info(f"Torchvision: {tv_ver}")
    except ImportError:
        raise PreflightError("torchvision not installed")

    # CUDA
    cuda_available = torch.cuda.is_available()
    log_info(f"CUDA available: {cuda_available}")
    if cuda_available:
        cuda_ver = torch.version.cuda
        cudnn_ver = torch.backends.cudnn.version()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        log_info(f"CUDA runtime: {cuda_ver}")
        log_info(f"cuDNN: {cudnn_ver}")
        log_info(f"GPU: {gpu_name} (count: {gpu_count})")

    # Resolved device
    device = resolve_device("auto")
    log_info(f"Resolved device: {device}")

    # Verify tested environment contract
    # We tested with PyTorch 2.5.1 + Torchvision 0.20.1
    expected_torch = "2.5.1"
    expected_tv = "0.20.1"
    if not torch_ver.startswith(expected_torch):
        raise PreflightError(
            f"PyTorch version mismatch: expected {expected_torch}.x, got {torch_ver}. "
            "This combination was not tested."
        )
    if not tv_ver.startswith(expected_tv):
        raise PreflightError(
            f"Torchvision version mismatch: expected {expected_tv}.x, got {tv_ver}. "
            "This combination was not tested."
        )

    # Fail if CUDA requested but unavailable
    if not cuda_available:
        raise PreflightError("CUDA is not available but GPU training is required for Milestone B")

    log_pass("Environment contract satisfied")
    return device


def verify_git_state():
    """Verify git commit and dirty state."""
    log_step("Git Repository State")

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, cwd=REPO_ROOT
        ).strip()
        log_info(f"Git commit: {commit}")
    except Exception as e:
        raise PreflightError(f"Failed to get git commit: {e}")

    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True, cwd=REPO_ROOT
        ).strip()
        is_dirty = bool(dirty)
        if is_dirty:
            log_info("Git working tree: DIRTY")
            log_info(f"Dirty files:\n{dirty}")
        else:
            log_info("Git working tree: CLEAN")
        log_pass("Git state recorded")
    except Exception as e:
        raise PreflightError(f"Failed to get git status: {e}")

    return commit, is_dirty


def discover_dataset():
    """Discover and verify dataset/weights in /kaggle/input or local data/metadit/."""
    log_step("Dataset Discovery")

    # Search paths in order of priority
    search_paths = [
        Path("/kaggle/input"),
        REPO_ROOT / "data" / "metadit",
    ]

    found = None
    for base in search_paths:
        if not base.exists():
            continue
        # Look for directories containing the required files
        for candidate in base.rglob("*"):
            if candidate.is_dir():
                required = [
                    "split_data/train_set.mat",
                    "split_data/val_set.mat",
                    "split_data/test_set.mat",
                    "weights/metadit-small.bin",
                    "weights/spec_encoder.pth",
                    "weights/surrogate_model.bin",
                ]
                if all((candidate / r).exists() for r in required):
                    if found is not None:
                        raise PreflightError(
                            f"Multiple dataset locations found:\n  1. {found}\n  2. {candidate}\n"
                            "Explicit selection required. Set DATA_ROOT environment variable or "
                            "ensure only one valid dataset is mounted."
                        )
                    found = candidate
                    log_info(f"Found dataset at: {candidate}")

    if found is None:
        raise PreflightError(
            "No valid MetaDiT dataset found. Searched:\n"
            + "\n".join(f"  {p}" for p in search_paths)
            + "\nEnsure dataset is mounted (Kaggle: attach as input; Colab: mount Drive)."
        )

    # Verify file sizes and load one sample
    log_info("Verifying dataset structure and loading sample...")
    train_path = found / "split_data" / "train_set.mat"
    val_path = found / "split_data" / "val_set.mat"

    # Quick load test
    ds = MetaDiTDataset(str(train_path), max_samples=1, seed=0)
    G, S = ds[0]
    log_info(f"Geometry shape: {tuple(G.shape)} (expected [3, 64, 64])")
    log_info(f"Spectrum shape: {tuple(S.shape)} (expected [2, 301])")

    if G.shape != (3, 64, 64):
        raise PreflightError(f"Geometry shape mismatch: {G.shape} != (3, 64, 64)")
    if S.shape != (2, 301):
        raise PreflightError(f"Spectrum shape mismatch: {S.shape} != (2, 301)")

    log_pass("Dataset discovered and verified")
    return str(found)


def verify_model_objective(device, data_root):
    """Construct model and objective, verify shapes and device placement."""
    log_step("Model + Objective Construction")

    cfg = {
        "model": {"hidden": 384, "num_heads": 6, "geo_depth": 6, "predictor_depth": 8,
                  "goal_tokens": 16, "num_predictor_heads": 6,
                  "ema_momentum_start": 0.996, "ema_momentum_end": 0.999},
        "weights": {
            "spectrum": str(Path(data_root) / "weights" / "spec_encoder.pth"),
            "metadit": str(Path(data_root) / "weights" / "metadit-small.bin"),
        },
    }

    model = build_model(
        cfg["model"],
        cfg["weights"]["spectrum"],
        device=device,
        init_from_metadit=True,
        metadit_weights=cfg["weights"]["metadit"],
    )
    log_info("Model built successfully")

    # Verify model on correct device
    model_device = next(model.parameters()).device
    if model_device != device:
        raise PreflightError(f"Model on {model_device}, expected {device}")
    log_info(f"Model device: {model_device}")

    objective = build_objective(
        "jepa_vicreg", {},
        projector_input_dim=cfg["model"]["hidden"],
    ).to(device)
    log_info("Objective built successfully")

    obj_device = next(objective.parameters()).device
    if obj_device != device:
        raise PreflightError(f"Objective on {obj_device}, expected {device}")
    log_info(f"Objective device: {obj_device}")

    # Test forward pass shapes
    B = 2
    G = torch.randn(B, 3, 64, 64, device=device)
    S = torch.randn(B, 2, 301, device=device)
    masker = BlockMasker(placement="random", grid=16, min_side=3, k_range=(1, 4), seed=12345)
    M = masker.sample(G, 0.5).to(device)

    model.train()
    objective.train()
    out = model(G, S, M, goal_mode="real", with_target=True)

    required_keys = {"z_hat", "z_x", "mask", "c_physics", "a_goal", "z_y_raw", "z_y_normalized"}
    missing = required_keys - set(out.keys())
    if missing:
        raise PreflightError(f"Model output missing keys: {missing}")

    log_info(f"z_hat: {tuple(out['z_hat'].shape)} (expected [B, 256, 384])")
    log_info(f"z_y_raw: {tuple(out['z_y_raw'].shape)} (expected [B, 256, 384])")
    log_info(f"mask: {tuple(out['mask'].shape)} (expected [B, 256], bool)")
    log_info(f"c_physics: {tuple(out['c_physics'].shape)} (expected [B, 384])")
    log_info(f"a_goal: {tuple(out['a_goal'].shape)} (expected [B, 16, 384])")

    if out["z_hat"].shape != (B, 256, 384):
        raise PreflightError(f"z_hat shape mismatch: {out['z_hat'].shape}")
    if out["z_y_raw"].shape != (B, 256, 384):
        raise PreflightError(f"z_y_raw shape mismatch: {out['z_y_raw'].shape}")

    # Test objective forward
    res = objective(model, G, S, M)
    total = res["total_loss"]
    if not torch.isfinite(total):
        raise PreflightError(f"Non-finite loss: {total}")
    log_info(f"Total loss: {total.item():.6f}")
    log_info(f"Loss components: {list(res['components'].keys())}")

    log_pass("Model + Objective verified")
    return model, objective, masker, cfg


def run_tiny_training(model, objective, masker, device):
    """Run tiny training: forward, loss, backward, EMA guard, optimizer step, EMA update, scheduler step."""
    log_step("Tiny Training Verification")

    set_seed(0)

    trainable = [p for p in model.parameters() if p.requires_grad] + \
                [p for p in objective.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
    total_steps = 2
    scheduler = build_scheduler(optimizer, 1e-3, 0, total_steps)
    model.ema.set_total_steps(total_steps)

    # Check EMA not in optimizer
    ema_ids = {id(p) for p in model.ema.parameters()}
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    if ema_ids & opt_ids:
        raise PreflightError("EMA parameters leaked into optimizer")

    # Batch
    B = 2
    G = torch.randn(B, 3, 64, 64, device=device)
    S = torch.randn(B, 2, 301, device=device)
    M = masker.sample(G, 0.5).to(device)

    model.train()
    objective.train()

    # Forward
    res = objective(model, G, S, M)
    total = res["total_loss"]
    if not torch.isfinite(total):
        raise PreflightError(f"Non-finite loss: {total}")

    # Backward
    total.backward()

    # EMA gradient guard
    leaked = [n for n, p in model.ema.named_parameters() if p.grad is not None]
    if leaked:
        raise PreflightError(f"EMA target received gradients: {leaked}")
    log_pass("EMA gradient guard: no leaks")

    # Optimizer step
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # EMA update
    objective.on_optimizer_step(model, 0)

    # Scheduler step
    scheduler.step()

    log_pass("Tiny training step completed (forward/backward/optimizer/EMA/scheduler)")
    return optimizer, scheduler


def run_validation(model, objective, masker, device, data_root):
    """Run production FixedValidation path."""
    log_step("Validation Path Verification")

    val_path = Path(data_root) / "split_data" / "val_set.mat"
    val_ds = MetaDiTDataset(str(val_path))

    fv = fixed_validation_from_loader(
        val_ds, n_samples=4, batch_size=2, device=device, ratio=0.5, mask_seed=12345
    )
    log_info(f"Fixed validation: {fv.mask_statistics['n_samples']} samples, "
             f"{fv.mask_statistics['n_batches']} batches")

    # Build deterministic reference
    cfg = {
        "model": {"hidden": 384, "num_heads": 6, "geo_depth": 6, "predictor_depth": 8,
                  "goal_tokens": 16, "num_predictor_heads": 6,
                  "ema_momentum_start": 0.996, "ema_momentum_end": 0.999},
        "weights": {
            "spectrum": str(Path(data_root) / "weights" / "spec_encoder.pth"),
            "metadit": str(Path(data_root) / "weights" / "metadit-small.bin"),
        },
    }
    ref_model = build_deterministic_reference(
        lambda: build_model(cfg["model"], cfg["weights"]["spectrum"], device=device, init_from_metadit=True,
                            metadit_weights=cfg["weights"]["metadit"])
    )
    refs = healthy_references(ref_model, fv, objective=objective)

    # Evaluate
    model.eval()
    objective.eval()
    metrics, health = fv.evaluate(model, objective, refs["raw"], refs["proj"])

    log_info(f"Validation metrics: {metrics}")
    log_info(f"Health status: {health['status']}")

    if health["status"] not in ("HEALTHY", "WARNING", "UNAVAILABLE", "COLLAPSED"):
        raise PreflightError(f"Unknown health status: {health['status']}")

    # Null gap
    gap_metrics = fv.null_gap(model, objective)
    log_info(f"Null gap metrics: {gap_metrics}")

    log_pass("Validation path verified")
    return metrics, health, gap_metrics


def run_checkpoint_resume(model, objective, optimizer, scheduler, masker, device, cfg):
    """Save and reload a temporary full checkpoint, then resume one step."""
    log_step("Checkpoint Save / Load / Resume")

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "preflight_ckpt.pt"

        # Collect EMA state
        ema_state = collect_ema_state(model)

        # Save full checkpoint
        save_checkpoint(
            str(ckpt_path), model, objective, optimizer, scheduler, cfg,
            global_step=0, epoch=0, micro_step=0, is_epoch_end=True,
            metrics={}, health=None,
            ema_state=ema_state,
            best_prediction={}, best_healthy_prediction={},
            masker_rng_state=masker.get_rng_state(),
            device=device, artifact_type="full",
        )

        # Verify file exists
        if not ckpt_path.exists():
            raise PreflightError("Checkpoint file not created")
        log_info(f"Checkpoint saved: {ckpt_path} ({ckpt_path.stat().st_size} bytes)")

        # Verify loadable
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        required_keys = [
            "schema_version", "objective_name", "step", "epoch", "micro_step", "is_epoch_end",
            "cfg", "best_prediction", "best_healthy_prediction",
            "model", "objective_state", "optimizer", "optimizer_param_shapes",
            "scheduler_state", "ema_state", "rng_state", "masker_rng_state",
            "git_commit", "git_dirty", "env_versions", "device_info", "artifact_type",
        ]
        for k in required_keys:
            if k not in obj:
                raise PreflightError(f"Checkpoint missing required key: {k}")
        log_pass("Checkpoint schema validated")

        # Reconstruct fresh model/objective
        model2 = build_model(
            cfg["model"],
            cfg["weights"]["spectrum"],
            device=device,
            init_from_metadit=True,
            metadit_weights=cfg["weights"]["metadit"],
        )
        objective2 = build_objective(
            "jepa_vicreg", {},
            projector_input_dim=cfg["model"]["hidden"],
        ).to(device)
        trainable2 = [p for p in model2.parameters() if p.requires_grad] + \
                     [p for p in objective2.parameters() if p.requires_grad]
        optimizer2 = torch.optim.AdamW(trainable2, lr=1e-3, weight_decay=1e-4)
        scheduler2 = build_scheduler(optimizer2, 1e-3, 0, 2)
        model2.ema.set_total_steps(2)

        masker2 = BlockMasker(placement="random", grid=16, min_side=3, k_range=(1, 4), seed=12345)

        # Load checkpoint
        loaded = load_checkpoint(str(ckpt_path), model2, objective2, optimizer2, scheduler2, device)
        restore_ema_state(model2, loaded.get("ema_state"))
        if loaded.get("masker_rng_state") is not None:
            masker2.set_rng_state(loaded["masker_rng_state"])

        # Verify model state restored
        for k in model.state_dict():
            if not torch.allclose(model.state_dict()[k], model2.state_dict()[k], atol=1e-6):
                raise PreflightError(f"Model param {k} not restored correctly")
        log_pass("Model state restored")

        # Verify objective state restored
        for k in objective.state_dict():
            if not torch.allclose(objective.state_dict()[k], objective2.state_dict()[k], atol=1e-6):
                raise PreflightError(f"Objective param {k} not restored correctly")
        log_pass("Objective state restored")

        # Verify optimizer state restored
        opt_state1 = optimizer.state_dict()
        opt_state2 = optimizer2.state_dict()
        if opt_state1["param_groups"] != opt_state2["param_groups"]:
            raise PreflightError("Optimizer param_groups not restored")
        for k in opt_state1["state"]:
            if k not in opt_state2["state"]:
                raise PreflightError(f"Optimizer state key {k} missing after restore")
            for sk in opt_state1["state"][k]:
                if not torch.equal(opt_state1["state"][k][sk], opt_state2["state"][k][sk]):
                    raise PreflightError(f"Optimizer state {k}.{sk} not restored")
        log_pass("Optimizer state restored")

        # Verify scheduler state restored
        if scheduler.state_dict() != scheduler2.state_dict():
            raise PreflightError("Scheduler state not restored")
        log_pass("Scheduler state restored")

        # Verify EMA state restored
        if model.ema.total_steps != model2.ema.total_steps:
            raise PreflightError("EMA total_steps not restored")
        log_pass("EMA state restored")

        # Resume: take one further step
        model2.train()
        objective2.train()
        G = torch.randn(2, 3, 64, 64, device=device)
        S = torch.randn(2, 2, 301, device=device)
        M = masker2.sample(G, 0.5).to(device)

        res = objective2(model2, G, S, M)
        total = res["total_loss"]
        total.backward()

        leaked = [n for n, p in model2.ema.named_parameters() if p.grad is not None]
        if leaked:
            raise PreflightError(f"EMA target received gradients after resume: {leaked}")

        trainable2 = [p for p in model2.parameters() if p.requires_grad] + \
                     [p for p in objective2.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable2, 1.0)
        optimizer2.step()
        optimizer2.zero_grad(set_to_none=True)
        objective2.on_optimizer_step(model2, 1)
        scheduler2.step()

        log_pass("Resume step completed without corruption")

    return True


def run_physics_controls(model, masker, device):
    """Run physics controls to verify canonical metrics."""
    log_step("Physics Controls Verification")

    set_seed(42)
    G = torch.randn(4, 3, 64, 64, device=device)
    S = torch.randn(4, 2, 301, device=device)
    M = masker.sample(G, 0.5).to(device)

    # Build a minimal objective for projector
    class _StubObjective:
        projector = torch.nn.Identity()

    objective = _StubObjective()

    # Test compute_physics_metrics
    metrics = compute_physics_metrics(model, G, S, M, objective=objective, device=device)
    required = {"L_real", "L_null", "L_shuffled", "gap_null", "gap_shuffled",
                "sensitivity_null", "sensitivity_shuffled"}
    missing = required - set(metrics.keys())
    if missing:
        raise PreflightError(f"Physics metrics missing: {missing}")
    log_info(f"Physics metrics: {metrics}")

    # Test validate_goal_mode
    try:
        validate_goal_mode("real")
        validate_goal_mode("null")
    except ValueError:
        raise PreflightError("validate_goal_mode failed for valid modes")

    try:
        validate_goal_mode("invalid")
        raise PreflightError("validate_goal_mode should reject 'invalid'")
    except ValueError:
        pass

    log_pass("Physics controls verified")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Milestone B Preflight Check")
    parser.add_argument("--config", default="configs/milestone_b.yaml", help="Config file path")
    parser.add_argument("--data-root", default=None, help="Override dataset root")
    args = parser.parse_args()

    print("="*60)
    print("MILESTONE B PREFLIGHT CHECK")
    print("="*60)

    try:
        # 1. Environment
        device = verify_environment()

        # 2. Git state
        commit, dirty = verify_git_state()

        # 3. Dataset
        data_root = args.data_root or discover_dataset()

        # 4. Model + Objective
        model, objective, masker, cfg = verify_model_objective(device, data_root)

        # 5. Tiny training
        optimizer, scheduler = run_tiny_training(model, objective, masker, device)

        # 6. Validation
        metrics, health, gap_metrics = run_validation(model, objective, masker, device, data_root)

        # 7. Physics controls
        physics_metrics = run_physics_controls(model, masker, device)

        # 8. Checkpoint + Resume
        run_checkpoint_resume(model, objective, optimizer, scheduler, masker, device, cfg)

        # 9. Config validation
        log_step("Config Validation")
        total_steps = 2
        try:
            validate_config(cfg, total_steps)
            log_pass("Config validation passed")
        except ValueError as e:
            raise PreflightError(f"Config validation failed: {e}")

        print("\n" + "="*60)
        print("ALL PREFLIGHT CHECKS PASSED")
        print("="*60)
        print(f"Git commit: {commit}")
        print(f"Git dirty: {dirty}")
        print(f"Device: {device}")
        print(f"Dataset: {data_root}")
        print(f"Health: {health['status']}")
        return 0

    except PreflightError as e:
        print(f"\n[PREFLIGHT FAILED] {e}")
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n[PREFLIGHT ERROR] Unexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())