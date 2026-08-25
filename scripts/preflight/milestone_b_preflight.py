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
import math
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


def verify_environment(allow_cpu=False, device_request="auto"):
    """Verify Python, PyTorch, Torchvision, CUDA, GPU, device contract.

    device_request comes from the CLI (--device); 'auto' keeps the historical
    resolve_device('auto') behavior. The resolved selection — not an internal
    hardcoded one — is what every later preflight stage must use (A6)."""
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

    # Resolved device (honors --device from the CLI; A6)
    device = resolve_device(device_request)
    log_info(f"Requested device: {device_request}")
    log_info(f"Resolved device: {device}")

    # Verify tested environment contract
    # We tested with PyTorch 2.5.1 + Torchvision 0.20.1 on CUDA
    # On CPU, any compatible version is acceptable for preflight
    expected_torch = "2.5.1"
    expected_tv = "0.20.1"
    if cuda_available:
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
    else:
        log_info(f"CPU mode: skipping strict PyTorch/Torchvision version check (got {torch_ver}/{tv_ver})")

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
        # Check if the base itself contains the required files
        required = [
            "split_data/train_set.mat",
            "split_data/val_set.mat",
            "split_data/test_set.mat",
            "weights/metadit-small.bin",
            "weights/spec_encoder.pth",
            "weights/surrogate_model.bin",
        ]
        if all((base / r).exists() for r in required):
            if found is not None:
                raise PreflightError(
                    f"Multiple dataset locations found:\n  1. {found}\n  2. {base}\n"
                    "Explicit selection required. Set DATA_ROOT environment variable or "
                    "ensure only one valid dataset is mounted."
                )
            found = base
            log_info(f"Found dataset at: {base}")
            break
        
        # Also check subdirectories (for Kaggle where dataset might be in a subfolder)
        for candidate in base.rglob("*"):
            if candidate.is_dir():
                if all((candidate / r).exists() for r in required):
                    if found is not None:
                        raise PreflightError(
                            f"Multiple dataset locations found:\n  1. {found}\n  2. {candidate}\n"
                            "Explicit selection required. Set DATA_ROOT environment variable or "
                            "ensure only one valid dataset is mounted."
                        )
                    found = candidate
                    log_info(f"Found dataset at: {candidate}")
                    break

    if found is None:
        raise PreflightError(
            "No valid MetaDiT dataset found. Searched:\n"
            + "\n".join(f"  {p}" for p in search_paths)
            + "\nEnsure dataset is mounted (Kaggle: attach as input; Colab: mount Drive)."
        )

    # A9: pin the discovered dataset into the repo's canonical location
    # (<repo>/data/metadit) via a directory symlink, so the training driver and
    # configs can keep using the repo-relative path unchanged on Kaggle/Colab.
    # Guards: skip when the found root already IS the canonical path; refuse to
    # clobber a real (non-symlink) directory; surface OSError (e.g. Windows
    # symlink privilege) as a PreflightError telling the operator to use
    # --data-root instead.
    link = REPO_ROOT / "data" / "metadit"
    already_pinned = False
    try:
        if link.is_symlink() or link.exists():
            already_pinned = link.resolve() == found.resolve()
    except OSError:
        already_pinned = False
    if not already_pinned:
        try:
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                raise PreflightError(
                    f"{link} exists and is not a symlink; refusing to replace it. "
                    "Remove it manually or pass --data-root explicitly."
                )
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(found, target_is_directory=True)
            log_info(f"Pinned dataset: {link} -> {found}")
        except OSError as e:
            raise PreflightError(
                f"Could not create dataset symlink {link} -> {found}: {e}. "
                "Pass --data-root explicitly instead of relying on the pinned path."
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


import math

def compute_total_optimizer_steps(cfg, dataset_size):
    """Compute total optimizer steps from config and dataset size."""
    batch_size = int(cfg["train"]["batch_size"])
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    epochs = int(cfg["train"]["epochs"])

    micro_batches_per_epoch = dataset_size // batch_size
    if micro_batches_per_epoch < 1:
        raise ValueError("Dataset is smaller than one training batch")

    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / grad_accum
    )

    return optimizer_steps_per_epoch * epochs


def verify_model_objective(device, data_root, full_cfg):
    """Construct model and objective using real config, verify shapes and device placement."""
    log_step("Model + Objective Construction")

    model = build_model(
        full_cfg["model"],
        full_cfg["weights"]["spectrum"],
        device=device,
        init_from_metadit=True,
        metadit_weights=full_cfg["weights"]["metadit"],
    )
    log_info("Model built successfully")

    # Verify model on correct device
    model_device = next(model.parameters()).device
    if model_device != device:
        raise PreflightError(f"Model on {model_device}, expected {device}")
    log_info(f"Model device: {model_device}")

    # Use the CONFIGURED objective (A7): preflight must exercise the same
    # objective the training run will use, not a hardcoded default.
    objective_name = full_cfg.get("objective", "jepa_vicreg")
    objective = build_objective(
        objective_name, full_cfg.get("objective_params", {}).get(objective_name, {}),
        projector_input_dim=full_cfg["model"]["hidden"],
    ).to(device)
    log_info(f"Objective built successfully: {objective_name}")

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

    # Full model output shape verification (Bug #26)
    expected_shapes = {
        "z_hat": (B, 256, 384),
        "z_x": (B, 256, 384),
        "z_y_raw": (B, 256, 384),
        "z_y_normalized": (B, 256, 384),
        "mask": (B, 256),
        "c_physics": (B, 384),
        "a_goal": (B, 16, 384),
    }
    for key, expected in expected_shapes.items():
        if tuple(out[key].shape) != expected:
            raise PreflightError(
                f"{key}: got {tuple(out[key].shape)}, expected {expected}"
            )

    log_info(f"z_hat: {tuple(out['z_hat'].shape)} (expected [B, 256, 384])")
    log_info(f"z_y_raw: {tuple(out['z_y_raw'].shape)} (expected [B, 256, 384])")
    log_info(f"mask: {tuple(out['mask'].shape)} (expected [B, 256], bool)")
    log_info(f"c_physics: {tuple(out['c_physics'].shape)} (expected [B, 384])")
    log_info(f"a_goal: {tuple(out['a_goal'].shape)} (expected [B, 16, 384])")

    # Test objective forward
    res = objective(model, G, S, M)
    total = res["total_loss"]
    if not torch.isfinite(total):
        raise PreflightError(f"Non-finite loss: {total}")
    log_info(f"Total loss: {total.item():.6f}")
    log_info(f"Loss components: {list(res['components'].keys())}")

    log_pass("Model + Objective verified")
    return model, objective, masker, full_cfg


def run_real_dataset_sample(model, objective, masker, device, data_root, full_cfg):
    """Run one real MetaDiT dataset sample through the pipeline (Bug #25)."""
    log_step("Real Dataset Sample Verification")

    train_path = Path(data_root) / "split_data" / "train_set.mat"
    real_ds = MetaDiTDataset(str(train_path), max_samples=1, seed=0)
    G_real, S_real = real_ds[0]

    G_real = G_real.unsqueeze(0).to(device)
    S_real = S_real.unsqueeze(0).to(device)
    M_real = masker.sample(G_real, 0.5).to(device)

    model.eval()
    objective.eval()
    real_res = objective(model, G_real, S_real, M_real)

    if not torch.isfinite(real_res["total_loss"]):
        raise PreflightError("Real-data preflight loss is non-finite")

    log_info(f"Real data loss: {real_res['total_loss'].item():.6f}")
    log_pass("Real dataset sample verified")


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
    class _StubObjective(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projector = torch.nn.Identity()

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
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU mode for local testing")
    parser.add_argument("--device", default="auto",
                        help="Device selection passed to resolve_device "
                             "(e.g. 'auto', 'cpu', 'cuda:0'). Default: auto.")
    args = parser.parse_args()

    # Load the REAL config at startup (Bug #5)
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    print("="*60)
    print("MILESTONE B PREFLIGHT CHECK")
    print("="*60)

    try:
        # 1. Environment (device selection comes from --device, A6)
        device = verify_environment(allow_cpu=args.allow_cpu,
                                    device_request=args.device)

        # 2. Git state
        commit, dirty = verify_git_state()

        # 3. Dataset
        data_root = args.data_root or discover_dataset()

        # 4. Model + Objective (using REAL config)
        model, objective, masker, cfg = verify_model_objective(device, data_root, full_cfg)

        # Load real dataset to compute total steps (Bug #6)
        train_path = Path(data_root) / "split_data" / "train_set.mat"
        train_ds = MetaDiTDataset(str(train_path), max_samples=full_cfg["data"].get("max_train_samples", 0), seed=full_cfg["train"]["seed"])
        total_steps = compute_total_optimizer_steps(full_cfg, len(train_ds))

        # Print training setup info (Bug #6)
        log_step("Training Setup Verification")
        batch_size = full_cfg["train"]["batch_size"]
        grad_accum = full_cfg["train"].get("grad_accum", 1)
        epochs = full_cfg["train"]["epochs"]
        micro_batches_per_epoch = len(train_ds) // batch_size
        optimizer_steps_per_epoch = math.ceil(micro_batches_per_epoch / full_cfg["train"].get("grad_accum", 1))
        log_info(f"Training samples: {len(train_ds)}")
        log_info(f"Batch size: {batch_size}")
        log_info(f"Grad accum: {grad_accum}")
        log_info(f"Micro-batches/epoch: {micro_batches_per_epoch}")
        log_info(f"Optimizer steps/epoch: {optimizer_steps_per_epoch}")
        log_info(f"Epochs: {epochs}")
        log_info(f"TOTAL optimizer steps: {total_steps}")
        log_info(f"val_every_steps: {full_cfg['train'].get('val_every_steps', 0)}")

        # 5. Tiny training — reuses the model/objective/masker/cfg already built
        # by verify_model_objective above (A8: no duplicate construction, which
        # would double GPU memory and diverge from the verified instances).
        optimizer, scheduler = run_tiny_training(model, objective, masker, device)

        # 6. Validation
        metrics, health, gap_metrics = run_validation(model, objective, masker, device, data_root)

        # 7. Physics controls
        physics_metrics = run_physics_controls(model, masker, device)

        # 8. Checkpoint + Resume
        run_checkpoint_resume(model, objective, optimizer, scheduler, masker, device, cfg)

        # 9. Config validation with REAL total_steps (Bug #5, #6)
        log_step("Config Validation")
        try:
            validate_config(full_cfg, total_steps)
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