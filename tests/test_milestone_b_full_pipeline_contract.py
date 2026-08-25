"""End-to-end integration test for Milestone B full pipeline contract (hardening spec §13).

Exercises in one process:
  fresh build
  → dataset sample
  → mask
  → model
  → objective
  → optimizer
  → scheduler
  → forward
  → loss
  → backward
  → EMA gradient guard
  → optimizer step
  → EMA update
  → validation
  → checkpoint save
  → checkpoint load
  → resume
  → next mask
  → next optimizer step

Uses a tiny test configuration but real production modules.
"""

import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from assembly import build_model
from data.mask import BlockMasker
from losses.objectives import build_objective
from train.engine import (
    save_checkpoint, load_checkpoint, collect_ema_state, restore_ema_state,
    fixed_validation_from_loader, build_deterministic_reference, healthy_references,
)
from scripts.train.train_milestone_b import CosineWarmup, build_scheduler
from runtime.device import resolve_device


class _TinyMetaDiTDataset(torch.utils.data.Dataset):
    """Tiny synthetic dataset mimicking MetaDiT shapes for testing."""
    def __init__(self, n=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.G = torch.randn(n, 3, 64, 64, generator=g)
        self.S = torch.randn(n, 2, 301, generator=g)
    def __len__(self): return len(self.G)
    def __getitem__(self, i): return self.G[i], self.S[i]


def collate_batch(batch):
    G = torch.stack([b[0] for b in batch])
    S = torch.stack([b[1] for b in batch])
    return G, S


def test_milestone_b_full_pipeline_contract():
    """Full pipeline contract test with tiny config."""
    device = resolve_device("cpu")

    # ---- tiny config ----
    cfg = {
        "model": {
            "hidden": 384,
            "num_heads": 6,
            "geo_depth": 1,
            "predictor_depth": 1,
            "goal_tokens": 16,
            "num_predictor_heads": 2,
            "ema_momentum_start": 0.99,
            "ema_momentum_end": 0.999,
        },
        "weights": {
            "spectrum": os.path.join(REPO_ROOT, "data/metadit/weights/spec_encoder.pth"),
            "metadit": os.path.join(REPO_ROOT, "data/metadit/weights/metadit-small.bin"),
        },
        "mask": {"min_side": 3, "k_range": [1, 2], "mask_seed": 12345},
        "train": {
            "batch_size": 2,
            "grad_accum": 1,
            "epochs": 1,
            "lr": 1e-3,
            "wd": 1e-4,
            "warmup_steps": 0,
            "clip_grad_norm": 1.0,
            "val_every_steps": 1,
            "val_batches": 1,
        },
    }

    # ---- build ----
    model = build_model(
        cfg["model"],
        cfg["weights"]["spectrum"],
        device=device,
        init_from_metadit=False,  # skip MetaDiT weights for speed
    )
    objective = build_objective(
        "jepa_vicreg", {},
        projector_input_dim=cfg["model"]["hidden"],
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad] + \
                [p for p in objective.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["wd"])
    total_steps = 2
    scheduler = build_scheduler(optimizer, cfg["train"]["lr"], cfg["train"]["warmup_steps"], total_steps)
    model.ema.set_total_steps(total_steps)

    masker = BlockMasker(
        placement="random", grid=16,
        min_side=cfg["mask"]["min_side"],
        k_range=tuple(cfg["mask"]["k_range"]),
        seed=cfg["mask"]["mask_seed"],
    )

    # ---- dataset sample ----
    ds = _TinyMetaDiTDataset(n=4, seed=42)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_batch)
    G, S = next(iter(loader))
    G, S = G.to(device), S.to(device)

    # ---- mask ----
    ratio = 0.5
    M = masker.sample(G, ratio).to(device)

    # ---- model + objective + forward ----
    model.train()
    objective.train()
    out = model(G, S, M, goal_mode="real", with_target=True)
    assert "z_hat" in out and "z_y_raw" in out and "mask" in out

    # ---- loss ----
    res = objective(model, G, S, M)
    total = res["total_loss"]
    assert torch.isfinite(total)

    # ---- backward ----
    total.backward()

    # ---- EMA gradient guard ----
    leaked = [n for n, p in model.ema.named_parameters() if p.grad is not None]
    assert not leaked, f"EMA target received gradients: {leaked}"

    # ---- optimizer step ----
    torch.nn.utils.clip_grad_norm_(trainable, cfg["train"]["clip_grad_norm"])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # ---- EMA update ----
    step = 0
    objective.on_optimizer_step(model, step)
    scheduler.step()

    # ---- validation ----
    val_ds = _TinyMetaDiTDataset(n=2, seed=123)
    fv = fixed_validation_from_loader(val_ds, 2, cfg["train"]["batch_size"], device, ratio=ratio)
    ref_model = build_deterministic_reference(
        lambda: build_model(cfg["model"], cfg["weights"]["spectrum"], device=device, init_from_metadit=False)
    )
    refs = healthy_references(ref_model, fv, objective=objective)
    val_metrics, health = fv.evaluate(model, objective, refs["raw"], refs["proj"])
    assert "cos_err_r0.5" in val_metrics or any(k.startswith("cos_err_r") for k in val_metrics)
    assert health["status"] in ("HEALTHY", "WARNING", "UNAVAILABLE", "COLLAPSED")

    # ---- checkpoint save ----
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "test_ckpt.pt")
        ema_state = collect_ema_state(model)
        save_checkpoint(
            ckpt_path, model, objective, optimizer, scheduler, cfg,
            global_step=step, epoch=0, micro_step=0, is_epoch_end=True,
            metrics=val_metrics, health=health,
            ema_state=ema_state,
            best_prediction={"primary": val_metrics.get("cos_err_r0.5", 0.0), "metrics": val_metrics, "step": step, "health": health},
            best_healthy_prediction={},
            masker_rng_state=masker.get_rng_state(),
            device=device, artifact_type="full",
        )

        # ---- checkpoint load ----
        model2 = build_model(
            cfg["model"],
            cfg["weights"]["spectrum"],
            device=device,
            init_from_metadit=False,
        )
        objective2 = build_objective(
            "jepa_vicreg", {},
            projector_input_dim=cfg["model"]["hidden"],
        ).to(device)
        trainable2 = [p for p in model2.parameters() if p.requires_grad] + \
                     [p for p in objective2.parameters() if p.requires_grad]
        optimizer2 = torch.optim.AdamW(trainable2, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["wd"])
        scheduler2 = build_scheduler(optimizer2, cfg["train"]["lr"], cfg["train"]["warmup_steps"], total_steps)
        model2.ema.set_total_steps(total_steps)

        masker2 = BlockMasker(
            placement="random", grid=16,
            min_side=cfg["mask"]["min_side"],
            k_range=tuple(cfg["mask"]["k_range"]),
            seed=cfg["mask"]["mask_seed"],
        )

        obj = load_checkpoint(ckpt_path, model2, objective2, optimizer2, scheduler2, device)
        restore_ema_state(model2, obj.get("ema_state"))
        if "masker_rng_state" in obj and obj["masker_rng_state"] is not None:
            masker2.set_rng_state(obj["masker_rng_state"])

        # Verify model state restored
        for k in model.state_dict():
            assert torch.allclose(model.state_dict()[k], model2.state_dict()[k], atol=1e-6), f"Model param {k} not restored"

        # Verify objective state restored
        for k in objective.state_dict():
            assert torch.allclose(objective.state_dict()[k], objective2.state_dict()[k], atol=1e-6), f"Objective param {k} not restored"

        # Verify optimizer state restored
        opt_state1 = optimizer.state_dict()
        opt_state2 = optimizer2.state_dict()
        assert opt_state1["param_groups"] == opt_state2["param_groups"]
        for k in opt_state1["state"]:
            assert k in opt_state2["state"]
            for sk in opt_state1["state"][k]:
                assert torch.equal(opt_state1["state"][k][sk], opt_state2["state"][k][sk])

        # Verify scheduler state restored
        assert scheduler.state_dict() == scheduler2.state_dict()

        # Verify EMA state restored
        assert model.ema.total_steps == model2.ema.total_steps

        # ---- resume: next mask ----
        model2.train()
        objective2.train()
        G2, S2 = next(iter(loader))
        G2, S2 = G2.to(device), S2.to(device)
        M2 = masker2.sample(G2, ratio).to(device)
        out2 = model2(G2, S2, M2, goal_mode="real", with_target=True)
        assert "z_hat" in out2

        # ---- resume: next optimizer step ----
        res2 = objective2(model2, G2, S2, M2)
        total2 = res2["total_loss"]
        total2.backward()
        leaked2 = [n for n, p in model2.ema.named_parameters() if p.grad is not None]
        assert not leaked2
        optimizer2.step()
        optimizer2.zero_grad(set_to_none=True)
        objective2.on_optimizer_step(model2, step + 1)
        scheduler2.step()

    print("PASS: test_milestone_b_full_pipeline_contract")


# ---------------------------------------------------------------------------
# Real production CLI smoke test (production-path hardening A3).
#
# Executes the ACTUAL scripts/train/train_milestone_b.py in a subprocess:
#   python scripts/train/train_milestone_b.py --config <tiny-config> \
#       --device cpu --max-steps 2
# and asserts exit code 0, checkpoint existence, loadability, and full schema.
# The training loop is NOT recreated inside the test.
# ---------------------------------------------------------------------------

import json
import subprocess

import yaml

from train.engine import REQUIRED_CHECKPOINT_KEYS


def _make_tiny_mat(path, n=8, seed=0):
    """Write a hermetic tiny MetaDiT-convention .mat file."""
    import numpy as np
    from scipy import io as sio
    rng = np.random.RandomState(seed)
    pattern = (rng.rand(64, 64, n) > 0.5).astype("int8")
    sio.savemat(str(path), {
        "pattern": pattern,
        "parameter": rng.rand(n, 3),
        "real": rng.randn(n, 301),
        "imag": rng.randn(n, 301),
    })
    return str(path)


def _write_tiny_cli_config(tmpdir, train_mat, val_mat, out_subdir="out"):
    """Tiny deterministic production config exercising grad_accum=2 on CPU."""
    cfg = {
        "experiment": "minimal",
        "objective": "jepa_vicreg",
        "minimal": {"mask_ratio": 0.5, "mask_placement": "random"},
        "sweep": {"mask_ratios": [0.2, 0.4, 0.6, 0.8, 1.0],
                  "mask_placement": "random"},
        "objective_params": {
            "jepa_vicreg": {
                "projector": {"input_dim": 384, "hidden_dim": 384, "output_dim": 384},
                "lambda_inv": 25.0, "lambda_var": 25.0, "lambda_cov": 1.0,
                "gamma": 1.0, "eps": 1.0e-4,
            },
        },
        "model": {
            "variant": "jepa", "patch_size": 4, "token_grid": 16,
            "hidden": 384, "num_heads": 6, "num_predictor_heads": 2,
            "geo_depth": 1, "predictor_depth": 1,
            "goal_tokens": 16, "num_goal_heads": 4,
            "ema_momentum_start": 0.99, "ema_momentum_end": 0.999,
            "init_from_metadit": False,
        },
        "mask": {"min_side": 3, "k_range": [1, 2], "mask_seed": 12345},
        "data": {"train_split": train_mat, "val_split": val_mat,
                 "max_train_samples": 8, "num_workers": 0},
        "weights": {
            "metadit": os.path.join(REPO_ROOT, "data/metadit/weights/metadit-small.bin"),
            "spectrum": os.path.join(REPO_ROOT, "data/metadit/weights/spec_encoder.pth"),
            "surrogate": os.path.join(REPO_ROOT, "data/metadit/weights/surrogate_model.bin"),
        },
        "train": {
            "batch_size": 2, "grad_accum": 2, "epochs": 1,
            "lr": 1e-3, "wd": 1e-4, "warmup_steps": 0, "clip_grad_norm": 1.0,
            "save_optimizer": True, "ckpt_every_steps": 0,
            "val_every_steps": 1, "val_batches": 1, "log_every_steps": 1,
            "seed": 0, "device": "cpu",
        },
        "out_dir": os.path.join(tmpdir, out_subdir),
    }
    cfg_path = os.path.join(tmpdir, f"tiny_{out_subdir}.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path, cfg


def _run_training_cli(config_path, extra_args, env_threads="1"):
    """Run the actual production CLI as a subprocess. Returns CompletedProcess."""
    cmd = [sys.executable,
           os.path.join(REPO_ROOT, "scripts", "train", "train_milestone_b.py"),
           "--config", config_path,
           "--device", "cpu",
           *extra_args]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = env_threads
    env["MKL_NUM_THREADS"] = env_threads
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT,
                          env=env, timeout=900)


def test_production_cli_smoke_max_steps_2():
    """A3: the real CLI runs 2 optimizer steps end-to-end and writes a full-schema checkpoint."""
    import torch
    with tempfile.TemporaryDirectory() as td:
        train_mat = _make_tiny_mat(os.path.join(td, "train.mat"), n=8, seed=42)
        val_mat = _make_tiny_mat(os.path.join(td, "val.mat"), n=4, seed=123)
        cfg_path, _ = _write_tiny_cli_config(td, train_mat, val_mat)

        proc = _run_training_cli(cfg_path, ["--max-steps", "2"])
        assert proc.returncode == 0, (
            f"CLI failed (exit {proc.returncode})\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}")

        ckpt_path = os.path.join(td, "out", "minimal_jepa_vicreg_latest.pt")
        assert os.path.exists(ckpt_path), f"checkpoint not created: {ckpt_path}"

        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in obj]
        assert not missing, f"checkpoint missing schema keys: {missing}"
        # Two optimizer steps ran (global_step is the last completed step index)
        assert obj["step"] == 1, f"expected last completed step index 1, got {obj['step']}"
        assert obj["artifact_type"] == "final"
        assert obj["is_epoch_end"] is True  # 4 batches of 8 -> exactly one flushed epoch


if __name__ == "__main__":
    test_milestone_b_full_pipeline_contract()
    test_production_cli_smoke_max_steps_2()