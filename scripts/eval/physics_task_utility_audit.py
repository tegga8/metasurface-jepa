#!/usr/bin/env python3
"""
Physics task-utility audit for Metasurface-JEPA.

Trains the existing jepa_vicreg setup for a controlled short trajectory and,
at fixed intervals, evaluates the SAME held-out geometry + SAME mask + SAME
target under:
    real spectrum
    null spectrum
    shuffled spectrum

Primary metric:
    raw masked JEPA cosine loss = 1 - cosine(z_hat, z_y_raw)

Utility gaps:
    gap_null = L_null - L_real
    gap_shuffled = L_shuffled - L_real

Positive gap => the real spectrum helps the masked-prediction task.

This is deliberately separate from VICReg projected-space diagnostics.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from assembly import build_model
from data.dataset import MetaDiTDataset, collate_batch
from data.mask import BlockMasker
from losses.objectives import build_objective


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def raw_masked_cosine_loss(out):
    z_hat = out["z_hat"]
    z_y = out["z_y_raw"]
    mask = out["mask"]
    d = 1.0 - F.cosine_similarity(z_hat, z_y, dim=-1)
    vals = d[mask]
    if vals.numel() == 0:
        raise RuntimeError("Mask selected zero target tokens.")
    return vals.mean()


@torch.no_grad()
def evaluate_conditionality(model, fixed_batches, device):
    was_training = model.training
    model.eval()

    real_losses, null_losses, shuf_losses = [], [], []
    real_null_sens, real_shuf_sens = [], []

    for G, S, M in fixed_batches:
        G = G.to(device, non_blocking=True)
        S = S.to(device, non_blocking=True)
        M = M.to(device, non_blocking=True)

        # Deterministic permutation; ensure no sample keeps its own spectrum.
        if G.shape[0] == 1:
            perm = torch.tensor([0], device=device)
        else:
            perm = torch.roll(torch.arange(G.shape[0], device=device), shifts=1)
        S_shuf = S[perm]

        out_real = model(G, S, M, goal_mode="real")
        out_null = model(G, S, M, goal_mode="null")
        out_shuf = model(G, S_shuf, M, goal_mode="real")

        real_losses.append(raw_masked_cosine_loss(out_real).item())
        null_losses.append(raw_masked_cosine_loss(out_null).item())
        shuf_losses.append(raw_masked_cosine_loss(out_shuf).item())

        mask = out_real["mask"]
        real_null_sens.append(
            (out_real["z_hat"] - out_null["z_hat"]).norm(dim=-1)[mask].mean().item()
        )
        real_shuf_sens.append(
            (out_real["z_hat"] - out_shuf["z_hat"]).norm(dim=-1)[mask].mean().item()
        )

    if was_training:
        model.train()

    real = float(np.mean(real_losses))
    null = float(np.mean(null_losses))
    shuf = float(np.mean(shuf_losses))

    return {
        "raw_jepa_loss_real": real,
        "raw_jepa_loss_null": null,
        "raw_jepa_loss_shuffled": shuf,
        "utility_gap_null": null - real,
        "utility_gap_shuffled": shuf - real,
        "predictor_sensitivity_real_vs_null": float(np.mean(real_null_sens)),
        "predictor_sensitivity_real_vs_shuffled": float(np.mean(real_shuf_sens)),
    }


class CosineWarmup:
    def __init__(self, base_lr, warmup_steps, total_steps):
        self.base_lr = float(base_lr)
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))

    def lr(self, step):
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / max(1, self.warmup_steps)
        t = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        t = min(1.0, max(0.0, t))
        return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--report-every", type=int, default=200)
    p.add_argument("--subset", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--mask-seed", type=int, default=12345)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--objective", default="jepa_vicreg")
    p.add_argument(
        "--out-dir",
        default="checkpoints/milestone_b/physics_task_utility",
    )
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    set_seed(args.seed)

    spec = REPO_ROOT / cfg["weights"]["spectrum"]
    metadit = REPO_ROOT / cfg["weights"]["metadit"]
    train_path = REPO_ROOT / cfg["data"]["train_split"]
    val_path = REPO_ROOT / cfg["data"]["val_split"]

    for path in (spec, metadit, train_path, val_path):
        if not path.exists():
            raise FileNotFoundError(path)
        print("FOUND:", path)

    model = build_model(
        cfg["model"],
        str(spec),
        device=device,
        init_from_metadit=cfg["model"].get("init_from_metadit", True),
        metadit_weights=str(metadit),
    )

    objective = build_objective(
        args.objective,
        cfg.get("objective_params", {}).get(args.objective, {}),
        projector_input_dim=cfg["model"].get("hidden", 384),
    ).to(device)

    train_ds = MetaDiTDataset(
        str(train_path),
        max_samples=cfg["data"].get("max_train_samples", 8192),
        seed=args.seed,
    )
    val_ds = MetaDiTDataset(str(val_path))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg["data"].get("num_workers", 0),
        collate_fn=collate_batch,
    )

    masker = BlockMasker(
        placement="random",
        grid=16,
        min_side=cfg["mask"].get("min_side", 3),
        k_range=tuple(cfg["mask"].get("k_range", [1, 4])),
        seed=args.mask_seed,
    )

    # Fixed validation set: same geometry + same mask for all interventions.
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    fixed_batches = []
    seen = 0
    val_iter = iter(val_loader)
    while seen < args.subset:
        try:
            G, S = next(val_iter)
        except StopIteration:
            break
        keep = min(G.shape[0], args.subset - seen)
        if keep <= 0:
            break
        G, S = G[:keep].clone(), S[:keep].clone()
        M = masker.sample(G, args.mask_ratio).cpu()
        fixed_batches.append((G, S, M))
        seen += keep


    if seen < 2:
        raise RuntimeError("Need at least 2 fixed validation samples.")

    trainable = [
        p for p in model.parameters() if p.requires_grad
    ] + [
        p for p in objective.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["wd"]),
        betas=(0.9, 0.999),
    )

    ema_ids = {id(p) for p in model.ema.parameters()}
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    if ema_ids & opt_ids:
        raise RuntimeError("EMA parameters leaked into optimizer.")

    sched = CosineWarmup(
        cfg["train"]["lr"],
        cfg["train"].get("warmup_steps", 0),
        args.steps,
    )

    history = []
    model.train()
    objective.train()
    optimizer.zero_grad(set_to_none=True)

    train_iter = iter(train_loader)

    print("\nStarting physics TASK-UTILITY audit...\n")

    for step in range(args.steps):
        try:
            G, S = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            G, S = next(train_iter)

        G = G.to(device, non_blocking=True)
        S = S.to(device, non_blocking=True)
        M = masker.sample(G, args.mask_ratio).to(device)

        res = objective(model, G, S, M)
        total = res["total_loss"]

        if not torch.isfinite(total):
            raise RuntimeError(
                f"Non-finite VICReg loss at step {step}: {total.item()}"
            )

        total.backward()

        leaked = [
            n for n, p in model.ema.named_parameters()
            if p.grad is not None
        ]
        if leaked:
            raise RuntimeError(f"EMA received gradients: {leaked[:5]}")

        torch.nn.utils.clip_grad_norm_(
            trainable,
            float(cfg["train"].get("clip_grad_norm", 1.0)),
        )

        lr_now = sched.lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        objective.on_optimizer_step(model, step)

        if (
            step == 0
            or (step + 1) % args.report_every == 0
            or step + 1 == args.steps
        ):
            cond = evaluate_conditionality(model, fixed_batches, device)
            row = {
                "step": step + 1,
                "train_vicreg_total": float(total.item()),
                "lr": float(lr_now),
                **cond,
            }
            history.append(row)
            print(json.dumps(row))

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "physics_task_utility_audit.json"

    report = {
        "objective": args.objective,
        "steps": args.steps,
        "report_every": args.report_every,
        "subset": seen,
        "batch_size": args.batch_size,
        "mask_ratio": args.mask_ratio,
        "history": history,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n-> {path}")
    print("\nFINAL TASK-UTILITY RESULT")
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
