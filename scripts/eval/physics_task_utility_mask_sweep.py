#!/usr/bin/env python3
"""
Physics utility vs masking-ratio audit.

One VICReg training run, then evaluate the SAME trained model on a fixed
validation subset at mask ratios:
    0.25, 0.50, 0.75, 1.00

For every ratio, compare:
    real spectrum
    null spectrum
    shuffled spectrum

Primary metric:
    raw masked JEPA cosine loss = 1 - cosine(z_hat, z_y_raw)

Positive:
    utility_gap_null     = L_null - L_real
    utility_gap_shuffled = L_shuffled - L_real
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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def raw_masked_cosine_loss(out):
    z_hat = out["z_hat"]
    z_y = out["z_y_raw"]
    mask = out["mask"]
    d = 1.0 - F.cosine_similarity(z_hat, z_y, dim=-1)
    return d[mask].mean()


@torch.no_grad()
def evaluate_ratio(model, batches, device):
    was_training = model.training
    model.eval()

    real, null, shuf = [], [], []
    s_null, s_shuf = [], []

    for G, S, M in batches:
        G = G.to(device, non_blocking=True)
        S = S.to(device, non_blocking=True)
        M = M.to(device, non_blocking=True)

        if G.shape[0] > 1:
            perm = torch.roll(torch.arange(G.shape[0], device=device), 1)
        else:
            perm = torch.zeros(1, dtype=torch.long, device=device)

        S_shuf = S[perm]

        out_r = model(G, S, M, goal_mode="real")
        out_n = model(G, S, M, goal_mode="null")
        out_s = model(G, S_shuf, M, goal_mode="real")

        real.append(raw_masked_cosine_loss(out_r).item())
        null.append(raw_masked_cosine_loss(out_n).item())
        shuf.append(raw_masked_cosine_loss(out_s).item())

        mask = out_r["mask"]
        s_null.append(
            (out_r["z_hat"] - out_n["z_hat"]).norm(dim=-1)[mask].mean().item()
        )
        s_shuf.append(
            (out_r["z_hat"] - out_s["z_hat"]).norm(dim=-1)[mask].mean().item()
        )

    if was_training:
        model.train()

    lr = float(np.mean(real))
    ln = float(np.mean(null))
    ls = float(np.mean(shuf))

    return {
        "L_real": lr,
        "L_null": ln,
        "L_shuffled": ls,
        "gap_null": ln - lr,
        "gap_shuffled": ls - lr,
        "sensitivity_real_null": float(np.mean(s_null)),
        "sensitivity_real_shuffled": float(np.mean(s_shuf)),
    }


class CosineWarmup:
    def __init__(self, base_lr, warmup, total):
        self.base = float(base_lr)
        self.warmup = max(0, int(warmup))
        self.total = max(1, int(total))

    def __call__(self, step):
        if step < self.warmup:
            return self.base * (step + 1) / max(1, self.warmup)
        t = min(1.0, max(0.0, (step - self.warmup) /
                           max(1, self.total - self.warmup)))
        return self.base * 0.5 * (1.0 + math.cos(math.pi * t))


def collect_fixed_validation(val_loader, masker, subset, ratio, device):
    batches = []
    seen = 0
    it = iter(val_loader)

    while seen < subset:
        try:
            G, S = next(it)
        except StopIteration:
            break

        keep = min(G.shape[0], subset - seen)
        if keep <= 0:
            break

        G = G[:keep].clone()
        S = S[:keep].clone()
        M = masker.sample(G, ratio).cpu()

        batches.append((G, S, M))
        seen += keep

    if seen < subset:
        raise RuntimeError(
            f"Could only collect {seen}/{subset} validation samples."
        )
    return batches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--subset", type=int, default=32)
    p.add_argument("--report-every", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--objective", default="jepa_vicreg")
    p.add_argument("--mask-seed", type=int, default=12345)
    p.add_argument(
        "--ratios", nargs="+", type=float,
        default=[0.25, 0.50, 0.75, 1.00]
    )
    p.add_argument(
        "--out-dir",
        default="checkpoints/milestone_b/physics_mask_sweep"
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
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    # Prepare one fixed validation subset per mask ratio.
    fixed_by_ratio = {}
    for ratio in args.ratios:
        masker = BlockMasker(
            placement="random",
            grid=16,
            min_side=cfg["mask"].get("min_side", 3),
            k_range=tuple(cfg["mask"].get("k_range", [1, 4])),
            seed=args.mask_seed + int(round(ratio * 1000)),
        )
        fixed_by_ratio[ratio] = collect_fixed_validation(
            val_loader, masker, args.subset, ratio, device
        )

    train_masker = BlockMasker(
        placement="random",
        grid=16,
        min_side=cfg["mask"].get("min_side", 3),
        k_range=tuple(cfg["mask"].get("k_range", [1, 4])),
        seed=args.mask_seed,
    )

    params = (
        [p for p in model.parameters() if p.requires_grad] +
        [p for p in objective.parameters() if p.requires_grad]
    )

    optimizer = torch.optim.AdamW(
        params,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["wd"]),
        betas=(0.9, 0.999),
    )

    ema_ids = {id(p) for p in model.ema.parameters()}
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    if ema_ids & opt_ids:
        raise RuntimeError("EMA parameters leaked into optimizer.")

    schedule = CosineWarmup(
        cfg["train"]["lr"],
        cfg["train"].get("warmup_steps", 0),
        args.steps,
    )

    print("\nTraining one VICReg model, then sweeping mask ratios...\n")

    model.train()
    objective.train()
    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)
    train_history = []

    for step in range(args.steps):
        try:
            G, S = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            G, S = next(train_iter)

        G = G.to(device, non_blocking=True)
        S = S.to(device, non_blocking=True)
        M = train_masker.sample(G, 0.5).to(device)

        res = objective(model, G, S, M)
        loss = res["total_loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")

        loss.backward()

        if any(p.grad is not None for p in model.ema.parameters()):
            raise RuntimeError("EMA received gradients.")

        torch.nn.utils.clip_grad_norm_(
            params,
            float(cfg["train"].get("clip_grad_norm", 1.0)),
        )

        lr = schedule(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        objective.on_optimizer_step(model, step)

        if (
            step == 0
            or (step + 1) % args.report_every == 0
            or step + 1 == args.steps
        ):
            train_history.append({
                "step": step + 1,
                "vicreg_loss": float(loss.item()),
                "lr": float(lr),
            })
            print(json.dumps(train_history[-1]))

    sweep = {}
    for ratio, batches in fixed_by_ratio.items():
        result = evaluate_ratio(model, batches, device)
        sweep[str(ratio)] = result
        print(f"\nMASK {ratio:.2f}")
        print(json.dumps(result, indent=2))

    report = {
        "objective": args.objective,
        "training_steps": args.steps,
        "train_mask_ratio": 0.5,
        "eval_ratios": args.ratios,
        "subset": args.subset,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "mask_seed": args.mask_seed,
        "train_history": train_history,
        "sweep": sweep,
    }

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "physics_mask_sweep.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n-> {out_path}")
    print("\nFINAL SWEEP")
    print(json.dumps(sweep, indent=2))


if __name__ == "__main__":
    main()
