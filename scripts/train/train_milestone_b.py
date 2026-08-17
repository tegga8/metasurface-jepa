"""Milestone B — vanilla deterministic JEPA training / eval driver (§7 Phase 2, §7.2).

Run inside `notebooks/cloud_train_runner.ipynb` per CLOUD_TRAINING.md:

    python scripts/train/train_milestone_b.py --config configs/milestone_b.yaml
    python scripts/train/train_milestone_b.py --config configs/milestone_b.yaml \
        --experiment sweep --model-variant direct --resume checkpoints/milestone_b/latest.pt
    python scripts/train/train_milestone_b.py --config configs/milestone_b.yaml \
        --eval-only --null-goal --resume checkpoints/milestone_b/latest.pt

Standalone CLI, resumable, checkpoints every epoch (or ckpt_every_steps). The minimal
experiment (§7.2): fixed 50% block-masked context, random block placement only, L = L_J
(variant jepa) — compared against (a) the direct masked generator (variant direct,
masked-pixel L1, no JEPA latent objective) and (c) the JEPA model with the goal replaced
by a null token (eval-time intervention, --null-goal).
"""

import argparse
import json
import math
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import MetaDiTDataset, collate_batch
from data.mask import BlockMasker
from diagnostics.goal_token_entropy import goal_token_entropy
from assembly import build_model, load_into_model, saveable_state_dict

METADIT_SRC = os.path.join(REPO_ROOT, "external", "metadit")
if METADIT_SRC not in sys.path:
    sys.path.insert(0, METADIT_SRC)

from model.dit import DIT_MODEL  # noqa: E402
from model.surrogate import surrogate_s3  # noqa: E402
from diffusion import create_diffusion  # noqa: E402

PIXEL_GRID = 16


# ---------------------------------------------------------------------------
# schedule / helpers
# ---------------------------------------------------------------------------

class CosineWarmup:
    """Cosine decay to 0 with linear warmup; deterministic from step (resume-safe)."""

    def __init__(self, base_lr, warmup_steps, total_steps):
        self.base_lr = base_lr
        self.warmup = warmup_steps
        self.total = total_steps

    def get_lr(self, step):
        if step < self.warmup:
            return self.base_lr * (step + 1) / max(1, self.warmup)
        t = (step - self.warmup) / max(1, self.total - self.warmup)
        return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_surrogate(path, device):
    m = surrogate_s3()
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_released_dit(path, device):
    m = DIT_MODEL["metadit_s"](diffusion=create_diffusion("500", learn_sigma=False),
                               condition_channel=301)
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def released_vit_embeddings(dit, g_full, block=11):
    """Released-DiT embedding of the 32x32 quadrant: returns (B, 256, 384) block features."""
    b = g_full.shape[0]
    quad = g_full[:, :, :32, :32]
    y = torch.zeros(b, 2, 301, device=quad.device)
    t = torch.zeros(b, dtype=torch.long, device=quad.device)
    with torch.no_grad():
        x = dit.x_embedder(quad) + dit.pos_embed
        t_emb = dit.t_embedder(t)
        y_emb = dit.y_embedder(y, train=False)
        c = t_emb + y_emb.mean(1)
        for li, blk in enumerate(dit.blocks):
            x = blk(x, c, y_emb)
        return x


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _eval_ratio_masks(n_batches, batch_size, ratio, device, grid=PIXEL_GRID,
                      min_side=3, k_range=(1, 4)):
    """Deterministic random-placement masks for reproducible eval (own RNG seed)."""
    ev = BlockMasker(placement="random", grid=grid, min_side=min_side, k_range=k_range)
    ev.rng.manual_seed(12345)
    masks = []
    for _ in range(n_batches):
        masks.append(ev.sample(torch.zeros(batch_size, 3, 64, 64), ratio).to(device))
    return masks


def evaluate_jepa(model, loader, masker, ratios, val_batches, device, need_null=True):
    """Masked-position latent cosine error vs EMA target; goal-token entropy; + null-goal
    gap (the §7.2 cheap proxy for goal-ignoring collapse)."""
    model.eval()
    if isinstance(ratios, float):
        ratios = [ratios]
    agg = {r: {"cos_err": 0.0, "count": 0} for r in ratios}
    ent_sum, ent_count = 0.0, 0
    null_agg = {r: {"cos_err": 0.0, "gap": 0.0, "count": 0} for r in ratios}
    n_batches = min(val_batches, len(loader))
    with torch.no_grad():
        for bi, (G, S) in enumerate(loader):
            if bi >= n_batches:
                break
            G, S = G.to(device), S.to(device)
            for ratio in ratios:
                M = _eval_ratio_masks(1, G.shape[0], ratio, device,
                                      PIXEL_GRID, masker.min_side, masker.k_range)[0]
                mask = (M.view(M.shape[0], -1) == 0)
                out = model(G, S, M, need_attn=True)
                d = (1.0 - torch.nn.functional.cosine_similarity(
                    torch.nn.functional.normalize(out["z_hat"], dim=-1),
                    torch.nn.functional.normalize(out["z_y"], dim=-1), dim=-1)).clamp(min=0)
                d_masked = d[mask]
                agg[ratio]["cos_err"] += d_masked.mean().item()
                agg[ratio]["count"] += 1
                if out["attn_weights"]:
                    h, _ = goal_token_entropy(out["attn_weights"], mask)
                    ent_sum += h.item()
                    ent_count += 1
                if need_null:
                    out_n = model(G, S, M, goal_mode="null")
                    d_n = (1.0 - torch.nn.functional.cosine_similarity(
                        torch.nn.functional.normalize(out_n["z_hat"], dim=-1),
                        torch.nn.functional.normalize(out["z_y"], dim=-1), dim=-1)).clamp(min=0)
                    d_n_masked = d_n[mask]
                    gap = (out["z_hat"] - out_n["z_hat"]).norm(dim=-1)[mask].mean()
                    null_agg[ratio]["cos_err"] += d_n_masked.mean().item()
                    null_agg[ratio]["gap"] += gap.item()
                    null_agg[ratio]["count"] += 1
    metrics = {}
    for r in ratios:
        c = agg[r]["count"] or 1
        metrics[f"cos_err_r{r:g}"] = agg[r]["cos_err"] / c
        if need_null:
            nc = null_agg[r]["count"] or 1
            metrics[f"null_cos_err_r{r:g}"] = null_agg[r]["cos_err"] / nc
            metrics[f"null_gap_r{r:g}"] = null_agg[r]["gap"] / nc
    metrics["goal_token_entropy"] = ent_sum / max(1, ent_count)
    metrics["goal_token_log_entropy"] = math.log(max(metrics["goal_token_entropy"], 1e-9))
    return metrics


def evaluate_direct(model, loader, masker, ratios, val_batches, device, surrogate=None,
                    released_dit=None):
    """Baseline-2 metrics: masked-pixel L1, full-image L1, frozen-surrogate physics MAE,
    and the released-ViT latent axis (global + masked-region cosine error)."""
    model.eval()
    if isinstance(ratios, float):
        ratios = [ratios]
    n_batches = min(val_batches, len(loader))
    agg = {r: {"px_masked": 0.0, "px_full": 0.0, "phys": 0.0, "vit_global": 0.0,
               "vit_masked": 0.0, "count": 0} for r in ratios}
    with torch.no_grad():
        for bi, (G, S) in enumerate(loader):
            if bi >= n_batches:
                break
            G, S = G.to(device), S.to(device)
            for ratio in ratios:
                M = _eval_ratio_masks(masker, 1, G.shape[0], ratio, device)[0]
                out = model(G, S, M)
                g_hat = out["g_hat"]
                up = M.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2).unsqueeze(1)
                diff = (g_hat - G).abs()
                agg[ratio]["px_masked"] += diff[up == 0].mean().item()
                agg[ratio]["px_full"] += diff.mean().item()
                if surrogate is not None:
                    s_hat = surrogate(g_hat).prediction
                    agg[ratio]["phys"] += (s_hat - S).abs().mean().item()
                if released_dit is not None:
                    z_hat = released_vit_embeddings(released_dit, g_hat)
                    z_ref = released_vit_embeddings(released_dit, G)
                    sim = torch.nn.functional.cosine_similarity(
                        torch.nn.functional.normalize(z_hat, dim=-1),
                        torch.nn.functional.normalize(z_ref, dim=-1), dim=-1)
                    agg[ratio]["vit_global"] += (1.0 - sim).clamp(min=0).mean().item()
                    rmask = (M[:, None, None, :, :] == 0).expand(
                        -1, 2, 2, -1, -1).reshape(G.shape[0], PIXEL_GRID, PIXEL_GRID)
                    rmask = rmask.reshape(G.shape[0], -1)
                    d = (1.0 - sim).clamp(min=0)
                    agg[ratio]["vit_masked"] += d[rmask].mean().item()
                agg[ratio]["count"] += 1
    metrics = {}
    for r in ratios:
        c = agg[r]["count"] or 1
        for k in ("px_masked", "px_full", "phys", "vit_global", "vit_masked"):
            metrics[f"{k}_r{r:g}"] = agg[r][k] / c
    return metrics


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, step, epoch, cfg, best, extra=None):
    obj = {
        "step": step, "epoch": epoch, "cfg": cfg, "best": best,
        "model": saveable_state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
    }
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    # own training artifacts (weights + optimizer states); weights_only=False for
    # PyTorch >= 2.6 default change
    obj = torch.load(path, map_location="cpu", weights_only=False)
    load_into_model(model, obj["model"], device)
    if optimizer is not None and obj.get("optimizer"):
        optimizer.load_state_dict(obj["optimizer"])
    for p in optimizer.param_groups:
        p["lr"] = scheduler.get_lr(obj["step"]) if scheduler else p["lr"]
    if obj.get("torch_rng") is not None:
        torch.set_rng_state(obj["torch_rng"].cpu())
    if obj.get("numpy_rng") is not None:
        np.random.set_state(obj["numpy_rng"])
    return obj


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Milestone B JEPA training")
    p.add_argument("--config", required=True)
    p.add_argument("--experiment", choices=["minimal", "sweep"], default=None)
    p.add_argument("--model-variant", choices=["jepa", "direct"], default=None)
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    p.add_argument("--eval-only", action="store_true", help="run eval from a checkpoint and exit")
    p.add_argument("--null-goal", action="store_true", help="eval with goal replaced by null")
    p.add_argument("--smoke", action="store_true", help="tiny local crash-test run")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    exp = args.experiment or cfg["experiment"]
    cfg["experiment"] = exp
    exp_cfg = cfg[exp]
    ratios = (exp_cfg["mask_ratio"] if exp == "minimal"
              else exp_cfg["mask_ratios"])
    placement = exp_cfg["mask_placement"]

    if args.model_variant:
        cfg["model"]["variant"] = args.model_variant
    variant = cfg["model"]["variant"]

    if args.smoke:  # local dev-only crash test (batch 1, handful of steps)
        cfg["train"].update(batch_size=1, grad_accum=1, epochs=1, val_batches=1,
                            val_every_steps=1, log_every_steps=1, save_optimizer=False,
                            warmup_steps=0, max_steps=3, ckpt_every_steps=1)
        cfg["data"]["max_train_samples"] = 8
        cfg["data"]["num_workers"] = 0

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(cfg["train"].get("device", "cuda"))
    else:
        device = torch.device("cpu")
    if args.smoke:
        device = torch.device("cpu")

    seed = args.seed if args.seed is not None else cfg["train"]["seed"]
    set_seed(seed)

    out_dir = os.path.join(REPO_ROOT, cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    smoke_tag = "smoke" if args.smoke else variant
    run_tag = f"{exp}_{smoke_tag}"

    # data
    ds = MetaDiTDataset(os.path.join(REPO_ROOT, cfg["data"]["train_split"]),
                        max_samples=cfg["data"].get("max_train_samples", 0), seed=seed)
    val_ds = MetaDiTDataset(os.path.join(REPO_ROOT, cfg["data"]["val_split"]))
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        num_workers=cfg["data"].get("num_workers", 0),
                        drop_last=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=0, collate_fn=collate_batch)

    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * cfg["train"]["epochs"]

    # masker + frozen released components
    masker = BlockMasker(placement=placement, grid=PIXEL_GRID,
                         min_side=cfg["mask"].get("min_side", 3),
                         k_range=tuple(cfg["mask"].get("k_range", [1, 4])), seed=seed)
    surrogate = load_surrogate(os.path.join(REPO_ROOT, cfg["weights"]["surrogate"]),
                               device)
    released_dit = load_released_dit(os.path.join(REPO_ROOT, cfg["weights"]["metadit"]),
                                     device)

    # model
    model = build_model(cfg["model"],
                        os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
                        device=device,
                        init_from_metadit=cfg["model"].get("init_from_metadit", True),
                        metadit_weights=os.path.join(REPO_ROOT, cfg["weights"]["metadit"]))
    if variant == "jepa":
        model.ema.set_total_steps(total_steps)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in trainable)
    print(f"[milestone_b] variant={variant} exp={exp} placement={placement} "
          f"ratios={ratios} device={device} trainable_params={n_param:,} "
          f"steps_per_epoch={steps_per_epoch} total_steps={total_steps}")

    # optimizer + schedule
    opt_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(trainable, lr=float(opt_cfg["lr"]),
                                  weight_decay=float(opt_cfg["wd"]), betas=(0.9, 0.999))
    scheduler = CosineWarmup(float(opt_cfg["lr"]), int(opt_cfg["warmup_steps"]),
                             total_steps)

    start_step, start_epoch, best = 0, 0, {}
    ckpt_path = os.path.join(out_dir, f"{run_tag}_latest.pt")
    if args.resume:
        obj = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_step, start_epoch, best = obj["step"] + 1, obj["epoch"], obj.get("best", {})
        print(f"[milestone_b] resumed from {args.resume}: step={start_step} epoch={start_epoch}")

    if args.eval_only:
        metrics = run_eval(model, val_loader, masker, surrogate, released_dit, ratios,
                           cfg["train"]["val_batches"], device,
                           null_goal=args.null_goal, variant=variant)
        tag = "eval" + ("_null_goal" if args.null_goal else "")
        eval_path = os.path.join(out_dir, f"{run_tag}_{tag}_metrics.json")
        with open(eval_path, "w") as f:
            json.dump({"config": cfg, "metrics": metrics}, f, indent=2)
        print(f"[milestone_b] eval-only done -> {eval_path}")
        print(json.dumps(metrics, indent=2))
        return

    # ---- training loop ----
    print(f"[milestone_b] training ({exp}) ...")
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    t_start = time.time()
    accum = opt_cfg.get("grad_accum", 1)
    loss_accum = 0.0

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        for bi, (G, S) in enumerate(loader):
            G, S = G.to(device), S.to(device)
            ratio = ratios[step % len(ratios)] if isinstance(ratios, list) else ratios
            M = masker.sample(G, ratio, surrogate if placement == "half_sensitivity"
                              else None).to(device)

            loss, out = model.loss(G, S, M)
            loss = loss / accum
            loss.backward()
            loss_accum += loss.item() * accum

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, opt_cfg["clip_grad_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if variant == "jepa":
                    model.ema.update(model.geometry_encoder, step)
                for g in optimizer.param_groups:
                    g["lr"] = scheduler.get_lr(step)

            if step % opt_cfg["log_every_steps"] == 0:
                lr_now = scheduler.get_lr(step)
                frac_masked = 1.0 - M.mean().item()
                print(f"  [step {step}] loss={loss_accum:.4f} lr={lr_now:.2e} "
                      f"mask_frac={frac_masked:.2f} "
                      f"({(time.time() - t_start) / max(1, step + 1 - start_step):.2f} s/step)")
                loss_accum = 0.0

            if step % opt_cfg["val_every_steps"] == 0 or (
                    cfg["train"].get("ckpt_every_steps", 0)
                    and step % cfg["train"]["ckpt_every_steps"] == 0):
                val_metrics = run_eval(model, val_loader, masker, surrogate, released_dit,
                                       ratios, cfg["train"]["val_batches"], device,
                                       null_goal=False, variant=variant)
                primary = (val_metrics.get("cos_err_r0.5") if variant == "jepa"
                           else val_metrics.get("px_masked_r0.5", 0.0)) or 0.0
                if not best or primary < best.get("primary", float("inf")):
                    best = {"primary": primary, "metrics": val_metrics, "step": step}
                    torch.save(saveable_state_dict(model),
                               os.path.join(out_dir, f"{run_tag}_best_model.pt"))
                print(f"  [val @ step {step}] {json.dumps(val_metrics)}")

            ckpt_now = (cfg["train"].get("ckpt_every_steps", 0)
                        and step % cfg["train"]["ckpt_every_steps"] == 0) or \
                (bi == len(loader) - 1)
            if ckpt_now:
                save_checkpoint(ckpt_path, model, optimizer if opt_cfg.get(
                    "save_optimizer", True) else None, step, epoch, cfg, best)
                print(f"  [ckpt] saved {ckpt_path} (step {step})")

            step += 1
            if args.smoke and (step - start_step) >= cfg["train"].get("max_steps", 0) \
                    and cfg["train"].get("max_steps"):
                break
        else:
            continue
        break  # smoke early-break

    # final eval
    final = run_eval(model, val_loader, masker, surrogate, released_dit, ratios,
                     cfg["train"]["val_batches"], device, null_goal=False,
                     variant=variant)
    if variant == "jepa":
        final.update(run_eval(model, val_loader, masker, surrogate, released_dit, ratios,
                              cfg["train"]["val_batches"], device, null_goal=True,
                              variant=variant))
        final["null_gap"] = final.get("null_gap_r0.5", float("nan"))
    final_path = os.path.join(out_dir, f"{run_tag}_final_metrics.json")
    with open(final_path, "w") as f:
        json.dump({"config": cfg, "metrics": final, "best": best}, f, indent=2)
    print(f"[milestone_b] final metrics -> {final_path}")
    print(json.dumps(final, indent=2))
    save_checkpoint(ckpt_path, model, optimizer if opt_cfg.get("save_optimizer", True)
                    else None, step - 1, epoch, cfg, best)


def run_eval(model, val_loader, masker, surrogate, released_dit, ratios, val_batches,
             device, null_goal=False, variant="jepa"):
    if variant == "jepa":
        return evaluate_jepa(model, val_loader, masker, ratios, val_batches, device,
                             need_null=null_goal)
    return evaluate_direct(model, val_loader, masker, ratios, val_batches, device,
                           surrogate=surrogate, released_dit=released_dit)


if __name__ == "__main__":
    main()