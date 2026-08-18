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
    """Deterministic random-placement masks for the DIRECT baseline eval (own RNG seed).

    NOTE (2026-08-17, audit fix): kept ONLY for the pixel-space direct baseline, where
    there is no latent FixedValidation equivalent. The legacy JEPA evaluator that used
    this helper has been removed — all JEPA evaluation (adaptive and legacy paths) now
    runs through FixedValidation (see _jepa_fixed_val_metrics).
    """
    ev = BlockMasker(placement="random", grid=grid, min_side=min_side, k_range=k_range)
    ev.rng.manual_seed(12345)
    masks = []
    for _ in range(n_batches):
        masks.append(ev.sample(torch.zeros(batch_size, 3, 64, 64), ratio).to(device))
    return masks


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
                M = _eval_ratio_masks(1, G.shape[0], ratio, device)[0]
                out = model(G, S, M)
                g_hat = out["g_hat"]
                pmask = M.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2)
                up = pmask.unsqueeze(1).expand_as(g_hat)
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
                    rmask = (M == 0).reshape(G.shape[0], -1)
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
        # Bug #17: CUDA RNG saved too (None on CPU-only machines, skipped safely)
        "torch_cuda_rng": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
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
    cuda = obj.get("torch_cuda_rng")
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda])
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

    if cfg.get("adaptive_training", {}).get("enabled", False):
        seed = args.seed or int(time.time())
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        train_adaptive(cfg, args, device, seed)
        return

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
        fixed_vals, refs = _build_legacy_fixed_vals(
            cfg, device, val_ds, cfg["train"]["val_batches"], cfg["train"]["batch_size"],
            ratios, model, mask_seed=cfg["mask"].get("mask_seed", 12345))
    else:
        fixed_vals, refs = None, None

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
        if variant == "jepa":
            metrics = _jepa_fixed_val_metrics(model, fixed_vals, refs,
                                              need_null=args.null_goal)
        else:
            metrics = run_eval(model, val_loader, masker, surrogate, released_dit,
                               ratios, cfg["train"]["val_batches"], device,
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
                if variant == "jepa":
                    val_metrics = _jepa_fixed_val_metrics(model, fixed_vals, refs)
                else:
                    val_metrics = run_eval(model, val_loader, masker, surrogate,
                                           released_dit, ratios,
                                           cfg["train"]["val_batches"], device,
                                           null_goal=False, variant=variant)
                ratio0 = ratios[0] if isinstance(ratios, list) else ratios
                primary = (val_metrics.get(f"cos_err_r{ratio0:g}") if variant == "jepa"
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
    if variant == "jepa":
        final = _jepa_fixed_val_metrics(model, fixed_vals, refs, need_null=True)
        gaps = [final[k] for k in final if k.startswith("null_gap_r")
                and isinstance(final[k], float)]
        final["null_gap"] = sum(gaps) / len(gaps) if gaps else float("nan")
    else:
        final = run_eval(model, val_loader, masker, surrogate, released_dit, ratios,
                         cfg["train"]["val_batches"], device, null_goal=False,
                         variant=variant)
    final_path = os.path.join(out_dir, f"{run_tag}_final_metrics.json")
    with open(final_path, "w") as f:
        json.dump({"config": cfg, "metrics": final, "best": best}, f, indent=2)
    print(f"[milestone_b] final metrics -> {final_path}")
    print(json.dumps(final, indent=2))
    save_checkpoint(ckpt_path, model, optimizer if opt_cfg.get("save_optimizer", True)
                    else None, step - 1, epoch, cfg, best)


def run_eval(model, val_loader, masker, surrogate, released_dit, ratios, val_batches,
             device, null_goal=False, variant="jepa"):
    """Direct-variant evaluation only (pixel-space baseline; no latent FixedValidation
    equivalent). JEPA-variant evaluation is handled by _jepa_fixed_val_metrics — the
    legacy own-mask/need_attn JEPA evaluator was removed (audit fix 2026-08-17)."""
    assert variant == "direct", "JEPA evaluation must go through FixedValidation"
    return evaluate_direct(model, val_loader, masker, ratios, val_batches, device,
                           surrogate=surrogate, released_dit=released_dit)


def _jepa_fixed_val_metrics(model, fixed_vals, refs, need_null=False):
    """JEPA metrics via FixedValidation: one shared metric path for in-loop validation,
    eval-only, and final eval (audit fix 2026-08-17 — removes the legacy evaluator's
    own masks + need_attn=True divergence). fixed_vals: {ratio: FixedValidation},
    refs: {ratio: healthy_references dict}."""
    metrics = {}
    for r, fv in sorted(fixed_vals.items()):
        m, _ = fv.evaluate(model, refs[r]["raw"], refs[r]["proj"])
        metrics[f"cos_err_r{r:g}"] = m["cos_err_r0.5"]
        if need_null:
            real, null, gap = fv.null_gap(model)
            metrics[f"null_cos_err_r{r:g}"] = null
            metrics[f"null_gap_r{r:g}"] = gap
    return metrics


def _build_legacy_fixed_vals(cfg, device, val_ds, val_batches, batch_size, ratios,
                             model, mask_seed=12345):
    """Per-ratio FixedValidation sets for the legacy (non-adaptive) JEPA path, plus the
    corresponding healthy released-init references on the SAME masks/batches.

    model is the TRAINING candidate: its trainable proj head defines the
    projection-space coordinate system the healthy references are measured in
    (Bug #14 — the released-init reference's own proj is a different random
    initialization and must not be the comparison head)."""
    from train.engine import fixed_validation_from_loader, healthy_references
    n = max(1, min(val_batches * batch_size, len(val_ds)))
    ratios_list = ratios if isinstance(ratios, list) else [ratios]
    fixed_vals, refs = {}, {}
    refs_model = None
    for r in ratios_list:
        fv = fixed_validation_from_loader(val_ds, n, batch_size, device,
                                          ratio=float(r), mask_seed=mask_seed)
        if refs_model is None:
            refs_model = build_model(cfg["model"],
                                     os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
                                     device=device,
                                     init_from_metadit=cfg["model"].get("init_from_metadit", True),
                                     metadit_weights=os.path.join(REPO_ROOT,
                                                                  cfg["weights"]["metadit"]))
            refs_model.eval()
        fixed_vals[r] = fv
        refs[r] = healthy_references(refs_model, fv, proj_source=model)
    return fixed_vals, refs


# ---------------------------------------------------------------------------
# adaptive loss ladder (operator spec: "Milestone B — Adaptive Loss Switching")
# one training loop, pluggable objective, one health diagnostic, one controller
# ---------------------------------------------------------------------------

def _adaptive_smoke_overrides(acfg):
    """Tiny local crash-test config (batch 1, handful of steps, jepa only)."""
    return {**acfg, "max_total_steps": 6, "val_every_steps": 2, "log_every_steps": 1,
            "lr_warmup_steps": 0, "warmup_steps": 1, "plateau_patience": 1,
            "min_delta": 1e-6, "collapse_patience": 2, "fixed_val_subset": 4,
            "val_batch_size": 1, "objectives": ["jepa"], "batch_size": 1}


class _CosineWarmup:
    """Linear warmup then cosine decay over the global budget (per-phase schedule,
    deterministic in global step). Self-contained: no dependency on a scheduler
    helper outside this script."""

    def __init__(self, lr, warmup, total):
        self.lr = float(lr)
        self.warmup = max(0, int(warmup))
        self.total = max(1, int(total))

    def get_lr(self, step):
        if step < self.warmup:
            return self.lr * (step + 1) / max(1, self.warmup)
        t = (step - self.warmup) / max(1, self.total - self.warmup)
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))


def _objective_kwargs(acfg, name):
    params = dict(acfg.get("objective_params", {}).get(name, {}))
    if name == "jepa_vicreg":
        return {k: params.get(k, d) for k, d in
                (("lambda_var", 0.1), ("lambda_cov", 0.04), ("gamma", 1.0),
                 ("cov_on", True))}
    if name == "lejepa":
        return {k: params.get(k, d) for k, d in
                (("lambda_sigreg", 0.1), ("num_slices", 8), ("num_points", 256),
                 ("seed", 0))}
    return {}


def _normalized_repr_dist(health, refs, cfg):
    from diagnostics.representation_health import COLLAPSE_CFG_DEFAULTS
    c = dict(COLLAPSE_CFG_DEFAULTS, **(cfg or {}))
    raw, hr = health["raw"], refs["raw"]
    return (abs(raw["eff_rank_frac"] - hr["eff_rank_frac"]) / c["near_rank"]
            + abs(raw["pairwise_cos"]["p05"] - hr["pairwise_cos"]["p05"]) / c["near_p05"]
            + abs(raw["same_token_cos"] - hr["same_token_cos"]) / c["near_same"])


def train_adaptive(cfg, args, device, seed):
    """Run the adaptive loss ladder: jepa -> jepa_vicreg -> lejepa, switching only on
    plateau / collapse / instability evidence, within a global step budget."""
    from losses.objectives import OBJECTIVES
    from train.adaptive import AdaptiveController, select_winner
    from train.engine import (fixed_validation_from_loader, healthy_references,
                              IntervalLossAccumulator, load_phase_checkpoint,
                              save_phase_checkpoint, write_ladder_summary,
                              write_phase_report)
    acfg = dict(cfg["adaptive_training"])
    if args.smoke:
        acfg = _adaptive_smoke_overrides(acfg)
        cfg["data"]["max_train_samples"] = 8
        cfg["data"]["num_workers"] = 0
    objectives_order = list(acfg.get("objectives", ["jepa", "jepa_vicreg", "lejepa"]))
    for o in objectives_order:
        assert o in OBJECTIVES, f"unknown objective in ladder: {o}"
    assert cfg["model"]["variant"] == "jepa", "adaptive ladder requires variant jepa"

    out_dir = os.path.join(REPO_ROOT, cfg.get("out_dir", "checkpoints/milestone_b"),
                           "adaptive")
    if args.smoke:
        out_dir = os.path.join(out_dir, "_smoke")
    os.makedirs(out_dir, exist_ok=True)
    lr = float(acfg.get("lr", cfg.get("train", {}).get("lr", 1e-4)))
    wd = float(acfg.get("wd", cfg.get("train", {}).get("wd", 0.05)))
    accum = int(acfg.get("grad_accum", 1))
    batch_size = int(acfg.get("batch_size", cfg.get("train", {}).get("batch_size", 64)))
    val_every = int(acfg.get("val_every_steps", 100))
    log_every = int(acfg.get("log_every_steps", 25))
    lr_warmup = int(acfg.get("lr_warmup_steps", 200))
    clip = float(acfg.get("clip_grad_norm", cfg.get("train", {}).get("clip_grad_norm", 1.0)))
    fixed_n = int(acfg.get("fixed_val_subset", 512))
    val_bs = int(acfg.get("val_batch_size", 32))
    ratio = float(acfg.get("mask_ratio", 0.5))
    mask_seed = int(acfg.get("mask_seed", 12345))
    collapse_cfg = acfg.get("collapse", {})

    # model + immutable base initialization (B1+B2: released init, EMA resynced)
    model = build_model(cfg["model"],
                        os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
                        device=device,
                        init_from_metadit=cfg["model"].get("init_from_metadit", True),
                        metadit_weights=os.path.join(REPO_ROOT, cfg["weights"]["metadit"]))
    model.ema.set_total_steps(int(acfg["max_total_steps"]))
    base_path = os.path.join(out_dir, "base_initialization.pt")
    if os.path.exists(base_path):
        base_obj = torch.load(base_path, map_location="cpu", weights_only=False)
        load_into_model(model, base_obj["model"], device)
    else:
        torch.save({"model": saveable_state_dict(model),
                    "meta": {"synthetic": False, "base_init": True,
                             "objective": "jepa (immutable base)",
                             "b1_resync": True}}, base_path)
    print(f"[adaptive] base initialization -> {base_path}")

    # fixed validation set + masks (deterministic, shared across objectives)
    val_ds = MetaDiTDataset(os.path.join(REPO_ROOT, cfg["data"]["val_split"]))
    fixed_val = fixed_validation_from_loader(val_ds, fixed_n, val_bs, device,
                                             ratio=ratio, mask_seed=mask_seed,
                                             collapse_cfg=collapse_cfg)

    # healthy released-init references on the SAME fixed validation set.
    # Bug #14: projected through the CANDIDATE model's proj head (proj_source),
    # never refs_model.proj — the released-init reference's own head is a separate
    # random initialization, i.e. a different learned coordinate system.
    refs_model = build_model(cfg["model"],
                             os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
                             device=device,
                             init_from_metadit=cfg["model"].get("init_from_metadit", True),
                             metadit_weights=os.path.join(REPO_ROOT, cfg["weights"]["metadit"]))
    refs_model.ema.set_total_steps(int(acfg["max_total_steps"]))
    refs_model.eval()
    refs = healthy_references(refs_model, fixed_val, proj_source=model)

    ds = MetaDiTDataset(os.path.join(REPO_ROOT, cfg["data"]["train_split"]),
                        max_samples=cfg["data"].get("max_train_samples", 0), seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=cfg["data"].get("num_workers", 0),
                        drop_last=True, collate_fn=collate_batch)
    masker = BlockMasker(placement="random", grid=int(cfg["model"]["token_grid"]),
                         min_side=cfg["mask"].get("min_side", 3),
                         k_range=tuple(cfg["mask"].get("k_range", [1, 4])), seed=seed)

    controller = AdaptiveController(acfg, objectives_order)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in trainable)
    print(f"[adaptive] objectives={objectives_order} max_steps={acfg['max_total_steps']} "
          f"trainable_params={n_param:,} fixed_val={fixed_val.mask_statistics['n_samples']}")

    jsonl_path = os.path.join(out_dir, "ladder_training_diagnostics.jsonl")
    resumed = None
    if args.resume:
        resumed_meta = torch.load(args.resume, map_location="cpu",
                                  weights_only=False)["adaptive_meta"]
        resumed = {"phase_idx": resumed_meta["phase"], "objective": resumed_meta["objective"],
                   "global_step": resumed_meta["global_step"] + 1,
                   "best_metric": resumed_meta["best_metric"],
                   "best_step": resumed_meta["best_step"]}
        print(f"[adaptive] resume -> {args.resume} (phase {resumed['phase_idx']}, "
              f"global step {resumed['global_step']})")

    phase_reports = []
    global_step = resumed["global_step"] if resumed else 0
    phase_idx = 0
    resumed_phase_idx = resumed["phase_idx"] if resumed else -1
    phase_report_json = lambda i, o: os.path.join(
        out_dir, f"phase_{i:02d}_{o}_report.json")
    keep_running = True
    jf = open(jsonl_path, "w" if args.resume is None else "a")

    while keep_running and phase_idx < len(objectives_order):
        if resumed is not None and phase_idx < resumed_phase_idx:
            phase_idx += 1
            continue
        objective_name = objectives_order[phase_idx]
        if not controller.enough_budget(global_step):
            print(f"[adaptive] global budget exhausted at step {global_step} before "
                  f"phase {objective_name}")
            break

        phase = controller.start_phase(objective_name, global_step)
        objective = OBJECTIVES[objective_name](**_objective_kwargs(acfg, objective_name))
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd, betas=(0.9, 0.999))
        scheduler = _CosineWarmup(lr, lr_warmup, int(acfg["max_total_steps"]))
        start_g = global_step

        if resumed is not None and resumed["phase_idx"] == phase.idx:
            ckpt_obj = load_phase_checkpoint(args.resume, model, optimizer,
                                             scheduler, device)
            # Bug #12: restore the FULL controller state (patience counters,
            # histories, transitions, phase bookkeeping), not just the tuple.
            controller.load_state_dict(ckpt_obj["adaptive_meta"]["controller_state"])
            if controller.phase is not None:
                phase = controller.phase
            global_step = resumed["global_step"]
            start_g = global_step
            resumed = None  # resume consumed
            print(f"[adaptive] phase {phase.idx} resumed at global step {global_step}")

        # Bug #16: iterate the loader with explicit epochs, NEVER via
        # itertools.cycle: the cycle-wrapper caches and replays the FIRST
        # iteration's batches, so shuffle=True shuffled once and then never
        # again. A fresh `for G, S in loader:` per pass gives DataLoader its
        # normal per-epoch reshuffle. Budget checks at the top
        # of each epoch AND inside the epoch give exact global-step termination at
        # any budget (phase_done exits both loops).
        optimizer.zero_grad(set_to_none=True)
        # Bug #18: exact per-interval training-loss means — two independent
        # accumulators so a log report and a val record never steal each other's
        # interval; each is reset after its own report. Counts micro-batches, so
        # the mean is exact under grad_accum=1, grad_accum=2, and phase boundaries
        # (fresh instances per phase drop a partial interval cleanly).
        log_loss = IntervalLossAccumulator()
        val_loss = IntervalLossAccumulator()
        comp_sums = {}
        comp_counts = {}
        t_start = time.time()
        unstable_steps = 0
        decision = None
        best_health, best_repr_d = None, float("inf")
        if phase.best_health is not None:
            best_health = phase.best_health  # survive resume for the phase report (Bug #12)
        lr_history, ema_history = [], []
        mb = 0  # micro-batch counter; accum micro-batches == 1 optimizer step

        phase_done = False
        while controller.enough_budget(global_step) and not phase_done:
            for G, S in loader:
                if not controller.enough_budget(global_step):
                    phase_done = True
                    break
                G, S = G.to(device), S.to(device)
                M = masker.sample(G, ratio).to(device)
                res = objective(model, G, S, M)
                total = res["total_loss"]
                if not torch.isfinite(total):
                    unstable_steps += 1
                    print(f"  [adaptive] NON-FINITE loss at step {global_step} "
                          f"({objective_name})")
                    save_phase_checkpoint(
                        os.path.join(out_dir, f"phase_{phase.idx:02d}_{objective_name}_latest.pt"),
                        model, optimizer if acfg.get("save_optimizer", True) else None,
                        cfg, controller, phase, global_step, {"cos_err_r0.5": float("nan")},
                        {"status": "UNSTABLE"})
                    decision = controller.on_unstable(global_step)
                    phase_done = True
                    break
                total = total / accum
                total.backward()
                mb += 1
                log_loss.add(total.item() * accum)
                val_loss.add(total.item() * accum)
                for k, v in res["components"].items():
                    if isinstance(v, torch.Tensor):
                        comp_sums[k] = comp_sums.get(k, 0.0) + v.item()
                        comp_counts[k] = comp_counts.get(k, 0) + 1

                if mb % accum != 0:
                    continue  # still accumulating this optimizer step (Bug #10: global
                              # step counts optimizer steps, not micro-batches)
                torch.nn.utils.clip_grad_norm_(trainable, clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                objective.on_optimizer_step(model, global_step)
                for g in optimizer.param_groups:
                    g["lr"] = scheduler.get_lr(global_step)

                if global_step % log_every == 0:
                    print(f"  [adaptive {objective_name} step {global_step}] "
                          f"loss={log_loss.report():.4f} "
                          f"lr={scheduler.get_lr(global_step):.2e} "
                          f"({(time.time() - t_start) / max(1, global_step + 1 - start_g):.2f} s/step)")

                if global_step % val_every == 0:
                    metrics, health = fixed_val.evaluate(model, refs["raw"], refs["proj"])
                    decision = controller.on_validation(metrics["cos_err_r0.5"],
                                                        health["status"], global_step)
                    if phase.best_step == global_step:
                        best_health = health
                        phase.best_health = health  # survive checkpoint/resume (Bug #12)
                    if phase.best_healthy_step == global_step:
                        # Bug #15: HEALTHY-gated checkpoints are the only ones a
                        # plateau transition (or the final winner) may load.
                        phase.best_healthy_health = health
                        phase.best_healthy_path = os.path.join(
                            out_dir, f"phase_{phase.idx:02d}_{objective_name}_best_healthy.pt")
                        save_phase_checkpoint(
                            phase.best_healthy_path, model, optimizer, cfg, controller,
                            phase, global_step, metrics, health)
                        print(f"    [best-healthy] saved {phase.best_healthy_path} "
                              f"(cos_err {phase.best_healthy_metric:.6g})")
                    d_repr = _normalized_repr_dist(health, refs, collapse_cfg)
                    lr_history.append(scheduler.get_lr(global_step))
                    ema_history.append(model.ema.current_momentum(global_step))
                    record = {"step": global_step,
                              "train_loss": val_loss.report(),
                              **metrics,
                              "target_effective_rank": health["raw"]["eff_rank_unnorm"],
                              "target_effective_rank_fraction": health["raw"]["eff_rank_frac"],
                              "target_mean_std": health["raw"]["token_std"],
                              "target_pairwise_cosine_mean": health["raw"]["pairwise_cos"]["mean"],
                              "target_pairwise_cosine_p05": health["raw"]["pairwise_cos"]["p05"],
                              "target_pairwise_cosine_median": health["raw"]["pairwise_cos"]["median"],
                              "target_pairwise_cosine_p95": health["raw"]["pairwise_cos"]["p95"],
                              "target_same_position_cosine_mean": health["raw"]["same_token_cos"],
                              **{k: v for k, v in health["goal"].items()},
                              **{k: v for k, v in health["attention"].items()},
                              "objective": objective_name,
                              "representation_status": health["status"]}
                    jf.write(json.dumps(record) + "\n")
                    jf.flush()
                    print(f"  [adaptive val @ {global_step}] cos_err={metrics['cos_err_r0.5']:.6g} "
                          f"status={health['status']} votes={health['signals']['votes']}")

                    if phase.best_step == global_step:  # controller just recorded a new best
                        best_path = os.path.join(
                            out_dir, f"phase_{phase.idx:02d}_{objective_name}_best.pt")
                        save_phase_checkpoint(
                            best_path,
                            model, optimizer, cfg, controller, phase, global_step,
                            metrics, health)
                        print(f"    [best] saved phase {phase.idx} best ckpt (cos_err "
                              f"{phase.best_metric:.6g})")
                    if d_repr < best_repr_d:
                        best_repr_d = d_repr
                        save_phase_checkpoint(
                            os.path.join(out_dir, f"phase_{phase.idx:02d}_{objective_name}_best_repr.pt"),
                            model, optimizer, cfg, controller, phase, global_step,
                            metrics, health)
                    save_phase_checkpoint(
                        os.path.join(out_dir, f"phase_{phase.idx:02d}_{objective_name}_latest.pt"),
                        model, optimizer if acfg.get("save_optimizer", True) else None,
                        cfg, controller, phase, global_step, metrics, health)
                global_step += 1   # Bug #10: one optimizer step == one global step
                if decision is not None and decision["action"] in ("switch", "stop"):
                    phase_done = True
                    break

        # phase teardown
        phase.phase_step = global_step - start_g
        mode = "continue" if decision is None else decision["action"]
        if mode == "continue":
            phase.stop_reason = "global_budget"
        elif mode == "switch":
            phase.stop_reason = decision["reason"]
        elif mode == "stop":
            phase.stop_reason = decision["reason"]
            keep_running = False
        print(f"[phase-transition] from={objective_name} reason={phase.stop_reason} "
              f"global_step={global_step} phase_steps={phase.phase_step}")
        finishing_idx, finishing_name = phase.idx, objective_name

        report = {
            "objective": objective_name, "phase": phase.idx,
            "start_global_step": start_g, "end_global_step": global_step,
            "stop_reason": phase.stop_reason,
            "best_cos_err": phase.best_metric if math.isfinite(phase.best_metric)
            else None,
            "step_of_best_cos_err": phase.best_step,
            # Bug #15: best-HEALTHY fields (deployment candidates) vs prediction
            # best (diagnostic only). best_healthy_cos_err is the ONLY metric the
            # final winner selection may use; None = no HEALTHY checkpoint existed.
            "best_healthy_cos_err": (phase.best_healthy_metric
                                     if math.isfinite(phase.best_healthy_metric)
                                     else None),
            "step_of_best_healthy": phase.best_healthy_step,
            "best_healthy_health": phase.best_healthy_health,
            "best_healthy_checkpoint": (
                phase.best_healthy_path
                or (os.path.join(out_dir,
                                 f"phase_{phase.idx:02d}_{objective_name}_best_healthy.pt")
                    if phase.best_healthy_step is not None else None)),
            "raw_target_health_at_best": best_health["raw"] if best_health else None,
            "projected_target_health_at_best": best_health["proj"] if best_health else None,
            "prediction_health_at_best": best_health["pred"] if best_health else None,
            "goal_token_health": best_health["goal"] if best_health else None,
            "goal_attention_health": best_health["attention"] if best_health else None,
            "mask_statistics": fixed_val.mask_statistics,
            "loss_components": {k: float(v / max(1, comp_counts.get(k, 1)))
                                for k, v in comp_sums.items()},
            "loss_components_config": _objective_kwargs(acfg, objective_name),
            "learning_rate_history": lr_history,
            "ema_momentum_history": ema_history,
            "representation_status": phase.representation_status,
            "unstable_steps": unstable_steps,
        }
        phase_reports.append(report)
        write_phase_report(phase_report_json(phase.idx, objective_name), report)

        if not keep_running:
            break
        phase_idx += 1
        if mode == "continue":
            keep_running = False
            break
        trans = decision["transition"]
        if trans["restart"] == "base_init":
            base_obj = torch.load(base_path, map_location="cpu", weights_only=False)
            load_into_model(model, base_obj["model"], device)
            print(f"  [adaptive] reset to base init for {trans['next']} "
                  f"(reason {trans['reason']})")
        else:
            # Bug #15: a healthy-plateau transition may ONLY load the best-HEALTHY
            # checkpoint, never the prediction-best (that may be a WARNING/COLLAPSED
            # step). If no HEALTHY checkpoint exists, do NOT silently fall back:
            # report explicitly and stop the ladder instead of transitioning.
            best_path = phase.best_healthy_path or os.path.join(
                out_dir, f"phase_{finishing_idx:02d}_{finishing_name}_best_healthy.pt")
            if os.path.exists(best_path):
                obj = load_phase_checkpoint(best_path, model, None, None, device)
                print(f"  [adaptive] load best healthy ckpt {best_path} for "
                      f"{trans['next']} (optimizer/scheduler reset, EMA preserved)")
            else:
                report["transition_blocked"] = "no_healthy_checkpoint"
                write_phase_report(phase_report_json(finishing_idx, finishing_name),
                                   report)
                print(f"  [adaptive] NO HEALTHY CHECKPOINT available at {best_path} - "
                      f"stopping the ladder instead of transitioning to "
                      f"{trans['next']} (no silent fallback to best-prediction)")
                keep_running = False
                break

    jf.close()

    # ---- final method selection + summary (§22) ----
    improved = "improved" if any(r.get("step_of_best_cos_err") is not None for r in phase_reports) else "n/a"
    # Bug #15: winner comes ONLY from a HEALTHY-gated checkpoint; if no phase
    # produced one, report no-clean-winner — never the best WARNING/COLLAPSED
    # prediction-best result.
    winner = select_winner(phase_reports)
    if winner is None:
        winner = {"no_clean_winner": True,
                  "reason": "no HEALTHY checkpoint across all phases; "
                            "best-prediction results are diagnostic-only, "
                            "not deployment candidates",
                  "selection_priority": "healthy-only>stable>improvement>goal-conditioning>lower-error"}
    write_ladder_summary(out_dir, phase_reports, controller.summary(),
                         fixed_val.mask_statistics, winner)
    print("\n" + "=" * 50)
    print("MILESTONE B ADAPTIVE TRAINING SUMMARY")
    print("=" * 50)
    print(f"stopped_reason: {controller.summary()['final_phase']}")
    print("per-objective:")
    for r in phase_reports:
        print(f"  {r['objective']:<12} steps={r['end_global_step'] - r['start_global_step']:>4} "
              f"best_cos_err={r['best_cos_err']:.6g} status={r['representation_status']} "
              f"unstable={r['unstable_steps']}")
    print(f"winner: {winner}")
    print(f"collapse_detected: {any(r['representation_status'] == 'COLLAPSED' for r in phase_reports)}")
    print(f"summary -> {out_dir}/LOSS_LADDER_SUMMARY.md / .json")

    # final eval with null-goal gap on the winner's weights (only a real
    # HEALTHY-gated winner checkpoint; a no-clean-winner cannot be evaluated)
    obj_path = (winner.get("checkpoint") if winner and not winner.get("no_clean_winner")
                else None)
    if obj_path is None and winner is not None and winner.get("no_clean_winner"):
        print("NO CLEAN WINNER: no HEALTHY checkpoint across all phases - "
              "skipping final winner eval")
    if obj_path:
        load_phase_checkpoint(obj_path, model, None, None, device)
        model.eval()
        real, null, gaps = fixed_val.null_gap(model)
        final_metrics = {"cos_err_r0.5": real,
                         "null_cos_err_r0.5": null,
                         "null_gap": gaps}
        final_path = os.path.join(out_dir, "final_winner_metrics.json")
        with open(final_path, "w") as f:
            json.dump({"winner": winner, "metrics": final_metrics}, f, indent=2)
        print(f"final winner eval (fixed val): {json.dumps(final_metrics)}")
        print(f"-> {final_path}")


if __name__ == "__main__":
    main()