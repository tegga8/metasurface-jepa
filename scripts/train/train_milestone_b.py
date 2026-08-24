"""Milestone B — vanilla deterministic JEPA training / eval driver (§7 Phase 2, §7.2).

Runs ONE objective from the final three-registry {jepa_vicreg, jepa_barlow, lejepa}
(§23/§24) through the shared engine (`src/train/engine.py`): fixed validation,
healthy released-init references, resumable checkpoints with objective state, and
EMA momentum counters (§30).

Run inside `notebooks/cloud_train_runner.ipynb` per CLOUD_TRAINING.md:

    python scripts/train/train_milestone_b.py --config configs/milestone_b.yaml
    python scripts/train/train_milestone_b.py --config configs/milestone_b.yaml \
        --experiment sweep --objective lejepa --resume checkpoints/milestone_b/latest.pt
    python scripts/train/train_milestone_b.py --config configs/milestone_b.yaml \
        --eval-only --null-goal --resume checkpoints/milestone_b/latest.pt

The minimal experiment (§7.2): fixed 50% block-masked context, random block
placement only, L = L_J + objective regularization — compared against the direct
masked generator (Baseline 2, `src/reference/direct_masked_generator.py`, evaluated
by the comparison script) and the null-goal intervention (--null-goal).

The historical adaptive ladder, `direct` model variant, released-DiT baseline eval,
and per-phase checkpointing have all been removed (architecture-repair spec).
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
from assembly import build_model, saveable_state_dict
from losses.objectives import build_objective

PIXEL_GRID = 16

# NOTE: this script does NOT import src/reference/direct_masked_generator — the
# Baseline-2 comparison is a separate eval concern (see scripts/eval/).


# ---------------------------------------------------------------------------
# schedule / helpers
# ---------------------------------------------------------------------------

class CosineWarmup:
    """Cosine decay to 0 with linear warmup; deterministic from step (resume-safe)."""

    def __init__(self, base_lr, warmup_steps, total_steps):
        self.base_lr = base_lr
        self.warmup = warmup_steps
        self.total = total_steps

    def factor(self, step):
        """Multiplier over base_lr — consumed by the LambdaLR wrapper."""
        if step < self.warmup:
            return (step + 1) / max(1, self.warmup)
        t = (step - self.warmup) / max(1, self.total - self.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))


def build_scheduler(optimizer, base_lr, warmup_steps, total_steps):
    """LambdaLR wrapper around CosineWarmup so the scheduler owns a state_dict
    (restored on resume, spec §30) instead of the legacy manual get_lr loop."""
    cos = CosineWarmup(base_lr, warmup_steps, total_steps)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: cos.factor(max(0, int(e))))


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_surrogate(path, device):
    """Frozen forward-EM surrogate, required only for half_sensitivity mask
    placement (§2: half of batches masked over resonance-relevant regions)."""
    METADIT_SRC = os.path.join(REPO_ROOT, "external", "metadit")
    if METADIT_SRC not in sys.path:
        sys.path.insert(0, METADIT_SRC)
    # ``external/metadit`` is added to ``sys.path`` at runtime, so importing
    # through the module loader avoids a false unresolved-import diagnostic.
    import importlib

    surrogate_s3 = importlib.import_module("model.surrogate").surrogate_s3

    m = surrogate_s3()
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _assert_no_ema_gradients(model, objective_name, step):
    """Per-step EMA-frozen guard: the EMA target encoder must receive no
    gradient through the objective (spec §9)."""
    leaked = [n for n, p in model.ema.named_parameters() if p.grad is not None]
    assert not leaked, (
        f"[{objective_name} step {step}] EMA target encoder received a "
        f"gradient on: {leaked[:5]}")


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _jepa_fixed_val_metrics(model, objective, fixed_vals, refs, need_null=False):
    """Shared metric path via the engine's FixedValidation: in-loop validation,
    eval-only, and final eval all use the exact same batches/masks/metric
    (projection through objective.projector, never a model attribute). refs
    must contain a healthy released-init reference per ratio (precomputed)."""
    metrics = {}
    for r, fv in sorted(fixed_vals.items()):
        ref = refs[r]
        m, health = fv.evaluate(model, objective, ref["raw"], ref["proj"])
        metrics[f"cos_err_r{r:g}"] = m["cos_err_r0.5"]
        metrics[f"health_r{r:g}"] = health["status"]
        if need_null:
            real, null, gap = fv.null_gap(model, objective)
            metrics[f"null_cos_err_r{r:g}"] = null
            metrics[f"null_gap_r{r:g}"] = gap
    return metrics


# ---------------------------------------------------------------------------
# checkpoints (engine, spec §30)
# ---------------------------------------------------------------------------

def _load_train_checkpoint(path, model, objective, optimizer, scheduler, device):
    from train.engine import load_checkpoint, restore_ema_state
    obj = load_checkpoint(path, model, objective, optimizer, scheduler, device)
    restore_ema_state(model, obj.get("ema_state"))
    return obj


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Milestone B JEPA training")
    p.add_argument("--config", required=True)
    p.add_argument("--experiment", choices=["minimal", "sweep"], default=None)
    p.add_argument("--objective", default=None,
                   help="objective name (override config; must be in the "
                        "final registry: jepa_vicreg | jepa_barlow | lejepa)")
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    p.add_argument("--eval-only", action="store_true", help="eval from a checkpoint and exit")
    p.add_argument("--null-goal", action="store_true", help="eval with goal replaced by null")
    p.add_argument("--smoke", action="store_true", help="tiny local crash-test run")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-steps", type=int, default=None,
                   help="hard step cap (smoke/debug); default: epochs from config")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    exp = args.experiment or cfg["experiment"]
    cfg["experiment"] = exp
    exp_cfg = cfg[exp]
    ratios = ([exp_cfg["mask_ratio"]] if exp == "minimal"
              else exp_cfg["mask_ratios"])
    placement = exp_cfg["mask_placement"]

    objective_name = args.objective or cfg.get("objective", "jepa_vicreg")

    if args.smoke:  # local dev-only crash test (batch 1, handful of steps)
        cfg["train"].update(batch_size=1, grad_accum=1, epochs=1, val_batches=1,
                            val_every_steps=1, log_every_steps=1, save_optimizer=False,
                            warmup_steps=0, ckpt_every_steps=1, max_steps=3)
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
    run_tag = f"{exp}_{objective_name}" + ("_smoke" if args.smoke else "")

    # data
    ds = MetaDiTDataset(os.path.join(REPO_ROOT, cfg["data"]["train_split"]),
                        max_samples=cfg["data"].get("max_train_samples", 0), seed=seed)
    val_ds = MetaDiTDataset(os.path.join(REPO_ROOT, cfg["data"]["val_split"]))
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        num_workers=cfg["data"].get("num_workers", 0),
                        drop_last=True, collate_fn=collate_batch)

    steps_per_epoch = max(1, len(loader))
    accum = int(cfg["train"].get("grad_accum", 1))
    if accum < 1:
        raise ValueError(f"grad_accum must be >= 1, got {accum}")

    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accum)
    total_steps = args.max_steps or optimizer_steps_per_epoch * cfg["train"]["epochs"]
    # masker (+ frozen surrogate only if half_sensitivity placement needs it)
    masker = BlockMasker(placement=placement, grid=PIXEL_GRID,
                         min_side=cfg["mask"].get("min_side", 3),
                         k_range=tuple(cfg["mask"].get("k_range", [1, 4])), seed=seed)
    surrogate = None
    if placement == "half_sensitivity":
        surrogate = load_surrogate(os.path.join(REPO_ROOT, cfg["weights"]["surrogate"]),
                                   device)

    # model + objective (final registry)
    model = build_model(cfg["model"],
                        os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
                        device=device,
                        init_from_metadit=cfg["model"].get("init_from_metadit", True),
                        metadit_weights=os.path.join(REPO_ROOT, cfg["weights"]["metadit"]))
    model.ema.set_total_steps(total_steps)
    # Phase-2 plumbing fix A: the objective owns trainable projector parameters
    # (VICReg/Barlow/LeJEPA projectors) and must live on the SAME device as the
    # model — a CPU-resident objective crashes on the first CUDA forward
    # ("Expected all tensors to be on the same device").
    objective = build_objective(
        objective_name, cfg.get("objective_params", {}).get(objective_name, {}),
        projector_input_dim=cfg["model"].get("hidden", 384),
    ).to(device)
    objective_params = list(objective.parameters())
    if objective_params:  # every registered objective owns projector parameters
        objective_device = objective_params[0].device
        assert objective_device == device, (
            f"Objective parameters are on {objective_device}, expected {device}")

    # fixed validation + healthy released-init references (per ratio, same masks)
    from train.engine import (build_deterministic_reference,
                              fixed_validation_from_loader, healthy_references)
    val_bs = cfg["train"]["val_batches"]
    batch_size = cfg["train"]["batch_size"]
    n = max(1, min(val_bs * batch_size, len(val_ds)))
    mask_seed = cfg["mask"].get("mask_seed", 12345)
    fixed_vals, refs, refs_model = {}, {}, None
    for r in ratios:
        fv = fixed_validation_from_loader(val_ds, n, batch_size, device,
                                          ratio=float(r), mask_seed=mask_seed)
        if refs_model is None:
            refs_model = build_deterministic_reference(
                lambda: build_model(cfg["model"],
                                    os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
                                    device=device,
                                    init_from_metadit=cfg["model"].get("init_from_metadit", True),
                                    metadit_weights=os.path.join(REPO_ROOT,
                                                                 cfg["weights"]["metadit"])))
            refs_model.eval()
        fixed_vals[r] = fv
        refs[r] = healthy_references(refs_model, fv, objective=objective)

    # optimizer owns model trainable params + objective trainable params, never EMA
    trainable = [p for p in model.parameters() if p.requires_grad] \
        + [p for p in objective.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in trainable)
    opt_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(trainable, lr=float(opt_cfg["lr"]),
                                  weight_decay=float(opt_cfg["wd"]), betas=(0.9, 0.999))
    ema_ids = {id(p) for p in model.ema.parameters()}
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    leaked = ema_ids & opt_ids
    assert not leaked, f"EMA target parameters inside optimizer: {len(leaked)}"
    scheduler = build_scheduler(optimizer, float(opt_cfg["lr"]),
                                int(opt_cfg["warmup_steps"]), total_steps)

    print(f"[milestone_b] objective={objective_name} exp={exp} placement={placement} "
          f"ratios={ratios} device={device} trainable_params={n_param:,} "
          f"steps_per_epoch={steps_per_epoch} total_steps={total_steps}")

    from train.engine import collect_ema_state, save_checkpoint
    ckpt_path = os.path.join(out_dir, f"{run_tag}_latest.pt")
    best_path = os.path.join(out_dir, f"{run_tag}_best_model.pt")
    start_step, start_epoch, best = 0, 0, {}
    if args.resume:
        obj = _load_train_checkpoint(args.resume, model, objective, optimizer,
                                     scheduler, device)
        start_step = obj["step"] + 1
        start_epoch = obj["epoch"] + 1
        best = obj.get("best", {})
        print(f"[milestone_b] resumed from {args.resume}: step={start_step} "
              f"epoch={start_epoch} objective={objective_name}")

    if args.eval_only:
        metrics = _jepa_fixed_val_metrics(model, objective, fixed_vals, refs,
                                          need_null=args.null_goal)
        tag = "eval" + ("_null_goal" if args.null_goal else "")
        eval_path = os.path.join(out_dir, f"{run_tag}_{tag}_metrics.json")
        with open(eval_path, "w") as f:
            json.dump({"config": cfg, "metrics": metrics}, f, indent=2)
        print(f"[milestone_b] eval-only done -> {eval_path}")
        print(json.dumps(metrics, indent=2))
        return

    # ---- training loop ----
    print(f"[milestone_b] training objective={objective_name} ({exp}) ...")
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    micro_step = 0
    t_start = time.time()
    accum = int(opt_cfg.get("grad_accum", 1))
    if accum < 1:
        raise ValueError(f"grad_accum must be >= 1, got {accum}")

    loss_accum = 0.0
    comp_sums, comp_counts = {}, {}
    sigreg_info = None

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        for bi, (G, S) in enumerate(loader):            
            if args.max_steps and step >= args.max_steps:
                break
            G, S = G.to(device), S.to(device)
            ratio = ratios[step % len(ratios)]
            M = masker.sample(G, ratio, surrogate).to(device)

            res = objective(model, G, S, M)
            total = res["total_loss"]
            if isinstance(res["components"].get("sigreg_info"), dict):
                sigreg_info = res["components"]["sigreg_info"]
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite total loss at step {step}: {total.item()}")
            total = total / accum
            total.backward()
            micro_step += 1
            _assert_no_ema_gradients(model, objective_name, step)
            loss_accum += total.item() * accum
            for k, v in res["components"].items():
                if isinstance(v, torch.Tensor):
                    comp_sums[k] = comp_sums.get(k, 0.0) + v.item()
                    comp_counts[k] = comp_counts.get(k, 0) + 1

            if micro_step % accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, opt_cfg["clip_grad_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            objective.on_optimizer_step(model, step)
            scheduler.step()

            if step % opt_cfg["log_every_steps"] == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                frac_masked = 1.0 - M.mean().item()
                print(f"  [step {step}] loss={loss_accum:.4f} lr={lr_now:.2e} "
                      f"mask_frac={frac_masked:.2f} "
                      f"({(time.time() - t_start) / max(1, step + 1 - start_step):.2f} s/step)")
                loss_accum = 0.0

            if step % opt_cfg["val_every_steps"] == 0:
                val_metrics = _jepa_fixed_val_metrics(model, objective, fixed_vals,
                                                      refs, need_null=False)
                primary = val_metrics.get(f"cos_err_r{ratios[0]:g}", float("inf"))
                if not best or primary < best.get("primary", float("inf")):
                    best = {"primary": primary, "metrics": val_metrics,
                            "step": step}
                    torch.save(saveable_state_dict(model), best_path)
                print(f"  [val @ step {step}] {json.dumps(val_metrics)}")

            ckpt_now = (cfg["train"].get("ckpt_every_steps", 0)
                        and step % cfg["train"]["ckpt_every_steps"] == 0) or \
                (bi == len(loader) - 1)
            if ckpt_now:
                save_checkpoint(ckpt_path, model, objective, optimizer, scheduler,
                                cfg, step, epoch, metrics=val_metrics if step % opt_cfg[
                                    "val_every_steps"] == 0 else {},
                                health=None, ema_state=collect_ema_state(model),
                                best_state=best)
                print(f"  [ckpt] saved {ckpt_path} (step {step})")

            step += 1
            if args.smoke and (step - start_step) >= cfg["train"].get("max_steps", 0):
                break
        if micro_step % accum != 0:
                        raise RuntimeError(
                            f"Epoch {epoch} ended with {micro_step % accum} "
                            f"unconsumed micro-batches for grad_accum={accum}"
                        )    
        if args.max_steps and step >= args.max_steps:
            break

    # final eval
    final = _jepa_fixed_val_metrics(model, objective, fixed_vals, refs,
                                    need_null=True)
    gaps = [final[k] for k in final if k.startswith("null_gap_r")
            and isinstance(final[k], float)]
    final["null_gap"] = sum(gaps) / len(gaps) if gaps else float("nan")
    final_path = os.path.join(out_dir, f"{run_tag}_final_metrics.json")
    with open(final_path, "w") as f:
        json.dump({"config": cfg, "metrics": final, "best": best,
                   "objective": objective_name,
                   "loss_components": {k: float(v / max(1, comp_counts.get(k, 1)))
                                       for k, v in comp_sums.items()},
                   "sigreg_info": sigreg_info}, f, indent=2)
    print(f"[milestone_b] final metrics -> {final_path}")
    print(json.dumps(final, indent=2))
    save_checkpoint(ckpt_path, model, objective, optimizer, scheduler, cfg,
                    step - 1, epoch, metrics=final, health=None,
                    ema_state=collect_ema_state(model), best_state=best)


if __name__ == "__main__":
    main()