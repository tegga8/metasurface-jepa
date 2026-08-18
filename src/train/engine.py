"""Reusable training-engine pieces for the Milestone B adaptive ladder.

Owns everything the training loop must NOT re-derive:
  - FixedValidation: one fixed validation subset + one fixed mask set (reproducible
    across objectives and calls), mask statistics recording, and the full
    validation computation (prediction metric cos_err_r0.5 through the projection,
    goal-token statistics, goal->spectrum attention statistics, and the three-space
    representation-health stats). The spectrum-diagnostics call never changes model
    outputs (the SDPA path is untouched; weights are computed separately).
  - Phase checkpoint save/load with the mandatory metadata (objective, phase, global
    step, best metric, representation status, loss config) so filenames and metadata
    together make it impossible to confuse which objective produced the weights.
  - Per-phase JSON reports (§18 schema) and the LOSS_LADDER_SUMMARY writers.

The strategy (when to switch objectives, budget enforcement) lives in
src/train/adaptive.py; the loop lives in scripts/train/train_milestone_b.py.
"""

import json
import os

import torch

from diagnostics.representation_health import (
    classify_health, eff_ranks, goal_attention_stats, goal_token_stats,
    pairwise_cos_stats, token_space_stats,
)
from data.mask import BlockMasker

PIXEL_GRID = 16


# ---------------------------------------------------------------------------
# fixed validation
# ---------------------------------------------------------------------------

def _pooled_pred_stats(Z_pooled):
    """Z_pooled: (B, D) mean-pooled masked predictions -> projection-space stats."""
    return {
        "mean_std": Z_pooled.std(dim=0).mean().item(),
        "pairwise_cos": pairwise_cos_stats(Z_pooled),
        **eff_ranks(Z_pooled),
    }


class FixedValidation:
    """One fixed subset of batches + fixed masks; deterministic metric computation."""

    def __init__(self, batches, ratio=0.5, grid=PIXEL_GRID, min_side=3,
                 k_range=(1, 4), mask_seed=12345, device="cpu", collapse_cfg=None,
                 n_goal=16):
        """batches: list of (G, S) tensors already on `device` (fixed order)."""
        self.batches = batches
        self.ratio = ratio
        self.device = device
        self.collapse_cfg = collapse_cfg or {}
        self.n_goal = n_goal
        masker = BlockMasker(placement="random", grid=grid, min_side=min_side,
                             k_range=k_range, seed=mask_seed)
        self.masks = []
        fracs = []
        for (G, S) in batches:
            M = masker.sample(G, ratio).to(device)
            self.masks.append(M)
            fracs.append((1.0 - M.mean()).item())
        self.mask_statistics = {
            "requested_mask_ratio": float(ratio),
            "actual_mask_ratio_mean": float(sum(fracs) / len(fracs)),
            # population std: unbiased std of a single observation is NaN (Bug #9)
            "actual_mask_ratio_std": float(torch.tensor(fracs).std(unbiased=False).item()),
            "actual_mask_ratio_min": float(min(fracs)),
            "actual_mask_ratio_max": float(max(fracs)),
            "n_batches": len(batches),
            "n_samples": sum(G.shape[0] for G, _ in batches),
        }

    def _acc_stats(self, model, healthy_raw, healthy_proj, goal_mode="real"):
        """Forward all batches, aggregate (cos_err, token buffers, goal diagnostics).

        Runs in eval mode (deterministic; training-mode DropPath/dropout would make
        the validation metric incomparable with the final winner eval), restoring
        the model's previous mode afterwards.

        NOTE (2026-08-17, audit fix): forwards run with need_attn=False — the
        prediction metric must not touch the predictor's attention path (the SDPA
        path with weight extraction was a divergence source between in-loop and
        winner eval). Goal-token utilization is monitored via the spectrum-path
        attention weights below, which never modify model outputs.
        """
        was_training = model.training
        model.eval()
        proj = getattr(model, "proj", None)
        cos_errs = []
        zy_raw, zy_proj, zh_pooled = [], [], []
        goal_tokens, goal_attns = [], []
        with torch.no_grad():
            for (G, S), M in zip(self.batches, self.masks):
                out = model(G, S, M, goal_mode=goal_mode, need_attn=False)
                mask = out["mask"]
                z_hat, z_y = out["z_hat"], out["z_y"]
                if proj is not None:
                    ph_, pt_ = proj(z_hat), proj(z_y)
                else:
                    ph_, pt_ = z_hat, z_y
                d = (1.0 - torch.nn.functional.cosine_similarity(
                    torch.nn.functional.normalize(ph_, dim=-1),
                    torch.nn.functional.normalize(pt_, dim=-1), dim=-1)).clamp(min=0)
                cos_errs.append(d[mask].mean().item())
                zy_raw.append(z_y.cpu())
                zy_proj.append(pt_.cpu())
                mw = mask.float()
                zh_pooled.append(((z_hat * mw.unsqueeze(-1)).sum(1)
                                  / mw.sum(1, keepdim=True).clamp(min=1)).cpu())
                _, a_goal, w = model.spectrum_path(S, goal_mode, need_weights=True)
                goal_tokens.append(a_goal.cpu())
                goal_attns.append(w.cpu())
        if was_training:
            model.train()
        metrics = {
            "cos_err_r0.5": float(sum(cos_errs) / len(cos_errs)),
        }
        raw = token_space_stats(torch.cat(zy_raw, dim=0))
        proj_stats = token_space_stats(torch.cat(zy_proj, dim=0))
        pred = _pooled_pred_stats(torch.cat(zh_pooled, dim=0))
        status, signals = classify_health(raw, proj_stats, healthy_raw,
                                          healthy_proj, self.collapse_cfg)
        health = {
            "status": status,
            "signals": signals,
            "raw": raw,
            "proj": proj_stats,
            "pred": pred,
            "goal": goal_token_stats(torch.cat(goal_tokens, dim=0)),
            "attention": goal_attention_stats(torch.cat(goal_attns, dim=0)),
        }
        return metrics, health

    def evaluate(self, model, healthy_raw, healthy_proj, goal_mode="real"):
        """Returns (metrics, health); health includes raw/proj/pred/goal/attention."""
        return self._acc_stats(model, healthy_raw, healthy_proj, goal_mode=goal_mode)

    def null_gap(self, model):
        """cos_err for real goal, cos_err for null goal, and ||z_hat_real - z_hat_null||
        on masked tokens — the single-forward-pass goal-utilization diagnostic, computed
        identically (eval mode, same batches/masks/metric) as evaluate()."""
        was_training = model.training
        model.eval()
        real, null, gaps = [], [], []
        proj = getattr(model, "proj", None)
        with torch.no_grad():
            for (G, S), M in zip(self.batches, self.masks):
                o1 = model(G, S, M, goal_mode="real")
                o2 = model(G, S, M, goal_mode="null")
                if proj is not None:
                    p1, p2, pt = proj(o1["z_hat"]), proj(o2["z_hat"]), proj(o1["z_y"])
                else:
                    p1, p2, pt = o1["z_hat"], o2["z_hat"], o1["z_y"]
                d1 = (1.0 - torch.nn.functional.cosine_similarity(
                    torch.nn.functional.normalize(p1, dim=-1),
                    torch.nn.functional.normalize(pt, dim=-1), dim=-1)).clamp(min=0)
                d2 = (1.0 - torch.nn.functional.cosine_similarity(
                    torch.nn.functional.normalize(p2, dim=-1),
                    torch.nn.functional.normalize(pt, dim=-1), dim=-1)).clamp(min=0)
                real.append(d1[o1["mask"]].mean().item())
                null.append(d2[o2["mask"]].mean().item())
                gaps.append((o1["z_hat"] - o2["z_hat"]).norm(dim=-1)[o1["mask"]].mean().item())
        if was_training:
            model.train()
        return (sum(real) / len(real), sum(null) / len(null),
                sum(gaps) / len(gaps))


def fixed_validation_from_loader(val_ds, n_samples, batch_size, device, ratio=0.5,
                                 mask_seed=12345, **kwargs):
    """Deterministic fixed subset: the first n_samples of the val dataset (val loader
    is shuffle=False), then FixedValidation with pre-generated masks.

    Honors n_samples EXACTLY: the final batch is trimmed to the remaining count so
    n_samples need not be a multiple of batch_size (and n_samples < batch_size
    yields fewer than batch_size samples, not more — Bug #8).
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    batches, n = [], min(n_samples, len(val_ds))
    total = 0
    for G, S in loader:
        want = n - total
        if want <= 0:
            break
        batches.append((G[:want].to(device), S[:want].to(device)))
        total += G.shape[0]
    return FixedValidation(batches, ratio=ratio, device=device,
                           mask_seed=mask_seed, **kwargs)


# ---------------------------------------------------------------------------
# healthy reference (released-init model, same fixed validation set)
# ---------------------------------------------------------------------------

def healthy_references(model, fixed_val):
    """Stats of a fresh released-init build on the SAME fixed validation set.

    model must be in eval mode; only the EMA target (raw) and its projection are
    measured. Deterministic given the fixed validation set.

    Device contract (Bug #7): z_y stays on the model's device through the
    projection; only the recorded stats tensors move to CPU. Projecting a CPU
    tensor with a CUDA projection head here previously crashed (or silently
    round-tripped via a .cuda() hack).
    """
    zy_raw, zy_proj, zh_pooled = [], [], []
    proj = getattr(model, "proj", None)
    with torch.no_grad():
        for (G, S), M in zip(fixed_val.batches, fixed_val.masks):
            z_y = model.ema(G)
            zy_raw.append(z_y.cpu())
            if proj is not None:
                zy_proj.append(proj(z_y).cpu())
            out = model(G, S, M)
            m = out["mask"].float()
            zh_pooled.append(((out["z_hat"] * m.unsqueeze(-1)).sum(1)
                              / m.sum(1, keepdim=True).clamp(min=1)).cpu())
    raw = token_space_stats(torch.cat(zy_raw, dim=0))
    proj_stats = token_space_stats(torch.cat(zy_proj, dim=0))
    pred = _pooled_pred_stats(torch.cat(zh_pooled, dim=0))
    return {"raw": raw, "proj": proj_stats, "pred": pred}


# ---------------------------------------------------------------------------
# adaptive checkpoints (metadata mandatory)
# ---------------------------------------------------------------------------

def save_phase_checkpoint(path, model, optimizer, cfg, controller, phase,
                          global_step, metrics, health, extra=None):
    obj = {
        "step": global_step, "epoch": 0, "cfg": cfg, "best": {
            "primary": metrics.get("cos_err_r0.5", 0.0), "metrics": metrics,
            "step": global_step, "health": health,
        },
        "model": _saveable(model),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": __import__("numpy").random.get_state(),
        "python_rng": __import__("random").getstate(),
        "adaptive_meta": {
            "objective": phase.objective, "phase": phase.idx,
            "global_step": global_step, "phase_step": phase.phase_step,
            "best_metric": phase.best_metric, "best_step": phase.best_step,
            "representation_status": health.get("status"),
            "controller": controller.summary(),
            # full controller state for exact resume (Bug #12): counters,
            # histories, transitions — not just the summary tuple
            "controller_state": controller.state_dict(),
        },
    }
    if extra:
        obj.update(extra)
    torch.save(obj, path)
    return path


def _saveable(model):
    from assembly import saveable_state_dict
    return saveable_state_dict(model)


def load_phase_checkpoint(path, model, optimizer, scheduler, device):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    from assembly import load_into_model
    load_into_model(model, obj["model"], device)
    if optimizer is not None and obj.get("optimizer"):
        optimizer.load_state_dict(obj["optimizer"])
    if scheduler is not None and "step" in obj:
        for g in optimizer.param_groups:
            g["lr"] = scheduler.get_lr(obj["step"])
    if obj.get("torch_rng") is not None:
        torch.set_rng_state(obj["torch_rng"].cpu())
    if obj.get("numpy_rng") is not None:
        __import__("numpy").random.set_state(obj["numpy_rng"])
    if obj.get("python_rng") is not None:
        __import__("random").setstate(obj["python_rng"])
    return obj


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------

def write_phase_report(path, phase_report):
    with open(path, "w") as f:
        json.dump(phase_report, f, indent=2, default=float)
    return path


def write_ladder_summary(out_dir, reports, controller_summary, mask_statistics,
                         winner):
    json_path = os.path.join(out_dir, "LOSS_LADDER_SUMMARY.json")
    with open(json_path, "w") as f:
        json.dump({"reports": reports, "controller": controller_summary,
                   "mask_statistics": mask_statistics, "winner": winner}, f,
                  indent=2, default=float)
    lines = ["# LOSS_LADDER_SUMMARY — Milestone B adaptive screening",
             "",
             f"- mask statistics: {json.dumps(mask_statistics)}",
             f"- controller: {json.dumps(controller_summary)}",
             "",
             "| objective | steps | best_cos_err | repr status | proj status | "
             "goal_pairwise | stability |",
             "|---|---|---|---|---|---|---|"]
    for r in reports:
        lines.append(
            f"| {r['objective']} | {r['end_global_step'] - r['start_global_step']} | "
            f"{r['best_cos_err']:.6g} | {r['representation_status']} | "
            f"{(r.get('projected_target_health_at_best') or {}).get('status', 'n/a')} | "
            f"{(r.get('goal_token_health') or {}).get('goal_token_pairwise_cosine_mean', float('nan')):.4f} | "
            f"{'stable' if not r.get('unstable_steps') else str(r['unstable_steps']) + ' unstable steps'} |")
    lines += ["", f"**Winner (priority: healthy-only > stable > meaningful improvement > "
                  f"goal conditioning > lower error):** {json.dumps(winner)}", ""]
    md_path = os.path.join(out_dir, "LOSS_LADDER_SUMMARY.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    return json_path, md_path