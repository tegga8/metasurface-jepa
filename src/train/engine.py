"""Reusable training-engine pieces shared by all Milestone B objectives and the
fixed-validation / short-audit evaluators (architecture-repair spec §25/§30).

Owns everything the training loop and evaluators must NOT re-derive:

  - FixedValidation: one fixed validation subset + one fixed mask set
    (reproducible across objectives and calls), mask-statistics recording, and
    the shared evaluation: prediction metric cos_err_r{ratio}, goal-token
    statistics, goal->spectrum attention statistics, and the three-space
    representation-health stats. Projection is provided by the OBJECTIVE's own
    projector — there is no `model.proj` anywhere (§17).
  - Healthy released-init references on the SAME fixed validation set, projected
    through the candidate objective's projector (never a random-reference head).
  - Checkpoint save/load with mandatory metadata (objective_name,
    objective_state, optimizer param-shape ownership, scheduler state, EMA
    momentum counters, RNG state, masker RNG state, git commit, env versions)
    and strict objective-name / optimizer-ownership validation on load (§30).
  - RNG collect/restore for exact resume (Bug #17).
  - IntervalLossAccumulator for exact per-interval loss means (Bug #18).

The strategy/loop lives in the milestone training scripts; this module contains
no adaptive-ladder, phase, or LOSS_LADDER machinery (all removed in the repair).
"""

import json
import os
import random
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch

from diagnostics.representation_health import (
    classify_health, eff_ranks, goal_attention_stats, goal_token_stats,
    pairwise_cos_stats, token_space_stats,
)
from data.mask import BlockMasker
from runtime.reproducibility import collect_rng_state as _collect_rng_state
from runtime.reproducibility import restore_rng_state as _restore_rng_state

PIXEL_GRID = 16

CHECKPOINT_SCHEMA_VERSION = 1
REQUIRED_CHECKPOINT_KEYS = (
    "schema_version",
    "objective_name",
    "step",
    "epoch",
    "micro_step",
    "batch_index",
    "is_epoch_end",
    "cfg",
    "best_prediction",
    "best_healthy_prediction",
    "model",
    "objective_state",
    "optimizer",
    "optimizer_param_shapes",
    "scheduler_state",
    "ema_state",
    "rng_state",
    "masker_rng_state",
    "git_commit",
    "git_dirty",
    "env_versions",
    "device_info",
    "artifact_type",
)


class IntervalLossAccumulator:
    """Exact per-interval training-loss mean (Bug #18).

    Sums un-divided per-micro-batch losses; report() returns sum/count and resets.
    Correct under any grad_accum (every optimizer step contributes `accum`
    micro-batches, so the per-micro-batch mean equals the per-step mean) and under
    interval boundaries (fresh instance drops the partial interval cleanly).
    """

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def add(self, value):
        self.sum += float(value)
        self.count += 1

    def report(self):
        mean = self.sum / max(1, self.count)
        self.reset()
        return mean

    def reset(self):
        self.sum = 0.0
        self.count = 0


def collect_rng_state():
    """CPU + numpy + python + (when CUDA available) CUDA RNG state for exact
    resume (Bug #17). CUDA state is None on CPU-only machines, never an error.
    Delegates to runtime.reproducibility for canonical implementation."""
    return _collect_rng_state()


def restore_rng_state(state):
    """Inverse of collect_rng_state. Missing/None entries are skipped; a CUDA
    state saved on a GPU machine is skipped safely when restoring on CPU.
    Delegates to runtime.reproducibility for canonical implementation."""
    _restore_rng_state(state)


# ---------------------------------------------------------------------------
# fixed validation
# ---------------------------------------------------------------------------

def _objective_projection(objective):
    """The projection operator used by the shared evaluator.

    Spec §17: no `model.proj` fallback. Every registered objective owns its
    projector (§24); if no objective is supplied (or it has none) the evaluator
    falls back to an identity projection so raw and projected statistics are
    well-defined and identical — never a model attribute."""
    if objective is None:
        return None
    return getattr(objective, "projector", None)


def _eval_mode_restore(model, objective):
    """Phase-2 plumbing fix B: switch model AND objective to eval mode for an
    evaluation/reference pass and return a zero-arg callable restoring BOTH
    previous modes (never a blind .train()).

    Duck-typed on purpose: real nn.Modules always expose the mode API and get
    the full semantics; bare test-stub objects without .eval/.train/.training
    are left untouched (exactly their pre-fix behavior).
    """
    has_model_api = (callable(getattr(model, "eval", None))
                     and callable(getattr(model, "train", None)))
    was_model_training = getattr(model, "training", None)
    was_objective_training = getattr(objective, "training", None)
    has_objective_api = objective is not None and \
        callable(getattr(objective, "eval", None))
    if has_model_api:
        model.eval()
    if was_objective_training is not None and has_objective_api:
        objective.eval()

    def _restore():
        if has_model_api and was_model_training is not None:
            model.train(was_model_training)
        if was_objective_training is not None and has_objective_api:
            objective.train(was_objective_training)

    return _restore


def _pooled_pred_stats(Z_pooled):
    """Z_pooled: (B, D) mean-pooled masked predictions -> projection-space stats.

    n < 2: NaN markers (Bug #21) — the stats are undefined and the raw-side
    n_geoms guard in classify_health produces the UNAVAILABLE verdict."""
    if Z_pooled.shape[0] < 2:
        return {"mean_std": float("nan"),
                "pairwise_cos": {"mean": float("nan"), "median": float("nan"),
                                 "p05": float("nan"), "p95": float("nan"),
                                 "min": float("nan")},
                "eff_rank_unnorm": float("nan"), "eff_rank_frac": float("nan"),
                "participation": float("nan"), "top_eig_frac": float("nan")}
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

    def _project(self, objective, x):
        proj = _objective_projection(objective)
        return proj(x) if proj is not None else x

    def _acc_stats(self, model, objective, healthy_raw, healthy_proj,
                   goal_mode="real"):
        """Forward all batches, aggregate (cos_err, token buffers, goal diagnostics).

        Projection is through `objective.projector` (spec §17: never a model
        attribute). Runs in eval mode (deterministic; training-mode
        DropPath/dropout would make the validation metric incomparable with the
        final winner eval), restoring both previous modes afterwards.

        Phase-2 plumbing fix B: the OBJECTIVE is switched to eval too — its
        projector contains BatchNorm layers whose running statistics would be
        contaminated by train-mode validation forwards (and batch statistics
        would make the metric nondeterministic). Both prior modes are restored
        in a finally block; neither module is left forced into either mode.

        NOTE (2026-08-17, audit fix): forwards run with need_attn=False — the
        prediction metric must not touch the predictor's attention path (the SDPA
        path with weight extraction was a divergence source between in-loop and
        winner eval). Goal-token utilization is monitored via the spectrum-path
        attention weights below, which never modify model outputs.
        """
        restore_modes = _eval_mode_restore(model, objective)
        try:
            # Bug #19: aggregate globally (loss_sum / mask_count), NOT per-batch means —
            # averaging per-batch means makes the metric depend on batch partitioning.
            loss_sum = 0.0
            mask_count = 0
            zy_raw, zy_proj, zh_pooled = [], [], []
            goal_tokens, goal_attns = [], []
            with torch.no_grad():
                for (G, S), M in zip(self.batches, self.masks):
                    out = model(G, S, M, goal_mode=goal_mode, need_attn=False)
                    mask = out["mask"]
                    z_hat, z_y = out["z_hat"], out["z_y_raw"]
                    ph_ = self._project(objective, z_hat)
                    pt_ = self._project(objective, z_y)
                    d = (1.0 - torch.nn.functional.cosine_similarity(
                        torch.nn.functional.normalize(ph_, dim=-1),
                        torch.nn.functional.normalize(pt_, dim=-1), dim=-1)).clamp(min=0)
                    dm = d[mask]
                    loss_sum += float(dm.sum(dtype=torch.float64).item())
                    mask_count += int(dm.numel())
                    zy_raw.append(z_y.cpu())
                    zy_proj.append(pt_.cpu())
                    mw = mask.float()
                    # Bug #13: prediction-health stats pool the PROJECTED prediction
                    # (same space the cos_err metric lives in), not raw z_hat — raw
                    # pooling is invariant to the learned projection and reports stats
                    # in a space nothing else uses.
                    zh_pooled.append(((ph_ * mw.unsqueeze(-1)).sum(1)
                                      / mw.sum(1, keepdim=True).clamp(min=1)).cpu())
                    _, a_goal, w = model.spectrum_path(S, goal_mode, need_weights=True)
                    goal_tokens.append(a_goal.cpu())
                    goal_attns.append(w.cpu())
            # Dynamic ratio key per the hardening spec
            ratio_key = f"cos_err_r{self.ratio:g}"
            metrics = {
                ratio_key: float(loss_sum / max(1, mask_count)),
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
        finally:
            restore_modes()
        return metrics, health

    def evaluate(self, model, objective, healthy_raw, healthy_proj,
                 goal_mode="real"):
        """Returns (metrics, health); health includes raw/proj/pred/goal/attention."""
        return self._acc_stats(model, objective, healthy_raw, healthy_proj,
                               goal_mode=goal_mode)

    def null_gap(self, model, objective):
        """cos_err for real goal, cos_err for null goal, and ||z_hat_real - z_hat_null||
        on masked tokens — the single-forward-pass goal-utilization diagnostic, computed
        identically (eval mode — model AND objective, see _acc_stats Phase-2 fix B;
        same batches/masks/metric) as evaluate()."""
        restore_modes = _eval_mode_restore(model, objective)
        try:
            # Bug #19: per-batch means replaced by global aggregation (identical rule
            # to _acc_stats) so real/null/gap metrics are batch-partition invariant.
            real_sum, null_sum, gap_sum, mask_count = 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for (G, S), M in zip(self.batches, self.masks):
                    o1 = model(G, S, M, goal_mode="real")
                    o2 = model(G, S, M, goal_mode="null")
                    p1 = self._project(objective, o1["z_hat"])
                    p2 = self._project(objective, o2["z_hat"])
                    pt = self._project(objective, o1["z_y_raw"])
                    d1 = (1.0 - torch.nn.functional.cosine_similarity(
                        torch.nn.functional.normalize(p1, dim=-1),
                        torch.nn.functional.normalize(pt, dim=-1), dim=-1)).clamp(min=0)
                    d2 = (1.0 - torch.nn.functional.cosine_similarity(
                        torch.nn.functional.normalize(p2, dim=-1),
                        torch.nn.functional.normalize(pt, dim=-1), dim=-1)).clamp(min=0)
                    m1 = o1["mask"]
                    m2 = o2["mask"]
                    real_sum += float(d1[m1].sum(dtype=torch.float64).item())
                    null_sum += float(d2[m2].sum(dtype=torch.float64).item())
                    gap_sum += float(((o1["z_hat"] - o2["z_hat"]).norm(dim=-1)[m1])
                                     .sum(dtype=torch.float64).item())
                    mask_count += int(m1.sum().item())
        finally:
            restore_modes()
        ratio_key = f"cos_err_r{self.ratio:g}"
        return {
            f"real_{ratio_key}": float(real_sum / max(1, mask_count)),
            f"null_{ratio_key}": float(null_sum / max(1, mask_count)),
            f"gap_{ratio_key}": float(gap_sum / max(1, mask_count)),
        }


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

REFS_SEED_DEFAULT = 2026


def build_deterministic_reference(build_fn, seed=REFS_SEED_DEFAULT):
    """Build the released-init healthy-reference model under a FIXED RNG context.

    The healthy reference must be ONE deterministic released-init state (final
    pre-training pass, FIX B): the released geometry encoder loads MetaDiT
    weights (deterministic), but the reference's remaining random components
    (context encoder, predictor) consume the ambient RNG, which differs run to
    run (e.g. from a time-based seed). Two otherwise-identical runs then
    measured against different references could classify the same state HEALTHY
    in one run and WARNING in another — a nondeterministic health gate is not
    scientific.

    The build runs inside torch.random.fork_rng() with a dedicated
    manual_seed(seed), so the reference is a pure function of `seed` — never of
    the ambient stream. The ambient RNG state is restored untouched afterwards:
    the reference build never perturbs training randomness.

    build_fn: zero-arg callable constructing the reference model (typically
    assembly.build_model with released init). Returns whatever build_fn returns.
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return build_fn()


def healthy_references(ref_model, fixed_val, objective=None):
    """Stats of a fresh released-init build on the SAME fixed validation set.

    Runs with ref_model AND objective (when supplied) in eval mode — Phase-2
    plumbing fix B: the reference projection passes through the candidate
    objective's projector, whose BatchNorm layers must neither consume batch
    statistics nor update their running statistics during reference
    construction. Both modules' previous modes are restored in a finally block.

    Only the EMA target (raw) and its projection are measured. Deterministic
    given the fixed validation set.

    Projection is through the candidate OBJECTIVE's projector (spec §17, Bug #14
    analog): the released-init reference's raw EMA embeddings and pooled
    predictions are projected through `objective.projector` — never through a
    model attribute — because the reference's own head would be a separate
    random initialization (a different learned coordinate system), so a collapse
    verdict compared across the two heads is not meaningful. objective=None keeps
    a raw-only measurement: proj stats equal raw stats (identity projection).

    Device contract (Bug #7): z_y stays on the model's device through the
    projection; only the recorded stats tensors move to CPU.
    """
    proj = _objective_projection(objective)
    restore_modes = _eval_mode_restore(ref_model, objective)
    try:
        zy_raw, zy_proj, zh_pooled = [], [], []
        with torch.no_grad():
            for (G, S), M in zip(fixed_val.batches, fixed_val.masks):
                z_y = ref_model.ema(G)
                zy_raw.append(z_y.cpu())
                if proj is not None:
                    zy_proj.append(proj(z_y).cpu())
                out = ref_model(G, S, M)
                m = out["mask"].float()
                # Bug #13 (same convention as _acc_stats): prediction-health stats pool
                # the PROJECTED prediction, in the same space the cos_err metric lives.
                z_hat = proj(out["z_hat"]) if proj is not None else out["z_hat"]
                zh_pooled.append(((z_hat * m.unsqueeze(-1)).sum(1)
                                  / m.sum(1, keepdim=True).clamp(min=1)).cpu())
        raw = token_space_stats(torch.cat(zy_raw, dim=0))
        if proj is not None:
            proj_stats = token_space_stats(torch.cat(zy_proj, dim=0))
        else:
            proj_stats = raw
        pred = _pooled_pred_stats(torch.cat(zh_pooled, dim=0))
    finally:
        restore_modes()
    return {"raw": raw, "proj": proj_stats, "pred": pred}


# ---------------------------------------------------------------------------
# git / environment metadata
# ---------------------------------------------------------------------------

def _git_info() -> dict[str, str]:
    """Collect git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        is_dirty = bool(dirty)
    except Exception:
        is_dirty = False
    return {"git_commit": commit, "git_dirty": is_dirty}


def _env_versions() -> dict[str, str]:
    """Collect key environment versions."""
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _device_info(device: torch.device | str | None = None) -> dict[str, Any]:
    """Collect device information."""
    dev = device if isinstance(device, torch.device) else torch.device(device or "cpu")
    info = {"device_type": dev.type}
    if dev.type == "cuda":
        info["device_index"] = dev.index
        info["device_name"] = torch.cuda.get_device_name(dev.index or 0)
    return info


# ---------------------------------------------------------------------------
# checkpoint schema validation
# ---------------------------------------------------------------------------

def _validate_checkpoint_schema(obj: dict, path: str) -> None:
    """Validate checkpoint has all required keys. Fails loudly on mismatch."""
    missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in obj]
    if missing:
        raise RuntimeError(
            f"Checkpoint {path} missing required keys: {missing}. "
            f"Expected schema version {CHECKPOINT_SCHEMA_VERSION}."
        )
    if obj.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Checkpoint {path} has schema version {obj.get('schema_version')}, "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )


# ---------------------------------------------------------------------------
# checkpoints (spec §30)
# ---------------------------------------------------------------------------

def _saveable(model):
    from assembly import saveable_state_dict
    return saveable_state_dict(model)


def collect_ema_state(model):
    """EMA momentum counters live as plain attributes (not in state_dict) — they
    must be carried explicitly in the checkpoint for exact resume (§30). The EMA
    TARGET ENCODER WEIGHTS are equally required: the JEPA loss predicts the
    target encoder's output, so a resume that rebuilds them from fresh init
    silently trains against wrong targets. Stored CPU-cloned for portability.

    Also collects scalar_mlp_ema if the model exposes one (unified JEPA path)."""
    ema = model.ema
    state = {"momentum_start": ema.momentum_start,
             "momentum_end": ema.momentum_end,
             "total_steps": ema.total_steps}
    target = getattr(ema, "target", None)
    if target is not None:
        state["target"] = {k: v.detach().cpu().clone()
                           for k, v in target.state_dict().items()}
    scalar_ema = getattr(model, "scalar_mlp_ema", None)
    if scalar_ema is not None:
        scalar_state = {
            "momentum_start": scalar_ema.momentum_start,
            "momentum_end": scalar_ema.momentum_end,
            "total_steps": scalar_ema.total_steps,
        }
        scalar_target = getattr(scalar_ema, "target", None)
        if scalar_target is not None:
            scalar_state["target"] = {
                k: v.detach().cpu().clone()
                for k, v in scalar_target.state_dict().items()
            }
        state["scalar_mlp_ema"] = scalar_state
    return state


def restore_ema_state(model, ema_state):
    """Inverse of collect_ema_state; no-op if ema_state is missing/empty.

    A legacy checkpoint without 'target' cannot reconstruct the evolved EMA
    weights — warn loudly rather than silently resuming against a freshly
    initialized target encoder."""
    if not ema_state:
        return
    ema = model.ema
    ema.momentum_start = float(ema_state["momentum_start"])
    ema.momentum_end = float(ema_state["momentum_end"])
    if "total_steps" in ema_state:
        ema.set_total_steps(ema_state["total_steps"])
    saved_target = ema_state.get("target")
    target = getattr(ema, "target", None)
    if target is not None:
        if saved_target is None:
            print("[checkpoint] WARNING: ema_state has no 'target' weights "
                  "(legacy checkpoint) — EMA target encoder left at its "
                  "current init; resumed training will NOT match an "
                  "uninterrupted run.")
        else:
            target.load_state_dict(saved_target)
    # Restore scalar_mlp_ema if the model exposes one (unified JEPA path).
    scalar_ema_state = ema_state.get("scalar_mlp_ema")
    if scalar_ema_state is not None:
        scalar_ema = getattr(model, "scalar_mlp_ema", None)
        if scalar_ema is not None:
            scalar_ema.momentum_start = float(scalar_ema_state["momentum_start"])
            scalar_ema.momentum_end = float(scalar_ema_state["momentum_end"])
            if "total_steps" in scalar_ema_state:
                scalar_ema.set_total_steps(scalar_ema_state["total_steps"])
            scalar_target = getattr(scalar_ema, "target", None)
            if scalar_target is not None:
                saved_scalar = scalar_ema_state.get("target")
                if saved_scalar is None:
                    print("[checkpoint] WARNING: scalar_mlp_ema has no 'target' "
                          "weights — left at current init.")
                else:
                    scalar_target.load_state_dict(saved_scalar)


def _optimizer_param_shapes(optimizer):
    """Ordered per-group parameter shapes of the live optimizer (used for the
    §30 ownership check on load: the optimizer must own exactly the same
    parameter list, in the same order, or it would silently train different
    weights)."""
    if optimizer is None:
        return None
    return [[tuple(p.shape) for p in group["params"]]
            for group in optimizer.param_groups]


def _check_optimizer_ownership(optimizer, saved_shapes):
    if saved_shapes is None:
        return
    cur = _optimizer_param_shapes(optimizer)
    if cur != saved_shapes:
        raise RuntimeError(
            "optimizer parameter-shape fingerprint does not match the "
            "checkpoint — loading its state would train different weights "
            "(spec §30 ownership check)")


def save_checkpoint(path, model, objective, optimizer, scheduler, cfg, global_step,
                    epoch=0, micro_step=0, batch_index=0, is_epoch_end=False, metrics=None, health=None,
                    ema_state=None, best_prediction=None, best_healthy_prediction=None,
                    masker_rng_state=None, device=None, artifact_type="full", extra=None):
    """Save a resumable checkpoint (§30). Mandatory metadata: objective_name,
    objective_state, optimizer state + param-shape ownership fingerprint,
    scheduler state, EMA momentum counters, RNG state, masker RNG state,
    git commit, env versions, device info, cfg, step, epoch, micro_step,
    batch_index, is_epoch_end, best_prediction, best_healthy_prediction, artifact_type.

    Writes atomically: writes to a temporary file then renames.
    """
    metrics = metrics or {}
    git = _git_info()
    env = _env_versions()
    dev_info = _device_info(device)

    # Separate best_prediction and best_healthy_prediction per hardening spec
    if best_prediction is None:
        ratio_key = f"cos_err_r{metrics.get('ratio', 0.5):g}" if 'ratio' in metrics else "cos_err_r0.5"
        best_prediction = {
            "primary": metrics.get(ratio_key, 0.0),
            "metrics": metrics,
            "step": global_step,
            "health": health,
        }
    if best_healthy_prediction is None:
        best_healthy_prediction = {}

    obj = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "objective_name": objective.name,
        "step": global_step,
        "epoch": epoch,
        "micro_step": micro_step,
        "batch_index": batch_index,
        "is_epoch_end": is_epoch_end,
        "cfg": cfg,
        "best_prediction": best_prediction,
        "best_healthy_prediction": best_healthy_prediction,
        "model": _saveable(model),
        "objective_state": objective.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "optimizer_param_shapes": _optimizer_param_shapes(optimizer),
        "scheduler_state": (scheduler.state_dict()
                            if scheduler is not None else None),
        "ema_state": ema_state,
        "rng_state": collect_rng_state(),
        "masker_rng_state": masker_rng_state,
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        "env_versions": env,
        "device_info": dev_info,
        "artifact_type": artifact_type,
    }
    if extra:
        obj.update(extra)

    # Atomic write: write to temp file then rename
    dirname = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(dir=dirname, delete=False, suffix=".pt") as tmp:
        tmp_path = tmp.name
        torch.save(obj, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_checkpoint(path, model, objective, optimizer, scheduler, device,
                    strict_objective=True, strict_optimizer=True, masker=None):
    """Load a checkpoint saved by save_checkpoint. Fails loudly (§30) if the
    objective name does not match (strict) or the optimizer's parameter list has
    diverged from what the checkpoint was saved with. Validates schema.
    Also restores masker RNG state if masker is provided."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint_schema(obj, path)

    saved_name = obj.get("objective_name")
    if strict_objective and saved_name != objective.name:
        raise RuntimeError(
            f"checkpoint {path} was saved for objective {saved_name!r} but "
            f"{objective.name!r} is being loaded — refusing to cross objectives "
            f"(spec §30 strict objective-name match)")
    # Validate optimizer ownership BEFORE any checkpoint state is restored
    # into the model/objective/optimizer: the live optimizer's parameter
    # fingerprint must match the checkpoint's, or loading its state would
    # train different weights. Checking first means a genuine mismatch is
    # raised while all runtime modules are still in their pre-load state
    # (a mis-constructed optimizer is caught before any state mutation).
    if optimizer is not None and obj.get("optimizer") is not None \
            and strict_optimizer:
        _check_optimizer_ownership(optimizer, obj.get("optimizer_param_shapes"))
    from assembly import load_into_model
    load_into_model(model, obj["model"], device)
    if "objective_state" in obj:
        # The unified objective (UnifiedJEPALoss) registers the externally
        # loaded frozen MetaDiT surrogate as a submodule, so its state dict
        # contains "surrogate.*" keys. Phase-B checkpoints predate that
        # submodule and have no such keys; the surrogate is ALWAYS loaded
        # authoritatively from data/metadit/weights/surrogate_model.bin by the
        # trainer, never from a checkpoint. Ignore "surrogate.*" keys on BOTH
        # sides — the checkpoint's (whether absent or stale) and the live
        # objective's (so strict loading does not demand them) — and load
        # every other objective key strictly, so an unrelated missing key
        # still fails loudly. The surrogate module itself is left completely
        # untouched (frozen weights, differentiable input path intact).
        objective_state = {
            k: v for k, v in obj["objective_state"].items()
            if not k.startswith("surrogate.")
        }
        # Load strictly against the filtered expectation by temporarily
        # removing the surrogate submodule, then re-attaching the original.
        # strict=True still raises on any missing non-surrogate objective key.
        surr = getattr(objective, "surrogate", None)
        if surr is not None:
            objective.surrogate = None
        try:
            objective.load_state_dict(objective_state, strict=True)
        finally:
            if surr is not None:
                objective.surrogate = surr
    elif any(p.requires_grad for p in objective.parameters()):
        raise RuntimeError(
            f"checkpoint {path} is missing objective_state for "
            f"{objective.name} which owns trainable parameters — refusing to "
            f"continue with a freshly-initialized projector (spec §12/§30: "
            f"fail loudly)")
    if optimizer is not None and obj.get("optimizer") is not None:
        optimizer.load_state_dict(obj["optimizer"])
    if scheduler is not None and obj.get("scheduler_state") is not None:
        scheduler.load_state_dict(obj["scheduler_state"])
    restore_rng_state(obj.get("rng_state", {}))
    
    # Restore masker RNG state internally (Bug #9)
    if masker is not None:
        masker_state = obj.get("masker_rng_state")
        if masker_state is not None:
            masker.set_rng_state(masker_state)
    
    return obj


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------

def write_json_report(path, report):
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    return path