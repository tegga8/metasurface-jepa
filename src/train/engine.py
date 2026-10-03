"""Reusable training-engine pieces shared by the unified training loop and evaluators.

Owns everything the training loop and evaluators must NOT re-derive:

  - Checkpoint save/load with mandatory metadata (objective_name,
    objective_state, optimizer param-shape ownership, scheduler state, EMA
    momentum counters, RNG state, masker RNG state, git commit, env versions)
    and strict objective-name / optimizer-ownership validation on load (§30).
  - EMA state collect/restore (occupancy EMA + scalar_mlp_ema) for exact resume.
  - RNG collect/restore for exact resume (Bug #17).

The strategy/loop lives in `scripts/train/train_unified.py`; this module contains
no adaptive-ladder, phase, or LOSS_LADDER machinery.
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

from runtime.reproducibility import collect_rng_state as _collect_rng_state
from runtime.reproducibility import restore_rng_state as _restore_rng_state

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


def saveable_objective_state(objective):
    """Objective state WITHOUT the frozen surrogate.

    When physics is active the objective registers the released EM surrogate as a
    submodule, so a plain `objective.state_dict()` serialises ~25 MB of frozen
    weights into every checkpoint — measured: the full-epoch checkpoint is 188 MB
    versus 163 MB before physics was switched on. The loader deliberately IGNORES
    `surrogate.*` keys (the surrogate is always re-loaded from
    data/metadit/weights/surrogate_model.bin), so those bytes are pure waste.
    Filtering on the way out makes the two sides symmetric; checkpoints written
    before this change still load, because the loader's own filter is unchanged.
    """
    if objective is None:
        return None
    return {k: v for k, v in objective.state_dict().items()
            if not k.startswith("surrogate.")}


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
        "objective_state": saveable_objective_state(objective),
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