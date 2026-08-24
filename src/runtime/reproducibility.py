"""Canonical RNG and reproducibility utilities.

Single source of truth for seed setting and RNG state capture/restore.
All first-party training/evaluation code must use these utilities.
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all RNG seeds deterministically.

    Args:
        seed: Base seed value. Used directly for torch, numpy, random;
              CUDA gets seed_all(seed) for all visible devices.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_rng_state() -> dict:
    """Collect complete RNG state for exact resume.

    Returns:
        Dict with keys:
        - torch_rng: CPU torch RNG state (ByteTensor)
        - numpy_rng: numpy RandomState state tuple
        - python_rng: Python random module state tuple
        - torch_cuda_rng: list of per-device CUDA RNG states (ByteTensors),
          or None if CUDA not available
    """
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    return {
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "torch_cuda_rng": cuda_state,
    }


def restore_rng_state(state: dict) -> None:
    """Restore RNG state from collect_rng_state() output.

    Args:
        state: Dict as returned by collect_rng_state(). Missing/None
               entries are skipped safely. CUDA state saved on a GPU
               machine is skipped when restoring on CPU-only.
    """
    if state.get("torch_rng") is not None:
        torch.set_rng_state(state["torch_rng"].cpu())
    if state.get("numpy_rng") is not None:
        np.random.set_state(state["numpy_rng"])
    if state.get("python_rng") is not None:
        random.setstate(state["python_rng"])
    cuda = state.get("torch_cuda_rng")
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda)


def fork_rng(seed: int | None = None, devices=None):
    """Context manager for isolated RNG fork (like torch.random.fork_rng).

    Args:
        seed: If provided, manual_seed(seed) is called inside the fork.
        devices: CUDA devices to fork (default: all visible).

    Returns:
        Context manager that restores RNG state on exit.
    """
    return torch.random.fork_rng(devices=devices, enabled=True)


def deterministic_reference_build(build_fn, seed: int = 2026):
    """Build a deterministic reference model under a fixed RNG context.

    The reference build consumes the ambient RNG (for context encoder,
    predictor init, etc.). To make the reference a pure function of `seed`,
    run inside fork_rng with manual_seed(seed). The ambient RNG state
    is restored untouched afterwards.

    Args:
        build_fn: Zero-arg callable that constructs the reference model.
        seed: Seed for the reference build (default 2026 per spec).

    Returns:
        Whatever build_fn returns.
    """
    with fork_rng():
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return build_fn()