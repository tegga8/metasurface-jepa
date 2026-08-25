"""Centralized physics-conditioning controls (hardening spec §8).

Canonical real/null/shuffled-spectrum controls with derangement for shuffled.
Canonical metric names for consistent reporting across evaluators.
"""

import torch
from runtime.device import resolve_device, assert_tensor_device


def derangement_permutation(batch_size: int, device: torch.device | str,
                            seed: int | None = None,
                            generator: torch.Generator | None = None) -> torch.Tensor:
    """Generate a derangement permutation (perm[i] != i for all i).

    A derangement is a permutation with no fixed points. This ensures that
    in the shuffled-spectrum control, no sample receives its own spectrum.

    Args:
        batch_size: Number of samples (must be >= 2).
        device: Target device for the permutation tensor.
        seed: Optional seed for reproducibility (creates new generator).
        generator: Optional existing generator to use.

    Returns:
        Tensor of shape (batch_size,) with deranged indices.

    Raises:
        ValueError: If batch_size < 2.
    """
    if batch_size < 2:
        raise ValueError("derangement requires batch_size >= 2")

    device = resolve_device(device)
    if generator is None:
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

    # Simple rejection sampling for derangement
    # For small batch sizes this is efficient; for large sizes use more sophisticated algorithms
    max_attempts = 100
    for _ in range(max_attempts):
        perm = torch.randperm(batch_size, generator=generator, device=device)
        if not torch.any(perm == torch.arange(batch_size, device=device)):
            return perm
    # Fallback: cyclic shift (guaranteed derangement for n >= 2)
    return torch.roll(torch.arange(batch_size, device=device), shifts=1)


def make_shuffled_spectrum(S: torch.Tensor, generator: torch.Generator | None = None,
                           seed: int | None = None) -> torch.Tensor:
    """Create shuffled spectrum tensor with derangement.

    Args:
        S: Spectrum tensor of shape (B, ...).
        generator: Optional RNG generator.
        seed: Optional seed.

    Returns:
        Shuffled spectrum tensor S_shuf where S_shuf[i] = S[perm[i]] and perm is a derangement.
    """
    b = S.shape[0]
    if b < 2:
        return S  # Can't derange a single sample
    perm = derangement_permutation(b, S.device, seed=seed, generator=generator)
    return S[perm]


# Canonical metric names for physics conditioning (hardening spec §8)
PHYSICS_METRICS = {
    "L_real": "raw_jepa_loss_real",
    "L_null": "raw_jepa_loss_null",
    "L_shuffled": "raw_jepa_loss_shuffled",
    "gap_null": "utility_gap_null",
    "gap_shuffled": "utility_gap_shuffled",
    "sensitivity_null": "predictor_sensitivity_real_vs_null",
    "sensitivity_shuffled": "predictor_sensitivity_real_vs_shuffled",
}


def compute_physics_metrics(model, G, S, M, objective=None, projector=None,
                            device=None, generator=None, seed=None):
    """Compute canonical physics-conditioning metrics for one batch.

    Args:
        model: GoalConditionedJEPA model.
        G: Geometry tensor (B, 3, 64, 64).
        S: Spectrum tensor (B, ...).
        M: Mask tensor (B, 16, 16), 1=visible, 0=masked.
        objective: Optional objective (for projector).
        projector: Optional explicit projector (if not using objective.projector).
        device: Device (resolved from model if None).
        generator: Optional RNG generator for shuffled control.
        seed: Optional seed for shuffled control.

    Returns:
        Dict with canonical metric names:
        - L_real, L_null, L_shuffled: raw masked JEPA cosine loss
        - gap_null, gap_shuffled: L_null - L_real, L_shuffled - L_real
        - sensitivity_null, sensitivity_shuffled: mean ||z_hat(real) - z_hat(null/shuffled)||
    """
    device = resolve_device(device or next(model.parameters()).device)
    G = G.to(device, non_blocking=True)
    S = S.to(device, non_blocking=True)
    M = M.to(device, non_blocking=True)

    # Verify device contract (Bug #17)
    from runtime.device import assert_module_device, assert_tensor_device
    assert_module_device(model, device, "model")
    if objective is not None:
        assert_module_device(objective, device, "objective")
    assert_tensor_device(G, device, "G")
    assert_tensor_device(S, device, "S")
    assert_tensor_device(M, device, "M")

    proj = projector or (objective.projector if objective is not None else None)

    S_shuf = make_shuffled_spectrum(S, generator=generator, seed=seed)

    # Use eval mode for projector (Bug #16) - ensure deterministic BatchNorm
    was_model_training = model.training
    was_objective_training = objective.training if objective is not None else None
    model.eval()
    if objective is not None:
        objective.eval()
    try:
        with torch.no_grad():
            out_r = model(G, S, M, goal_mode="real")
            out_n = model(G, S, M, goal_mode="null")
            out_s = model(G, S_shuf, M, goal_mode="real")

            mask = out_r["mask"]

            def _loss(out):
                z_hat = out["z_hat"]
                z_y = out["z_y_raw"]
                if proj is not None:
                    z_hat = proj(z_hat)
                    z_y = proj(z_y)
                d = (1.0 - torch.nn.functional.cosine_similarity(
                    torch.nn.functional.normalize(z_hat, dim=-1),
                    torch.nn.functional.normalize(z_y, dim=-1), dim=-1)).clamp(min=0)
                return d[mask].mean().item()

            L_real = _loss(out_r)
            L_null = _loss(out_n)
            L_shuf = _loss(out_s)

            sens_null = (out_r["z_hat"] - out_n["z_hat"]).norm(dim=-1)[mask].mean().item()
            sens_shuf = (out_r["z_hat"] - out_s["z_hat"]).norm(dim=-1)[mask].mean().item()
    finally:
        # Restore original modes
        model.train(was_model_training)
        if objective is not None and was_objective_training is not None:
            objective.train(was_objective_training)

    return {
        "L_real": L_real,
        "L_null": L_null,
        "L_shuffled": L_shuf,
        "gap_null": L_null - L_real,
        "gap_shuffled": L_shuf - L_real,
        "sensitivity_null": sens_null,
        "sensitivity_shuffled": sens_shuf,
    }


def validate_goal_mode(goal_mode: str) -> None:
    """Validate goal_mode is one of the allowed values.

    Args:
        goal_mode: String to validate.

    Raises:
        ValueError: If goal_mode not in {"real", "null"}.
    """
    if goal_mode not in ("real", "null"):
        raise ValueError(
            f"goal_mode must be 'real' or 'null', got {goal_mode!r}"
        )