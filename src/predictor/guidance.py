"""Classifier-free goal guidance (architecture_v5.md §3.5.1, Phase 4 MD §3.5.1).

During training: replace A_goal with learned null token A_∅ with probability
~10% (goal dropout). At inference: combine

    Z_guided = P(Z_x, A_∅) + w · [P(Z_x, A_goal) − P(Z_x, A_∅)]

The model's forward already supports goal_mode in {"real", "null", "shuffled"}
via spectrum_path. This module provides the CFG combine logic and the
guidance-gap diagnostic (§20.3).
"""

import torch


def cfg_combine(z_real, z_null, w):
    """Combine real-goal and null-goal predictions via classifier-free guidance.

    Z_guided = z_null + w * (z_real - z_null)

    Args:
        z_real:  prediction with real goal, shape (B, ...)
        z_null:  prediction with null goal, shape (B, ...)
        w:       guidance scale (float or tensor)

    Returns:
        z_guided: same shape as inputs.
    """
    return z_null + w * (z_real - z_null)


def goal_dropout(goal_mode, p, rng=None):
    """Randomly replace goal_mode with "null" for classifier-free training.

    Args:
        goal_mode: current goal mode string.
        p:         dropout probability (e.g. 0.1).
        rng:       optional torch.Generator for reproducibility.

    Returns:
        Possibly modified goal_mode string.
    """
    if goal_mode == "null":
        return "null"
    if p <= 0:
        return goal_mode
    if rng is not None:
        roll = torch.rand(1, generator=rng).item()
    else:
        roll = torch.rand(1).item()
    return "null" if roll < p else goal_mode


@torch.no_grad()
def cfg_forward(model, occ, sv, sk, spec, mask, w, device="cpu"):
    """Run classifier-free guidance inference.

    Performs two forward passes (real goal and null goal) and combines
    the predicted z_hat via cfg_combine. Uses with_target=False for speed
    (no EMA target needed at inference).

    Args:
        model:     UnifiedJEPA instance (in eval mode).
        occ:       [B,1,64,64] occupancy.
        sv:        [B,3] scalar values (true values for conditioning).
        sk:        [B,3] bool known flags.
        spec:      [B,2,301] spectrum.
        mask:      [B,16,16] visibility mask.
        w:         guidance scale.
        device:    target device.

    Returns:
        z_hat_guided: (B, 256, hidden) guided prediction.
        info: dict with raw z_hat_real, z_hat_null, and guidance gap scalar.
    """
    model.eval()

    # Real-goal forward
    out_real = model(occ, sv, sk, spec, mask,
                     goal_mode="real", with_target=False)
    z_real = out_real["z_hat"]

    # Null-goal forward
    out_null = model(occ, sv, sk, spec, mask,
                     goal_mode="null", with_target=False)
    z_null = out_null["z_hat"]

    # Combine
    z_guided = cfg_combine(z_real, z_null, w)

    # Guidance gap: ||z_real - z_null|| / sigma(z_real)
    diff = (z_real - z_null)
    gap = diff.abs().mean().item()
    std_real = z_real.std().item()
    norm_gap = gap / max(std_real, 1e-6)

    info = {
        "z_hat_real": z_real,
        "z_hat_null": z_null,
        "guidance_gap": gap,
        "normalized_guidance_gap": norm_gap,
    }
    return z_guided, info
