"""Normalized classifier-free guidance diagnostic (§20.3, Phase 4 MD §3.5.1).

Computes the normalized guidance gap:

    gap / sigma(Z_x) = ||P(Z_x, A_goal) - P(Z_x, A_∅)|| / std(P(Z_x, A_goal))

This is a single-forward-pair diagnostic: one forward with real goal,
one with null goal, measure the normalized difference. A small or
constant gap across very different targets indicates Failure Mode 2
(predictor ignores the spectrum — §13).

When run across a curriculum sweep of mask ratios (20/40/60/80/100%),
the curve should rise with mask ratio — at 0% mask the goal is
redundant (context is complete), at 100% mask the goal is the only
input and collapse → zero gap is a severe failure.
"""

import torch


@torch.no_grad()
def compute_guidance_gap(model, occ, sv, sk, spec, mask, device="cpu"):
    """Single (occ, spec, mask) → normalized guidance gap scalar.

    Args:
        model: UnifiedJEPA (eval mode).
        occ:  [B,1,64,64] occupancy.
        sv:   [B,3] scalar values.
        sk:   [B,3] bool known flags.
        spec: [B,2,301] spectrum.
        mask: [B,16,16] visibility mask.
        device: target device.

    Returns:
        dict with:
            guidance_gap:          float (raw L2-ish mean abs diff)
            normalized_guidance_gap: float (gap / std of real prediction)
            z_real_std:            float
            z_null_std:            float
    """
    model.eval()

    out_real = model(occ, sv, sk, spec, mask,
                     goal_mode="real", with_target=False)
    out_null = model(occ, sv, sk, spec, mask,
                     goal_mode="null", with_target=False)

    z_real = out_real["z_hat"]
    z_null = out_null["z_hat"]

    diff = z_real - z_null
    gap = diff.abs().mean()
    z_real_std = z_real.std()
    z_null_std = z_null.std()

    return {
        "guidance_gap": gap.item(),
        "normalized_guidance_gap": (gap / z_real_std.clamp(min=1e-6)).item(),
        "z_real_std": z_real_std.item(),
        "z_null_std": z_null_std.item(),
    }


@torch.no_grad()
def guidance_gap_sweep(model, occ, sv, spec, masker, ratios, device="cpu"):
    """Compute normalized guidance gap across mask-ratio buckets.

    Args:
        model:     UnifiedJEPA (eval mode).
        occ:       [B,1,64,64] base occupancy.
        sv:        [B,3] scalar values.
        spec:      [B,2,301] spectrum.
        masker:    BlockMasker instance.
        ratios:    list of mask ratios (e.g. [0.2, 0.4, 0.6, 0.8, 1.0]).
        device:    target device.

    Returns:
        dict: {ratio: normalized_guidance_gap}
    """
    sk = torch.ones(occ.shape[0], 3, dtype=torch.bool, device=device)
    results = {}
    for ratio in ratios:
        M = masker.sample(occ, ratio)
        gap_info = compute_guidance_gap(model, occ, sv, sk, spec, M, device)
        results[ratio] = gap_info["normalized_guidance_gap"]
    return results
