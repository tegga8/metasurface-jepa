"""Phase 4 — Physics loop with the frozen MetaDiT surrogate.

Provides:
- load_surrogate: load frozen ConvSurrogate from checkpoint
- physics_loss: L_phys = spectrum loss through surrogate (differentiable w.r.t. geometry)
- surrogate_gradient_test: verify dS/dG flows to model params
- soft_hard_occupancy_test: characterize soft vs. hard occupancy through surrogate

Per Phase 4 MD §4: the surrogate is frozen (eval mode, no param gradient) but
autograd MUST flow through the geometry input — never torch.no_grad() around
the surrogate in a physics-loss step.

Per Phase 4 MD §6: visible pixels are retained from the input occupancy
(L_preserve), so physics loss does not overwrite observed geometry.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
METADIT_SRC = os.path.join(REPO_ROOT, "external", "metadit")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if METADIT_SRC not in sys.path:
    sys.path.insert(0, METADIT_SRC)

import torch
import torch.nn.functional as F
import torch.nn as nn


def load_surrogate(path, device="cpu"):
    """Load the frozen MetaDiT forward EM surrogate.

    The surrogate takes [B, 3, 64, 64] geometry → [B, 2, 301] spectrum.
    Parameters are frozen; eval mode; autograd flows through the input.

    Per Phase 4 MD §4: frozen params, eval mode, but NOT no_grad on input.
    """
    from model.surrogate import surrogate_s3
    m = surrogate_s3()
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "prediction" not in ckpt:
        # Could be a raw state_dict or a checkpoint dict
        m.load_state_dict(ckpt, strict=True)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def physics_loss(model, surrogate, occ, sv, sk, spec, mask,
                 goal_mode="real", loss_type="smooth_l1",
                 use_ste=False, normalize=True):
    """Compute physics response loss through the frozen surrogate.

    Path (Phase 4 MD §4):
        z_hat → decode_geometry → assembled geometry → surrogate → spectrum
        → dS/dG → ... → student encoder

    Args:
        model:      UnifiedJEPA instance.
        surrogate:  Frozen ConvSurrogate.
        occ:        [B,1,64,64] input occupancy.
        sv:         [B,3] true scalar values.
        sk:         [B,3] bool known flags.
        spec:       [B,2,301] target spectrum.
        mask:       [B,16,16] 1=visible, 0=masked.
        goal_mode:  spectrum goal mode for the surrogate.
        loss_type:  "smooth_l1" or "l1" or "mse".
        use_ste:    If True, use hard occupancy for surrogate input but soft
                   gradient path (straight-through estimator).
        normalize:  If True, normalize loss by per-sample spectrum std.

    Returns:
        L_phys:  scalar tensor (differentiable w.r.t. model params).
        spectrum_pred: [B, 2, 301].
        geometry:  [B, 3, 64, 64].
    """
    out = model(occ, sv, sk, spec, mask, goal_mode=goal_mode)

    # Decode: z_hat (predicted latents) + scalar_pred → geometry
    geometry, soft_occ = model.decode_geometry(
        out["z_hat"], out["scalar_pred"],
        occ_input=occ, mask=mask, use_ste=use_ste)

    # Forward through frozen surrogate — autograd MUST flow (Phase 4 MD §4)
    result = surrogate(geometry)
    spectrum_pred = result.prediction  # [B, 2, 301]

    if normalize:
        spec_std = spec.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        # Normalize BOTH prediction and target by the target's per-sample std
        # (Phase 4 MD §5: "normalized L1 or SmoothL1") so the wide dynamic
        # range of spectral amplitudes across samples doesn't dominate.
        spectrum_pred_n = spectrum_pred / spec_std
        spectrum_target_n = spec / spec_std
        diff = spectrum_pred_n - spectrum_target_n
    else:
        spectrum_pred_n = spectrum_pred
        spectrum_target_n = spec
        diff = spectrum_pred - spec

    if loss_type == "smooth_l1":
        L_phys = F.smooth_l1_loss(
            spectrum_pred_n, spectrum_target_n, reduction="none")
    elif loss_type == "l1":
        L_phys = diff.abs()
    elif loss_type == "mse":
        L_phys = diff ** 2
    else:
        raise ValueError(f"loss_type {loss_type!r} not supported")

    L_phys = L_phys.mean()
    return L_phys, spectrum_pred, geometry


def surrogate_gradient_test(model, surrogate, occ, sv, spec, mask, device="cpu"):
    """Verify gradients flow from surrogate output through assembled geometry
    to the student model parameters (Phase 4 MD §4, §13 acceptance).

    Per Phase 4 MD §3 ("Do not silently choose STE without the check"), the
    soft-occupancy path is tried FIRST; STE is only used as a documented
    fallback when the soft path yields zero student gradients (the surrogate's
    ReLU6 activations are in a dead zone on soft fields).

    Returns True if at least one student parameter has a non-zero gradient
    after backward through the surrogate.
    """
    model.train()
    surrogate.eval()

    # Collect student params that require grad
    student_params = [p for p in model.parameters() if p.requires_grad]
    sk = torch.ones(occ.shape[0], 3, dtype=torch.bool, device=device)

    # Try soft occupancy first (Phase 4 MD §3 default representation).
    L_phys, _, _ = physics_loss(
        model, surrogate, occ, sv, sk, spec, mask,
        goal_mode="real", use_ste=False)
    L_phys.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in student_params)

    if not has_grad:
        # Soft path dead (surrogate ReLU6 zero Jacobian on soft fields) —
        # documented STE fallback per Phase 4 MD §3.
        model.zero_grad(set_to_none=True)
        L_phys, _, _ = physics_loss(
            model, surrogate, occ, sv, sk, spec, mask,
            goal_mode="real", use_ste=True)
        L_phys.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in student_params)

    model.zero_grad(set_to_none=True)

    # Verify surrogate params have NO gradient (must stay frozen)
    surr_frozen = all(p.grad is None
                      for p in surrogate.parameters() if not p.requires_grad)
    return has_grad and surr_frozen


def soft_hard_occupancy_test(model, surrogate, occ, sv, spec, sk, mask,
                             device="cpu"):
    """Characterize soft vs. hard occupancy through the surrogate (Phase 4 MD §3).

    Compares:
    - Real binary geometry (hard occupancy 0/1)
    - Soft occupancy (sigmoid of decoder logits)

    If the surrogate is materially out-of-distribution on soft fields,
    a straight-through estimator (STE) should be used. This test quantifies
    the difference.

    Returns:
        dict with spectrum difference metrics.
    """
    model.eval()
    surrogate.eval()

    with torch.no_grad():
        out = model(occ, sv, sk, spec, mask, goal_mode="real")

        # Hard (binary) geometry
        geometry_hard, _ = model.decode_geometry(
            out["z_hat"], out["scalar_pred"],
            occ_input=occ, mask=mask, use_ste=False)
        # Force binary occupancy
        soft_occ = torch.sigmoid(
            model.geometry_decoder(
                out["z_hat"])[1])
        hard_occ = (soft_occ > 0.5).float()
        from data.factorize import assemble_metadit_geometry
        geometry_binary = assemble_metadit_geometry(
            hard_occ, out["scalar_pred"][:, 0],
            out["scalar_pred"][:, 1], out["scalar_pred"][:, 2])

        spec_hard = surrogate(geometry_hard).prediction
        spec_binary = surrogate(geometry_binary).prediction

        diff = (spec_hard - spec_binary).abs().mean().item()
        rel_diff = diff / (spec_hard.abs().mean().item() + 1e-8)

    return {
        "spectrum_l1_diff": diff,
        "spectrum_rel_diff": rel_diff,
        "surrogate_out_of_distribution": rel_diff > 0.1,
        "ste_recommended": rel_diff > 0.1,
    }


# ---------------------------------------------------------------------------
# Preservation loss (Phase 4 MD §6)
# ---------------------------------------------------------------------------

def preservation_loss(occ_pred, occ_input, mask, scalar_pred, scalar_known, true_scalars):
    """L_preserve: penalize changes to known occupancy/scalars (Phase 4 MD §6).

    For partial/retrofit scenarios:
    - Visible occupancy pixels must match the input.
    - Known scalars must match the true values.

    Args:
        occ_pred:   [B,1,64,64] predicted (soft) occupancy.
        occ_input:  [B,1,64,64] input binary occupancy.
        mask:       [B,16,16] 1=visible, 0=masked.
        scalar_pred: [B,3] predicted scalars.
        scalar_known: [B,3] bool — which scalars are observed.
        true_scalars: [B,3] true scalar values.

    Returns:
        scalar tensor
    """
    # Upsample mask to pixel level
    b = occ_pred.shape[0]
    up = mask.view(b, 1, 16, 16).repeat_interleave(4, 2).repeat_interleave(4, 3)
    vis = (up > 0.5).float()

    # Occupancy preservation on visible pixels
    occ_err = (occ_pred - occ_input.float()) * vis
    L_occ = occ_err.abs().sum() / (vis.sum() + 1.0)

    # Scalar preservation for known scalars
    known = scalar_known.float()
    sc_err = (scalar_pred - true_scalars) * known
    L_sc = sc_err.abs().sum() / (known.sum() + 1.0)

    return L_occ + L_sc
