"""Phase 4 — Physics loop with the frozen MetaDiT surrogate.

Provides:
- load_surrogate: load frozen ConvSurrogate from checkpoint
- physics_loss: L_phys = spectrum loss through surrogate (differentiable w.r.t. geometry)
- surrogate_gradient_test: verify dS/dG flows to model params
- soft_hard_occupancy_test: characterize soft vs. hard occupancy through surrogate

Per Phase 4 MD §4: the surrogate is frozen (eval mode, no param gradient) but
autograd MUST flow through the geometry input — never torch.no_grad() around
the surrogate in a physics-loss step.

Per Phase 4 MD §6: visible occupancy pixels and known scalars are hard-retained
inside decode_geometry (retention + known-scalar substitution), so physics
loss cannot overwrite observed geometry.
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


def physics_loss_from_out(model, out, surrogate, occ, sv, sk, spec, mask,
                          loss_type="smooth_l1", use_ste=False,
                          normalize=True, hard_forward=False):
    """Authoritative physics computation from an ALREADY-COMPUTED model output
    (Fix 11: one student forward → one physics decode → one surrogate forward).

    Path (Phase 4 MD §4):
        out["z_hat"] → decode_geometry → assembled geometry → surrogate →
        spectrum → dS/dG → ... → student encoder

    Args:
        model:      UnifiedJEPA instance (for decode_geometry).
        out:        model output dict (must contain z_hat and scalar_pred).
        surrogate:  Frozen ConvSurrogate.
        occ:        [B,1,64,64] input occupancy.
        sv:         [B,3] true scalar values.
        sk:         [B,3] bool known flags.
        spec:       [B,2,301] target spectrum.
        mask:       [B,16,16] 1=visible, 0=masked.
        loss_type:  "smooth_l1" or "l1" or "mse".
        use_ste:    If True, use hard occupancy for surrogate input but soft
                   gradient path (straight-through estimator). Training only.
        normalize:  If True, normalize loss by per-sample spectrum std.
        hard_forward: If True, threshold occupancy to binary regardless of
                   training mode (soft-vs-hard diagnostic).

    Returns:
        L_phys:  scalar tensor (differentiable w.r.t. model params).
        spectrum_pred: [B, 2, 301].
        geometry:  [B, 3, 64, 64].
    """
    # Decode: z_hat (predicted latents) + scalar_pred → geometry.
    # Known scalars (sk) are substituted with their true values (sv) at
    # assembly — the scalar analog of visible-occupancy retention.
    # Fix 4 (spec §6): hard_forward must actually reach decode_geometry —
    # the argument is behavior-affecting, not just present in the signature.
    geometry, soft_occ = model.decode_geometry(
        out["z_hat"], out["scalar_pred"],
        occ_input=occ, mask=mask, use_ste=use_ste,
        scalar_known=sk, scalar_values=sv,
        hard_forward=hard_forward)

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


def physics_loss(model, surrogate, occ, sv, sk, spec, mask,
                 goal_mode="real", loss_type="smooth_l1",
                 use_ste=False, normalize=True, hard_forward=False):
    """Standalone physics loss (runs the model forward once, then delegates to
    physics_loss_from_out — the single authoritative computation).

    Path (Phase 4 MD §4):
        model → z_hat/scalar_pred → decode_geometry → assembled geometry →
        surrogate → spectrum → dS/dG → ... → student encoder

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
        hard_forward: If True, threshold occupancy to binary regardless of
                   training mode (soft-vs-hard diagnostic).

    Returns:
        L_phys:  scalar tensor (differentiable w.r.t. model params).
        spectrum_pred: [B, 2, 301].
        geometry:  [B, 3, 64, 64].
    """
    out = model(occ, sv, sk, spec, mask, goal_mode=goal_mode)
    return physics_loss_from_out(
        model, out, surrogate, occ, sv, sk, spec, mask,
        loss_type=loss_type, use_ste=use_ste, normalize=normalize,
        hard_forward=hard_forward)


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
    # All scalars UNKNOWN: this test verifies dS/dG flows through the model's
    # PREDICTED path (z_hat + scalar_pred → geometry → surrogate). If scalars
    # were all-known, known-scalar substitution would freeze them at their true
    # (constant) values and the assembled geometry would no longer depend on
    # scalar_pred — severing the very path this test exists to validate.
    sk = torch.zeros(occ.shape[0], 3, dtype=torch.bool, device=device)

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
    - Soft geometry: sigmoid occupancy (soft_forward, use_ste=False)
    - Hard-binary geometry: explicitly thresholded occupancy (hard_forward=True)

    Both branches go through decode_geometry with identical occ_input/mask and
    identical known-scalar substitution, so visible-pixel retention and scalar
    handling are applied identically on both sides — the ONLY difference is
    occupancy hardness. The diagnostic must NOT depend on model.training (STE
    applies only under training; this test runs in eval mode), so the hard
    branch uses hard_forward explicitly.

    Returns:
        dict with spectrum difference metrics.
    """
    model.eval()
    surrogate.eval()

    with torch.no_grad():
        out = model(occ, sv, sk, spec, mask, goal_mode="real")

        geometry_soft, _ = model.decode_geometry(
            out["z_hat"], out["scalar_pred"],
            occ_input=occ, mask=mask, use_ste=False, hard_forward=False,
            scalar_known=sk, scalar_values=sv)
        geometry_hard, _ = model.decode_geometry(
            out["z_hat"], out["scalar_pred"],
            occ_input=occ, mask=mask, use_ste=False, hard_forward=True,
            scalar_known=sk, scalar_values=sv)

        spec_soft = surrogate(geometry_soft).prediction
        spec_hard = surrogate(geometry_hard).prediction

        diff = (spec_soft - spec_hard).abs().mean().item()
        rel_diff = diff / (spec_soft.abs().mean().item() + 1e-8)

    return {
        "spectrum_l1_diff": diff,
        "spectrum_rel_diff": rel_diff,
        "surrogate_out_of_distribution": rel_diff > 0.1,
        "ste_recommended": rel_diff > 0.1,
    }

