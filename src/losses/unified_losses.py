"""Phase 3 — Unified JEPA loss components (architecture_v5.md §3-§5), Phase 4
MD §4 (physics loss integration).

Combines:
- L_inv  : MSE on masked occupancy tokens (via shared VICReg projector)
- L_var  : VICReg variance penalty
- L_cov  : VICReg covariance penalty
- L_scalar: L1 regression on UNKNOWN scalar positions only
- L_phys : physics-response loss through frozen MetaDiT surrogate (Phase 4 MD §4)

Full objective:
    L = lambda_inv  * L_inv
      + lambda_var  * L_var
      + lambda_cov  * L_cov
      + lambda_scalar * L_scalar
      + lambda_phys * L_phys

Per Phase 3 MD §5: staged training starts with lambda_phys=0 (physics loss
disabled) until the no-physics architecture is numerically stable. Phase 4
MD §4.1 ramps lambda_phys from 0 over lambda_phys_ramp_steps.

The projector is owned by the objective (spec §17: no model.proj). On
optimizer-step it drives BOTH EMA updates (occupancy + scalar_mlp).
"""

import torch
from torch import nn
import torch.nn.functional as F

from losses.jepa_loss import jepa_loss, ProjectionMLP
from losses.vicreg import vicreg_branch_terms
from losses.objective_modules import VICRegProjector


class OccupancyTokenLoss(nn.Module):
    """JEPA / invariance loss on masked occupancy tokens (standalone).

    Computes MSE between predicted and target latents on masked positions,
    optionally through a projector. Returns (scalar_loss, per_sample).
    """

    def __init__(self, hidden=192, use_proj=True):
        super().__init__()
        self.projector = ProjectionMLP(hidden=hidden) if use_proj else None

    def forward(self, z_hat, z_y_raw, mask):
        return jepa_loss(z_hat, z_y_raw, mask, proj=self.projector)


class ScalarPredictionLoss(nn.Module):
    """L1 regression on scalar parameters at positions marked unknown.

    Only positions where scalar_known is False contribute. Known positions
    are excluded to avoid double-supervision through the known/unknown flag
    (Phase 1 MD §3: missingness is explicit).
    """

    def __init__(self, loss_type="l1"):
        super().__init__()
        assert loss_type in ("l1", "huber"), f"loss_type {loss_type!r} not supported"
        self.loss_type = loss_type

    def forward(self, scalar_pred, scalar_values, scalar_known):
        unknown = ~scalar_known  # (B, 3) bool
        if self.loss_type == "huber":
            err = F.huber_loss(scalar_pred, scalar_values, reduction="none")
        else:
            err = (scalar_pred - scalar_values).abs()
        err = err * unknown.float()
        n_unknown = unknown.sum().clamp(min=1)
        return err.sum() / n_unknown


class PhysicsSpectrumLoss(nn.Module):
    """Physics-response loss (placeholder — disabled in Phase 3).

    When enabled (lambda_phys > 0), the predicted geometry is decoded and
    passed through the frozen MetaDiT EM surrogate to compute spectrum
    error against the target. Currently returns zero; the training loop
    gates activation via lambda_phys > 0.

    Per Phase 3 MD §6: when physics loss is active, the MetaDiT forward
    MUST remain differentiable w.r.t. geometry input — this class should
    only be enabled after the no-physics architecture is numerically
    stable.
    """

    def __init__(self):
        super().__init__()
        self._enabled = False

    def forward(self, spectrum_pred, spectrum_target):
        if not self._enabled:
            return torch.zeros((), device=spectrum_pred.device)
        return F.mse_loss(spectrum_pred, spectrum_target)

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False


class UnifiedJEPALoss(nn.Module):
    """Combined JEPA + VICReg + scalar + (optional) physics objective.

    Architecture: unified_occ_param_spectrum_jepa_v1 (192-D throughout).
    Owns its projector (spec §17: no model.proj). Shares the projector
    between JEPA (invariance) and VICReg (variance + covariance), matching
    the canonical VICReg topology (Bardes et al. 2021) adapted to token-level
    masked geometry.

    Projector gradient ownership (Fix 12, explicitly documented): the
    objective-owned projector is trained from BOTH branches — p_hat =
    projector(z_hat) and p_y = projector(z_y) both flow gradients into the
    projector. This matches the Milestone-B VICRegObjective and canonical
    VICReg: the gradient stops at the EMA target encoder because z_y_raw is
    already detached (stop-grad at the EMA boundary, architecture_v5.md §3.6);
    the projector itself is a shared learnable head. The target branch is NOT
    wrapped in torch.no_grad() because that would freeze the projector's
    target-side updates, diverging from the tested Milestone-B behavior.

    on_optimizer_step updates BOTH EMA targets (occupancy + scalar_mlp)
    per Phase 2 §6.
    """

    name = "unified_jepa"
    term_names = ("L_inv", "L_var", "L_cov", "L_scalar", "L_phys")

    def __init__(self, hidden=192, lambda_inv=25.0, lambda_var=25.0,
                 lambda_cov=1.0, lambda_scalar=1.0, lambda_phys=0.0,
                 gamma=1.0, eps=1e-4, scalar_loss_type="l1",
                 surrogate=None, physics_use_ste=True):
        super().__init__()
        self.projector = VICRegProjector(
            input_dim=hidden, hidden_dim=hidden, output_dim=hidden,
        )
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_scalar = lambda_scalar
        self.lambda_phys = lambda_phys
        self.gamma = gamma
        self.eps = eps
        self.surrogate = surrogate  # frozen MetaDiT EM surrogate (Phase 4)
        # Phase 4 MD §3: STE choice is DOCUMENTED, not silent. The frozen
        # surrogate's ReLU6 activations have a zero Jacobian on soft occupancy
        # fields (verified by soft_hard_occupancy_test), so STE is the
        # empirical default; set physics_use_ste=False only after re-running
        # that check on a surrogate that accepts soft input.
        self.physics_use_ste = physics_use_ste

        self.occupancy_loss = OccupancyTokenLoss(hidden=hidden, use_proj=False)
        self.scalar_loss = ScalarPredictionLoss(loss_type=scalar_loss_type)
        self.physics_loss = PhysicsSpectrumLoss()

    def forward(self, model, occupancy, scalar_values, scalar_known,
                spectrum, mask, goal_mode="real"):
        out = model(
            occupancy, scalar_values, scalar_known, spectrum,
            mask, goal_mode=goal_mode,
        )
        mask_bool = out["mask"]
        z_hat = out["z_hat"]
        z_y = out["z_y_raw"]

        # Projected space (shared projector, single forward per branch)
        p_hat_full = self.projector(z_hat)
        p_y_full = self.projector(z_y)
        p_hat = p_hat_full[mask_bool]
        p_y = p_y_full[mask_bool]

        # JEPA (invariance) + VICReg (var + cov) on masked tokens
        L_inv, L_var, L_cov = vicreg_branch_terms(
            p_hat, p_y, gamma=self.gamma, eps=self.eps)

        L_inv_w = self.lambda_inv * L_inv
        L_var_w = self.lambda_var * L_var
        L_cov_w = self.lambda_cov * L_cov

        # Scalar L1 on unknown positions
        L_scalar = self.scalar_loss(
            out["scalar_pred"], scalar_values, scalar_known)

        # Physics loss: decode geometry → surrogate → spectrum error (Phase 4 MD §4).
        # Reuses the ALREADY-COMPUTED out (z_hat/scalar_pred) via
        # physics_loop.physics_loss_from_out — exactly one student forward per
        # step, one physics decode, one surrogate forward (Fix 11). Delegates
        # to the single authoritative physics implementation.
        if self.lambda_phys > 0 and self.surrogate is not None and model.training:
            from physics.physics_loop import physics_loss_from_out
            L_phys, _, _ = physics_loss_from_out(
                model, out, self.surrogate, occupancy, scalar_values,
                scalar_known, spectrum, mask, loss_type="smooth_l1",
                use_ste=self.physics_use_ste, normalize=True)
        else:
            L_phys = self.physics_loss(
                out.get("spectrum_target", spectrum), spectrum)

        total = (L_inv_w + L_var_w + L_cov_w
                 + self.lambda_scalar * L_scalar
                 + self.lambda_phys * L_phys)

        out["loss_components"] = {
            "L_inv": float(L_inv.detach()), "L_var": float(L_var.detach()),
            "L_cov": float(L_cov.detach()),
            "L_scalar": float(L_scalar.detach()), "L_phys": float(L_phys.detach()),
            "L_inv_weighted": float(L_inv_w.detach()),
            "L_var_weighted": float(L_var_w.detach()),
            "L_cov_weighted": float(L_cov_w.detach()),
            "L_total": float(total.detach()),
        }
        return {
            "total_loss": total,
            "components": out["loss_components"],
            "out": out,
            "projector_inputs": {"z_hat": z_hat, "z_y": z_y},
            "projector_outputs": {"p_hat": p_hat_full, "p_y": p_y_full},
        }

    def on_optimizer_step(self, model, step):
        """Update both EMA targets after optimizer step (Phase 2 §6)."""
        model.ema.update(model.occupancy_encoder, step)
        model.scalar_mlp_ema.update(model.scalar_encoder, step)
