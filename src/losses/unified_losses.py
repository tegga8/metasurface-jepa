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

from losses.vicreg import VICRegProjector, build_projector, vicreg_branch_terms


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


class SummaryScalarReadout(nn.Module):
    """Auxiliary read-out on the scalar-summary token (door (a) of the scalar
    investigation, operator decision 2026-09-13).

    The scalar encoder produces a pooled summary token that the fusion encoder
    hands to the predictor as a key/value entry. That token had **no objective of
    its own** — it is a pure intermediate — so nothing pushed it to carry the
    conditioning. Measured consequences (REPORT.md §17, §19, §20): the token varies
    by only ~11 % when the scalars change, the predictor attends to it at 34 % of
    the scalar query's attention (93.6x uniform), yet `scalar_pred` moves 0.000145
    under a scalar perturbation against 0.246 under a spectrum perturbation; and
    `L_scalar` delivers 0.0036 of gradient to the scalar encoder against 0.6932 to
    the scalar decoder. Scaling the token 100x at inference did not fix it (§20.1),
    which ruled out magnitude as the binding constraint.

    Reconstructing the KNOWN scalar values from the token gives the encoder a short,
    direct gradient — the only path from a scalar objective to the encoder that does
    not run through the predictor. It is scored only where a scalar is KNOWN: where
    the input was zeroed there is nothing to recover, and the head never sees the raw
    values, so it cannot learn to copy them.
    """

    def __init__(self, hidden=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, summary_token):
        """summary_token: (B, 1, hidden) -> (B, 3) known-scalar estimates."""
        return self.net(summary_token.squeeze(1))


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
    term_names = ("L_inv", "L_var", "L_cov", "L_scalar", "L_occ", "L_phys",
                  "L_summary")

    def __init__(self, hidden=192, lambda_inv=25.0, lambda_var=25.0,
                 lambda_cov=1.0, lambda_scalar=1.0, lambda_phys=0.0,
                 lambda_occ=0.0,
                 gamma=1.0, eps=1e-4, scalar_loss_type="l1",
                 surrogate=None, physics_use_ste=True, lambda_summary=0.0,
                 projector_type="mlp_bn"):
        super().__init__()
        # Phase 3b: objective-owned projector, ablable via loss.projector_type
        # (none | linear | mlp | mlp_bn). Spec §17: there is no model.proj.
        self.projector = build_projector(projector_type, hidden)
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_scalar = lambda_scalar
        self.lambda_phys = lambda_phys
        self.lambda_occ = lambda_occ
        self.gamma = gamma
        self.eps = eps
        self.surrogate = surrogate  # frozen MetaDiT EM surrogate (Phase 4)
        # Phase 4 MD §3: STE choice is DOCUMENTED, not silent. The frozen
        # surrogate's ReLU6 activations have a zero Jacobian on soft occupancy
        # fields (verified by soft_hard_occupancy_test), so STE is the
        # empirical default; set physics_use_ste=False only after re-running
        # that check on a surrogate that accepts soft input.
        self.physics_use_ste = physics_use_ste
        # Door (a): weight of the summary-token read-out. 0.0 keeps the shipped
        # behaviour bit-identical (the term is then exactly zero).
        self.lambda_summary = lambda_summary

        # Fix 6 (spec §8): self.occupancy_loss removed — it was constructed
        # but never called; forward() computes the projected JEPA/VICReg
        # terms inline via the shared objective-owned projector. Keeping an
        # unused module here would be dead, misleading code.
        self.scalar_loss = ScalarPredictionLoss(loss_type=scalar_loss_type)
        self.summary_readout = SummaryScalarReadout(hidden=hidden)

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
        #
        # Audit B5: null-goal (CFG dropout) steps SKIP the physics term. Its
        # target is the sample's true spectrum — the very condition that was
        # dropped — so training it there would push the unconditional branch
        # toward outputs it cannot infer (goal-ignoring / mode-collapse
        # pressure, architecture_v5.md §8.3 check 8). The spectrum-free
        # objectives (VICReg invariance/variance/covariance, scalar regression)
        # still train that branch; the conditional branch keeps L_phys.
        physics_active = (
            self.lambda_phys > 0 and self.surrogate is not None
            and model.training and goal_mode != "null"
        )
        if physics_active:
            from physics.physics_loop import physics_loss_from_out
            L_phys, _, _ = physics_loss_from_out(
                model, out, self.surrogate, occupancy, scalar_values,
                scalar_known, spectrum, mask, loss_type="smooth_l1",
                use_ste=self.physics_use_ste, normalize=True)
        else:
            # The physics term is inactive (lambda_phys = 0, no surrogate, eval
            # mode, or a goal-dropped step). It is genuinely zero here, not
            # "measured as zero" - validation reports it as not-evaluated for
            # that reason (audit B28).
            L_phys = torch.zeros((), device=z_hat.device)

        # Occupancy BCE (architecture_v5.md §4.1; operator decision 2026-09-13):
        # direct supervision of the decoder on MASKED pixels. Without it the
        # decoder is trained only through the physics path, so it receives no
        # gradient at all while lambda_phys = 0 (staging B). Visible pixels are
        # retained at assembly (never overwritten), so the decoder is asked to
        # infer only what was masked — the same masked-region convention as the
        # latent objective.
        if self.lambda_occ > 0:
            occ_logits = model.decode_occupancy_logits(
                z_hat, out["scalar_pred"],
                scalar_known=scalar_known, scalar_values=scalar_values)
            b = occ_logits.shape[0]
            masked_px = out["mask"].view(b, 1, 16, 16).repeat_interleave(
                4, 2).repeat_interleave(4, 3) > 0.5          # (B,1,64,64)
            if masked_px.any():
                L_occ = F.binary_cross_entropy_with_logits(
                    occ_logits[masked_px], occupancy[masked_px])
            else:
                L_occ = torch.zeros((), device=occ_logits.device)
        else:
            L_occ = torch.zeros((), device=z_hat.device)

        # Door (a): reconstruct the KNOWN scalars from the summary token, so the
        # scalar encoder is trained to make that token carry the conditioning.
        # Scored on known positions only: unknown values were zeroed before the
        # encoder saw them, so there is nothing to recover there.
        if self.lambda_summary > 0 and scalar_known.any():
            readout = self.summary_readout(out["scalar_summary"])
            summary_err = (readout - scalar_values).abs() * scalar_known.float()
            L_summary = summary_err.sum() / scalar_known.sum().clamp(min=1)
        else:
            L_summary = torch.zeros((), device=z_hat.device)

        total = (L_inv_w + L_var_w + L_cov_w
                 + self.lambda_scalar * L_scalar
                 + self.lambda_occ * L_occ
                 + self.lambda_phys * L_phys
                 + self.lambda_summary * L_summary)

        # Per-term WEIGHTED losses, exposed so the trainer can attribute the
        # gradient budget live (Phase 2a of the roadmap — until now the physics
        # share was only measured post-hoc). Behaviour-neutral: `total` is
        # unchanged; these tensors are read only by the gradient-share probe.
        out["term_losses"] = {
            "L_inv": L_inv_w, "L_var": L_var_w, "L_cov": L_cov_w,
            "L_scalar": self.lambda_scalar * L_scalar,
            "L_occ": self.lambda_occ * L_occ,
            "L_phys": self.lambda_phys * L_phys,
            "L_summary": self.lambda_summary * L_summary,
        }

        out["loss_components"] = {
            "L_inv": float(L_inv.detach()), "L_var": float(L_var.detach()),
            "L_cov": float(L_cov.detach()),
            "L_scalar": float(L_scalar.detach()), "L_occ": float(L_occ.detach()),
            "L_phys": float(L_phys.detach()),
            "L_inv_weighted": float(L_inv_w.detach()),
            "L_var_weighted": float(L_var_w.detach()),
            "L_cov_weighted": float(L_cov_w.detach()),
            "L_occ_weighted": float((self.lambda_occ * L_occ).detach()),
            "L_phys_weighted": float((self.lambda_phys * L_phys).detach()),
            "L_summary": float(L_summary.detach()),
            "L_summary_weighted": float((self.lambda_summary * L_summary).detach()),
            "L_total": float(total.detach()),
        }
        return {
            "total_loss": total,
            "components": out["loss_components"],
            "term_losses": out["term_losses"],
            "out": out,
            "projector_inputs": {"z_hat": z_hat, "z_y": z_y},
            "projector_outputs": {"p_hat": p_hat_full, "p_y": p_y_full},
        }

    def train(self, mode=True):
        """Override: the frozen MetaDiT surrogate registered as self.surrogate
        must stay in EVAL mode regardless of the objective's training mode.

        Diagnostic-protocol finding: objective.train() recursively put the
        surrogate's 38 BatchNorm2d layers into train mode, so every training
        physics forward normalized by BATCH statistics (batch size 2) and
        mutated BN running stats — the physics loss trained against a
        corrupted surrogate, and surrogate outputs for identical geometry
        differed before/after the first train-mode call (measured: spectrum
        error 19.58 eval vs 0.59 train-BN for the same geometry). Eval-mode
        BN remains differentiable w.r.t. its input, so the physics gradient
        path is unchanged.
        """
        super().train(mode)
        if self.surrogate is not None:
            self.surrogate.eval()
        return self

    def on_optimizer_step(self, model, step):
        """Update both EMA targets after optimizer step (Phase 2 §6)."""
        model.ema.update(model.occupancy_encoder, step)
        model.scalar_mlp_ema.update(model.scalar_encoder, step)
