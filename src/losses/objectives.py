"""Pluggable training objectives for the Milestone B adaptive loss ladder (§1–§7).

One interface, one objective per ladder rung, zero objective-specific math in the
training loop:

    jepa        : L_J (masked positions only, projection applied, EMA target stop-grad)
    jepa_var    : adds a variance floor only: L_J + lambda_var * L_var
    jepa_vicreg : L_J + lambda_var * L_var + lambda_cov * L_cov   (projected masked preds)
    jepa_vicreg2: same variance+covariance pressure on BOTH projected branches
                  (masked predictions AND masked EMA targets; the target encoder is
                  frozen, so this path intentionally trains the shared projector only)
    jepa_barlow : L_J + lambda_bt * L_BT (Barlow cross-correlation on the same two
                  projected branches — redundancy reduction via the off-diagonal)
    lejepa      : L_J (student target, no EMA) + lambda_sigreg * L_SIGReg  (sliced ECF)

Each objective returns, per step:
    {"total_loss": tensor, "components": {name: tensor/float, ...}, "out": model out}
and owns the per-step auxiliary updates via `on_optimizer_step` (EMA momentum update
for the two EMA-based objectives; no-op for LeJEPA which has no teacher copy).

Component logging contract (screening ladder): every regularized rung reports its
unweighted regularizer terms (L_var/L_cov split per branch for jepa_vicreg2, L_BT with
diag/off-diag, L_SIGReg), the lambda-weighted regularizer (`*_weighted`), and the
regularizer's share of the total loss (`var_ratio` / `cov_ratio` / `barlow_ratio` /
`sigreg_ratio`), all as tensors so the training loop's tensor-only accumulator picks
them up. The plain `jepa` rung has no regularizer and reports L_J only.
"""

import torch
from torch import nn

from losses.barlow import barlow_twins_loss
from losses.jepa_loss import jepa_loss
from losses.objective_modules import VICRegProjector
from losses.sigreg import sigreg_loss
from losses.vicreg import (covariance_loss, invariance_loss, variance_loss,
                           vicreg_branch_terms, vicreg_loss)


def _mask_from_M(M):
    return (M.view(M.shape[0], -1) == 0)  # (B, 256) bool, 1 = masked


class JEPAObjective:
    """Phase 0 — current EMA-JEPA, exactly the approved B1+B2 objective:

    L_J = (1/|M|) * sum_{i in M} [1 - cos(P(z_hat_i), P(z_i^y))]
    P = projection head, z^y = EMA target (stop-gradient), masked positions only.

    The corrected 256x384 architecture has NO model.proj (the refactor removed
    the shared projection head); `proj` therefore falls back to None (raw-space
    cosine, matching GoalConditionedJEPA.loss) when the model does not own one.
    """

    name = "jepa"

    def __call__(self, model, G, S, M):
        out = model(G, S, M)
        mask = out["mask"] if "mask" in out else _mask_from_M(M)
        L_J, _ = jepa_loss(out["z_hat"], out["z_y"], mask,
                           proj=getattr(model, "proj", None))
        return {"total_loss": L_J, "components": {"L_J": L_J}, "out": out}

    def on_optimizer_step(self, model, step):
        model.ema.update(model.geometry_encoder, step)


class JEPAVICRegObjective:
    """Phase 1 — JEPA + small VICReg-style variance/covariance regularization.

    L = L_J + lambda_var * L_var + lambda_cov * L_cov on the PROJECTED predictions
    of masked tokens (the same features the JEPA cosine sees). The EMA target is
    untouched (stop-gradient preserved).
    """

    name = "jepa_vicreg"

    def __init__(self, lambda_var=0.1, lambda_cov=0.04, gamma=1.0, cov_on=True):
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma
        self.cov_on = cov_on

    def __call__(self, model, G, S, M):
        out = model(G, S, M)
        mask = out["mask"] if "mask" in out else _mask_from_M(M)
        L_J, _ = jepa_loss(out["z_hat"], out["z_y"], mask, proj=model.proj)
        zh = model.proj(out["z_hat"])[mask]                 # (N, D) masked predictions
        L_var, L_cov = vicreg_loss(zh, gamma=self.gamma, cov_on=self.cov_on)
        L_var_w = self.lambda_var * L_var
        L_cov_w = self.lambda_cov * L_cov
        total = L_J + L_var_w + L_cov_w
        var_ratio = L_var_w / total.clamp_min(1e-8)
        cov_ratio = L_cov_w / total.clamp_min(1e-8)
        return {"total_loss": total,
                "components": {"L_J": L_J, "L_var": L_var, "L_cov": L_cov,
                               "L_var_weighted": L_var_w, "L_cov_weighted": L_cov_w,
                               "var_ratio": var_ratio, "cov_ratio": cov_ratio,
                               "lambda_var": self.lambda_var, "lambda_cov": self.lambda_cov},
                "out": out}

    def on_optimizer_step(self, model, step):
        model.ema.update(model.geometry_encoder, step)


class JEPAVICRegDualObjective:
    """Corrected branch-symmetric VICReg (ladder rung `jepa_vicreg2`).

    The historical jepa_vicreg (Phase 1, Kaggle: COLLAPSED at 400/500 under
    lambda_var=0.1, lambda_cov=0.04, gamma=1.0) regularizes only the prediction
    branch. The review's core point: the diversity pressure must be applied to
    BOTH projected branches. Here the variance/covariance regularization is
    computed independently on the projected masked predictions AND on the
    projected masked EMA targets.

    Gradient boundary (design-critical, tested separately): the EMA *target
    encoder* is frozen, so gradient through the target encoder is impossible;
    the shared projector is a separate learnable head, and this target
    regularization path intentionally trains it. The original JEPA cosine path
    inside jepa_loss() remains detached (Bug #1 contract, test_jepa_loss.py) —
    it is never switched back to a target-gradient path.
    """

    name = "jepa_vicreg2"

    def __init__(self, lambda_var=0.1, lambda_cov=0.04, gamma=1.0, cov_on=True):
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma
        self.cov_on = cov_on

    def __call__(self, model, G, S, M):
        out = model(G, S, M)
        mask = out["mask"] if "mask" in out else _mask_from_M(M)

        # Existing JEPA loss; target path remains detached here.
        L_J, _ = jepa_loss(
            out["z_hat"],
            out["z_y"],
            mask,
            proj=model.proj,
        )

        # Prediction branch regularization path.
        zh = model.proj(out["z_hat"])[mask]

        # Separate target regularization path.
        # Target encoder is frozen; projector gradient is intentionally allowed.
        zt = model.proj(out["z_y"])[mask]

        L_var_p, L_cov_p = vicreg_loss(
            zh,
            gamma=self.gamma,
            cov_on=self.cov_on,
        )

        L_var_t, L_cov_t = vicreg_loss(
            zt,
            gamma=self.gamma,
            cov_on=self.cov_on,
        )

        L_var = L_var_p + L_var_t
        L_cov = L_cov_p + L_cov_t

        L_var_w = self.lambda_var * L_var
        L_cov_w = self.lambda_cov * L_cov

        total = L_J + L_var_w + L_cov_w
        var_ratio = L_var_w / total.clamp_min(1e-8)
        cov_ratio = L_cov_w / total.clamp_min(1e-8)

        return {
            "total_loss": total,
            "components": {
                "L_J": L_J,
                "L_var_pred": L_var_p,
                "L_var_target": L_var_t,
                "L_cov_pred": L_cov_p,
                "L_cov_target": L_cov_t,
                "L_var": L_var,
                "L_cov": L_cov,
                "L_var_weighted": L_var_w,
                "L_cov_weighted": L_cov_w,
                "var_ratio": var_ratio,
                "cov_ratio": cov_ratio,
                "lambda_var": self.lambda_var,
                "lambda_cov": self.lambda_cov,
            },
            "out": out,
        }

    def on_optimizer_step(self, model, step):
        model.ema.update(model.geometry_encoder, step)


class VICRegObjective(nn.Module):
    """Faithful EMA-JEPA + VICReg-style regularization (candidate `jepa_vicreg`).

    Canonical VICReg (Bardes et al. 2021) uses two trainable views through a
    shared encoder/projector and combines invariance + variance + covariance on
    the PROJECTED branch outputs. This project's topology is instead:

        masked geometry -> context encoder -> predictor      -> z_hat
        full geometry   -> EMA target encoder (FROZEN)       -> z_y

    so the faithful adaptation is: z_hat and z_y both pass through the
    objective-owned VICReg projector (the original implementation has a
    dedicated projector), and

        L_total = lambda_inv * L_inv + lambda_var * L_var + lambda_cov * L_cov

    with

        L_inv = MSE(p_hat, p_y)                                   (both branches)
        L_var = 0.5 * (var_penalty(p_hat) + var_penalty(p_y))     (both branches)
        L_cov = 0.5 * (cov_penalty(p_hat) + cov_penalty(p_y))     (both branches)

    computed over MASKED geometry tokens — the deliberate token-level
    adaptation (§6 of the CODEX spec; canonical VICReg is image-level). The
    EMA target encoder stays frozen: its parameters receive no gradient and
    are updated only through the EMA operation. The raw 384-D latent is NOT
    the projector output; the projector defines the VICReg objective space
    only, and raw-space health is reported separately.

    Gradient boundary (tested): L_inv/L_var/L_cov each flow student ->
    projector; the target branch reaches the objective-owned projector but
    never the frozen EMA encoder.
    """

    name = "jepa_vicreg"

    def __init__(self, lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0,
                 gamma=1.0, eps=1e-4,
                 projector_input_dim=384, projector_hidden_dim=384,
                 projector_output_dim=384):
        super().__init__()
        self.projector = VICRegProjector(
            input_dim=projector_input_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
        )
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma
        self.eps = eps

    def forward(self, model, G, S, M):
        out = model(G, S, M)
        mask = out["mask"] if "mask" in out else _mask_from_M(M)
        z_hat, z_y = out["z_hat"], out["z_y"]

        # Projector applied once per branch; masked tokens and geometry-level
        # pooled vectors are derived from the SAME projected outputs (never
        # re-running the projector, which would double-update BN statistics).
        p_hat_full = self.projector(z_hat)                 # (B, 256, D)
        p_y_full = self.projector(z_y)                     # (B, 256, D)
        p_hat = p_hat_full[mask]                           # (N, D) masked tokens
        p_y = p_y_full[mask]                               # (N, D) masked tokens

        # Token-level statistics are the loss (deliberate adaptation, §6).
        L_inv, L_var, L_cov = vicreg_branch_terms(
            p_hat, p_y, gamma=self.gamma, eps=self.eps)

        L_inv_w = self.lambda_inv * L_inv
        L_var_w = self.lambda_var * L_var
        L_cov_w = self.lambda_cov * L_cov
        total = L_inv_w + L_var_w + L_cov_w

        # Geometry-level health (§6: "Never rely only on token-level
        # statistics"): masked tokens mean-pooled per geometry -> (B, D),
        # then the same cross-geometry statistics. Reported as components,
        # not added to the loss.
        mw = mask.float()
        p_hat_g = (p_hat_full * mw.unsqueeze(-1)).sum(1) \
            / mw.sum(1, keepdim=True).clamp(min=1)        # (B, D)
        p_y_g = (p_y_full * mw.unsqueeze(-1)).sum(1) \
            / mw.sum(1, keepdim=True).clamp(min=1)        # (B, D)
        geo_inv, geo_var, geo_cov = vicreg_branch_terms(
            p_hat_g, p_y_g, gamma=self.gamma, eps=self.eps)

        inv_ratio = L_inv_w / total.clamp_min(1e-8)
        var_ratio = L_var_w / total.clamp_min(1e-8)
        cov_ratio = L_cov_w / total.clamp_min(1e-8)

        return {
            "total_loss": total,
            "components": {
                "L_inv": L_inv, "L_var": L_var, "L_cov": L_cov,
                "L_inv_weighted": L_inv_w,
                "L_var_weighted": L_var_w,
                "L_cov_weighted": L_cov_w,
                "inv_ratio": inv_ratio,
                "var_ratio": var_ratio,
                "cov_ratio": cov_ratio,
                "lambda_inv": self.lambda_inv,
                "lambda_var": self.lambda_var,
                "lambda_cov": self.lambda_cov,
                # geometry-level (pooled) health statistics
                "geo_inv": geo_inv, "geo_var": geo_var, "geo_cov": geo_cov,
            },
            "out": out,
            "projector_inputs": {"z_hat": z_hat, "z_y": z_y},
            "projector_outputs": {"p_hat": p_hat_full, "p_y": p_y_full},
        }

    def on_optimizer_step(self, model, step):
        model.ema.update(model.geometry_encoder, step)


class JEPABarlowObjective:
    """Barlow-style redundancy reduction (ladder rung `jepa_barlow`).

    L = L_J + lambda_bt * L_BT with L_BT the Barlow Twins cross-correlation
    loss between the projected masked predictions and the projected masked EMA
    targets (both standardized, diag -> match, off-diag -> de-redundancy). The
    diagonal targets the observed dimensional-collapse pathology (low effective
    rank); the off-diagonal penalizes redundant dimensions directly.

    Gradient boundary (same design as jepa_vicreg2, tested): the JEPA cosine
    path inside jepa_loss() remains detached; the Barlow term runs on a
    SEPARATE projection path, where the shared projector may be trained from
    the frozen EMA target's output (target encoder parameters stay frozen).
    """

    name = "jepa_barlow"

    def __init__(self, lambda_bt=1.0, alpha=0.005):
        self.lambda_bt = lambda_bt
        self.alpha = alpha

    def __call__(self, model, G, S, M):
        out = model(G, S, M)
        mask = out["mask"] if "mask" in out else _mask_from_M(M)

        L_J, _ = jepa_loss(
            out["z_hat"],
            out["z_y"],
            mask,
            proj=model.proj,
        )

        zh = model.proj(out["z_hat"])[mask]
        zt = model.proj(out["z_y"])[mask]

        L_BT, info = barlow_twins_loss(
            zh,
            zt,
            alpha=self.alpha,
        )

        L_BT_w = self.lambda_bt * L_BT
        total = L_J + L_BT_w
        barlow_ratio = L_BT_w / total.clamp_min(1e-8)
        # info returns python floats; re-wrap as tensors so the training loop's
        # tensor-only component accumulator picks them up (Batch 4 logging fix).
        bt_diag = zh.new_tensor(info["diag_term"])
        bt_off_diag = zh.new_tensor(info["off_diag_term"])

        return {
            "total_loss": total,
            "components": {
                "L_J": L_J,
                "L_BT": L_BT,
                "L_BT_weighted": L_BT_w,
                "barlow_ratio": barlow_ratio,
                "lambda_bt": self.lambda_bt,
                "bt_diag": bt_diag,
                "bt_off_diag": bt_off_diag,
            },
            "out": out,
        }

    def on_optimizer_step(self, model, step):
        model.ema.update(model.geometry_encoder, step)


class LeJEPAObjective:
    """Phase 2 — design-doc §3.3 Variant B: no EMA/teacher copy; SIGReg-style
    distribution regularization on the embeddings directly.

    L = L_J (student encoder output as target, no stop-grad — the regularizer
    prevents collapse) + lambda_sigreg * L_SIGReg on the projected masked
    predictions. See src/losses/sigreg.py for the exact sliced-ECF math and its
    reported hyperparameters.
    """

    name = "lejepa"

    def __init__(self, lambda_sigreg=0.1, num_slices=8, num_points=256, seed=0):
        self.lambda_sigreg = lambda_sigreg
        self.sigreg_kwargs = dict(num_slices=num_slices, num_points=num_points, seed=seed)

    def __call__(self, model, G, S, M):
        mask = _mask_from_M(M)
        out = model(G, S, M, with_target=False)            # student z_hat, no EMA
        z_hat = out["z_hat"]                               # (B, 256, 384)
        z_y = model.geometry_encoder(G)                     # student target, no EMA
        L_J, _ = jepa_loss(z_hat, z_y, mask, proj=model.proj,
                           stop_grad_target=False)  # LeJEPA: no stop-grad by design
        zh = model.proj(z_hat)[mask]
        L_sig, info = sigreg_loss(zh, **self.sigreg_kwargs)
        L_sig_w = self.lambda_sigreg * L_sig
        total = L_J + L_sig_w
        sigreg_ratio = L_sig_w / total.clamp_min(1e-8)
        return {"total_loss": total,
                "components": {"L_J": L_J, "L_SIGReg": L_sig,
                               "L_SIGReg_weighted": L_sig_w,
                               "sigreg_ratio": sigreg_ratio,
                               "lambda_sigreg": self.lambda_sigreg,
                               "sigreg_info": info},
                "out": {"z_hat": z_hat, "mask": mask, "z_y": z_y}}

    def on_optimizer_step(self, model, step):
        pass  # no teacher copy in this variant


class JEPAVarianceObjective:
    """Screening rung between `jepa` and `jepa_vicreg`: isolates the variance
    term alone (no covariance), to test whether a variance floor by itself is
    sufficient to prevent target collapse, before attributing any effect to
    covariance/decorrelation pressure.

    L = L_J + lambda_var * L_var, on the PROJECTED predictions of masked
    tokens — same feature space jepa_vicreg regularizes, for a controlled
    comparison where only the objective differs.
    """

    name = "jepa_var"

    def __init__(self, lambda_var=1.0, gamma=1.0):
        self.lambda_var = lambda_var
        self.gamma = gamma

    def __call__(self, model, G, S, M):
        out = model(G, S, M)
        mask = out["mask"] if "mask" in out else _mask_from_M(M)
        L_J, _ = jepa_loss(out["z_hat"], out["z_y"], mask, proj=model.proj)
        zh = model.proj(out["z_hat"])[mask]                 # (N, D) masked predictions
        L_var, _ = vicreg_loss(zh, gamma=self.gamma, cov_on=False)
        L_var_w = self.lambda_var * L_var                   # weighted regularizer
        total = L_J + L_var_w
        var_ratio = L_var_w / total.clamp_min(1e-8)         # share of total loss
        return {"total_loss": total,
                "components": {"L_J": L_J, "L_var": L_var, "L_var_weighted": L_var_w,
                               "var_ratio": var_ratio,
                               "lambda_var": self.lambda_var},
                "out": out}

    def on_optimizer_step(self, model, step):
        model.ema.update(model.geometry_encoder, step)


OBJECTIVES = {
    "jepa": JEPAObjective,
    "jepa_var": JEPAVarianceObjective,
    # jepa_vicreg is the FAITHFUL EMA-JEPA + VICReg-style objective
    # (VICRegObjective: objective-owned projector, invariance+variance+
    # covariance on both projected branches). The historical single-branch
    # JEPAVICRegObjective class remains importable for the regression tests
    # that compare against it, but is no longer a ladder rung.
    "jepa_vicreg": VICRegObjective,
    "jepa_vicreg2": JEPAVICRegDualObjective,
    "jepa_barlow": JEPABarlowObjective,
    "lejepa": LeJEPAObjective,
}
