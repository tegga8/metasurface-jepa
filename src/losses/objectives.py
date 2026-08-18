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

from losses.barlow import barlow_twins_loss
from losses.jepa_loss import jepa_loss
from losses.sigreg import sigreg_loss
from losses.vicreg import vicreg_loss


def _mask_from_M(M):
    return (M.view(M.shape[0], -1) == 0)  # (B, 256) bool, 1 = masked


class JEPAObjective:
    """Phase 0 — current EMA-JEPA, exactly the approved B1+B2 objective:

    L_J = (1/|M|) * sum_{i in M} [1 - cos(P(z_hat_i), P(z_i^y))]
    P = projection head, z^y = EMA target (stop-gradient), masked positions only.
    """

    name = "jepa"

    def __call__(self, model, G, S, M):
        out = model(G, S, M)
        mask = out["mask"] if "mask" in out else _mask_from_M(M)
        L_J, _ = jepa_loss(out["z_hat"], out["z_y"], mask, proj=model.proj)
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
    "jepa_vicreg": JEPAVICRegObjective,
    "jepa_vicreg2": JEPAVICRegDualObjective,
    "jepa_barlow": JEPABarlowObjective,
    "lejepa": LeJEPAObjective,
}
