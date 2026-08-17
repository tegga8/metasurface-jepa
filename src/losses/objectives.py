"""Pluggable training objectives for the Milestone B adaptive loss ladder (§1–§7).

One interface, three objective formulations, zero objective-specific math in the
training loop:

    jepa        : L_J (masked positions only, projection applied, EMA target stop-grad)
    jepa_vicreg : L_J + lambda_var * L_var + lambda_cov * L_cov   (projected masked preds)
    lejepa      : L_J (student target, no EMA) + lambda_sigreg * L_SIGReg  (sliced ECF)

Each objective returns, per step:
    {"total_loss": tensor, "components": {name: tensor/float, ...}, "out": model out}
and owns the per-step auxiliary updates via `on_optimizer_step` (EMA momentum update
for the two EMA-based objectives; no-op for LeJEPA which has no teacher copy).
"""

import torch

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
        total = L_J + self.lambda_var * L_var + self.lambda_cov * L_cov
        return {"total_loss": total,
                "components": {"L_J": L_J, "L_var": L_var, "L_cov": L_cov,
                               "lambda_var": self.lambda_var, "lambda_cov": self.lambda_cov},
                "out": out}

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
        L_J, _ = jepa_loss(z_hat, z_y, mask, proj=model.proj)
        zh = model.proj(z_hat)[mask]
        L_sig, info = sigreg_loss(zh, **self.sigreg_kwargs)
        total = L_J + self.lambda_sigreg * L_sig
        return {"total_loss": total,
                "components": {"L_J": L_J, "L_SIGReg": L_sig,
                               "lambda_sigreg": self.lambda_sigreg,
                               "sigreg_info": info},
                "out": {"z_hat": z_hat, "mask": mask, "z_y": z_y}}

    def on_optimizer_step(self, model, step):
        pass  # no teacher copy in this variant


OBJECTIVES = {
    "jepa": JEPAObjective,
    "jepa_vicreg": JEPAVICRegObjective,
    "lejepa": LeJEPAObjective,
}
