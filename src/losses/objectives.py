"""Pluggable training objectives — final registry (§23 of the architecture-repair spec).

Exactly three research objectives:

    jepa_vicreg : faithful EMA-JEPA + VICReg (invariance + variance + covariance
                  on both projected branches; EMA target frozen)
    jepa_barlow : EMA-JEPA + Barlow Twins (projected branches, per-feature
                  standardization, cross-correlation diagonal match + off-diagonal
                  redundancy; EMA target frozen)
    lejepa      : teacher-free LeJEPA (no EMA, student-as-target with NO stop-grad,
                  SIGReg-style distributional regularization on projected features)

One interface, one objective per name, zero objective-specific math in the
training loop (§24): every objective is callable as `objective(model, G, S, M)`
and returns:

    {"total_loss": tensor, "components": {name: tensor, ...}, "out": model out}

plus optional `projector_inputs` / `projector_outputs` (raw vs projected spaces
for the shared representation-health evaluator, §25/§27), and owns per-step
auxiliary updates via `on_optimizer_step` (EMA momentum update for the two
EMA-based objectives; no-op for LeJEPA which has no teacher copy).

Every objective owns its projector. There is no `model.proj` anywhere in this
module and no silent fallback (§17): if an objective needs projected features,
it projects through its own head or fails loudly.

No obsolete ladder rungs: `jepa`, `jepa_var`, `jepa_vicreg2` and the historical
model.proj-dependent classes are gone.
"""

import torch
from torch import nn

from losses.barlow import barlow_twins_loss
from losses.jepa_loss import jepa_loss
from losses.objective_modules import BarlowProjector, LeJEPAProjector, VICRegProjector
from losses.sigreg import sigreg_loss
from losses.vicreg import vicreg_branch_terms


def _mask_from_M(M):
    return (M.view(M.shape[0], -1) == 0)  # (B, 256) bool, 1 = masked


class VICRegObjective(nn.Module):
    """Faithful EMA-JEPA + VICReg-style regularization (`jepa_vicreg`).

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
        L_cov = cov_penalty(p_hat) + cov_penalty(p_y)             (both branches)

    per-branch forms are canonical VICReg: var_penalty is the hinge
    relu(gamma - std).mean() (NOT squared) and cov_penalty is the off-diagonal
    squared sum / D. L_var averages the two branches (official /2 per branch);
    L_cov SUMS them (official cov_loss = cov_x + cov_y, no 0.5 factor).

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
    # unweighted component names whose isolated gradients the short audit measures
    term_names = ("L_inv", "L_var", "L_cov")

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
        z_hat, z_y = out["z_hat"], out["z_y_raw"]

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
        # not added to the loss. With fewer than 2 geometries the pooled
        # statistics are undefined: they are NaN-MARKED (Bug #21 convention),
        # never silently zeroed — the hard N>=2 raise applies to the LOSS
        # terms only.
        mw = mask.float()
        p_hat_g = (p_hat_full * mw.unsqueeze(-1)).sum(1) \
            / mw.sum(1, keepdim=True).clamp(min=1)        # (B, D)
        p_y_g = (p_y_full * mw.unsqueeze(-1)).sum(1) \
            / mw.sum(1, keepdim=True).clamp(min=1)        # (B, D)
        if p_hat_g.shape[0] >= 2:
            geo_inv, geo_var, geo_cov = vicreg_branch_terms(
                p_hat_g, p_y_g, gamma=self.gamma, eps=self.eps)
        else:
            nan = torch.full((), float("nan"), device=p_hat_g.device)
            geo_inv, geo_var, geo_cov = nan, nan, nan

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


class BarlowObjective(nn.Module):
    """EMA-JEPA + Barlow Twins redundancy reduction (`jepa_barlow`, §21).

    Independent implementation of the Barlow Twins (Zbinden et al. 2021)
    cross-correlation objective — NOT a one-line variant of the VICReg
    objective: the two projected branches are standardized per feature, the
    cross-correlation matrix is formed, the DIAGONAL is matched to identity
    (invariance) and the OFF-DIAGONAL is driven to zero (redundancy
    reduction), with a single `alpha` weighting between the two:

        zp = (p_hat - mean)/std ;  zt = (p_y - mean)/std
        C  = (zp.T @ zt) / N
        L_BT = mean_i (1 - C_ii)^2  +  alpha * mean_{i != j} C_ij^2

    (Mean-form per-entry terms, a documented scaling deviation from the
    canonical sum-form so L_BT is O(1) per dimension rather than O(D);
    reported in the final report §J.)

    The projector is objective-owned (BarlowProjector, never shared with
    VICReg). The EMA target encoder stays frozen; the target branch reaches
    the objective projector only. The raw 384-D latent is NOT the projector
    output; raw-space health is reported separately.
    """

    name = "jepa_barlow"
    term_names = ("L_BT",)

    def __init__(self, lambda_bt=1.0, alpha=0.005,
                 projector_input_dim=384, projector_hidden_dim=384,
                 projector_output_dim=384):
        super().__init__()
        self.projector = BarlowProjector(
            input_dim=projector_input_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
        )
        self.lambda_bt = lambda_bt
        self.alpha = alpha

    def forward(self, model, G, S, M):
        out = model(G, S, M)
        mask = out["mask"] if "mask" in out else _mask_from_M(M)
        z_hat, z_y = out["z_hat"], out["z_y_raw"]

        p_hat_full = self.projector(z_hat)                 # (B, 256, D)
        p_y_full = self.projector(z_y)                     # (B, 256, D)
        p_hat = p_hat_full[mask]                           # (N, D) masked tokens
        p_y = p_y_full[mask]                               # (N, D) masked tokens

        L_BT, info = barlow_twins_loss(p_hat, p_y, alpha=self.alpha)

        L_BT_w = self.lambda_bt * L_BT
        total = L_BT_w
        barlow_ratio = L_BT_w / total.clamp_min(1e-8)

        # info returns python floats; re-wrap as tensors so the training loop's
        # tensor-only component accumulator picks them up.
        bt_diag = p_hat.new_tensor(info["diag_term"])
        bt_off_diag = p_hat.new_tensor(info["off_diag_term"])

        return {
            "total_loss": total,
            "components": {
                "L_BT": L_BT,
                "L_BT_weighted": L_BT_w,
                "barlow_ratio": barlow_ratio,
                "lambda_bt": self.lambda_bt,
                "alpha": self.alpha,
                "bt_diag": bt_diag,
                "bt_off_diag": bt_off_diag,
            },
            "out": out,
            "projector_inputs": {"z_hat": z_hat, "z_y": z_y},
            "projector_outputs": {"p_hat": p_hat_full, "p_y": p_y_full},
        }

    def on_optimizer_step(self, model, step):
        model.ema.update(model.geometry_encoder, step)


class LeJEPAObjective(nn.Module):
    """Teacher-free LeJEPA (`lejepa`, §22; design doc §3.3 Variant B).

    No EMA/teacher copy: the model is run with `with_target=False`, and the
    target is the STUDENT geometry encoder's own output on the full geometry —

        z_hat = predictor(masked context, physics)
        z_y   = student_encoder(G)          (no teacher, no stop-grad)

    with

        L = L_J(z_hat, z_y; stop_grad_target=False)
          + lambda_sigreg * 0.5 * (L_SIGReg(p_hat) + L_SIGReg(p_y))

    where p_hat/p_y are the objective-owned projector's outputs on masked
    tokens and L_SIGReg is the sliced-ECF Gaussianity test in
    `src/losses/sigreg.py`. Because there is no teacher, BOTH learnable
    branches receive distributional pressure (a frozen target would collapse
    nothing; here the target is a student output and needs the same guard).

    `on_optimizer_step` is a no-op: there is no teacher copy to update. The
    path never touches `model.ema` — verified by tests (test_lejepa_collapse).
    """

    name = "lejepa"
    term_names = ("L_J", "L_SIGReg")

    def __init__(self, lambda_sigreg=0.1, num_slices=8, num_points=256, seed=0,
                 projector_input_dim=384, projector_hidden_dim=384,
                 projector_output_dim=384):
        super().__init__()
        self.projector = LeJEPAProjector(
            input_dim=projector_input_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
        )
        self.lambda_sigreg = lambda_sigreg
        self.sigreg_kwargs = dict(num_slices=num_slices, num_points=num_points, seed=seed)

    def forward(self, model, G, S, M):
        mask = _mask_from_M(M)
        out = model(G, S, M, with_target=False)            # student z_hat, no EMA
        z_hat = out["z_hat"]                               # (B, 256, 384)
        z_y = model.geometry_encoder(G)                    # student target, no EMA
        L_J, _ = jepa_loss(z_hat, z_y, mask, proj=None,
                           stop_grad_target=False)  # LeJEPA: no stop-grad by design

        p_hat_full = self.projector(z_hat)
        p_y_full = self.projector(z_y)
        p_hat = p_hat_full[mask]
        p_y = p_y_full[mask]

        L_sig_p, info_p = sigreg_loss(p_hat, **self.sigreg_kwargs)
        L_sig_t, info_t = sigreg_loss(p_y, **self.sigreg_kwargs)
        L_sig = 0.5 * (L_sig_p + L_sig_t)

        L_sig_w = self.lambda_sigreg * L_sig
        total = L_J + L_sig_w
        sigreg_ratio = L_sig_w / total.clamp_min(1e-8)
        return {"total_loss": total,
                "components": {"L_J": L_J, "L_SIGReg": L_sig,
                               "L_SIGReg_pred": L_sig_p,
                               "L_SIGReg_target": L_sig_t,
                               "L_SIGReg_weighted": L_sig_w,
                               "sigreg_ratio": sigreg_ratio,
                               "lambda_sigreg": self.lambda_sigreg,
                               "sigreg_info": {"pred": info_p, "target": info_t}},
                "out": {"z_hat": z_hat, "z_y_raw": z_y, "mask": mask},
                "projector_inputs": {"z_hat": z_hat, "z_y": z_y},
                "projector_outputs": {"p_hat": p_hat_full, "p_y": p_y_full}}

    def on_optimizer_step(self, model, step):
        pass  # no teacher copy in this variant


OBJECTIVES = {
    "jepa_vicreg": VICRegObjective,
    "jepa_barlow": BarlowObjective,
    "lejepa": LeJEPAObjective,
}


def build_objective(name, params=None, projector_input_dim=384,
                    projector_hidden_dim=384, projector_output_dim=384):
    """Instantiate one registered objective from a config dict (§24).

    One interface: `objective(model, G, S, M)` + `on_optimizer_step(model, step)`.
    `params` is the config's `objective_params.<name>` dict; the optional nested
    `projector: {input_dim, hidden_dim, output_dim}` overrides the model-latent
    defaults. Unknown top-level keys raise (typo guard) rather than being
    silently ignored.
    """
    if name not in OBJECTIVES:
        raise KeyError(
            f"unknown objective {name!r}; registered: {sorted(OBJECTIVES)}")
    params = dict(params or {})
    proj = dict(params.pop("projector", None) or {})
    kwargs = dict(
        projector_input_dim=proj.get("input_dim", projector_input_dim),
        projector_hidden_dim=proj.get("hidden_dim", projector_hidden_dim),
        projector_output_dim=proj.get("output_dim", projector_output_dim),
    )
    return OBJECTIVES[name](**{**params, **kwargs})