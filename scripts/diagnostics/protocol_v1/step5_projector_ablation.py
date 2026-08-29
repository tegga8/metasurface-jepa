"""Protocol step 5: projector ablation (TRIGGERED by step 2's absorption signature).

Three 500-step runs, identical data/seed/steps/lambda, ONE variable changed:
  A — current objective (projector trained, as shipped)
  B — projector frozen at init (requires_grad=False, excluded from optimizer)
  C — A + auxiliary raw-latent term  lambda_raw * MSE(z_hat[mb], z_y[mb])
      (diagnostic-only; official objective untouched)

lambda_raw = 1.0 chosen so the auxiliary term (~5-7 raw) is non-negligible
next to L_var_weighted (~13-15); a "small" weight 100x below the dominant term
would make C indistinguishable from A and the experiment uninformative.
Recorded as an experiment config, not a shipped hyperparameter.
"""
import os, sys, time, json
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "train"))
os.chdir(REPO)

import torch
import yaml
import torch.nn.functional as F

from train_unified import (
    training_step, _ensure_spectrum_weights, make_synthetic_dataset,
    _build_scalar_masker_bank, build_scheduler,
)
from assembly import build_unified_model
from data.mask import BlockMasker
from physics.physics_loop import load_surrogate
from losses.unified_losses import UnifiedJEPALoss
from runtime.reproducibility import set_seed


class _RegimeLoggerStub:
    def record(self, ratio, regime):
        pass


def fixed_batch(seed):
    torch.manual_seed(seed)
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    occ[:, :, :32, :32] = 1.0
    sv = torch.rand(2, 3) * 10 + 1
    spec = torch.randn(2, 2, 301)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=seed + 1)
    M = masker.sample(occ, 1.0)
    sk = torch.zeros(2, 3, dtype=torch.bool)
    return occ, sv, spec, sk, M


def metrics_for_batch(model, objective, occ, sv, spec, sk, M):
    was_m, was_o = model.training, objective.training
    model.eval(); objective.eval()
    try:
        with torch.no_grad():
            out = model(occ, sv, sk, spec, M, goal_mode="real")
            mb = out["mask"]
            zh, zy = out["z_hat"][mb], out["z_y_raw"][mb]
            ph = objective.projector(out["z_hat"])[mb]
            py = objective.projector(out["z_y_raw"])[mb]
            def pair(a, b):
                return (torch.nn.functional.mse_loss(a, b).item(),
                        (1 - torch.nn.functional.cosine_similarity(
                            a, b, dim=-1).clamp(min=0)).mean().item(),
                        a.norm(dim=-1).mean().item(), b.norm(dim=-1).mean().item())
            rm, rc, rh, ry = pair(zh, zy)
            pm, pc, pnh, pny = pair(ph, py)
            return {"raw_mse": round(rm, 4), "raw_cos": round(rc, 4),
                    "raw_norm_hat": round(rh, 3), "raw_norm_y": round(ry, 3),
                    "proj_mse": round(pm, 4), "proj_cos": round(pc, 4)}
    finally:
        model.train(was_m); objective.train(was_o)


def iou_f1(pred_bin, true_bin):
    p, t = pred_bin.flatten(), true_bin.flatten()
    tp = ((p > 0.5) & (t > 0.5)).sum().item()
    fp = ((p > 0.5) & (t <= 0.5)).sum().item()
    fn = ((p <= 0.5) & (t > 0.5)).sum().item()
    return (tp / max(1, tp + fp + fn), 2 * tp / max(1, 2 * tp + fp + fn))


def run_variant(tag, freeze_projector, lambda_raw, steps=500):
    with open("configs/unified.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["loss"]["lambda_phys"] = 0.01
    cfg["train"]["total_steps"] = steps
    device = "cpu"
    set_seed(42)
    spec_weights = _ensure_spectrum_weights(cfg["weights"]["spectrum"], device,
                                            allow_dummy=False)
    model = build_unified_model(cfg, spec_weights, device=device)
    surrogate = load_surrogate(cfg["weights"]["surrogate"], device=device)
    objective = UnifiedJEPALoss(
        hidden=cfg["hidden"], lambda_inv=cfg["loss"]["lambda_inv"],
        lambda_var=cfg["loss"]["lambda_var"], lambda_cov=cfg["loss"]["lambda_cov"],
        lambda_scalar=cfg["loss"]["lambda_scalar"], lambda_phys=cfg["loss"]["lambda_phys"],
        gamma=cfg["loss"]["gamma"], eps=cfg["loss"]["eps"], surrogate=surrogate,
        physics_use_ste=cfg["staging"]["physics_use_ste"]).to(device)

    if freeze_projector:
        for p in objective.projector.parameters():
            p.requires_grad_(False)

    model_group = [p for p in model.parameters() if p.requires_grad]
    obj_group = [p for p in objective.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": model_group, "lr": cfg["train"]["lr"]},
         {"params": obj_group, "lr": cfg["train"]["lr"]}],
        weight_decay=cfg["train"].get("wd", 1e-4))
    scheduler = build_scheduler(optimizer, cfg["train"]["lr"],
                                cfg["train"].get("warmup_steps", 100), steps)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    bank = _build_scalar_masker_bank(cfg, seed=42)
    rng = torch.Generator().manual_seed(42)
    train_pool = make_synthetic_dataset(16, device, seed=42)
    data_iter = iter(train_pool * (steps // 16 + 2))

    b_in = fixed_batch(42)
    b_out = fixed_batch(777)
    snap = {}
    t0 = time.time()
    model.train(); objective.train()
    for step in range(steps + 1):
        if step in (0, 50, 100, 200, 500):
            snap[step] = {"in": metrics_for_batch(model, objective, *b_in),
                          "out": metrics_for_batch(model, objective, *b_out)}
            print(f"  [{tag} ckpt {step:3d}] raw_cos_in="
                  f"{snap[step]['in']['raw_cos']:.4f} raw_mse_in="
                  f"{snap[step]['in']['raw_mse']:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
        if step == steps:
            break
        occ, sv, spec = next(data_iter)
        result, M, sk = training_step(
            model, objective, occ, sv, spec, cfg, device, step, masker, rng,
            _RegimeLoggerStub(), surrogate=surrogate, scalar_masker_bank=bank)
        total = result["total_loss"]
        if lambda_raw > 0:
            out_d = result["out"]
            mb = out_d["mask"]
            raw_term = F.mse_loss(out_d["z_hat"][mb], out_d["z_y_raw"][mb])
            total = total + lambda_raw * raw_term
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model_group + obj_group, 1.0)
        optimizer.step(); scheduler.step()
        objective.on_optimizer_step(model, step)

    # final geometry + spectrum diagnostics on the fixed held-out batch
    model.eval(); objective.eval()
    occ, sv, spec, sk, M = b_out
    with torch.no_grad():
        out = model(occ, sv, sk, spec, M, goal_mode="real", with_target=True)
        z_y = out["z_y_raw"]
        hard, soft = None, None
        geom, soft = model.decode_geometry(z_y, sv)      # condition A
        hard = (soft > 0.5).float()
        iou, f1 = iou_f1(hard, occ)
        spec_std = spec.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        spred = surrogate(geom).prediction
        spec_err = ((spred - spec) / spec_std).abs().mean().item()
        # decoder gradient magnitude from weighted physics at the end state
    model.train(); objective.train()
    out = model(occ, sv, sk, spec, M, goal_mode="real")
    from physics.physics_loop import physics_loss_from_out
    Lp, _, _ = physics_loss_from_out(model, out, surrogate, occ, sv, sk, spec,
                                     M, loss_type="smooth_l1",
                                     use_ste=objective.physics_use_ste,
                                     normalize=True)
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    (cfg["loss"]["lambda_phys"] * Lp).backward()
    gd = sum(p.grad.norm().item() ** 2 for p in model.geometry_decoder.parameters()
             if p.grad is not None) ** 0.5

    final = {"geometry_A_iou": round(iou, 4), "geometry_A_f1": round(f1, 4),
             "geometry_A_occ_frac": round(hard.mean().item(), 4),
             "spectrum_err_A": round(spec_err, 4),
             "decoder_grad_from_Lphys_w": round(gd, 6)}
    print(f"[{tag}] FINAL: {json.dumps(final)}", flush=True)
    return {"snapshots": snap, "final": final}


def main():
    out = {}
    out["A_current"] = run_variant("A", freeze_projector=False, lambda_raw=0.0)
    out["B_frozen_proj"] = run_variant("B", freeze_projector=True, lambda_raw=0.0)
    out["C_raw_aux"] = run_variant("C", freeze_projector=False, lambda_raw=1.0)
    sp = os.environ.get("COMMANDCODE_SCRATCHPAD", ".")
    with open(os.path.join(sp, "step5_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
