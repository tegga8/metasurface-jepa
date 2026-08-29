"""Protocol step 7: controlled lambda_phys sweep {0.01, 0.1, 1, 10}.

Short 100-step runs (synthetic pool, seed 42), all evaluated on the FIXED REAL
batch (seed 777, full mask = hard stratum). One variable changed per run.
Records: decoder grad from lambda*L_phys, encoder/predictor physics grad,
L_var/L_cov stability, decoded occupancy fraction (collapse indicator),
spectrum error, hard-stratum real-vs-shuffled gap.
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
from data.factorize import factorize_geometry
from data.dataset import MetaDiTDataset, collate_batch
from physics.physics_loop import load_surrogate, physics_loss_from_out
from losses.unified_losses import UnifiedJEPALoss
from runtime.physics_controls import make_shuffled_spectrum
from runtime.reproducibility import set_seed


class _RegimeLoggerStub:
    def record(self, ratio, regime):
        pass


def fixed_real_batch():
    ds = MetaDiTDataset("data/metadit/split_data/train_set.mat",
                        max_samples=2, seed=777)
    G, S = collate_batch([ds[0], ds[1]])
    occ, sv = factorize_geometry(G)
    spec = S
    sk = torch.zeros(2, 3, dtype=torch.bool)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=4242)
    M = masker.sample(occ, 1.0)
    return occ, sv, spec, sk, M


def grad_norms_for(model, out, surrogate, occ, sv, sk, spec, M, lam):
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        if p.grad is not None:
            p.grad = None
    Lp, _, _ = physics_loss_from_out(model, out, surrogate, occ, sv, sk, spec,
                                     M, loss_type="smooth_l1",
                                     use_ste=True, normalize=True)
    (lam * Lp).backward()
    def gn(name):
        s = 0.0
        for p in getattr(model, name).parameters():
            if p.grad is not None:
                s += p.grad.norm().item() ** 2
        return round(s ** 0.5, 6)
    return {"decoder": gn("geometry_decoder"), "encoder": gn("occupancy_encoder"),
            "predictor": gn("predictor")}


def evaluate_batch(model, surrogate, occ, sv, spec, sk, M, lam, out):
    """Gradient norms + physics/spectrum/occupancy/collapse + RNS gap."""
    grads = grad_norms_for(model, out, surrogate, occ, sv, sk, spec, M, lam)
    model.eval()
    with torch.no_grad():
        geom, soft = model.decode_geometry(out["z_hat"], out["scalar_pred"])
        occ_frac = float((soft > 0.5).float().mean().item())
        spec_std = spec.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        sp = surrogate(geom).prediction
        spec_err = ((sp - spec) / spec_std).abs().mean().item()
        # hard-stratum real vs shuffled (spectrum conditioning)
        def err_with(spec_cond):
            o = model(occ, sv, sk, spec_cond, M, goal_mode="real")
            g, _ = model.decode_geometry(o["z_hat"], o["scalar_pred"])
            return ((surrogate(g).prediction - spec) / spec_std).abs().mean().item()
        r = err_with(spec)
        sh = err_with(make_shuffled_spectrum(spec, seed=0))
    model.train()
    return {**grads, "occ_frac": round(occ_frac, 4),
            "spectrum_err": round(spec_err, 4),
            "real_err": round(r, 4), "shuffled_err": round(sh, 4),
            "real_beats_shuffled": bool(r < sh)}


def run_lambda(lam, steps=100):
    with open("configs/unified.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["loss"]["lambda_phys"] = lam
    cfg["train"]["total_steps"] = steps
    device = "cpu"
    set_seed(42)
    sw = _ensure_spectrum_weights(cfg["weights"]["spectrum"], device, allow_dummy=False)
    model = build_unified_model(cfg, sw, device=device)
    surrogate = load_surrogate(cfg["weights"]["surrogate"], device=device)
    objective = UnifiedJEPALoss(
        hidden=cfg["hidden"], lambda_inv=cfg["loss"]["lambda_inv"],
        lambda_var=cfg["loss"]["lambda_var"], lambda_cov=cfg["loss"]["lambda_cov"],
        lambda_scalar=cfg["loss"]["lambda_scalar"], lambda_phys=lam,
        gamma=cfg["loss"]["gamma"], eps=cfg["loss"]["eps"], surrogate=surrogate,
        physics_use_ste=cfg["staging"]["physics_use_ste"]).to(device)
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad],
          "lr": cfg["train"]["lr"]},
         {"params": [p for p in objective.parameters() if p.requires_grad],
          "lr": cfg["train"]["lr"]}],
        weight_decay=cfg["train"].get("wd", 1e-4))
    scheduler = build_scheduler(optimizer, cfg["train"]["lr"],
                                cfg["train"].get("warmup_steps", 20), steps)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    bank = _build_scalar_masker_bank(cfg, seed=42)
    rng = torch.Generator().manual_seed(42)
    pool = make_synthetic_dataset(16, device, seed=42)
    data_iter = iter(pool * 10)
    occ, sv, spec, sk, M = fixed_real_batch()

    traj = []
    t0 = time.time()
    model.train(); objective.train()
    for step in range(steps + 1):
        if step in (0, 50, 100):
            out = model(occ, sv, sk, spec, M, goal_mode="real")
            ev = evaluate_batch(model, surrogate, occ, sv, spec, sk, M, lam, out)
            ev["step"] = step
            c = result["components"] if step > 0 and result is not None else {}
            ev["L_var"] = round(c.get("L_var", float("nan")), 4)
            ev["L_cov"] = round(c.get("L_cov", float("nan")), 4)
            ev["L_phys"] = round(c.get("L_phys", float("nan")), 4)
            traj.append(ev)
            print(f"  [lam={lam} step {step:3d}] {json.dumps(ev)} ({time.time()-t0:.0f}s)",
                  flush=True)
        if step == steps:
            break
        b_occ, b_sv, b_spec = next(data_iter)
        result, Mtr, sktr = training_step(
            model, objective, b_occ, b_sv, b_spec, cfg, device, step, masker,
            rng, _RegimeLoggerStub(), surrogate=surrogate, scalar_masker_bank=bank)
        optimizer.zero_grad(set_to_none=True)
        result["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] +
            [p for p in objective.parameters() if p.requires_grad], 1.0)
        optimizer.step(); scheduler.step()
        objective.on_optimizer_step(model, step)
    return traj


def main():
    out = {}
    for lam in (0.01, 0.1, 1.0, 10.0):
        print(f"=== lambda_phys = {lam} ===", flush=True)
        out[str(lam)] = run_lambda(lam)
    sp = os.environ.get("COMMANDCODE_SCRATCHPAD", ".")
    with open(os.path.join(sp, "step7_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
