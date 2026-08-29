"""Protocol step 2 (v2): paired raw-vs-projected diagnostic over a 500-step run.

Two fixed batches, each with a fixed mask, evaluated at {0,50,100,200,500}:
  - IN-TRAINING batch (from the training pool, seed 42)
  - HELD-OUT batch (seed 777, never trained on)
Also records the training loss components at each checkpoint.
"""
import os, sys, time, json
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "train"))
os.chdir(REPO)

import torch
import yaml

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
                return (
                    torch.nn.functional.mse_loss(a, b).item(),
                    (1 - torch.nn.functional.cosine_similarity(
                        a, b, dim=-1).clamp(min=0)).mean().item(),
                    a.norm(dim=-1).mean().item(),
                    b.norm(dim=-1).mean().item(),
                )
            rm, rc, rh, ry = pair(zh, zy)
            pm, pc, pnh, pny = pair(ph, py)
            return {"raw_mse": rm, "raw_cos": rc, "raw_norm_hat": rh,
                    "raw_norm_y": ry, "proj_mse": pm, "proj_cos": pc,
                    "proj_norm_hat": pnh, "proj_norm_y": pny}
    finally:
        model.train(was_m); objective.train(was_o)


def fixed_batch(seed):
    torch.manual_seed(seed)
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    occ[:, :, :32, :32] = 1.0
    sv = torch.rand(2, 3) * 10 + 1
    spec = torch.randn(2, 2, 301)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=seed + 1)
    M = masker.sample(occ, 1.0)              # hard stratum, fixed mask
    sk = torch.zeros(2, 3, dtype=torch.bool)
    return occ, sv, spec, sk, M


def main():
    with open("configs/unified.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["loss"]["lambda_phys"] = 0.01
    cfg["train"]["total_steps"] = 500
    device = "cpu"
    set_seed(cfg["train"].get("seed", 42))

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

    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad],
          "lr": cfg["train"]["lr"]},
         {"params": [p for p in objective.parameters() if p.requires_grad],
          "lr": cfg["train"]["lr"]}],
        weight_decay=cfg["train"].get("wd", 1e-4))
    scheduler = build_scheduler(optimizer, cfg["train"]["lr"],
                                cfg["train"].get("warmup_steps", 100),
                                cfg["train"]["total_steps"])
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=cfg["train"].get("seed", 42))
    bank = _build_scalar_masker_bank(cfg, seed=cfg["train"].get("seed", 42))
    rng = torch.Generator().manual_seed(cfg["train"].get("seed", 42))

    train_pool = make_synthetic_dataset(16, device, seed=cfg["train"]["seed"])
    data_iter = iter(train_pool * 40)

    batch_in = fixed_batch(42)      # from the training distribution (seed 42)
    batch_out = fixed_batch(777)    # held out
    CHECKPOINTS = [0, 50, 100, 200, 500]
    results = {"in_train": {}, "held_out": {}, "train_components": {}}
    t0 = time.time()
    model.train(); objective.train()
    last_components = None
    for step in range(0, 501):
        if step in CHECKPOINTS:
            results["in_train"][step] = metrics_for_batch(model, objective, *batch_in)
            results["held_out"][step] = metrics_for_batch(model, objective, *batch_out)
            if last_components is not None:
                results["train_components"][step] = dict(last_components)
            print(f"[ckpt {step:3d}] in={json.dumps(results['in_train'][step])}",
                  flush=True)
            print(f"          out={json.dumps(results['held_out'][step])}", flush=True)
            if last_components:
                print(f"          train={json.dumps(results['train_components'][step])}",
                      flush=True)
        if step == 500:
            break
        occ, sv, spec = next(data_iter)
        result, M, sk = training_step(
            model, objective, occ, sv, spec, cfg, device, step, masker, rng,
            _RegimeLoggerStub(), surrogate=surrogate, scalar_masker_bank=bank)
        optimizer.zero_grad(set_to_none=True)
        result["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] +
            [p for p in objective.parameters() if p.requires_grad], 1.0)
        optimizer.step(); scheduler.step()
        objective.on_optimizer_step(model, step)
        last_components = result["components"]
        if step % 100 == 0:
            c = result["components"]
            print(f"  [train {step:3d}] L_total={c['L_total']:.4f} "
                  f"L_inv={c['L_inv']:.5f} L_var={c['L_var']:.4f} "
                  f"L_cov={c['L_cov']:.4f} L_scalar={c['L_scalar']:.4f} "
                  f"L_phys={c['L_phys']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    sp = os.environ.get("COMMANDCODE_SCRATCHPAD", ".")
    with open(os.path.join(sp, "step2_results_v2.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
