"""Protocol steps 3 + 4: geometry-realization isolation and gradient attribution.

No training. One fixed batch (seed 777), real released spectrum encoder +
real surrogate, unified architecture exactly as shipped.

Step 3 conditions (decoder fed the EMA target latent z_y):
  A: z_y + TRUE scalars          (decoder capability, best case)
  B: z_y + PREDICTED scalars     (isolates scalar-decoder quality)
  C: perturbed z_y + TRUE scalars (sanity: decoder must degrade gracefully)

Step 4: backward each loss term separately (zero grad between terms), record
per-module grad norms; verify the structural zero (JEPA/VICReg terms must
produce exactly zero gradient in geometry_decoder / scalar_decoder); compute
gradient cosine between weighted L_phys and L_inv+L_var+L_cov on shared params.
"""
import os, sys, json
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "train"))
os.chdir(REPO)

import torch
import yaml
import torch.nn.functional as F

from train_unified import _ensure_spectrum_weights
from assembly import build_unified_model
from data.factorize import factorize_geometry
from data.mask import BlockMasker
from data.dataset import MetaDiTDataset, collate_batch
from physics.physics_loop import load_surrogate, physics_loss_from_out
from losses.unified_losses import UnifiedJEPALoss
from losses.vicreg import vicreg_branch_terms
from runtime.reproducibility import set_seed

REAL_AVAILABLE = os.path.exists(os.path.join(REPO, "data/metadit/split_data/train_set.mat"))

MODULES = ["occupancy_encoder", "fusion_encoder", "predictor",
           "geometry_decoder", "scalar_decoder", "scalar_encoder"]


def iou_f1(pred_bin, true_bin):
    p, t = pred_bin.flatten(), true_bin.flatten()
    tp = ((p > 0.5) & (t > 0.5)).sum().item()
    fp = ((p > 0.5) & (t <= 0.5)).sum().item()
    fn = ((p <= 0.5) & (t > 0.5)).sum().item()
    iou = tp / max(1, tp + fp + fn)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return iou, f1


def main():
    with open("configs/unified.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["loss"]["lambda_phys"] = 0.01
    device = "cpu"
    set_seed(42)

    spec_weights = _ensure_spectrum_weights(cfg["weights"]["spectrum"], device,
                                            allow_dummy=False)
    model = build_unified_model(cfg, spec_weights, device=device)
    surrogate = load_surrogate(cfg["weights"]["surrogate"], device=device)
    objective = UnifiedJEPALoss(
        hidden=cfg["hidden"], lambda_phys=cfg["loss"]["lambda_phys"],
        gamma=cfg["loss"]["gamma"], eps=cfg["loss"]["eps"],
        surrogate=surrogate,
        physics_use_ste=cfg["staging"]["physics_use_ste"]).to(device)

    # ---- fixed batch: REAL data if available, else synthetic (seed 777) ----
    if REAL_AVAILABLE:
        ds = MetaDiTDataset("data/metadit/split_data/train_set.mat",
                            max_samples=2, seed=777)
        G, S = collate_batch([ds[0], ds[1]])
        occ_true, sv_true = factorize_geometry(G)
        spec = S
        data_kind = "real(2 samples, seed 777)"
    else:
        torch.manual_seed(777)
        occ_true = (torch.rand(2, 1, 64, 64) > 0.5).float()
        occ_true[:, :, :32, :32] = 1.0
        sv_true = torch.rand(2, 3) * 10 + 1
        spec = torch.randn(2, 2, 301)
        data_kind = "synthetic(seed 777)"
    occ_true, sv_true, spec = occ_true.to(device), sv_true.to(device), spec.to(device)
    print(f"DATA: {data_kind}", flush=True)

    sk_all_unknown = torch.zeros(2, 3, dtype=torch.bool)
    sk_all_known = torch.ones(2, 3, dtype=torch.bool)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=4242)
    M = masker.sample(occ_true, 1.0)   # full mask (hard stratum)

    # ================= STEP 3: geometry realization from z_y =================
    model.eval(); objective.eval()
    with torch.no_grad():
        out_full = model(occ_true, sv_true, sk_all_known, spec, M,
                         goal_mode="real", with_target=True)
        z_y = out_full["z_y_raw"]                       # EMA target latent
        scalar_pred = out_full["scalar_pred"]           # model's scalar guess
        assert z_y.shape == (2, 256, 192)

        def decode(z, scalars):
            geom, soft = model.decode_geometry(z, scalars)   # no retention
            return (soft > 0.5).float(), soft, geom

        results3 = {}
        # A: z_y + TRUE scalars
        hardA, softA, geomA = decode(z_y, sv_true)
        iouA, f1A = iou_f1(hardA, occ_true)
        bceA = F.binary_cross_entropy_with_logits(
            model.geometry_decoder(z_y, sv_true), occ_true).item()
        results3["A_true_scalars"] = {
            "iou": round(iouA, 4), "f1": round(f1A, 4),
            "pred_occ_frac": round(hardA.mean().item(), 4),
            "bce_px": round(bceA, 4)}
        # B: z_y + PREDICTED scalars
        hardB, softB, geomB = decode(z_y, scalar_pred)
        iouB, f1B = iou_f1(hardB, occ_true)
        bceB = F.binary_cross_entropy_with_logits(
            model.geometry_decoder(z_y, scalar_pred), occ_true).item()
        results3["B_pred_scalars"] = {
            "iou": round(iouB, 4), "f1": round(f1B, 4),
            "pred_occ_frac": round(hardB.mean().item(), 4),
            "bce_px": round(bceB, 4),
            "scalar_mae": round((scalar_pred - sv_true).abs().mean().item(), 4)}
        # C: perturbed z_y + TRUE scalars (0.5 * per-token std noise)
        noise_scale = 0.5 * z_y.std().item()
        z_pert = z_y + noise_scale * torch.randn_like(z_y)
        hardC, softC, geomC = decode(z_pert, sv_true)
        iouC, f1C = iou_f1(hardC, occ_true)
        results3["C_perturbed_latent"] = {
            "iou": round(iouC, 4), "f1": round(f1C, 4),
            "pred_occ_frac": round(hardC.mean().item(), 4),
            "noise_scale": round(noise_scale, 4)}
    print("STEP3:", json.dumps(results3, indent=2), flush=True)

    # ================= STEP 4: gradient attribution ==========================
    model.train(); objective.train()
    sk4 = sk_all_unknown
    out = model(occ_true, sv_true, sk4, spec, M, goal_mode="real")
    mb = out["mask"]
    p_hat = objective.projector(out["z_hat"])[mb]
    p_y = objective.projector(out["z_y_raw"])[mb]
    L_inv, L_var, L_cov = vicreg_branch_terms(p_hat, p_y,
                                              gamma=objective.gamma,
                                              eps=objective.eps)
    L_scalar = objective.scalar_loss(out["scalar_pred"], sv_true, sk4)
    L_phys_raw, _, _ = physics_loss_from_out(
        model, out, surrogate, occ_true, sv_true, sk4, spec, M,
        loss_type="smooth_l1", use_ste=objective.physics_use_ste,
        normalize=True)
    lam = cfg["loss"]["lambda_phys"]
    L_phys_w = lam * L_phys_raw
    terms = {"L_inv": L_inv, "L_var": L_var, "L_cov": L_cov,
             "L_scalar": L_scalar, "L_phys_raw": L_phys_raw,
             "L_phys_weighted": L_phys_w}
    vals = {k: round(v.item(), 6) for k, v in terms.items()}

    params = [p for p in model.parameters() if p.requires_grad] + \
             [p for p in objective.parameters() if p.requires_grad]

    def module_grad_norms():
        norms = {}
        for name in MODULES:
            mod = getattr(model, name, None)
            if mod is None:
                continue
            s = 0.0
            for p in mod.parameters():
                if p.grad is not None:
                    s += p.grad.norm().item() ** 2
            norms[name] = round(s ** 0.5, 8)
        return norms

    grad_table = {}
    for tname, t in terms.items():
        optimizer = None  # zero grads directly
        for p in params:
            if p.grad is not None:
                p.grad = None
        t.backward(retain_graph=True)
        grad_table[tname] = module_grad_norms()
    print("STEP4 term values:", json.dumps(vals), flush=True)
    print("STEP4 grad norms:", json.dumps(grad_table, indent=2), flush=True)

    # structural zero check
    struct = {}
    for t in ("L_inv", "L_var", "L_cov"):
        gd = grad_table[t]["geometry_decoder"]
        sd = grad_table[t]["scalar_decoder"]
        struct[t] = {"geometry_decoder": gd, "scalar_decoder": sd,
                     "structural_zero_ok": (gd == 0.0 and sd == 0.0)}
    print("STEP4 structural:", json.dumps(struct, indent=2), flush=True)

    # gradient cosine: weighted L_phys vs L_inv+L_var+L_cov on encoder+predictor
    for p in params:
        if p.grad is not None:
            p.grad = None
    (L_inv + L_var + L_cov).backward(retain_graph=True)
    shared = [n for n in ("occupancy_encoder", "predictor")]
    def flat(mods):
        vecs = []
        for n in mods:
            for p in getattr(model, n).parameters():
                if p.requires_grad:
                    vecs.append(p.grad.detach().flatten()
                                if p.grad is not None else
                                torch.zeros(p.numel()))
        return torch.cat(vecs)
    g_jepa = flat(shared)
    for p in params:
        if p.grad is not None:
            p.grad = None
    L_phys_w.backward(retain_graph=True)
    g_phys = flat(shared)
    cos = torch.nn.functional.cosine_similarity(
        g_jepa.unsqueeze(0), g_phys.unsqueeze(0)).item()
    print(f"STEP4 grad cosine(L_phys_w, JEPA terms) on {shared}: "
          f"{cos:.6f}  |g_jepa|={g_jepa.norm():.4f} |g_phys|={g_phys.norm():.4f}",
          flush=True)

    out = {"step3": results3, "step4_values": vals, "step4_grads": grad_table,
           "step4_structural": struct, "step4_grad_cosine": cos,
           "data": data_kind}
    sp = os.environ.get("COMMANDCODE_SCRATCHPAD", ".")
    with open(os.path.join(sp, "step34_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
