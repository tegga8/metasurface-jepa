"""Design-set round-trip through our model (independent physics in the loop).

Modes:
  calib   : 3 calibration items, fed (a) their true CST spectra and (b) their
            MEEP spectra (post-convention) -> IoU vs the true geometry.
  novel   : set 1 (totally new designs) MEEP spectra -> IoU decoded vs original.
  variant : set 2 (dataset-derived variants) MEEP spectra -> IoU decoded vs the
            EDITED variant and vs its dataset SOURCE (memorization probe).
  symbol  : the 9 symbolic designs (symbol_designs.npz) -> MEEP spectra -> IoU
            decoded vs the original glyph.
  real    : K items (default 200) from the released split (default test) with
            their true CST spectra -> IoU decoded vs the true geometry
            (in-distribution round-trip at scale, all models).
Controls: 12 val items with true CST spectra through the same pipeline.

Scenario-A inputs (full mask, all scalars unknown), eval mode.
Usage: python _novel_modeltest.py calib|novel|variant|symbol|real [sp] [split] [K]
"""

import json
import os
import sys

import numpy as np
import torch
import yaml
from scipy import io

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "train"))
sys.path.insert(0, os.path.join(REPO, "scripts", "eval"))

import eval_scenarios as es
from assembly import build_unified_model
from data.mask import BlockMasker
from losses.unified_losses import UnifiedJEPALoss
from physics.physics_loop import load_surrogate
from train.engine import load_checkpoint

SPLIT = os.path.join(REPO, "data", "metadit", "split_data", "val_set.mat")
TRAIN_SPLIT = os.path.join(REPO, "data", "metadit", "split_data", "train_set.mat")
FREQS_N = 301


def resolved(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def load_model(cfg_path, ckpt_path, device="cpu"):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    lc = cfg.get("loss", {})
    surrogate = load_surrogate(resolved(cfg["weights"]["surrogate"]), device=device)
    model = build_unified_model(cfg, resolved(cfg["weights"]["spectrum"]), device=device)
    objective = UnifiedJEPALoss(
        hidden=cfg["hidden"],
        lambda_inv=lc.get("lambda_inv", 25.0), lambda_var=lc.get("lambda_var", 25.0),
        lambda_cov=lc.get("lambda_cov", 1.0), lambda_scalar=lc.get("lambda_scalar", 1.0),
        lambda_phys=lc.get("lambda_phys", 0.0),
        lambda_summary=lc.get("lambda_summary", 0.0),
        gamma=lc.get("gamma", 1.0), eps=lc.get("eps", 1e-4),
        scalar_loss_type=lc.get("scalar_loss_type", "l1"),
        lambda_occ=lc.get("lambda_occ", 0.0),
        projector_type=lc.get("projector_type", "mlp_bn"),
        lambda_cond=lc.get("lambda_cond", 0.0), lambda_scal_t=lc.get("lambda_scal_t", 0.0),
        objective=cfg.get("objective", "jepa"), surrogate=surrogate,
        physics_use_ste=cfg.get("staging", {}).get("physics_use_ste", True),
    ).to(device)
    load_checkpoint(ckpt_path, model, objective, optimizer=None, scheduler=None,
                    device=device, strict_optimizer=False)
    model.eval()
    return model, surrogate


def scenario_a_decode(model, surrogate, occ, sv, spec, device="cpu"):
    with torch.no_grad():
        b = occ.shape[0]
        sk = torch.zeros(b, 3, dtype=torch.bool, device=device)
        masker = BlockMasker(placement="random", grid=16, min_side=3,
                             k_range=(1, 4), seed=42)
        M = masker.sample(occ, ratio=1.0).to(device)
        out = model(occ, sv, sk, spec, M, goal_mode="real", with_target=False)
        prob = model.decode_occupancy_prob(out["z_hat"], out["scalar_pred"],
                                           scalar_known=sk, scalar_values=sv)
        geom, _ = model.decode_geometry(out["z_hat"], out["scalar_pred"],
                                        occ_input=occ, mask=M, scalar_known=sk,
                                        scalar_values=sv, hard_forward=True)
        spec_pred = surrogate(geom).prediction
        err = es._spectrum_error_per_sample(spec_pred, spec)
    return prob, err


def iou_f1(prob, occ_true):
    pb = (prob > 0.5)
    tb = (occ_true > 0.5)
    inter = float((pb & tb).sum())
    union = float((pb | tb).sum())
    tp = inter
    fp = float((pb & ~tb).sum())
    fn = float((~pb & tb).sum())
    iou = inter / union if union > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-12)
    return iou, f1, float(pb.float().mean()), float(tb.float().mean())


def apply_convention(t_re, t_im, conv):
    t = np.array(t_re) + 1j * np.array(t_im)
    if conv["conj"]:
        t = np.conj(t)
    n = len(t)
    u = np.arange(n) / (n - 1)
    t = t * np.exp(1j * (conv["a"] + conv["b"] * u))
    return torch.from_numpy(np.stack([t.real, t.imag]).astype(np.float32))


def meep_spec_from_entry(dd, conv):
    tag = conv.get("mode_tag", "def")
    m = dd["modes"][tag]
    return apply_convention(m["re"], m["im"], conv)


def main():
    mode = sys.argv[1]
    sp = sys.argv[2]
    conv = json.load(open(os.path.join(sp, "_meep_convention.json")))
    print("convention:", conv, flush=True)

    d = io.loadmat(SPLIT)
    pat, par, real, imag = d["pattern"], d["parameter"], d["real"], d["imag"]

    models = [
        ("S2 slim", os.path.join(REPO, "configs/scaling/unified_s2_slim.yaml"),
         os.path.join(sp, "s2_out", "s2", "seed0.pt")),
        ("S1 small", os.path.join(REPO, "configs/scaling/unified_s1_small.yaml"),
         os.path.join(sp, "s1_out", "s1", "seed0.pt")),
        ("L1 wide", os.path.join(REPO, "configs/scaling/unified_l1_wide.yaml"),
         os.path.join(sp, "l1a_ckpt", "l1", "seed0.pt")),
        ("base(v2,s0)", os.path.join(REPO, "configs/unified.yaml"),
         os.path.join(sp, "schedfix_ckpt", "schedfix", "seed0.pt")),
    ]

    if mode == "calib":
        calib = json.load(open(os.path.join(sp, "meep_calib_out.json")))
        idxs = [it["idx"] for it in calib]
        occ = torch.stack([torch.from_numpy(
            (pat[:, :, i] == 1).astype(np.float32))[None, :, :] for i in idxs])
        svs = torch.tensor([par[i].tolist() for i in idxs], dtype=torch.float32)
        spec_cst = torch.stack([torch.from_numpy(
            np.stack([real[i], imag[i]]).astype(np.float32)) for i in idxs])
        spec_meep = torch.stack([meep_spec_from_entry(it, conv) for it in calib])
        for name, cfgp, ckp in models:
            model, surrogate = load_model(cfgp, ckp)
            for tag, spec in (("CST", spec_cst), ("MEEP", spec_meep)):
                prob, err = scenario_a_decode(model, surrogate, occ, svs, spec)
                print(f"[{name} | {tag}-fed]", flush=True)
                for k, i in enumerate(idxs):
                    iou, f1, pf, tf = iou_f1(prob[k], occ[k])
                    print(f"  item{i}: IoU={iou:.3f} F1={f1:.3f} "
                          f"occ_pred={pf:.3f} (true {tf:.3f}) "
                          f"surrogate_sperr={float(err[k]):.4f}", flush=True)

    elif mode in ("novel", "variant", "symbol"):
        if mode == "novel":
            npz = os.path.join(REPO, "_novel_designs.npz")
            mj = os.path.join(sp, "meep_set1_out.json")
        elif mode == "variant":
            npz = os.path.join(REPO, "_variant_designs.npz")
            mj = os.path.join(sp, "meep_set2_out.json")
        else:
            npz = os.path.join(REPO, "symbol_designs.npz")
            mj = os.path.join(sp, "meep_symbols_out.json")
        z = np.load(npz, allow_pickle=True)
        pats = z["patterns"]
        sc = z["scalars"]
        per_item = sc.ndim > 1
        names = z["names"].tolist() if "names" in z else None
        meep = json.load(open(mj))
        n_use = min(len(meep["designs"]), len(pats))
        if n_use < len(pats):
            print(f"NOTE: partial set — {n_use}/{len(pats)} designs completed",
                  flush=True)
        pats = pats[:n_use]
        if per_item:
            sc = sc[:n_use]
        occ = torch.stack([torch.from_numpy(p.astype(np.float32))[None, :, :]
                           for p in pats])
        svs = torch.tensor(
            [list(map(float, sc[k])) if per_item else list(map(float, sc))
             for k in range(len(pats))], dtype=torch.float32)
        spec_meep = torch.stack([meep_spec_from_entry(dd, conv)
                                 for dd in meep["designs"]])

        src_occ = None
        if mode == "variant":
            dt = io.loadmat(TRAIN_SPLIT)
            train = dt["pattern"]
            src_idx = z["source_idx"]
            src_occ = torch.stack([torch.from_numpy(
                (train[:, :, int(i)] == 1).astype(np.float32))[None, :, :]
                for i in src_idx])

        occ_n = pat.reshape(-1, pat.shape[-1]).sum(axis=0)
        cand = np.where((occ_n > 0.28 * 4096) & (occ_n < 0.55 * 4096))[0]
        pick = cand[np.linspace(0, len(cand) - 1, 12).astype(int)]
        occ_c = torch.stack([torch.from_numpy(
            (pat[:, :, i] == 1).astype(np.float32))[None, :, :] for i in pick])
        svs_c = torch.tensor([par[i].tolist() for i in pick], dtype=torch.float32)
        spec_c = torch.stack([torch.from_numpy(
            np.stack([real[i], imag[i]]).astype(np.float32)) for i in pick])

        for name, cfgp, ckp in models:
            model, surrogate = load_model(cfgp, ckp)
            print(f"=== {name} ===", flush=True)
            prob, err = scenario_a_decode(model, surrogate, occ, svs, spec_meep)
            dec_path = os.path.join(sp, f"symbol_dec_{name.split()[0]}.npz")
            np.savez(dec_path, prob=prob.detach().cpu().numpy(),
                     occ=occ.detach().cpu().numpy())
            print(f"  saved decodes -> {os.path.basename(dec_path)}", flush=True)
            ious = []
            for k in range(len(pats)):
                iou, f1, pf, tf = iou_f1(prob[k], occ[k])
                ious.append(iou)
                label = names[k] if names else f"{mode}{k}"
                extra = ""
                if src_occ is not None:
                    iou_s, _, _, _ = iou_f1(prob[k], src_occ[k])
                    extra = f" IoU_vs_source={iou_s:.3f}"
                print(f"  {label}: IoU={iou:.3f} F1={f1:.3f} "
                      f"occ_pred={pf:.3f} (true {tf:.3f}) "
                      f"surrogate_sperr={float(err[k]):.4f}{extra}", flush=True)
            print(f"  {mode.upper()} mean IoU = {np.mean(ious):.3f} "
                  f"+/- {np.std(ious):.3f}", flush=True)
            prob, err = scenario_a_decode(model, surrogate, occ_c, svs_c, spec_c)
            ious = []
            for k in range(len(pick)):
                iou, f1, pf, tf = iou_f1(prob[k], occ_c[k])
                ious.append(iou)
            print(f"  CONTROL (val, CST-fed) mean IoU = {np.mean(ious):.3f} "
                  f"+/- {np.std(ious):.3f}", flush=True)
    elif mode == "real":
        split = sys.argv[3] if len(sys.argv) > 3 else "test"
        k_items = int(sys.argv[4]) if len(sys.argv) > 4 else 200
        dd = io.loadmat(os.path.join(REPO, "data", "metadit", "split_data",
                                     f"{split}_set.mat"))
        pat_r, par_r, real_r, imag_r = (dd["pattern"], dd["parameter"],
                                        dd["real"], dd["imag"])
        occ_r = pat_r.reshape(-1, pat_r.shape[-1]).sum(axis=0)
        cand_r = np.where((occ_r > 0.28 * 4096) & (occ_r < 0.55 * 4096))[0]
        pick_r = cand_r[np.linspace(0, len(cand_r) - 1, k_items).astype(int)]
        occ = torch.stack([torch.from_numpy(
            (pat_r[:, :, i] == 1).astype(np.float32))[None, :, :] for i in pick_r])
        svs = torch.tensor([par_r[i].tolist() for i in pick_r],
                           dtype=torch.float32)
        spec = torch.stack([torch.from_numpy(
            np.stack([real_r[i], imag_r[i]]).astype(np.float32)) for i in pick_r])
        print(f"split={split} n={len(pick_r)} "
              f"(occupancy-stratified, true CST spectra)", flush=True)

        for name, cfgp, ckp in models:
            model, surrogate = load_model(cfgp, ckp)
            prob, err = scenario_a_decode(model, surrogate, occ, svs, spec)
            ious, f1s = [], []
            for k in range(len(pick_r)):
                iou, f1, pf, tf = iou_f1(prob[k], occ[k])
                ious.append(iou)
                f1s.append(f1)
            print(f"[{name}] {split} n={len(pick_r)}: "
                  f"IoU={np.mean(ious):.3f} +/- {np.std(ious):.3f}  "
                  f"F1={np.mean(f1s):.3f}  "
                  f"surrogate_sperr_mean={float(err.mean()):.4f}", flush=True)

    else:
        raise SystemExit("mode must be calib|novel|variant|symbol|real")


if __name__ == "__main__":
    main()
