"""Masked-fill verification for the unified JEPA (architecture_v5.md §8.3 #11).

Answers two questions about partial-occupancy completion, on top of the existing
IoU/F1 diagnostics:

  1. Is ONLY the masked part filled, and is the fill consistent with the retained
     border?  -> visible_identity (hard), filled-region IoU/F1, seam_mismatch,
     texture stats (roughness + neighbour agreement vs the true geometry).
  2. Does the latent localise the edit (fix only that part), rather than reshaping
     the whole structure?  -> locality_probe: perturb the NEAREST vs the FARTHEST
     visible pixels (equal counts) and measure how much the predicted masked
     region moves. A locality-respecting latent is far less sensitive to distant
     context; a global one moves equally.

`visible_identity` is guaranteed by construction (decode_geometry retains visible
pixels: occ_input*vis + pred*(1-vis)) so it passes trivially -- the scientific
content is the seam/texture/locality numbers, which are currently unmeasured.

The pure helpers take tensors and are unit-tested with stubs; the CLI wires the
real model. Convention: mask (B,16,16), 1 = visible, 0 = masked.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

GRID = 16
PATCH = 4
RES = GRID * PATCH  # 64


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def upsample_vis(mask):
    """(B,16,16) 1=visible -> (B,1,64,64) bool."""
    up = mask.view(mask.shape[0], 1, GRID, GRID).repeat_interleave(
        PATCH, 2).repeat_interleave(PATCH, 3)
    return up > 0.5


def boundary_band(mask, width=1):
    """Masked pixels within `width` pixels of a visible pixel (seam neighbourhood)."""
    vis = upsample_vis(mask)                                   # (B,1,64,64)
    k = 2 * width + 1
    vis_dil = F.max_pool2d(vis.float(), kernel_size=k, stride=1, padding=width) > 0.5
    return (~vis) & vis_dil


def visible_identity_maxdiff(deployed_occ, occ_input, mask):
    """Max |deployed - input| over VISIBLE pixels; 0.0 == visible region untouched."""
    vis = upsample_vis(mask)
    d = (deployed_occ - occ_input).abs() * vis.float()
    return float(d.max().item())


def region_iou_f1(pred_bin, true_bin, region):
    """Micro IoU/F1/precision/recall of pred vs true restricted to `region` (bool)."""
    p = pred_bin[region]
    t = true_bin[region]
    tp = float((p & t).sum().item())
    fp = float((p & ~t).sum().item())
    fn = float((~p & t).sum().item())
    iou = tp / max(1.0, tp + fp + fn)
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall,
            "n_pixels": int(region.sum().item())}


def _pair_regions(region):
    rx = region[:, :, :, 1:] & region[:, :, :, :-1]
    ry = region[:, :, 1:, :] & region[:, :, :-1, :]
    return rx, ry


def roughness(x, region):
    """Mean |adjacent difference| over adjacent pairs both inside `region`.

    Low = oversmoothed/blurred fill; high = noisy fill. Compare pred vs true.
    """
    rx, ry = _pair_regions(region)
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    num = dx[rx].sum() + dy[ry].sum() if rx.any() else torch.zeros(())
    den = rx.sum() + ry.sum()
    return float((num / den.clamp(min=1)).item())


def neighbor_agreement(x_bin, region):
    """Fraction of adjacent pixel pairs (both in `region`) that share a value.

    A first-order periodicity/continuity proxy: periodic meta-atom patterns give
    high agreement; a blurred fill gives an unrealistically high value, noise low.
    """
    rx, ry = _pair_regions(region)
    ax = (x_bin[:, :, :, 1:] == x_bin[:, :, :, :-1])
    ay = (x_bin[:, :, 1:, :] == x_bin[:, :, :-1, :])
    agree = ax[rx].float().sum() + ay[ry].float().sum() if rx.any() else torch.zeros(())
    den = rx.sum() + ry.sum()
    return float((agree / den.clamp(min=1)).item())


def masked_fill_report(deployed_occ, occ_input, true_occ, mask, boundary_width=1):
    """All masked-fill checks for one batch.

    deployed_occ / occ_input / true_occ: (B,1,64,64) occupancy (binary or soft;
    thresholded at 0.5 for the binary metrics).
    """
    vis = upsample_vis(mask)
    filled = ~vis
    pred_bin = deployed_occ > 0.5
    true_bin = true_occ > 0.5
    band = boundary_band(mask, boundary_width)

    rough_pred = roughness(deployed_occ, filled)
    rough_true = roughness(true_occ, filled)
    return {
        "visible_identity_maxdiff": visible_identity_maxdiff(
            deployed_occ, occ_input, mask),
        "filled_region": region_iou_f1(pred_bin, true_bin, filled),
        "seam_band": region_iou_f1(pred_bin, true_bin, band),
        "seam_mismatch": (float((pred_bin[band] != true_bin[band]).float().mean().item())
                          if band.any() else 0.0),
        "filled_roughness_pred": rough_pred,
        "filled_roughness_true": rough_true,
        "filled_roughness_ratio": rough_pred / rough_true if rough_true > 0 else None,
        "filled_agreement_pred": neighbor_agreement(pred_bin, filled),
        "filled_agreement_true": neighbor_agreement(true_bin, filled),
    }


# ---------------------------------------------------------------------------
# locality probe
# ---------------------------------------------------------------------------

@torch.no_grad()
def locality_probe(predict_fn, occ, mask, k=64):
    """Near-vs-far visible perturbation sensitivity of the predicted masked region.

    For each sample: take the k visible pixels NEAREST and the k FARTHEST from the
    filled region's centroid (equal counts), flip each set in turn, and measure the
    mean absolute change in the prediction over the filled region.

      locality_ratio = change_far / change_near
        ~0  -> the latent fixes only the local part (locality-respecting)
        ~1  -> the prediction depends on the whole image (global)

    predict_fn(occ, mask) -> (B,1,64,64) occupancy logits/prob.
    """
    b = occ.shape[0]
    vis = upsample_vis(mask)                       # (B,1,64,64)
    filled = ~vis
    base = predict_fn(occ, mask)

    n = filled.float().sum(dim=(1, 2, 3), keepdim=True).clamp(min=1)
    yy, xx = torch.meshgrid(
        torch.arange(RES, dtype=occ.dtype), torch.arange(RES, dtype=occ.dtype),
        indexing="ij")
    yy = yy.view(1, 1, RES, RES).to(occ.device)
    xx = xx.view(1, 1, RES, RES).to(occ.device)
    ff = filled.float()
    cy = (yy * ff).sum(dim=(1, 2, 3), keepdim=True) / n
    cx = (xx * ff).sum(dim=(1, 2, 3), keepdim=True) / n
    dist = ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt()

    vis_flat = vis.view(b, -1)
    d_flat = dist.view(b, -1)
    min_vis = int(vis_flat.sum(dim=1).min().item())
    if min_vis < 2:
        return {"applicable": False, "k": 0, "change_near": None, "change_far": None,
                "locality_ratio": None, "localized": None,
                "reason": "no visible context (e.g. full mask) — locality undefined"}
    k = max(1, min(int(k), min_vis))
    d_near = torch.where(vis_flat, d_flat, torch.full_like(d_flat, 1e9))
    d_far = torch.where(vis_flat, d_flat, torch.full_like(d_flat, -1.0))
    near_idx = d_near.topk(k, dim=1, largest=False).indices
    far_idx = d_far.topk(k, dim=1).indices

    def _select(idx):
        m = torch.zeros_like(vis_flat)
        m.scatter_(1, idx, True)
        return m.view(b, 1, RES, RES)

    near, far = _select(near_idx), _select(far_idx)
    occ_near = torch.where(near, 1.0 - occ, occ)
    occ_far = torch.where(far, 1.0 - occ, occ)
    ch_near = float((predict_fn(occ_near, mask) - base).abs()[filled].mean().item())
    ch_far = float((predict_fn(occ_far, mask) - base).abs()[filled].mean().item())
    ratio = ch_far / (ch_near + 1e-12)
    return {
        "applicable": True,
        "k": k,
        "change_near": ch_near,
        "change_far": ch_far,
        "locality_ratio": ratio,
        "localized": bool(ratio < 0.5),
    }


# ---------------------------------------------------------------------------
# CLI (requires a checkpoint; see docs/benchmarking)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Masked-fill verification")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "eval"))
    import yaml
    import eval_scenarios as es

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model, surrogate = es._load_eval(cfg, args.checkpoint, args.device)
    model.eval()
    occ, sv, spec = es._load_val_batch(cfg, torch.device(args.device),
                                       smoke=False, n_samples=args.samples)
    masker = es.BlockMasker(placement="random", grid=16, min_side=3,
                            k_range=(1, 4), seed=999)
    out = {}
    with torch.no_grad():
        for name, ratio, sk in (("A", 1.0, torch.zeros(occ.shape[0], 3, dtype=torch.bool)),
                                ("B", 0.5, es._scenario_b_known_flags(occ.shape[0], occ.device)),
                                ("C", 0.25, torch.ones(occ.shape[0], 3, dtype=torch.bool))):
            sk = sk.to(occ.device)
            M = masker.sample(occ, ratio=ratio).to(occ.device)
            res = model(occ, sv, sk, spec, M, goal_mode="real", with_target=False)
            _, deployed = model.decode_geometry(
                res["z_hat"], res["scalar_pred"], occ_input=occ, mask=M,
                scalar_known=sk, scalar_values=sv, hard_forward=True)

            def predict_fn(o, m):
                o_out = model(o, sv, sk, spec, m, goal_mode="real", with_target=False)
                return model.decode_occupancy_prob(o_out["z_hat"], o_out["scalar_pred"],
                                                   scalar_known=sk, scalar_values=sv)

            report = masked_fill_report(deployed, occ, occ, M)
            report["locality"] = locality_probe(predict_fn, occ, M)
            out[f"scenario_{name}"] = report
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
