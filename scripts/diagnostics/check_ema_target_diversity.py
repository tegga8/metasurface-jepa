"""EMA target representation diversity diagnostic — Milestone B collapse gate.

Loads a Milestone B 'jepa' checkpoint, runs the EMA target encoder over held-out
geometries, and reports input-dependence statistics:
  - token variance/std (variance across samples at a fixed spatial token)
  - sample variance/std (variance across tokens within a sample)
  - different-sample pairwise cosine on mean-pooled embeddings (mean/median/p05/min)
  - same-spatial-token cross-sample cosine
  - entropy effective rank / participation rank / top eigenvalue fraction

Judged against two anchors per the operator-approved fix plan:
  (a) the collapsed run (Kaggle checkpoint step 2687): cross-sample cosine ~0.99987,
      effective rank ~2.6/384, participation 2.10, top eigenvalue fraction 0.631
  (b) Milestone A healthy signals: z_S cross-sample mean cosine 0.184, block-11
      clustering ARI 0.397
plus two in-script healthy references: the released MetaDiT ViT geometry encoder and
a fresh random-init encoder. The verdict is a comparison, not a fixed cutoff; the
human operator makes the final call from the reported numbers.

--check-ema-resync asserts B1: after build with released init, the EMA target must
reproduce the student exactly at step 0 (ξ(0) = θ(0)).

Usage:
  python scripts/diagnostics/check_ema_target_diversity.py \
      --checkpoint checkpoints/milestone_b/minimal_jepa_latest.pt --max-geoms 512
  python scripts/diagnostics/check_ema_target_diversity.py --check-ema-resync
"""

import argparse
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from assembly import build_model, load_into_model
from data.dataset import MetaDiTDataset, collate_batch
from encoders.geometry_encoder import GeometryEncoder

COLLAPSED_ANCHOR = {
    "pairwise_cos": 0.999870,
    "pairwise_p05": 0.999601,
    "same_token_cos": 0.999266,
    "eff_rank_unnorm": 2.5986,
    "eff_rank_frac": 2.5986 / 384.0,
    "participation": 2.10,
    "top_eig_frac": 0.6307,
}

MILESTONE_A_ANCHOR = {
    "zS_cross_sample_mean_cos": 0.184,
    "block11_clustering_ari": 0.397,
}


def eff_ranks(X):
    """Entropy effective rank (fraction + unnormalized), participation ratio, top
    eigenvalue fraction, over centered embeddings (B, D)."""
    Xc = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    e = (s ** 2).clamp(min=0.0)
    total = e.sum().item()
    if total <= 0.0:
        return {"eff_rank_unnorm": 0.0, "eff_rank_frac": 0.0,
                "participation": 0.0, "top_eig_frac": 0.0}
    p = e / e.sum()
    ent = -(p * torch.log(p + 1e-12)).sum().item()
    n = p.numel()
    part = total ** 2 / (e ** 2).sum().item()
    top = e.max().item() / total
    return {"eff_rank_unnorm": ent, "eff_rank_frac": ent / math.log(n),
            "participation": part, "top_eig_frac": top}


def pairwise_cos_stats(X_mean):
    """Different-sample pairwise cosine on mean-pooled (B, D) embeddings."""
    Xn = F.normalize(X_mean, dim=-1)
    G = Xn @ Xn.T
    idx = torch.triu_indices(Xn.shape[0], Xn.shape[0], offset=1)
    v = G[idx[0], idx[1]]
    return {"mean": v.mean().item(), "median": v.median().item(),
            "p05": v.quantile(0.05).item(), "min": v.min().item()}


def same_token_cos(X):
    """Per-token-position cross-sample pairwise cosine, averaged (B, T, D)."""
    Xn = F.normalize(X, dim=-1)
    G = torch.einsum("btd,bsd->tbs", Xn, Xn)
    idx = torch.triu_indices(Xn.shape[0], Xn.shape[0], offset=1)
    return G[:, idx[0], idx[1]].mean().item()


def var_stats(X):
    return {"token_var": X.var(dim=0).mean().item(),
            "token_std": X.var(dim=0).sqrt().mean().item(),
            "sample_var": X.var(dim=1).mean().item(),
            "sample_std": X.var(dim=1).sqrt().mean().item()}


def encoder_stats(encoder, geoms, device, max_geoms):
    stats = {}
    with torch.no_grad():
        embs = []
        for G in geoms:
            G = G.to(device)
            x = encoder(G)
            embs.append(x.cpu())
            if sum(e.shape[0] for e in embs) >= max_geoms:
                break
        X = torch.cat(embs, dim=0)[:max_geoms]
    stats.update(var_stats(X))
    stats["pairwise_cos"] = pairwise_cos_stats(X.mean(dim=1))
    stats["same_token_cos"] = same_token_cos(X)
    stats.update(eff_ranks(X.mean(dim=1)))
    stats["n_geoms"] = X.shape[0]
    return stats


def verdict(target, collapsed, refs):
    """Reference-relative, not absolute-cutoff: the target is non-degenerate when it
    behaves like the healthy released-ViT reference AND is far from the collapsed
    anchor on the sharp discriminators (effective-rank fraction, p05 cosine tail,
    same-token cosine)."""
    rel = refs["released_vit"]
    d_rank = abs(target["eff_rank_frac"] - rel["eff_rank_frac"])
    d_cos = abs(target["pairwise_cos"]["p05"] - rel["pairwise_cos"]["p05"])
    d_token = abs(target["same_token_cos"] - rel["same_token_cos"])
    near_released = d_rank <= 0.05 and d_cos <= 0.02 and d_token <= 0.05
    far_rank = target["eff_rank_frac"] / max(collapsed["eff_rank_frac"], 1e-9)
    far_cos = collapsed["pairwise_p05"] - target["pairwise_cos"]["p05"]
    non_deg = bool(near_released and far_rank > 5.0 and far_cos > 0.0)
    return {
        "clearly_non_degenerate": non_deg,
        "dist_to_released_vit": {"eff_rank_frac": d_rank, "p05_cos": d_cos,
                                 "same_token_cos": d_token},
        "near_released_vit": bool(near_released),
        "ratio_eff_rank_vs_collapsed": far_rank,
        "margin_p05_cos_vs_collapsed": far_cos,
    }


def build_fresh_model(cfg, device, spec_path, metadit_path, init_from_metadit=True):
    model = build_model(cfg["model"], spec_path, device=device,
                        init_from_metadit=init_from_metadit,
                        metadit_weights=metadit_path)
    model.eval()
    return model


def check_ema_resync(cfg, device, spec_path, metadit_path, geoms):
    model = build_fresh_model(cfg, device, spec_path, metadit_path)
    G = geoms[0].to(device)
    with torch.no_grad():
        z_student = model.geometry_encoder(G)
        z_target = model.ema(G)
    diff = (z_student - z_target).abs().max().item()
    cos = F.cosine_similarity(z_student.flatten(1), z_target.flatten(1)).mean().item()
    ok = diff < 1e-6 and cos > 1.0 - 1e-6
    print(f"[resync] student vs EMA target at step 0: max_abs_diff={diff:.3e} "
          f"mean_cos={cos:.10f} -> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "milestone_b.yaml"))
    p.add_argument("--max-geoms", type=int, default=512)
    p.add_argument("--out", default=None, help="optional json output path")
    p.add_argument("--device", default="cpu")
    p.add_argument("--check-ema-resync", action="store_true",
                   help="assert EMA target == released-init student at step 0 (B1)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.checkpoint is None and not args.check_ema_resync:
        p.error("need --checkpoint or --check-ema-resync")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device)
    spec_path = os.path.join(REPO_ROOT, cfg["weights"]["spectrum"])
    metadit_path = os.path.join(REPO_ROOT, cfg["weights"]["metadit"])
    data_path = os.path.join(REPO_ROOT, cfg["data"]["val_split"])
    torch.manual_seed(args.seed)

    ds = MetaDiTDataset(data_path)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0,
                        drop_last=True, collate_fn=collate_batch)
    geoms = []
    for G, _ in loader:
        geoms.append(G)
        if sum(g.shape[0] for g in geoms) >= args.max_geoms:
            break
    if not geoms:
        raise RuntimeError("no geometries loaded")

    if args.check_ema_resync:
        ok = check_ema_resync(cfg, device, spec_path, metadit_path, geoms)
        sys.exit(0 if ok else 1)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt["cfg"] if isinstance(ckpt, dict) and "cfg" in ckpt else cfg
    model = build_fresh_model(ckpt_cfg, device, spec_path, metadit_path)
    if isinstance(ckpt, dict) and "model" in ckpt:
        load_into_model(model, ckpt["model"], device)
    else:
        load_into_model(model, ckpt, device)

    released = GeometryEncoder(hidden=cfg["model"].get("hidden", 384),
                               num_heads=cfg["model"].get("num_heads", 6),
                               depth=cfg["model"].get("geo_depth", 6))
    released.init_from_metadit(torch.load(metadit_path, map_location="cpu"),
                               blocks_to_take=cfg["model"].get("geo_depth", 6))
    released.eval()
    random_enc = GeometryEncoder(hidden=cfg["model"].get("hidden", 384),
                                 num_heads=cfg["model"].get("num_heads", 6),
                                 depth=cfg["model"].get("geo_depth", 6))
    random_enc.eval()

    target = encoder_stats(model.ema, geoms, device, args.max_geoms)
    refs = {"released_vit": encoder_stats(released, geoms, device, args.max_geoms),
            "random_init": encoder_stats(random_enc, geoms, device, args.max_geoms)}
    v = verdict(target, COLLAPSED_ANCHOR, refs)

    print(f"\ncheckpoint: {args.checkpoint}")
    print(f"step/epoch: {ckpt.get('step') if isinstance(ckpt, dict) and 'step' in ckpt else 'n/a'} / "
          f"{ckpt.get('epoch') if isinstance(ckpt, dict) and 'epoch' in ckpt else 'n/a'}")
    print(f"geometries: {target['n_geoms']}\n")
    hdr = f"{'metric':<28}{'EMA target':>14}{'released ViT':>14}{'random init':>14}"
    print(hdr)
    print("-" * len(hdr))
    for key, label in [("token_var", "token var"), ("token_std", "token std"),
                       ("sample_var", "sample var"), ("sample_std", "sample std"),
                       ("pairwise_cos.mean", "pairwise cos mean"),
                       ("pairwise_cos.median", "pairwise cos median"),
                       ("pairwise_cos.p05", "pairwise cos p05"),
                       ("pairwise_cos.min", "pairwise cos min"),
                       ("same_token_cos", "same-token cos"),
                       ("eff_rank_unnorm", "entropy eff rank"),
                       ("eff_rank_frac", "eff rank frac"),
                       ("participation", "participation"),
                       ("top_eig_frac", "top eig frac")]:
        def get(d, k):
            v2 = d
            for part in k.split("."):
                v2 = v2[part]
            return v2
        row = f"{label:<28}"
        for d in (target, refs["released_vit"], refs["random_init"]):
            row += f"{get(d, key):>14.6g}"
        print(row)

    print("\nAnchors:")
    print(f"  collapsed run (step 2687): pairwise cos mean {COLLAPSED_ANCHOR['pairwise_cos']:.6f} "
          f"/ p05 {COLLAPSED_ANCHOR['pairwise_p05']:.6f}, "
          f"eff rank {COLLAPSED_ANCHOR['eff_rank_unnorm']:.2f}/384 "
          f"(frac {COLLAPSED_ANCHOR['eff_rank_frac']:.4f}), "
          f"participation {COLLAPSED_ANCHOR['participation']:.2f}, "
          f"top eig frac {COLLAPSED_ANCHOR['top_eig_frac']:.4f}")
    print(f"  Milestone A healthy signals: z_S cross-sample mean cos "
          f"{MILESTONE_A_ANCHOR['zS_cross_sample_mean_cos']:.3f}, "
          f"block-11 clustering ARI {MILESTONE_A_ANCHOR['block11_clustering_ari']:.3f}")
    d_rel = v["dist_to_released_vit"]
    print(f"\nVERDICT: {'CLEARLY NON-DEGENERATE' if v['clearly_non_degenerate'] else 'STILL COLLAPSED / DEGENERATE'}")
    print(f"  distance to healthy released-ViT reference: eff_rank_frac {d_rel['eff_rank_frac']:.4f} "
          f"(<= 0.05), p05 cos {d_rel['p05_cos']:.4f} (<= 0.02), "
          f"same-token cos {d_rel['same_token_cos']:.4f} (<= 0.05) -> "
          f"{'near' if v['near_released_vit'] else 'far'}")
    print(f"  margin p05 cos vs collapsed anchor: {v['margin_p05_cos_vs_collapsed']:.4f} (need > 0)")
    print(f"  eff-rank ratio vs collapsed anchor: {v['ratio_eff_rank_vs_collapsed']:.2f}x (need > 5x)")

    if args.out:
        result = {"checkpoint": args.checkpoint, "target": target, "refs": refs,
                  "anchors": {"collapsed": COLLAPSED_ANCHOR, "milestone_a": MILESTONE_A_ANCHOR},
                  "verdict": v}
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\njson -> {args.out}")


if __name__ == "__main__":
    main()
