"""Create a synthetic "collapsed EMA target" checkpoint for Milestone B diagnostics.

The original collapsed Kaggle checkpoint (step 2687) is NOT on this machine — only
its measured anchor statistics were preserved (eff rank ~2.6/384, participation 2.10,
pairwise cosine 0.99987, same-token 0.99927, top eig frac 0.6307). This script
rebuilds a checkpoint whose EMA target reproduces those signatures so the diversity
diagnostic (and any future collapse guard) can be exercised against a collapsed state
without the original artifact.

Construction: take the B1+B2 released-initialized build (MetaDiT geometry init + EMA
synchronized with the student + projection head), then corrupt ONLY the geometry
encoder / EMA target so the encoder output is nearly input-independent:
  - patch_embed.weight = rank1(u) + eps * randn   (rank-1 dominant + tiny noise)
  - patch_embed.bias = 0, pos_embed = 0
  - transformer blocks optionally scaled by gamma (grid-searched) to keep the output
    subspace low-dimensional
The EMA target is re-synchronized with the corrupted student (B1 semantics), so the
checkpoint is a valid loadable artifact for the diagnostic.

A small calibration grid (measured on a few dozen geometries) picks the corruption
strength closest to the historical anchors:
    eff_rank_unnorm ~ 2.6, eff_rank_frac ~ 0.0068, pairwise cos ~ 0.9999,
    same-token cos ~ 0.999.

Usage:
    python scripts/diagnostics/make_synthetic_collapsed_ckpt.py \
        --out checkpoints/milestone_b/synthetic_collapsed.pt
"""

import argparse
import copy
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import torch
import yaml
from torch.utils.data import DataLoader

from assembly import build_model, saveable_state_dict
from data.dataset import MetaDiTDataset, collate_batch

sys.path.insert(0, os.path.join(SCRIPTS_DIR, "diagnostics"))
from check_ema_target_diversity import (COLLAPSED_ANCHOR, encoder_stats,
                                        var_stats, eff_ranks,
                                        pairwise_cos_stats, same_token_cos)

DEFAULT_OUT = os.path.join(REPO_ROOT, "checkpoints", "milestone_b",
                           "synthetic_collapsed.pt")


def _summarize(encoder, geoms, device, max_geoms):
    with torch.no_grad():
        embs = []
        for G in geoms:
            x = encoder(G.to(device))
            embs.append(x.cpu())
            if sum(e.shape[0] for e in embs) >= max_geoms:
                break
        X = torch.cat(embs, dim=0)[:max_geoms]
    pooled = X.mean(dim=1)
    out = {"eff_rank_unnorm": eff_ranks(pooled)["eff_rank_unnorm"],
           "eff_rank_frac": eff_ranks(pooled)["eff_rank_frac"],
           "pairwise_mean": pairwise_cos_stats(pooled)["mean"],
           "pairwise_p05": pairwise_cos_stats(pooled)["p05"],
           "same_token_cos": same_token_cos(X)}
    return out


def _distance(stats):
    a = COLLAPSED_ANCHOR
    return (abs(stats["eff_rank_unnorm"] - a["eff_rank_unnorm"]) / a["eff_rank_unnorm"]
            + abs(stats["pairwise_mean"] - a["pairwise_cos"]) / 0.02
            + abs(stats["same_token_cos"] - a["same_token_cos"]) / 0.02)


def corrupt_encoder(encoder, eps, gamma, seed=7):
    """Return a deep-copied encoder with near-input-independent output."""
    e = copy.deepcopy(encoder)
    with torch.no_grad():
        w = e.patch_embed.weight                        # (384, 3, 4, 4)
        gen = torch.Generator().manual_seed(seed)
        noise = torch.randn(w.shape, generator=gen)
        u = torch.randn(w.shape[0], 1, 1, 1, generator=gen)
        rank1 = (u.expand_as(w) / math.sqrt(w.shape[1] * w.shape[2] * w.shape[3]))
        e.patch_embed.weight.copy_(rank1 + eps * noise)
        e.patch_embed.bias.zero_()
        e.pos_embed.zero_()
        for block in e.blocks:
            for p in block.parameters():
                p.mul_(gamma)
    return e


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "milestone_b.yaml"))
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--max-geoms", type=int, default=64, help="calibration size")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device)
    spec_path = os.path.join(REPO_ROOT, cfg["weights"]["spectrum"])
    metadit_path = os.path.join(REPO_ROOT, cfg["weights"]["metadit"])
    data_path = os.path.join(REPO_ROOT, cfg["data"]["val_split"])

    ds = MetaDiTDataset(data_path)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0,
                        drop_last=True, collate_fn=collate_batch)
    geoms = []
    for G, _ in loader:
        geoms.append(G)
        if sum(g.shape[0] for g in geoms) >= args.max_geoms:
            break

    model = build_model(cfg["model"], spec_path, device=device,
                        init_from_metadit=True, metadit_weights=metadit_path)
    model.eval()

    grid = [(eps, gamma) for eps in (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
            for gamma in (0.0, 0.1, 0.3, 1.0)]
    best, best_d = None, float("inf")
    print(f"calibrating {len(grid)} corruption candidates on {args.max_geoms} geoms ...")
    for eps, gamma in grid:
        enc = corrupt_encoder(model.geometry_encoder, eps, gamma)
        s = _summarize(enc, geoms, device, args.max_geoms)
        d = _distance(s)
        print(f"  eps={eps:.0e} gamma={gamma:.2f} -> eff_rank={s['eff_rank_unnorm']:.3f} "
              f"pairwise={s['pairwise_mean']:.6f} p05={s['pairwise_p05']:.6f} "
              f"same_token={s['same_token_cos']:.6f} d={d:.3f}")
        if d < best_d:
            best_d, best = d, (eps, gamma)

    eps, gamma = best
    print(f"\nselected: eps={eps:.0e} gamma={gamma:.2f} (distance {best_d:.3f})")
    enc = corrupt_encoder(model.geometry_encoder, eps, gamma)
    model.geometry_encoder.load_state_dict(enc.state_dict())
    model.ema.target.load_state_dict(model.geometry_encoder.state_dict())  # B1 resync

    final = _summarize(model.ema, geoms, device, args.max_geoms)
    print(f"final EMA-target stats: {json.dumps({k: round(v, 6) for k, v in final.items()})}")
    print(f"historical anchor:      {json.dumps({k: round(v, 6) for k, v in COLLAPSED_ANCHOR.items()})}")

    obj = {
        "step": 2687, "epoch": 0, "cfg": cfg, "best": {"primary": 0.0003186},
        "model": saveable_state_dict(model),
        "optimizer": None,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": __import__("numpy").random.get_state(),
        "meta": {
            "synthetic_collapsed": True,
            "description": "EMA target corrupted (rank-1 patch embed + zero pos_embed + "
                           "scaled blocks) to reproduce the step-2687 collapsed anchor's "
                           "pairwise-cosine signatures (p05 ~0.99977 vs anchor 0.99960, "
                           "same-token ~0.9995 vs 0.99927). Eff-rank entropy (H ~0.005) "
                           "is far below the healthy reference and below the anchor's "
                           "H=2.5986 (the anchor's long-tail spectrum is not reproduced "
                           "by rank-1 corruption); the diagnostic still classifies this "
                           "as collapsed. Original Kaggle checkpoint not on this machine.",
            "corruption": {"eps": eps, "gamma": gamma},
            "measured_stats": final,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(obj, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
