"""NN-scoping probe (approved plan: ~/.commandcode/plans/nn-scoping-probe.md).

Decides the framing question: is NN retrieval's advantage universal, or does it fail
where our model handles it? Three measurements, no training:

  1. baseline  -- ours (Scenario A) vs NN on the same held-out items (sanity);
  2. novelty   -- ours vs NN stratified by NN retrieval distance in spectrum space
                  (does retrieval's edge vanish when there is no close neighbour?);
  3. retrofit  -- Scenario C (25% mask + all scalars known): ours vs NN-with-retention.

All arms are scored by the frozen surrogate in MetaDiT MAE units.

Run (real, on a box with the checkpoint):
    python scripts/diagnostics/nn_scoping.py --config configs/unified.yaml \
        --checkpoint checkpoints/unified/latest.pt --split test --samples 512 \
        --pool 20000 --device cpu
Pipeline check (random init, synthetic data):
    python scripts/diagnostics/nn_scoping.py --config configs/unified.yaml --smoke
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "src")
BENCH = os.path.join(REPO_ROOT, "scripts", "benchmark")
EVAL = os.path.join(REPO_ROOT, "scripts", "eval")
for _p in (REPO_ROOT, SRC, BENCH, EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import yaml

import metadit_metrics as mm
import eval_scenarios as es
import benchmark_metadit as bm
from data.factorize import factorize_geometry, assemble_metadit_geometry


def novelty_quartiles(dists, *value_arrays):
    """Split items into 4 quartiles by ascending retrieval distance and return, per
    quartile, the mean of every value array (quartile 0 = closest neighbours)."""
    order = np.argsort(np.asarray(dists))
    out = []
    for q in np.array_split(order, 4):
        out.append(tuple(float(np.mean(np.asarray(v)[q])) for v in value_arrays))
    return out


def retain_visible(pred_occ, occ_input, mask):
    """occ_input on visible pixels, pred_occ on masked pixels (the model's rule)."""
    b = mask.shape[0]
    vis = mask.view(b, 1, 16, 16).repeat_interleave(4, 2).repeat_interleave(4, 3) > 0.5
    return torch.where(vis, occ_input, pred_occ)


def _load_held_out(cfg, split, n, device, smoke):
    if smoke:
        return es._make_synthetic_batch(max(2, n), device)
    from data.dataset import MetaDiTDataset, collate_batch
    from torch.utils.data import DataLoader
    ds = MetaDiTDataset(bm._split_path(cfg, split), max_samples=n, seed=0)
    occ_l, sv_l, spec_l = [], [], []
    for G, S in DataLoader(ds, batch_size=64, shuffle=False,
                           collate_fn=collate_batch, num_workers=0):
        occ, sv = factorize_geometry(G)
        occ_l.append(occ.to(device)); sv_l.append(sv.to(device)); spec_l.append(S.to(device))
    return torch.cat(occ_l), torch.cat(sv_l), torch.cat(spec_l)


def _load_pool(cfg, pool, device, smoke):
    if smoke:
        _, _, spec = es._make_synthetic_batch(max(2, pool), device)
        return spec, None, None
    from data.dataset import MetaDiTDataset, collate_batch
    from torch.utils.data import DataLoader
    ds = MetaDiTDataset(bm._split_path(cfg, "train"), max_samples=pool, seed=0)
    occ_l, sv_l, spec_l = [], [], []
    for G, S in DataLoader(ds, batch_size=128, shuffle=False,
                           collate_fn=collate_batch, num_workers=0):
        occ, sv = factorize_geometry(G)
        occ_l.append(occ.to(device)); sv_l.append(sv.to(device)); spec_l.append(S.to(device))
    return torch.cat(spec_l), torch.cat(occ_l), torch.cat(sv_l)


@torch.no_grad()
def _nearest(target_spec, pool_spec):
    """Per-item nearest pool index + mean-L1 distance (chunked cdist, memory-safe)."""
    n, p = target_spec.shape[0], pool_spec.shape[0]
    idx = torch.empty(n, dtype=torch.long)
    dist = torch.empty(n)
    a = target_spec.reshape(n, -1)
    bpool = pool_spec.reshape(p, -1)
    for s in range(0, n, 64):
        e = min(s + 64, n)
        d = torch.cdist(a[s:e], bpool, p=1) / a.shape[1]      # mean L1 (matches nn_metrics)
        mn, mi = d.min(dim=1)
        idx[s:e] = mi.cpu(); dist[s:e] = mn.cpu()
    return idx, dist


@torch.no_grad()
def _ours_occ(model, occ, sv, sk, spec, M):
    out = model(occ, sv, sk, spec, M, goal_mode="real", with_target=False)
    geom, _ = model.decode_geometry(out["z_hat"], out["scalar_pred"], occ_input=occ,
                                    mask=M, scalar_known=sk, scalar_values=sv,
                                    hard_forward=True)
    return geom                                                 # (B, 3, 64, 64)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="NN-scoping probe")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--pool", type=int, default=20000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="random-init model + synthetic data (pipeline check only)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device)
    model, surrogate = bm._load_model_and_surrogate(cfg, args.checkpoint, device,
                                                    args.smoke)

    occ, sv, spec = _load_held_out(cfg, args.split, args.samples, device, args.smoke)
    pool_spec, pool_occ, pool_sv = _load_pool(cfg, args.pool, device, args.smoke)
    n = occ.shape[0]
    idx, nn_dist = _nearest(spec, pool_spec)

    masker = es.BlockMasker(placement="random", grid=16, min_side=3,
                            k_range=(1, 4), seed=999)

    # --- Scenario A: baseline + novelty stratification ---
    skA = torch.zeros(n, 3, dtype=torch.bool, device=device)
    MA = masker.sample(occ, ratio=1.0).to(device)
    oursA, nnA = torch.zeros(n), torch.zeros(n)
    for i in range(n):
        t = spec[i:i + 1]
        ogeom = _ours_occ(model, occ[i:i + 1], sv[i:i + 1], skA[i:i + 1], t,
                          MA[i:i + 1])
        oursA[i] = mm.per_item_mae(t, surrogate(ogeom).prediction)
        j = int(idx[i])
        if args.smoke:
            nngeom = assemble_metadit_geometry(occ[i:i + 1], sv[i:i + 1, 0],
                                               sv[i:i + 1, 1], sv[i:i + 1, 2])
        else:
            nngeom = assemble_metadit_geometry(pool_occ[j:j + 1], pool_sv[j:j + 1, 0],
                                               pool_sv[j:j + 1, 1], pool_sv[j:j + 1, 2])
        nnA[i] = mm.per_item_mae(t, surrogate(nngeom).prediction)

    # --- Scenario C: retrofit, 25% mask + all scalars known ---
    skC = torch.ones(n, 3, dtype=torch.bool, device=device)
    MC = masker.sample(occ, ratio=0.25).to(device)
    oursC, nnC = [], []
    if not args.smoke:
        for i in range(n):
            t = spec[i:i + 1]
            ogeom = _ours_occ(model, occ[i:i + 1], sv[i:i + 1], skC[i:i + 1], t,
                              MC[i:i + 1])
            oursC.append(float(mm.per_item_mae(t, surrogate(ogeom).prediction)))
            j = int(idx[i])
            ret = retain_visible(pool_occ[j:j + 1], occ[i:i + 1], MC[i:i + 1])
            nngeom = assemble_metadit_geometry(ret, sv[i:i + 1, 0], sv[i:i + 1, 1],
                                               sv[i:i + 1, 2])
            nnC.append(float(mm.per_item_mae(t, surrogate(nngeom).prediction)))

    qs = novelty_quartiles(nn_dist.numpy(), oursA.numpy(), nnA.numpy())
    order = np.argsort(nn_dist.numpy())
    q_dist = [float(nn_dist.numpy()[c].mean()) for c in np.array_split(order, 4)]
    win = oursA < nnA   # headroom: how often ours beats NN, and by how much

    report = {
        "split": args.split, "n": int(n), "pool": int(pool_spec.shape[0]),
        "smoke": bool(args.smoke),
        "scenario_A_baseline": {
            "ours_MAE": float(oursA.mean()),
            "nn_MAE": float(nnA.mean()),
            "ours_beats_nn_fraction": float(win.double().mean()),
            # Headroom for a router/refiner between ours and NN: if ours wins on
            # only a few items, an oracle pick adds ~nothing -> a refiner must
            # produce designs strictly better than BOTH, not merely select.
            "oracle_min_MAE": float(torch.minimum(oursA, nnA).mean()),
            "mean_margin_when_ours_wins":
                float((nnA - oursA)[win].mean()) if bool(win.any()) else 0.0,
        },
        "novelty_quartiles_MAE": [
            {"quartile": q, "nn_dist_mean": q_dist[q], "ours": qs[q][0], "nn": qs[q][1]}
            for q in range(4)
        ],
        "scenario_C_retrofit": {
            "ours_MAE": float(np.mean(oursC)) if oursC else None,
            "nn_MAE": float(np.mean(nnC)) if nnC else None,
            "ours_beats_nn_fraction":
                float(np.mean([o < m for o, m in zip(oursC, nnC)])) if oursC else None,
        },
        "nn_dist_stats": {"mean": float(nn_dist.mean()),
                          "median": float(nn_dist.median())},
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
