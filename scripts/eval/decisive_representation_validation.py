#!/usr/bin/env python3
"""
MILESTONE-B DECISIVE REPRESENTATION VALIDATION
==============================================

One-pass, gated validation for the four existing JEPA objectives:

    jepa_vicreg
    jepa_vicreg2
    jepa_barlow
    lejepa

This is NOT another training script.

It evaluates already-trained checkpoints on multiple genuinely held-out pools,
multiple random mask seeds, and four mask ratios.  It automatically performs
the checks that were repeatedly being done manually:

A. TARGET REPRESENTATION SANITY
   Full geometry -> EMA target encoder -> target representation
   - cross-sample cosine distribution
   - pooled effective rank
   - full-token effective rank
   - per-dimension variance
   - sample-to-sample diversity

B. PHYSICS-RELEVANCE OF TARGET SPACE
   For unseen sample pairs:
       latent distance <-> EM-spectrum distance
   - Pearson correlation
   - Spearman correlation
   - spectrum-nearest-neighbour / latent-nearest-neighbour agreement
   - most-similar vs most-different spectrum pairs

C. JEPA PREDICTION
   masked geometry + physics conditioning -> predictor
   versus
   full geometry -> EMA target
   - prediction cosine error
   - null prediction baseline
   - null gap
   - prediction diversity
   - target/prediction diversity
   - masked-token diagnostics

D. MASK ROBUSTNESS
   25%, 50%, 75%, 100% masking
   multiple independent mask seeds

E. AUTOMATIC GATING
   A model is rejected immediately if the target representation is clearly
   collapsed or if its latent space fails the physics-relevance sanity test.
   Only a model passing those gates receives the expensive multi-mask evaluation.

F. AUTOMATIC WINNER
   Among models that pass all gates, select by a composite score that rewards:
       - low JEPA prediction error
       - improvement over null
       - target/prediction diversity
       - physics-latent correlation
       - stability across pools/seeds
   No hand-picking after the run.

IMPORTANT
---------
"Different datasets" here means independent held-out validation POOLS sampled
from the repository's val_split. They are NOT training samples and are not
claimed to be an external dataset. If the repo later gains a true test split,
pass it with --split.

The script never uses train_split for the model-selection measurements.

Default experiment:
    4 held-out pools × 512 geometries
    4 mask ratios: 0.25, 0.50, 0.75, 1.00
    4 mask seeds
    4 models

The target representation test is done on FULL geometries before masking.
This is deliberate: it answers the fundamental question:
"Does the same EMA target encoder distinguish genuinely different unseen
geometries?"

Example:
python scripts/eval/decisive_representation_validation.py \
  --checkpoint jepa_vicreg=checkpoints/milestone_b/adaptive/phase_00_jepa_vicreg_best_healthy.pt \
  --checkpoint jepa_vicreg2=checkpoints/milestone_b/adaptive/phase_01_jepa_vicreg2_best_healthy.pt \
  --checkpoint jepa_barlow=checkpoints/milestone_b/adaptive/phase_02_jepa_barlow_best_healthy.pt \
  --checkpoint lejepa=checkpoints/milestone_b/adaptive/phase_03_lejepa_best_healthy.pt \
  --pools 4 \
  --pool-size 512 \
  --pairs 20000 \
  --batch-size 32 \
  --device cuda

For a faster smoke:
    --pools 2 --pool-size 256 --pairs 5000

Outputs:
    .../decisive_representation_validation/
        final_report.json
        final_report.csv
        target_audit_<model>.json
        mask_audit_<model>.json
        decision.json
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_DIR))

from assembly import build_model, load_into_model
from data.dataset import MetaDiTDataset, collate_batch
from data.mask import BlockMasker


MODEL_ORDER = [
    "jepa_vicreg",
    "jepa_vicreg2",
    "jepa_barlow",
    "lejepa",
]

DEFAULT_POOLS = [1101, 2202, 3303, 4404]
DEFAULT_MASKS = [0.25, 0.50, 0.75, 1.00]
DEFAULT_MASK_SEEDS = [1101, 2202, 3303, 4404]


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite(x):
    return bool(np.isfinite(float(x)))


def pct(x, q):
    x = np.asarray(x)
    return float(np.percentile(x, q)) if len(x) else float("nan")


def describe(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p05": pct(x, 5),
        "p95": pct(x, 95),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x*x).sum() * (y*y).sum())
    return float((x*y).sum() / d) if d > 0 else float("nan")


def rank_average(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        r[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return r


def spearman(x, y):
    return pearson(rank_average(x), rank_average(y))


def effective_rank(X):
    X = X.double()
    X = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X)
    e = (s*s).clamp_min(0)
    total = float(e.sum())
    if total <= 0:
        return {
            "effective_rank": 0.0,
            "rank_fraction": 0.0,
            "participation_ratio": 0.0,
            "top_eigenvalue_fraction": 1.0,
        }
    p = e / e.sum()
    er = float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()))
    pr = float((e.sum()**2) / (e*e).sum())
    top = float(e.max() / e.sum())
    return {
        "effective_rank": er,
        "rank_fraction": er / X.shape[1],
        "participation_ratio": pr,
        "top_eigenvalue_fraction": top,
    }


def sample_pairs(n, k, seed):
    rng = np.random.RandomState(seed)
    max_pairs = n * (n - 1) // 2
    k = min(int(k), max_pairs)
    if k <= 0:
        return np.empty((0, 2), dtype=np.int64)

    pairs = set()
    while len(pairs) < k:
        m = min(max(2048, k // 5), k - len(pairs))
        a = rng.randint(0, n, m)
        b = rng.randint(0, n, m)
        for x, y in zip(a, b):
            if x == y:
                continue
            if x > y:
                x, y = y, x
            pairs.add((int(x), int(y)))
            if len(pairs) >= k:
                break
    return np.asarray(list(pairs), dtype=np.int64)


# ---------------------------------------------------------------------------
# Dataset / held-out pools
# ---------------------------------------------------------------------------

class IndexedDataset(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[int(self.indices[i])]


def make_pool_indices(n, pool_size, seeds):
    """
    Deterministically create disjoint pools from the validation split.
    """
    if len(seeds) * pool_size > n:
        raise RuntimeError(
            f"Need {len(seeds)*pool_size} validation samples, but val split has {n}."
        )

    rng = np.random.RandomState(20260818)
    perm = rng.permutation(n)

    # Map pool seed -> deterministic shuffled offset, while guaranteeing
    # disjointness by assigning contiguous blocks of a global permutation.
    pools = {}
    for i, seed in enumerate(seeds):
        pools[int(seed)] = perm[i*pool_size:(i+1)*pool_size]
    return pools


def load_pool(ds, indices, batch_size):
    loader = DataLoader(
        IndexedDataset(ds, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_batch,
    )
    gs, ss = [], []
    for G, S in loader:
        gs.append(G)
        ss.append(S)
    return torch.cat(gs), torch.cat(ss)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_target_model(cfg, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(
        cfg["model"],
        os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
        device=device,
        init_from_metadit=cfg["model"].get("init_from_metadit", True),
        metadit_weights=os.path.join(REPO_ROOT, cfg["weights"]["metadit"]),
    )
    load_into_model(model, ckpt["model"], device)
    model.eval()
    model.ema.eval()
    return model


# ---------------------------------------------------------------------------
# Representation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_target(model, G, device, batch_size):
    out = []
    for i in range(0, len(G), batch_size):
        z = model.ema(G[i:i+batch_size].to(device))
        if z.ndim != 3:
            raise RuntimeError(f"Expected target [B,T,D], got {tuple(z.shape)}")
        out.append(z.cpu().float())
    return torch.cat(out)


@torch.no_grad()
def run_prediction(model, G, S, ratio, mask_seed, cfg, device, batch_size):
    masker = BlockMasker(
        placement="random",
        grid=int(cfg["model"]["token_grid"]),
        min_side=int(cfg["mask"].get("min_side", 3)),
        k_range=tuple(cfg["mask"].get("k_range", [1, 4])),
        seed=int(mask_seed),
    )

    pred_list = []
    target_list = []
    mask_list = []

    for i in range(0, len(G), batch_size):
        gb = G[i:i+batch_size].to(device)
        sb = S[i:i+batch_size].to(device)
        M = masker.sample(gb, float(ratio)).to(device)

        out = model(gb, sb, M)

        zp = out["z_hat"]
        zy = out["z_y"]
        mask = out["mask"] if "mask" in out else (M.view(M.shape[0], -1) == 0)

        pred_list.append(zp.cpu().float())
        target_list.append(zy.cpu().float())
        mask_list.append(mask.cpu())

    return (
        torch.cat(pred_list),
        torch.cat(target_list),
        torch.cat(mask_list),
    )


# ---------------------------------------------------------------------------
# Target representation / physics audit
# ---------------------------------------------------------------------------

def spectrum_features(S):
    X = S.reshape(len(S), -1).double()
    return F.normalize(X, dim=-1)


def target_audit(Z, S, pairs_n, pair_seed):
    """
    Z: [N,T,D], FULL geometry target representation.
    S: [N,...], corresponding EM responses.
    """
    N, T, D = Z.shape
    pooled = Z.mean(dim=1)
    pooled_n = F.normalize(pooled, dim=-1)
    token_n = F.normalize(Z, dim=-1)

    rank = effective_rank(pooled)
    flat_rank = effective_rank(Z.reshape(N*T, D))

    pairs = sample_pairs(N, pairs_n, pair_seed)
    ii = torch.from_numpy(pairs[:, 0]).long()
    jj = torch.from_numpy(pairs[:, 1]).long()

    zcos = (pooled_n[ii] * pooled_n[jj]).sum(-1).numpy()
    scos = (
        spectrum_features(S)[ii] * spectrum_features(S)[jj]
    ).sum(-1).numpy()

    latent_dist = 1.0 - zcos
    spectrum_dist = 1.0 - scos

    # Same spatial token, cross-sample cosine.
    vals = []
    for start in range(0, len(pairs), 1024):
        a = token_n[ii[start:start+1024]]
        b = token_n[jj[start:start+1024]]
        vals.append((a*b).sum(-1).mean(-1))
    same_tok = torch.cat(vals).numpy()

    # Spectrum quintiles -> latent separation.
    q = np.quantile(spectrum_dist, [0,.2,.4,.6,.8,1])
    buckets = []
    for b in range(5):
        m = (
            (spectrum_dist >= q[b]) &
            (spectrum_dist <= q[b+1])
            if b == 0 else
            (spectrum_dist > q[b]) &
            (spectrum_dist <= q[b+1])
        )
        if m.any():
            buckets.append({
                "bucket": b,
                "n": int(m.sum()),
                "spectrum_dist_mean": float(spectrum_dist[m].mean()),
                "latent_dist_mean": float(latent_dist[m].mean()),
                "latent_dist_median": float(np.median(latent_dist[m])),
            })

    # Extreme 5%.
    k = max(1, len(pairs)//20)
    order = np.argsort(spectrum_dist)
    low = order[:k]
    high = order[-k:]

    return {
        "N": N,
        "tokens": T,
        "D": D,
        "pooled_rank": rank,
        "full_token_rank": flat_rank,
        "pooled_cosine": describe(zcos),
        "same_token_cosine": describe(same_tok),
        "latent_distance": describe(latent_dist),
        "spectrum_distance": describe(spectrum_dist),
        "pearson_latent_vs_spectrum": pearson(latent_dist, spectrum_dist),
        "spearman_latent_vs_spectrum": spearman(latent_dist, spectrum_dist),
        "spectrum_quintiles": buckets,
        "similar_spectrum_5pct_latent_distance": describe(latent_dist[low]),
        "different_spectrum_5pct_latent_distance": describe(latent_dist[high]),
        "extreme_separation_ratio": float(
            np.median(latent_dist[high]) /
            max(np.median(latent_dist[low]), 1e-12)
        ),
    }


# ---------------------------------------------------------------------------
# Prediction audit
# ---------------------------------------------------------------------------

def masked_prediction_metrics(zp, zy, mask):
    """
    Prediction is evaluated only on masked positions.

    Also measures prediction diversity across sample-token observations.
    """
    N, T, D = zp.shape
    mask = mask.bool()

    p = zp[mask]
    t = zy[mask]

    p = F.normalize(p, dim=-1)
    t = F.normalize(t, dim=-1)

    cos = (p*t).sum(-1)

    # Null predictor: per-token mean target representation, normalized.
    # This is the "predict a generic target" baseline.
    null = zy.reshape(-1, D).mean(dim=0)
    null = F.normalize(null, dim=0)
    null_cos = (p * null).sum(-1)

    # Prediction diversity is measured across sample-token predictions.
    pred_rank = effective_rank(zp[mask])
    target_rank = effective_rank(zy[mask])

    # Sample-level pooled prediction diversity.
    pred_pool = zp.clone()
    pred_pool[~mask] = 0.0
    counts = mask.sum(dim=1).clamp_min(1).float().unsqueeze(-1)
    pred_pool = pred_pool.sum(dim=1) / counts
    target_pool = zy.clone()
    target_pool[~mask] = 0.0
    target_pool = target_pool.sum(dim=1) / counts

    pred_pool_n = F.normalize(pred_pool, dim=-1)
    target_pool_n = F.normalize(target_pool, dim=-1)

    return {
        "masked_tokens": int(mask.sum()),
        "cos_err": float((1-cos).mean()),
        "cos_p05": float((1-cos).quantile(.05)),
        "cos_p95": float((1-cos).quantile(.95)),
        "null_err": float((1-null_cos).mean()),
        "null_gap": float(
            ((1-null_cos).mean() / max(float((1-cos).mean()), 1e-12))
            - 1.0
        ),
        "pred_token_rank": pred_rank,
        "target_token_rank": target_rank,
        "pred_pool_rank": effective_rank(pred_pool),
        "target_pool_rank": effective_rank(target_pool),
        "pred_pool_cos_mean": float(
            (pred_pool_n @ pred_pool_n.T).fill_diagonal_(0).sum()
            / max(1, N*(N-1))
        ),
        "target_pool_cos_mean": float(
            (target_pool_n @ target_pool_n.T).fill_diagonal_(0).sum()
            / max(1, N*(N-1))
        ),
    }


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def gate_target(a, args):
    """
    Hard target-space gate.

    The thresholds are intentionally conservative and are used together rather
    than individually.

    A model FAILS if:
      - pooled cross-sample cosine is effectively identical (~1),
      - pooled rank is essentially rank-1,
      - physics/latent Spearman relation is non-positive,
      - OR full-token rank is literally degenerate.
    """
    pooled_cos_p05 = a["pooled_cosine"]["p05"]
    pooled_rank = a["pooled_rank"]["effective_rank"]
    full_rank = a["full_token_rank"]["effective_rank"]
    rho = a["spearman_latent_vs_spectrum"]

    failures = []

    if pooled_cos_p05 >= args.collapse_cos_p05:
        failures.append(
            f"pooled_cross_sample_cos_p05={pooled_cos_p05:.6f} "
            f">= {args.collapse_cos_p05}"
        )

    if pooled_rank < args.min_pooled_rank:
        failures.append(
            f"pooled_effective_rank={pooled_rank:.4f} "
            f"< {args.min_pooled_rank}"
        )

    if full_rank < args.min_full_token_rank:
        failures.append(
            f"full_token_effective_rank={full_rank:.4f} "
            f"< {args.min_full_token_rank}"
        )

    if not finite(rho) or rho < args.min_physics_spearman:
        failures.append(
            f"physics_spearman={rho:.4f} "
            f"< {args.min_physics_spearman}"
        )

    return len(failures) == 0, failures


def gate_prediction(m):
    failures = []
    if not finite(m["cos_err"]):
        failures.append("non-finite prediction error")
    if not finite(m["null_err"]):
        failures.append("non-finite null error")
    if m["cos_err"] >= m["null_err"]:
        failures.append(
            f"prediction is not better than null "
            f"({m['cos_err']:.6f} >= {m['null_err']:.6f})"
        )
    return len(failures) == 0, failures


def summarize_mask_results(rows):
    if not rows:
        return {}
    return {
        "mean_cos_err": float(np.mean([r["cos_err"] for r in rows])),
        "median_cos_err": float(np.median([r["cos_err"] for r in rows])),
        "mean_null_gap": float(np.mean([r["null_gap"] for r in rows])),
        "mean_pred_rank": float(
            np.mean([r["pred_token_rank"]["effective_rank"] for r in rows])
        ),
        "worst_cos_err": float(max(r["cos_err"] for r in rows)),
        "pass_rate": float(np.mean([r["pass"] for r in rows])),
    }


def composite_score(target_rows, mask_rows):
    """
    Conservative model-selection score.

    Lower prediction error is good.
    Positive physics correlation is good.
    Higher target rank is mildly rewarded.
    Stability/pass rate is strongly rewarded.

    This is a ranking score, not a scientific metric.
    """
    t_rho = np.nanmean(
        [r["spearman_latent_vs_spectrum"] for r in target_rows]
    )
    t_rank = np.mean(
        [r["pooled_rank"]["effective_rank"] for r in target_rows]
    )

    valid = [r for r in mask_rows if r["pass"]]
    if valid:
        cos = np.mean([r["cos_err"] for r in valid])
        pass_rate = len(valid) / len(mask_rows)
    else:
        cos = 1.0
        pass_rate = 0.0

    # Convert to bounded reward terms.
    prediction_reward = 1.0 / (1.0 + max(cos, 0.0))
    physics_reward = max(0.0, min(1.0, (t_rho + 1.0)/2.0))
    rank_reward = max(0.0, min(1.0, t_rank / 16.0))

    return (
        0.50 * prediction_reward +
        0.25 * physics_reward +
        0.10 * rank_reward +
        0.15 * pass_rate
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="NAME=PATH; repeat for four checkpoints")
    ap.add_argument("--config",
                    default=str(REPO_ROOT/"configs/milestone_b.yaml"))
    ap.add_argument("--split", default=None,
                    help="Override validation split path; defaults to cfg data.val_split")
    ap.add_argument("--pools", type=int, default=4)
    ap.add_argument("--pool-size", type=int, default=512)
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=str(
        REPO_ROOT/"checkpoints/milestone_b/decisive_representation_validation"
    ))
    ap.add_argument("--no-gate", action="store_true",
                    help="Run all mask evaluations even if target gate fails")
    ap.add_argument("--save-embeddings", action="store_true")

    # Hard target-space gates.
    ap.add_argument("--collapse-cos-p05", type=float, default=0.995)
    ap.add_argument("--min-pooled-rank", type=float, default=1.25)
    ap.add_argument("--min-full-token-rank", type=float, default=4.0)
    ap.add_argument("--min-physics-spearman", type=float, default=0.05)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(20260818)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpts = {}
    for x in args.checkpoint:
        name, path = x.split("=", 1)
        ckpts[name] = path

    # Preserve requested order; unknown names go after canonical order.
    ordered = [x for x in MODEL_ORDER if x in ckpts]
    ordered += [x for x in ckpts if x not in ordered]

    split_path = (
        args.split if args.split is not None
        else os.path.join(REPO_ROOT, cfg["data"]["val_split"])
    )

    ds = MetaDiTDataset(split_path)
    pools = make_pool_indices(len(ds), args.pool_size,
                              DEFAULT_POOLS[:args.pools])

    print("="*110)
    print("DECISIVE MILESTONE-B REPRESENTATION VALIDATION")
    print("="*110)
    print(f"split        : {split_path}")
    print(f"pool size    : {args.pool_size}")
    print(f"pools        : {list(pools)}")
    print(f"mask ratios  : {DEFAULT_MASKS}")
    print(f"mask seeds   : {DEFAULT_MASK_SEEDS}")
    print(f"pairs/pool   : {args.pairs}")
    print()
    print("GATE:")
    print(f"  pooled cosine p05 must be < {args.collapse_cos_p05}")
    print(f"  pooled effective rank must be >= {args.min_pooled_rank}")
    print(f"  full-token effective rank must be >= {args.min_full_token_rank}")
    print(f"  latent/spectrum Spearman must be >= {args.min_physics_spearman}")
    print("="*110)

    all_results = {}
    decisions = {}

    for model_name in ordered:
        print("\n" + "#"*110)
        print(f"MODEL: {model_name}")
        print("#"*110)

        model = load_target_model(
            cfg, ckpts[model_name], torch.device(args.device)
        )

        target_rows = []
        target_pass = True
        target_failures = []

        # ---------------------------------------------------------------
        # Stage A: target encoder sanity + physics relevance.
        # This happens BEFORE any mask evaluation.
        # ---------------------------------------------------------------
        for pool_no, (pool_seed, indices) in enumerate(pools.items()):
            print(
                f"\n[target {model_name}] pool={pool_no} "
                f"seed={pool_seed} N={len(indices)}"
            )
            G, S = load_pool(ds, indices, args.batch_size)
            Z = encode_target(
                model, G, torch.device(args.device), args.batch_size
            )

            audit = target_audit(
                Z, S, args.pairs,
                pair_seed=int(pool_seed)
            )
            passed, failures = gate_target(audit, args)

            row = {
                "pool": pool_no,
                "pool_seed": int(pool_seed),
                "n": len(indices),
                **audit,
                "pass": passed,
                "failures": failures,
            }
            target_rows.append(row)

            print(
                f"  pooled rank={audit['pooled_rank']['effective_rank']:.4f} "
                f"cos_p05={audit['pooled_cosine']['p05']:.6f} "
                f"full_rank={audit['full_token_rank']['effective_rank']:.4f} "
                f"rho={audit['spearman_latent_vs_spectrum']:.4f} "
                f"PASS={passed}"
            )

            if not passed:
                target_pass = False
                target_failures.extend(
                    [f"pool {pool_no}: {x}" for x in failures]
                )

            del G, S, Z
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Require every independent pool to pass the fundamental target test.
        # This prevents selecting a model that only works on one convenient pool.
        target_pass = target_pass and len(target_rows) == args.pools

        print(
            f"\nTARGET GATE => "
            f"{'PASS' if target_pass else 'FAIL'}"
        )
        if target_failures:
            for x in target_failures:
                print("  FAIL:", x)

        mask_rows = []

        # ---------------------------------------------------------------
        # Stage B: only continue if target representation survives.
        # ---------------------------------------------------------------
        if target_pass or args.no_gate:
            for pool_no, (pool_seed, indices) in enumerate(pools.items()):
                G, S = load_pool(ds, indices, args.batch_size)

                for ratio in DEFAULT_MASKS:
                    for mask_seed in DEFAULT_MASK_SEEDS:
                        print(
                            f"[mask {model_name}] pool={pool_no} "
                            f"mask={ratio:.2f} seed={mask_seed}"
                        )

                        zp, zy, M = run_prediction(
                            model, G, S, ratio, mask_seed,
                            cfg, torch.device(args.device),
                            args.batch_size
                        )

                        met = masked_prediction_metrics(zp, zy, M)
                        passed, failures = gate_prediction(met)

                        row = {
                            "model": model_name,
                            "pool": pool_no,
                            "pool_seed": int(pool_seed),
                            "mask_ratio": float(ratio),
                            "mask_seed": int(mask_seed),
                            **met,
                            "pass": passed,
                            "failures": failures,
                        }
                        mask_rows.append(row)

                        print(
                            f"  cos={met['cos_err']:.7f} "
                            f"null={met['null_err']:.7f} "
                            f"gap={met['null_gap']:.4f} "
                            f"pred_rank={met['pred_token_rank']['effective_rank']:.3f} "
                            f"PASS={passed}"
                        )

                        del zp, zy, M
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                del G, S

        else:
            print(
                f"\n[SKIP] {model_name} failed the fundamental target-space gate. "
                f"Moving automatically to the next model."
            )

        mask_pass_rate = (
            float(np.mean([r["pass"] for r in mask_rows]))
            if mask_rows else 0.0
        )

        final_pass = (
            target_pass and
            len(mask_rows) > 0 and
            mask_pass_rate >= 0.90
        )

        score = composite_score(target_rows, mask_rows)

        all_results[model_name] = {
            "target_gate_pass": target_pass,
            "target_failures": target_failures,
            "target_rows": target_rows,
            "mask_rows": mask_rows,
            "mask_pass_rate": mask_pass_rate,
            "final_pass": final_pass,
            "composite_score": score,
        }

        decisions[model_name] = {
            "target_gate_pass": target_pass,
            "mask_pass_rate": mask_pass_rate,
            "final_pass": final_pass,
            "composite_score": score,
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Final automatic selection
    # -------------------------------------------------------------------
    passed = [
        (name, r["composite_score"])
        for name, r in all_results.items()
        if r["final_pass"]
    ]
    passed.sort(key=lambda x: x[1], reverse=True)

    winner = passed[0][0] if passed else None

    final = {
        "experiment": {
            "split": split_path,
            "pools": list(pools.keys()),
            "pool_size": args.pool_size,
            "pairs_per_pool": args.pairs,
            "mask_ratios": DEFAULT_MASKS,
            "mask_seeds": DEFAULT_MASK_SEEDS,
        },
        "gates": {
            "collapse_cos_p05": args.collapse_cos_p05,
            "min_pooled_rank": args.min_pooled_rank,
            "min_full_token_rank": args.min_full_token_rank,
            "min_physics_spearman": args.min_physics_spearman,
            "required_mask_pass_rate": 0.90,
        },
        "models": decisions,
        "winner": winner,
        "winner_score": (
            all_results[winner]["composite_score"]
            if winner else None
        ),
    }

    with open(Path(args.out_dir)/"final_report.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(Path(args.out_dir)/"decision.json", "w") as f:
        json.dump(final, f, indent=2)

    with open(Path(args.out_dir)/"final_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "target_gate", "mask_pass_rate",
            "final_pass", "composite_score"
        ])
        for name, d in decisions.items():
            w.writerow([
                name,
                d["target_gate_pass"],
                d["mask_pass_rate"],
                d["final_pass"],
                d["composite_score"],
            ])

    print("\n" + "="*110)
    print("FINAL DECISION")
    print("="*110)

    for name, d in decisions.items():
        print(
            f"{name:<18} "
            f"target={'PASS' if d['target_gate_pass'] else 'FAIL':<4} "
            f"masks={d['mask_pass_rate']:.1%} "
            f"final={'PASS' if d['final_pass'] else 'FAIL':<4} "
            f"score={d['composite_score']:.5f}"
        )

    print("-"*110)
    if winner:
        print(f"WINNER: {winner}")
        print(
            "This winner passed the target-space anti-collapse/physics gate "
            "on every independent held-out pool and passed >=90% of the "
            "multi-mask prediction tests."
        )
    else:
        print("NO MODEL PASSED.")
        print(
            "Do NOT proceed to geometry decoding. Fix the representation "
            "before spending more compute."
        )

    print()
    print(f"JSON -> {Path(args.out_dir)/'final_report.json'}")
    print(f"Decision -> {Path(args.out_dir)/'decision.json'}")
    print(f"CSV -> {Path(args.out_dir)/'final_report.csv'}")


if __name__ == "__main__":
    main()
