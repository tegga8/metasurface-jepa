"""
DECISIVE JEPA REPRESENTATION + PREDICTOR AUDIT
==============================================

Purpose
-------
One clean validation run for the three trained objectives:

    jepa_vicreg
    jepa_barlow
    lejepa

This script deliberately separates THREE questions that were previously mixed:

1. TARGET REPRESENTATION:
   Do different unseen geometries produce different EMA-target embeddings?
   Do those differences track different EM spectra?

2. DIRECT RAW-LATENT PREDICTION:
   Does z_hat itself approach z_target?
   This is the scientifically important representation test.

3. TRAINING/PROJECTED PREDICTION:
   Does P(z_hat) approach P(z_target)?
   This exactly reproduces the repository's JEPA objective-space metric.

It also tests whether the PHYSICS CONDITION actually helps:
    real physics-conditioned predictor vs null/no-physics predictor.

The predictor is considered useful only if the REAL predictor beats the NULL
predictor in the same metric space.

IMPORTANT:
- All model-selection samples come from the repository validation split.
- Pools are disjoint and never use train_split.
- The target audit uses FULL, unmasked geometries.
- Prediction tests use fresh deterministic masks for every pool/mask/seed.
- The raw and projected losses are both reported.
- The script never declares a model healthy merely because projected JEPA loss
  is tiny.

Default:
    4 disjoint pools x 512 unseen validation geometries
    mask ratios = 25%, 50%, 75%, 100%
    4 independent mask seeds
    => 64 mask evaluations per model, after target sanity passes.

Output is intentionally compact:
    - 4 target lines/model
    - 4 aggregated mask lines/model
    - 1 model summary
    - 1 final comparison table

Outputs:
    checkpoints/milestone_b/decisive_representation_validation/
        final_report.json
        final_report.csv
        decision.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from assembly import build_model
from data.dataset import MetaDiTDataset, collate_batch
from data.mask import BlockMasker
from losses.objectives import build_objective
from train.engine import load_checkpoint


MODELS = ["jepa_vicreg", "jepa_barlow", "lejepa"]
POOL_SEEDS = [1101, 2202, 3303, 4404]
MASK_RATIOS = [0.25, 0.50, 0.75, 1.00]
MASK_SEEDS = [1101, 2202, 3303, 4404]


# ---------------------------------------------------------------------------
# Small math utilities
# ---------------------------------------------------------------------------

def seed_all(seed=20260818):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "std": float(x.std()),
    }


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def rank_average(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    s = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and s[j] == s[i]:
            j += 1
        r[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return r


def spearman(x, y):
    return pearson(rank_average(np.asarray(x)), rank_average(np.asarray(y)))


def effective_rank(X):
    X = X.double()
    X = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X)
    e = (s * s).clamp_min(0)
    total = float(e.sum())
    if total <= 0:
        return 0.0
    p = e / e.sum()
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()))


def cosine_distance(a, b):
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


def normalized_cosine_distance(a, b):
    return cosine_distance(
        F.normalize(a, dim=-1),
        F.normalize(b, dim=-1),
    ).clamp_min(0)


def sample_pairs(n, k, seed):
    rng = np.random.RandomState(seed)
    k = min(int(k), n * (n - 1) // 2)
    pairs = set()
    while len(pairs) < k:
        a = rng.randint(0, n, size=max(2048, min(k - len(pairs), 8192)))
        b = rng.randint(0, n, size=len(a))
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
# Validation pools
# ---------------------------------------------------------------------------

class IndexedDataset(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[int(self.indices[i])]


def make_pools(n, pool_size):
    if len(POOL_SEEDS) * pool_size > n:
        raise RuntimeError(
            f"Validation split has {n} samples but "
            f"{len(POOL_SEEDS) * pool_size} are required."
        )
    rng = np.random.RandomState(20260818)
    perm = rng.permutation(n)
    return {
        seed: perm[i * pool_size:(i + 1) * pool_size]
        for i, seed in enumerate(POOL_SEEDS)
    }


def load_pool(ds, indices, batch_size):
    loader = DataLoader(
        IndexedDataset(ds, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_batch,
    )
    Gs, Ss = [], []
    for G, S in loader:
        Gs.append(G)
        Ss.append(S)
    return torch.cat(Gs), torch.cat(Ss)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(cfg, checkpoint, device, name):
    """Build model + objective and restore BOTH from the §30 checkpoint. The
    objective-owned projector defines the projected comparison space — there is
    no `model.proj` (§17), and a checkpoint missing the objective state fails
    loudly inside load_checkpoint."""
    objective = build_objective(
        name, cfg.get("objective_params", {}).get(name, {}),
        projector_input_dim=cfg["model"].get("hidden", 384))
    model = build_model(
        cfg["model"],
        os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
        device=device,
        init_from_metadit=cfg["model"].get("init_from_metadit", True),
        metadit_weights=os.path.join(
            REPO_ROOT, cfg["weights"]["metadit"]
        ),
    )
    load_checkpoint(checkpoint, model, objective, None, None, device)
    model.eval()
    model.ema.eval()
    return model, objective


# ---------------------------------------------------------------------------
# TARGET REPRESENTATION TEST
# ---------------------------------------------------------------------------

@torch.no_grad()
def target_embeddings(model, G, device, batch_size):
    zs = []
    for i in range(0, len(G), batch_size):
        z = model.ema(G[i:i + batch_size].to(device))
        if z.ndim != 3:
            raise RuntimeError(
                f"Target encoder returned {tuple(z.shape)}, expected [B,T,D]."
            )
        zs.append(z.cpu().float())
    return torch.cat(zs)


def target_audit(Z, S, pairs_n, seed):
    N, T, D = Z.shape

    pooled = Z.mean(1)
    pooled_n = F.normalize(pooled, dim=-1)

    pairs = sample_pairs(N, pairs_n, seed)
    ii = torch.from_numpy(pairs[:, 0]).long()
    jj = torch.from_numpy(pairs[:, 1]).long()

    latent_cos = (pooled_n[ii] * pooled_n[jj]).sum(-1).numpy()
    latent_dist = 1.0 - latent_cos

    sf = F.normalize(S.reshape(N, -1).double(), dim=-1)
    spectrum_cos = (sf[ii] * sf[jj]).sum(-1).numpy()
    spectrum_dist = 1.0 - spectrum_cos

    same_token = (
        F.normalize(Z[ii], dim=-1) *
        F.normalize(Z[jj], dim=-1)
    ).sum(-1).mean(-1).numpy()

    return {
        "pooled_rank": effective_rank(pooled),
        "full_token_rank": effective_rank(Z.reshape(N * T, D)),
        "pooled_cos_p05": float(np.percentile(latent_cos, 5)),
        "pooled_cos_mean": float(latent_cos.mean()),
        "same_token_cos_mean": float(same_token.mean()),
        "latent_spectrum_spearman": spearman(latent_dist, spectrum_dist),
        "latent_spectrum_pearson": pearson(latent_dist, spectrum_dist),
        "latent_dist_mean": float(latent_dist.mean()),
        "spectrum_dist_mean": float(spectrum_dist.mean()),
    }


# ---------------------------------------------------------------------------
# PREDICTION TEST
# ---------------------------------------------------------------------------

def make_masker(cfg, seed):
    return BlockMasker(
        placement="random",
        grid=int(cfg["model"]["token_grid"]),
        min_side=int(cfg["mask"].get("min_side", 3)),
        k_range=tuple(cfg["mask"].get("k_range", [1, 4])),
        seed=int(seed),
    )


@torch.no_grad()
def evaluate_condition(
    model, objective, G, S, ratio, mask_seed, cfg, device, batch_size
):
    """
    Evaluate:
      REAL: physics-conditioned predictor
      NULL: identical predictor with goal_mode='null'

    For BOTH branches compute:
      raw-space cosine distance: z_hat vs z_y
      projected-space cosine distance: P(z_hat) vs P(z_y)

    The projected metric exactly mirrors the repository's JEPA objective:
      project both sides, then cosine, masked positions only.
    P is the objective-owned projector (§17); there is no model.proj.
    """
    masker = make_masker(cfg, mask_seed)

    real_raw = []
    real_proj = []
    null_raw = []
    null_proj = []
    shifts = []
    target_raw = []
    pred_raw = []

    for i in range(0, len(G), batch_size):
        gb = G[i:i + batch_size].to(device)
        sb = S[i:i + batch_size].to(device)
        M = masker.sample(gb, float(ratio)).to(device)

        real = model(gb, sb, M, goal_mode="real", need_attn=False)
        null = model(gb, sb, M, goal_mode="null", need_attn=False)

        mask = real["mask"].bool()
        zr = real["z_hat"]
        zn = null["z_hat"]
        zt = real["z_y"]

        P = objective.projector if objective is not None else None
        if P is not None:
            pr = P(zr)
            pn = P(zn)
            pt = P(zt)
        else:
            pr, pn, pt = zr, zn, zt

        real_raw.append(normalized_cosine_distance(zr[mask], zt[mask]).cpu())
        null_raw.append(normalized_cosine_distance(zn[mask], zt[mask]).cpu())

        real_proj.append(
            normalized_cosine_distance(pr[mask], pt[mask]).cpu()
        )
        null_proj.append(
            normalized_cosine_distance(pn[mask], pt[mask]).cpu()
        )

        shifts.append((zr[mask] - zn[mask]).norm(dim=-1).cpu())
        target_raw.append(zt[mask].cpu())
        pred_raw.append(zr[mask].cpu())

    rr = torch.cat(real_raw).numpy()
    nr = torch.cat(null_raw).numpy()
    rp = torch.cat(real_proj).numpy()
    np_ = torch.cat(null_proj).numpy()
    shift = torch.cat(shifts).numpy()

    zt = torch.cat(target_raw)
    zp = torch.cat(pred_raw)

    raw_improvement = nr - rr
    proj_improvement = np_ - rp

    return {
        "raw_real": float(rr.mean()),
        "raw_null": float(nr.mean()),
        "raw_improvement": float(raw_improvement.mean()),
        "raw_win_rate": float((raw_improvement > 0).mean()),

        "proj_real": float(rp.mean()),
        "proj_null": float(np_.mean()),
        "proj_improvement": float(proj_improvement.mean()),
        "proj_win_rate": float((proj_improvement > 0).mean()),

        "predictor_physics_shift_mean": float(shift.mean()),
        "raw_pred_rank": effective_rank(zp),
        "raw_target_rank": effective_rank(zt),
        "actual_mask_tokens": int(len(rr)),
    }


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def target_pass(a):
    """
    These are screening gates, not claims that a rank of X is intrinsically
    'good'. A model must:
      - not be effectively rank-1,
      - not have near-identical pooled outputs,
      - show positive latent/EM-response association.
    """
    return (
        a["pooled_cos_p05"] < 0.995 and
        a["pooled_rank"] >= 1.25 and
        a["full_token_rank"] >= 4.0 and
        a["latent_spectrum_spearman"] >= 0.05
    )


def aggregate_conditions(rows):
    out = {}
    for ratio in MASK_RATIOS:
        rr = [x for x in rows if x["mask_ratio"] == ratio]
        out[f"{ratio:.2f}"] = {
            "n": len(rr),
            "raw_real": float(np.mean([x["raw_real"] for x in rr])),
            "raw_null": float(np.mean([x["raw_null"] for x in rr])),
            "raw_improvement": float(np.mean(
                [x["raw_improvement"] for x in rr]
            )),
            "raw_win_rate": float(np.mean(
                [x["raw_win_rate"] for x in rr]
            )),
            "proj_real": float(np.mean([x["proj_real"] for x in rr])),
            "proj_null": float(np.mean([x["proj_null"] for x in rr])),
            "proj_improvement": float(np.mean(
                [x["proj_improvement"] for x in rr]
            )),
            "proj_win_rate": float(np.mean(
                [x["proj_win_rate"] for x in rr]
            )),
            "predictor_shift": float(np.mean(
                [x["predictor_physics_shift_mean"] for x in rr]
            )),
            "raw_pred_rank": float(np.mean(
                [x["raw_pred_rank"] for x in rr]
            )),
        }
    return out


def final_decision(target_rows, mask_rows):
    tg = all(target_pass(x) for x in target_rows)

    if not mask_rows:
        return {
            "target_pass": tg,
            "prediction_pass": False,
            "final_pass": False,
            "reason": "target gate failed; prediction stage skipped",
        }

    raw_wins = np.mean([x["raw_improvement"] > 0 for x in mask_rows])
    proj_wins = np.mean([x["proj_improvement"] > 0 for x in mask_rows])
    raw_mean_improvement = np.mean(
        [x["raw_improvement"] for x in mask_rows]
    )
    proj_mean_improvement = np.mean(
        [x["proj_improvement"] for x in mask_rows]
    )

    # We require the real physics-conditioned predictor to beat the null
    # consistently in BOTH raw latent and actual training/projector space.
    pred_pass = (
        raw_wins >= 0.90 and
        proj_wins >= 0.90 and
        raw_mean_improvement > 0 and
        proj_mean_improvement > 0
    )

    return {
        "target_pass": bool(tg),
        "prediction_pass": bool(pred_pass),
        "final_pass": bool(tg and pred_pass),
        "raw_win_rate": float(raw_wins),
        "projected_win_rate": float(proj_wins),
        "raw_mean_improvement": float(raw_mean_improvement),
        "projected_mean_improvement": float(proj_mean_improvement),
        "reason": (
            "PASS: raw and projected real predictor beat null"
            if tg and pred_pass else
            "FAIL: predictor does not consistently beat null in both spaces"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="append", required=True)
    p.add_argument("--config",
                   default=str(REPO_ROOT / "configs/milestone_b.yaml"))
    p.add_argument("--split", default=None)
    p.add_argument("--pools", type=int, default=4)
    p.add_argument("--pool-size", type=int, default=512)
    p.add_argument("--pairs", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--out-dir",
        default=str(
            REPO_ROOT /
            "checkpoints/milestone_b/decisive_representation_validation"
        ),
    )
    args = p.parse_args()

    seed_all()
    device = torch.device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    checkpoints = {}
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        checkpoints[name] = path

    split = args.split or os.path.join(
        REPO_ROOT, cfg["data"]["val_split"]
    )
    ds = MetaDiTDataset(split)
    pool_seeds = POOL_SEEDS[:args.pools]
    pools = make_pools(len(ds), args.pool_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("DECISIVE REPRESENTATION + PREDICTOR AUDIT")
    print("=" * 100)
    print(f"validation split : {split}")
    print(f"unseen samples   : {args.pools * args.pool_size} / model")
    print(f"target pools     : {args.pools} x {args.pool_size}")
    print(f"target pairs     : {args.pairs} / pool")
    print(f"mask ratios      : {MASK_RATIOS}")
    print(f"mask seeds       : {MASK_SEEDS}")
    print()
    print("Prediction metrics:")
    print("  RAW  = 1-cos(z_hat, z_target)")
    print("  PROJ = 1-cos(P(z_hat), P(z_target))  [training objective]")
    print("  IMP  = null_error - real_error  [positive = physics helps]")
    print("=" * 100)

    results = {}

    for name in MODELS:
        if name not in checkpoints:
            continue

        print(f"\n{'#' * 100}\nMODEL: {name}\n{'#' * 100}")
        model, objective = load_model(cfg, checkpoints[name], device, name)

        target_rows = []

        # ---- Target representation stage ----
        for pi, pool_seed in enumerate(pool_seeds):
            G, S = load_pool(
                ds, pools[pool_seed], args.batch_size
            )
            Z = target_embeddings(
                model, G, device, args.batch_size
            )
            a = target_audit(
                Z, S, args.pairs, pool_seed
            )
            passed = target_pass(a)

            target_rows.append(a)

            print(
                f"TARGET pool={pi} "
                f"rank={a['pooled_rank']:.3f} "
                f"full_rank={a['full_token_rank']:.3f} "
                f"cos_p05={a['pooled_cos_p05']:.6f} "
                f"rho={a['latent_spectrum_spearman']:.3f} "
                f"{'PASS' if passed else 'FAIL'}"
            )

            del G, S, Z
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        tg = all(target_pass(x) for x in target_rows)

        print(f"TARGET GATE: {'PASS' if tg else 'FAIL'}")

        mask_rows = []

        if tg:
            # ---- Prediction stage ----
            for pi, pool_seed in enumerate(pool_seeds):
                G, S = load_pool(
                    ds, pools[pool_seed], args.batch_size
                )

                for ratio in MASK_RATIOS:
                    cond = []
                    for mask_seed in MASK_SEEDS:
                        m = evaluate_condition(
                            model, objective, G, S,
                            ratio, mask_seed,
                            cfg, device, args.batch_size
                        )
                        row = {
                            "pool": pi,
                            "pool_seed": pool_seed,
                            "mask_ratio": ratio,
                            "mask_seed": mask_seed,
                            **m,
                        }
                        mask_rows.append(row)
                        cond.append(m)

                    # Print one aggregate line per mask ratio, NOT every seed.
                    print(
                        f"MASK {ratio:>4.0%} "
                        f"raw={np.mean([x['raw_real'] for x in cond]):.5f} "
                        f"raw_null={np.mean([x['raw_null'] for x in cond]):.5f} "
                        f"raw_imp={np.mean([x['raw_improvement'] for x in cond]):+.5f} "
                        f"proj={np.mean([x['proj_real'] for x in cond]):.5f} "
                        f"proj_imp={np.mean([x['proj_improvement'] for x in cond]):+.5f}"
                    )

                del G, S

        decision = final_decision(target_rows, mask_rows)
        by_mask = aggregate_conditions(mask_rows)

        results[name] = {
            "target": target_rows,
            "mask": mask_rows,
            "by_mask": by_mask,
            "decision": decision,
        }

        print(
            f"DECISION {name}: "
            f"{'PASS' if decision['final_pass'] else 'FAIL'} | "
            f"raw_win={decision.get('raw_win_rate', 0):.1%} | "
            f"proj_win={decision.get('projected_win_rate', 0):.1%}"
        )

        del model, objective
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Compact final table ----
    print("\n" + "=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)
    print(
        f"{'MODEL':<18}"
        f"{'TARGET':>10}"
        f"{'RAW IMP':>12}"
        f"{'RAW WIN':>11}"
        f"{'PROJ IMP':>12}"
        f"{'PROJ WIN':>12}"
        f"{'FINAL':>10}"
    )
    print("-" * 100)

    for name, r in results.items():
        d = r["decision"]
        print(
            f"{name:<18}"
            f"{'PASS' if d['target_pass'] else 'FAIL':>10}"
            f"{d.get('raw_mean_improvement', float('nan')):>12.5f}"
            f"{d.get('raw_win_rate', float('nan')):>11.1%}"
            f"{d.get('projected_mean_improvement', float('nan')):>12.5f}"
            f"{d.get('projected_win_rate', float('nan')):>12.1%}"
            f"{'PASS' if d['final_pass'] else 'FAIL':>10}"
        )

    winners = [
        (name, r["decision"]["raw_mean_improvement"],
         r["decision"]["projected_mean_improvement"])
        for name, r in results.items()
        if r["decision"]["final_pass"]
    ]
    winners.sort(key=lambda x: (x[1], x[2]), reverse=True)
    winner = winners[0][0] if winners else None

    decision = {
        "winner": winner,
        "models": {
            name: r["decision"] for name, r in results.items()
        },
        "experiment": {
            "validation_split": split,
            "pools": pool_seeds,
            "pool_size": args.pool_size,
            "pairs_per_pool": args.pairs,
            "mask_ratios": MASK_RATIOS,
            "mask_seeds": MASK_SEEDS,
            "raw_metric": "1-cos(z_hat,z_target)",
            "projected_metric": "1-cos(P(z_hat),P(z_target))",
            "improvement": "null_error-real_error",
        },
    }

    with open(out_dir / "final_report.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    with open(out_dir / "final_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "target_pass", "raw_mean_improvement",
            "raw_win_rate", "projected_mean_improvement",
            "projected_win_rate", "final_pass"
        ])
        for name, r in results.items():
            d = r["decision"]
            w.writerow([
                name,
                d["target_pass"],
                d.get("raw_mean_improvement"),
                d.get("raw_win_rate"),
                d.get("projected_mean_improvement"),
                d.get("projected_win_rate"),
                d["final_pass"],
            ])

    print("-" * 100)
    print(f"WINNER: {winner if winner else 'NONE'}")
    if winner:
        print(
            "Winner passed target-space sanity and the physics-conditioned "
            "predictor beat the null in BOTH raw and projected spaces."
        )
    else:
        print(
            "NO MODEL PASSED. Do not proceed to geometry decoding yet."
        )

    print(f"\nJSON: {out_dir / 'final_report.json'}")
    print(f"DECISION: {out_dir / 'decision.json'}")
    print(f"CSV: {out_dir / 'final_report.csv'}")


if __name__ == "__main__":
    main()