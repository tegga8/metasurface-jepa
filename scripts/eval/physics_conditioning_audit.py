"""Physics-conditioning audit (architecture-repair spec §10 / §28).

Measures whether the physics condition inputs actually carry signal and whether
the predictor actually depends on them, for one checkpoint + objective:

  - c_physics  (global FiLM condition, (B, 384))  — mean feature std, pairwise
    cosine p05/p50/p95, entropy effective rank and rank fraction
  - a_goal     (16 goal tokens, (B, 16, 384))     — same stats on the mean-pooled
    per-sample vector AND on the 16-token stack
  - predictor sensitivity — mean |z_hat(real) - z_hat(null)| and
    |z_hat(real) - z_hat(shuffled-goal)| on masked tokens, plus real/null/shuffled
    cos_err in raw and projected space

Interpretation (Cases A/B/C, documented for human confirmation — the spec names
the case structure but does not fix numeric bars):

  Case A EMBEDDING_COLLAPSE : c_physics or a_goal is (near) rank-1 / zero-variance
                              across samples -> the physics input itself carries no
                              cross-sample signal; conditioning cannot work.
  Case B PREDICTOR_DEAD     : embeddings carry signal (Case A false) but the
                              predictor output is invariant to the goal
                              (real-vs-null or real-vs-shuffled delta ~ 0) ->
                              the predictor ignores physics (Failure Mode 2, §13).
  Case C HEALTHY            : embeddings carry signal AND the predictor responds
                              to the goal identity.

Run (local dev CPU ok for small subsets; full subsets on cloud GPU):

    python scripts/eval/physics_conditioning_audit.py \
        --config configs/milestone_b.yaml \
        --checkpoint checkpoints/milestone_b/sweep_jepa_vicreg_latest.pt
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from assembly import build_model
from data.dataset import MetaDiTDataset, collate_batch
from data.mask import BlockMasker
from losses.objectives import build_objective
from train.engine import load_checkpoint

PIXEL_GRID = 16


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rows(X):
    return X.reshape(-1, X.shape[-1]) if X.ndim == 3 else X


def eff_rank_stats(X):
    """Entropy effective rank and rank fraction of the centered (N, D) matrix."""
    Xr = _rows(X).double()
    Xc = Xr - Xr.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    e = (s ** 2).clamp(min=0.0)
    total = e.sum().item()
    if total <= 0.0:
        return {"eff_rank": 0.0, "rank_fraction": 0.0}
    p = e / e.sum()
    ent = -(p * torch.log(p + 1e-12)).sum().item()
    r = math.exp(ent)
    return {"eff_rank": r, "rank_fraction": r / Xr.shape[1]}


def feats_stats(X):
    """Per-feature std stats + pairwise cosine quantiles on (N, D) rows."""
    Xr = _rows(X)
    std = Xr.double().std(dim=0, unbiased=Xr.shape[0] >= 2)
    out = {
        "mean_feature_std": std.mean().item(),
        "min_feature_std": std.min().item(),
        "median_feature_std": std.median().item(),
        "frac_std_lt_0p1": (std < 0.1).double().mean().item(),
    }
    if Xr.shape[0] >= 2:
        Xn = F.normalize(Xr, dim=-1)
        G = Xn @ Xn.T
        idx = torch.triu_indices(Xn.shape[0], Xn.shape[0], offset=1, device=Xn.device)
        v = G[idx[0], idx[1]]
        out["pairwise_cos_p05"] = v.quantile(0.05).item()
        out["pairwise_cos_p50"] = v.median().item()
        out["pairwise_cos_p95"] = v.quantile(0.95).item()
    else:
        out.update({"pairwise_cos_p05": float("nan"),
                    "pairwise_cos_p50": float("nan"),
                    "pairwise_cos_p95": float("nan")})
    out.update(eff_rank_stats(Xr))
    return out


def cosine_err(a, b, mask):
    d = (1.0 - F.cosine_similarity(
        F.normalize(a, dim=-1), F.normalize(b, dim=-1), dim=-1)).clamp(min=0)
    return d[mask].mean().item()


def case_verdict(emb_c, emb_g, deltas, eps=1e-4):
    """Cases A/B/C. Embeds collapsed if rank fraction < 0.02 or
    mean feature std < 0.05. Predictor dead if both real-vs-null and
    real-vs-shuffled masked-token deltas < eps."""
    emb_collapsed = (
        (emb_c["rank_fraction"] < 0.02 or emb_c["mean_feature_std"] < 0.05)
        or (emb_g["rank_fraction"] < 0.02 or emb_g["mean_feature_std"] < 0.05))
    dead = (deltas["real_vs_null"] < eps and deltas["real_vs_shuffled"] < eps)
    if emb_collapsed:
        return "CASE_A_EMBEDDING_COLLAPSE"
    if dead:
        return "CASE_B_PREDICTOR_DEAD"
    return "CASE_C_HEALTHY"


def main():
    p = argparse.ArgumentParser(description="Physics-conditioning audit (spec §10)")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--objective", default=None)
    p.add_argument("--subset", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--mask-seed", type=int, default=12345)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default="checkpoints/milestone_b/physics_audit")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    name = args.objective or cfg.get("objective", "jepa_vicreg")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)

    model = build_model(cfg["model"],
                        os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
                        device=device,
                        init_from_metadit=cfg["model"].get("init_from_metadit", True),
                        metadit_weights=os.path.join(REPO_ROOT, cfg["weights"]["metadit"]))
    objective = build_objective(name, cfg.get("objective_params", {}).get(name, {}),
                                projector_input_dim=cfg["model"].get("hidden", 384))
    load_checkpoint(args.checkpoint, model, objective, None, None, device)
    model.eval()
    objective.eval()
    P = objective.projector

    val_ds = MetaDiTDataset(os.path.join(REPO_ROOT, cfg["data"]["val_split"]))
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_batch)
    masker = BlockMasker(placement="random", grid=PIXEL_GRID, min_side=3,
                         k_range=(1, 4), seed=args.mask_seed)

    c_phys_all, a_goal_all = [], []
    deltas = {"real_vs_null": [], "real_vs_shuffled": []}
    phys_err = {k: [] for k in ("real_raw", "null_raw", "shuf_raw",
                                "real_proj", "null_proj", "shuf_proj")}
    n_done = 0
    with torch.no_grad():
        for G, S in loader:
            if n_done >= args.subset:
                break
            G, S = G.to(device), S.to(device)
            n_done += G.shape[0]
            M = masker.sample(G, args.mask_ratio).to(device)
            perm = torch.randperm(G.shape[0], generator=torch.Generator(
                device=device).manual_seed(args.mask_seed))
            S_shuf = S[perm]

            c_real, a_real = model.spectrum_path(S, "real")
            _, a_null = model.spectrum_path(S, "null")
            out_r = model(G, S, M, goal_mode="real")
            out_n = model(G, S, M, goal_mode="null")
            out_s = model(G, S_shuf, M, goal_mode="real")
            mask = out_r["mask"]

            c_phys_all.append(c_real.cpu())
            a_goal_all.append(a_real.mean(1).cpu())          # (B, 384) pooled
            mw = mask.float()
            deltas["real_vs_null"].append(
                ((out_r["z_hat"] - out_n["z_hat"]).norm(dim=-1)[mask].mean().item()))
            deltas["real_vs_shuffled"].append(
                ((out_r["z_hat"] - out_s["z_hat"]).norm(dim=-1)[mask].mean().item()))

            zy = out_r["z_y"]
            p_zy, p_nz, p_sz = P(zy), P(out_n["z_hat"]), P(out_s["z_hat"])
            phys_err["real_raw"].append(cosine_err(out_r["z_hat"], zy, mask))
            phys_err["null_raw"].append(cosine_err(out_n["z_hat"], zy, mask))
            phys_err["shuf_raw"].append(cosine_err(out_s["z_hat"], zy, mask))
            phys_err["real_proj"].append(cosine_err(P(out_r["z_hat"]), p_zy, mask))
            phys_err["null_proj"].append(cosine_err(P(out_n["z_hat"]), p_nz, mask))
            phys_err["shuf_proj"].append(cosine_err(P(out_s["z_hat"]), p_sz, mask))

    c_phys = torch.cat(c_phys_all, dim=0)
    a_goal = torch.cat(a_goal_all, dim=0)
    stats_c = feats_stats(c_phys)
    stats_g = feats_stats(a_goal)
    avg_deltas = {k: float(np.mean(v)) for k, v in deltas.items()}
    errs = {k: float(np.mean(v)) for k, v in phys_err.items()}
    verdict = case_verdict(stats_c, stats_g, avg_deltas)

    report = {
        "objective": name,
        "checkpoint": args.checkpoint,
        "c_physics": stats_c,
        "a_goal": stats_g,
        "predictor_deltas": avg_deltas,
        "physics_cos_err": errs,
        "physics_raw_real_vs_null_improvement": errs["null_raw"] - errs["real_raw"],
        "physics_raw_real_vs_shuffled_improvement": errs["shuf_raw"] - errs["real_raw"],
        "case": verdict,
        "n_samples": n_done,
        "mask_ratio": args.mask_ratio,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"physics_audit_{name}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps(report, indent=2))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()