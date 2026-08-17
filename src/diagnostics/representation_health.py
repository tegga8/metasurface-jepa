"""Shared representation-health diagnostics — single source of truth.

All EMA-target / projection-space / goal-token / spectrum-attention statistics used
by Milestone B monitoring. The stat functions were moved here from
`scripts/diagnostics/check_ema_target_diversity.py` on 2026-08-17 together with the
`same_token_cos` correctness fix (see that module's docstring and the unit tests in
`tests/test_same_token_cos.py`).

Used by:
  - scripts/diagnostics/check_ema_target_diversity.py   (offline CLI, two-anchor verdict)
  - src/train/engine.py                                 (adaptive validation, HEALTHY/
                                                         WARNING/COLLAPSED per validation)
"""

import math

import torch
import torch.nn.functional as F

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

COLLAPSE_CFG_DEFAULTS = {
    "eff_rank_frac_div": 3.0,          # raw/proj eff_rank_frac <= healthy / div -> vote
    "p05_plus": 0.005,                 # raw/proj p05 >= healthy p05 + plus -> vote
    "same_token_plus": 0.005,          # same-token cos >= healthy + plus -> vote
    "std_div": 3.0,                    # token_std <= healthy token_std / div -> vote
    "collapse_votes": 3,               # votes >= this -> COLLAPSED
    "near_rank": 0.05, "near_p05": 0.02, "near_same": 0.05,   # HEALTHY tolerances
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
    idx = torch.triu_indices(Xn.shape[0], Xn.shape[0], offset=1, device=Xn.device)
    v = G[idx[0], idx[1]]
    return {"mean": v.mean().item(), "median": v.median().item(),
            "p05": v.quantile(0.05).item(), "p95": v.quantile(0.95).item(),
            "min": v.min().item()}


def same_token_cos(X):
    """Per-token-position cross-sample pairwise cosine, averaged (B, T, D).

    G[t, i, j] = cos(X[i, t], X[j, t]) — different SAMPLES i, j at the SAME spatial
    token t.

    NOTE (diagnostic correctness fix, 2026-08-17): the original implementation used
    `torch.einsum("btd,bsd->tbs", Xn, Xn)`. Because `b` is a shared free label in
    both operands, einsum aligns the batch axes and emits a single `b` axis — the
    second operand's `s` labels its dim 1, the TOKEN dim. The output was therefore
    (T, B, T), not (T, B, B): the triu batch indices then indexed the token dim.
    With B > T (e.g. --max-geoms 512) this raised
    `IndexError: index 256 is out of bounds for dimension 1 with size 256`; with
    B <= T it ran silently while computing within-sample token-pair cosines (wrong
    semantics). Replaced with an explicit batched matmul over the two batch axes,
    producing (T, B, B); the indices are taken from the actual batch size.
    """
    assert X.ndim == 3, f"expected (B, T, D), got {tuple(X.shape)}"
    b = X.shape[0]
    if b < 2:
        return float("nan")
    Xn = F.normalize(X, dim=-1)
    G = torch.bmm(Xn.transpose(0, 1), Xn.transpose(0, 1).transpose(1, 2))  # (T, B, B)
    idx = torch.triu_indices(b, b, offset=1, device=Xn.device)
    return G[:, idx[0], idx[1]].mean().item()


def var_stats(X):
    return {"token_var": X.var(dim=0).mean().item(),
            "token_std": X.var(dim=0).sqrt().mean().item(),
            "sample_var": X.var(dim=1).mean().item(),
            "sample_std": X.var(dim=1).sqrt().mean().item()}


def token_space_stats(X):
    """Full stats for (B, T, D) token embeddings — flat dict, same keys as the CLI's
    encoder_stats rows."""
    stats = dict(var_stats(X))
    stats["pairwise_cos"] = pairwise_cos_stats(X.mean(dim=1))
    stats["same_token_cos"] = same_token_cos(X)
    stats.update(eff_ranks(X.mean(dim=1)))
    stats["n_geoms"] = X.shape[0]
    return stats


def encoder_stats(encoder, geoms, device, max_geoms):
    """Run an encoder over geometry batches and collect token-space stats."""
    with torch.no_grad():
        embs = []
        for G in geoms:
            G = G.to(device)
            x = encoder(G)
            embs.append(x.cpu())
            if sum(e.shape[0] for e in embs) >= max_geoms:
                break
        X = torch.cat(embs, dim=0)[:max_geoms]
    return token_space_stats(X)


def goal_token_stats(a_goal):
    """a_goal: (B, 16, D) goal-token representations -> within-sample diversity stats."""
    b, g, d = a_goal.shape
    an = F.normalize(a_goal, dim=-1)
    sim = torch.einsum("bid,bjd->bij", an, an)          # (B, 16, 16)
    off = sim - torch.eye(g, device=sim.device).unsqueeze(0)
    idx = torch.triu_indices(g, g, offset=1, device=sim.device)
    vals = off[:, idx[0], idx[1]]                        # (B, pairs)
    per_sample_rank = []
    for i in range(b):
        ac = a_goal[i] - a_goal[i].mean(0, keepdim=True)  # (16, D)
        s = torch.linalg.svdvals(ac)
        e = (s ** 2).clamp(min=0.0)
        if e.sum() <= 0:
            per_sample_rank.append(0.0)
            continue
        p = e / e.sum()
        per_sample_rank.append(-(p * torch.log(p + 1e-12)).sum().item())
    return {
        "goal_token_pairwise_cosine_mean": vals.mean().item(),
        "goal_token_pairwise_cosine_min": vals.min().item(),
        "goal_token_pairwise_cosine_max": vals.max().item(),
        "goal_token_effective_rank": (sum(per_sample_rank) / max(1, b)
                                      if per_sample_rank else 0.0),
    }


def goal_attention_stats(w):
    """Spectrum-path goal attention weights (B, H, 16, 301) -> diversity stats.

    p (per sample, mean over heads) = (16, 301); entropy per goal token over the
    301 spectrum locations (nats); peak mass; pairwise overlap = sum min(p_i, p_j).
    """
    b, h, g, k = w.shape
    p = w.mean(dim=1)                                    # (B, 16, 301)
    logp = torch.log(p.clamp_min(1e-9))
    h_i = -(p * logp).sum(dim=-1)                        # (B, 16) entropy per token
    ent_mean = h_i.mean().item()
    ent_std = h_i.std(dim=-1).mean().item()
    peak = p.max(dim=-1).values.mean().item()
    overlaps = []
    idx = torch.triu_indices(g, g, offset=1, device=p.device)
    for i in range(b):
        pi, pj = p[i][idx[0]], p[i][idx[1]]
        overlaps.append(torch.minimum(pi, pj).sum(dim=-1).mean().item())
    return {
        "goal_attention_entropy_mean": ent_mean,
        "goal_attention_entropy_std": ent_std,
        "goal_attention_peak_mass": peak,
        "goal_attention_overlap_mean": (sum(overlaps) / max(1, b) if overlaps else 0.0),
    }


def classify_health(raw, proj, healthy_raw, healthy_proj, cfg=None):
    """Reference-relative HEALTHY / WARNING / COLLAPSED classification.

    raw: token_space_stats of the raw EMA target; proj: token_space_stats of the
    projected EMA target (B2 makes the JEPA loss live in projection space, so the
    projected space is mandatory); healthy_*: the same stats computed on a fresh
    released-init build over the SAME fixed validation set. Collapse is judged as
    "substantially more concentrated than the healthy reference" across multiple
    signals (never one arbitrary threshold), with the historical collapsed anchor
    as context only.

    Returns (status, signals) where signals maps each collapse-consistent signal to
    its boolean value and the numeric margin.
    """
    c = dict(COLLAPSE_CFG_DEFAULTS, **(cfg or {}))
    near_healthy = (
        abs(raw["eff_rank_frac"] - healthy_raw["eff_rank_frac"]) <= c["near_rank"]
        and abs(raw["pairwise_cos"]["p05"] - healthy_raw["pairwise_cos"]["p05"]) <= c["near_p05"]
        and abs(raw["same_token_cos"] - healthy_raw["same_token_cos"]) <= c["near_same"]
        and abs(proj["pairwise_cos"]["p05"] - healthy_proj["pairwise_cos"]["p05"]) <= c["near_p05"])
    signals = {
        "eff_rank_frac": raw["eff_rank_frac"] <= healthy_raw["eff_rank_frac"] / c["eff_rank_frac_div"],
        "p05": raw["pairwise_cos"]["p05"] >= healthy_raw["pairwise_cos"]["p05"] + c["p05_plus"],
        "same_token": raw["same_token_cos"] >= healthy_raw["same_token_cos"] + c["same_token_plus"],
        "std": raw["token_std"] <= healthy_raw["token_std"] / c["std_div"],
        "proj_p05": proj["pairwise_cos"]["p05"] >= healthy_proj["pairwise_cos"]["p05"] + c["p05_plus"],
        "proj_eff_rank_frac": proj["eff_rank_frac"] <= healthy_proj["eff_rank_frac"] / c["eff_rank_frac_div"],
    }
    votes = sum(1 for v in signals.values() if v)
    if votes >= c["collapse_votes"]:
        status = "COLLAPSED"
    elif near_healthy and votes == 0:
        status = "HEALTHY"
    else:
        status = "WARNING"
    margins = {
        "eff_rank_frac_ratio": raw["eff_rank_frac"] / max(healthy_raw["eff_rank_frac"], 1e-9),
        "p05_margin": raw["pairwise_cos"]["p05"] - healthy_raw["pairwise_cos"]["p05"],
        "same_token_margin": raw["same_token_cos"] - healthy_raw["same_token_cos"],
        "std_ratio": raw["token_std"] / max(healthy_raw["token_std"], 1e-9),
        "proj_eff_rank_frac_ratio": proj["eff_rank_frac"] / max(healthy_proj["eff_rank_frac"], 1e-9),
        "proj_p05_margin": proj["pairwise_cos"]["p05"] - healthy_proj["pairwise_cos"]["p05"],
    }
    return status, {"votes": votes, "signals": signals, "margins": margins,
                    "near_healthy": bool(near_healthy)}


def verdict(target, collapsed, refs):
    """Reference-relative CLI verdict (kept for the offline diagnostic): the target is
    non-degenerate when it behaves like the healthy released-ViT reference AND is far
    from the collapsed anchor on the sharp discriminators (effective-rank fraction,
    p05 cosine tail, same-token cosine)."""
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
