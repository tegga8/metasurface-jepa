"""Shared representation-health diagnostics — single source of truth.

All EMA-target / projection-space / goal-token / spectrum-attention statistics used
by Milestone B monitoring. The stat functions were moved here from
`scripts/diagnostics/check_ema_target_diversity.py` on 2026-08-17 together with the
`same_token_cos` correctness fix (see that module's docstring and the unit tests in
`tests/test_same_token_cos.py`).

Used by:
  - scripts/diagnostics/check_ema_target_diversity.py   (offline CLI, two-anchor verdict)
  - src/train/engine.py                                 (fixed-validation health, HEALTHY/
                                                          WARNING/COLLAPSED per validation)
  - scripts/eval/eval_vicreg_sanity.py                  (five-way collapse classification,
                                                          spec §26)
"""

import math

import torch
import torch.nn.functional as F

COLLAPSED_ANCHOR = {
    "pairwise_cos": 0.999870,
    "pairwise_p05": 0.999601,
    "same_token_cos": 0.999266,
    "eff_rank_unnorm": 13.444902,        # exp(2.5986) — effective rank in #-of-dims units
    "eff_rank_frac": 13.444902 / 384.0,
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
    """Entropy effective rank (exp(H), i.e. Roy–Vetterli effective rank in #-of-dims
    units), its fraction exp(H)/D, participation ratio, top eigenvalue fraction, over
    centered embeddings (B, D).

    NOTE (2026-08-17): previously returned the entropy H itself under the
    "eff_rank_unnorm" name, and H/log(D) under "eff_rank_frac" — both wrong scales.
    """
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
    return {"eff_rank_unnorm": math.exp(ent), "eff_rank_frac": math.exp(ent) / n,
            "participation": part, "top_eig_frac": top}


def pairwise_cos_stats(X_mean):
    """Different-sample pairwise cosine on mean-pooled (B, D) embeddings.

    n < 2 -> all-NaN dict (quantile of the empty pair set would raise; the NaN
    contract keeps this callable from token_space_stats for n_geoms < 2, which
    Bug #21 marks UNAVAILABLE downstream)."""
    if X_mean.shape[0] < 2:
        return {"mean": float("nan"), "median": float("nan"), "p05": float("nan"),
                "p95": float("nan"), "min": float("nan")}
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
    """unbiased=False for n_geoms == 1: unbiased variance of a single observation
    is NaN plus a UserWarning (both useless); n >= 2 keeps the historical
    unbiased=True behavior bit-identical."""
    ub = X.shape[0] >= 2
    return {"token_var": X.var(dim=0, unbiased=ub).mean().item(),
            "token_std": X.var(dim=0, unbiased=ub).sqrt().mean().item(),
            "sample_var": X.var(dim=1, unbiased=ub).mean().item(),
            "sample_std": X.var(dim=1, unbiased=ub).sqrt().mean().item()}


def token_space_stats(X):
    """Full stats for (B, T, D) token embeddings — flat dict, same keys as the CLI's
    encoder_stats rows.

    n_geoms < 2: pairwise/same-token/effective-rank diagnostics are undefined
    (Bug #21) and must not emit values that could be classified — NaN markers
    preserve the dict schema while classify_health's n_geoms guard produces the
    explicit UNAVAILABLE verdict."""
    n = X.shape[0]
    stats = dict(var_stats(X))
    if n < 2:
        stats["pairwise_cos"] = {"mean": float("nan"), "median": float("nan"),
                                 "p05": float("nan"), "p95": float("nan"),
                                 "min": float("nan")}
        stats["same_token_cos"] = float("nan")
        stats["eff_rank_unnorm"] = float("nan")
        stats["eff_rank_frac"] = float("nan")
        stats["participation"] = float("nan")
        stats["top_eig_frac"] = float("nan")
    else:
        stats["pairwise_cos"] = pairwise_cos_stats(X.mean(dim=1))
        stats["same_token_cos"] = same_token_cos(X)
        stats.update(eff_ranks(X.mean(dim=1)))
    stats["n_geoms"] = n
    return stats


def grouped_view(stats):
    """B1 (calibration spec): grouped schema over a flat token_space_stats dict.

    Reorganizes — never recomputes — the existing statistics into the three
    calibration groups:
      mean_pooled : cross-sample structure of geometry-level pooled embeddings
                    (computed on X.mean(dim=1) inside token_space_stats)
      token_level : per-token-position variability + same-token-position cosine
      n_geoms     : sample count the stats were computed over

    The flat dict remains the source of truth consumed by classify_health /
    classify_failure_mode; this view exists so calibration reports can present
    raw-vs-released-vs-random-vs-collapsed comparisons without flattening
    nested pairwise_cos sub-dicts ad hoc.
    """
    return {
        "mean_pooled": {
            "pairwise_cos": stats["pairwise_cos"],
            "eff_rank_unnorm": stats["eff_rank_unnorm"],
            "eff_rank_frac": stats["eff_rank_frac"],
            "participation": stats["participation"],
            "top_eig_frac": stats["top_eig_frac"],
        },
        "token_level": {
            "token_var": stats["token_var"],
            "token_std": stats["token_std"],
            "same_token_cos": stats["same_token_cos"],
        },
        "n_geoms": stats.get("n_geoms"),
    }


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
        per_sample_rank.append(math.exp(-(p * torch.log(p + 1e-12)).sum().item()))
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

    Bug #21: with n_geoms < 2 the pairwise/same-token/effective-rank diagnostics
    are undefined (they emit NaN via empty-pairwise reductions), and NaN silently
    flowing into this classifier would produce a WARNING/HEALTHY verdict with no
    explanation. Any of the four stats dicts with n_geoms < 2 (or 0) instead
    returns an explicit UNAVAILABLE status that can never classify as HEALTHY or
    COLLAPSED.
    """
    c = dict(COLLAPSE_CFG_DEFAULTS, **(cfg or {}))
    for tag, stats in (("raw", raw), ("proj", proj),
                       ("healthy_raw", healthy_raw), ("healthy_proj", healthy_proj)):
        n = stats.get("n_geoms")
        if n is not None and n < 2:
            return ("UNAVAILABLE",
                    {"votes": 0, "signals": {}, "margins": {},
                     "near_healthy": False,
                     "reason": f"{tag} n_geoms={n} < 2: pairwise/same-token/"
                               "effective-rank diagnostics undefined — refusing "
                               "a NaN-based health verdict"})
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


# ---------------------------------------------------------------------------
# five-way collapse classification (architecture-repair spec §26)
# ---------------------------------------------------------------------------

def _raw_collapse_votes(raw, healthy_raw, c):
    """Reference-relative collapse votes in the RAW EMA-target space."""
    signals = {
        "eff_rank_frac": raw["eff_rank_frac"] <= healthy_raw["eff_rank_frac"] / c["eff_rank_frac_div"],
        "p05": raw["pairwise_cos"]["p05"] >= healthy_raw["pairwise_cos"]["p05"] + c["p05_plus"],
        "same_token": raw["same_token_cos"] >= healthy_raw["same_token_cos"] + c["same_token_plus"],
        "std": raw["token_std"] <= healthy_raw["token_std"] / c["std_div"],
    }
    return sum(1 for v in signals.values() if v), signals


def _proj_collapse_votes(proj, healthy_proj, c):
    """Reference-relative collapse votes in the PROJECTED objective space."""
    signals = {
        "proj_p05": proj["pairwise_cos"]["p05"] >= healthy_proj["pairwise_cos"]["p05"] + c["p05_plus"],
        "proj_eff_rank_frac": proj["eff_rank_frac"] <= healthy_proj["eff_rank_frac"] / c["eff_rank_frac_div"],
    }
    return sum(1 for v in signals.values() if v), signals


def classify_failure_mode(raw, proj, healthy_raw, healthy_proj, cfg=None,
                          physics_gap=None, physics_shuffle_delta=None,
                          target_gradient_leak=False, invalid_implementation=None):
    """Five-way collapse classification (spec §26) with priority ordering.

    Inputs:
      raw/proj/healthy_*     token_space_stats dicts (as classify_health).
      physics_gap            null_cos_err - real_cos_err; > 0 means the real goal
                             beats the null goal (the predictor uses A_goal).
      physics_shuffle_delta  |cos_err(real) - cos_err(shuffled-goal)|; > 0 means
                             the predictor depends on WHICH goal, not merely on
                             a goal existing.
      target_gradient_leak   True if gradients reached the EMA target encoder.
      invalid_implementation None or a reason string — if set, the verdict is
                             INVALID_IMPLEMENTATION regardless of the numbers.

    Verdicts (priority order):
      INVALID_IMPLEMENTATION      objective/evaluator wiring broken (missing
                                  objective state, NaN signals, no projector)
      TARGET_GRADIENT_LEAK        gradients reached the frozen EMA target
      RAW_COLLAPSE                the raw EMA-target representation collapsed
      PROJECTOR_COLLAPSE          raw healthy but the projected space collapsed
      PHYSICS_CONDITIONING_FAILURE representation healthy but the predictor does
                                  not use the physics goal (gap ~ 0 and/or
                                  shuffle delta ~ 0)
      HEALTHY                     none of the above

    The physics bars are "essentially zero" gates (no dependency at all), NOT
    meaningful-improvement thresholds — the design doc does not fix those and
    they must be human-confirmed (Standing Rule 3).
    """
    c = dict(COLLAPSE_CFG_DEFAULTS, **(cfg or {}))

    if invalid_implementation:
        return {"verdict": "INVALID_IMPLEMENTATION",
                "reason": str(invalid_implementation),
                "raw_collapsed": False, "proj_collapsed": False,
                "physics_conditioning_failure": False,
                "target_gradient_leak": False,
                "raw_votes": 0, "proj_votes": 0,
                "physics_gap": physics_gap,
                "physics_shuffle_delta": physics_shuffle_delta}

    if target_gradient_leak:
        return {"verdict": "TARGET_GRADIENT_LEAK",
                "reason": "gradients reached the frozen EMA target encoder",
                "raw_collapsed": False, "proj_collapsed": False,
                "physics_conditioning_failure": False,
                "target_gradient_leak": True,
                "raw_votes": 0, "proj_votes": 0,
                "physics_gap": physics_gap,
                "physics_shuffle_delta": physics_shuffle_delta}

    raw_votes, raw_sig = _raw_collapse_votes(raw, healthy_raw, c)
    proj_votes, proj_sig = _proj_collapse_votes(proj, healthy_proj, c)
    raw_collapsed = raw_votes >= c.get("collapse_votes", 3)
    proj_collapsed = (not raw_collapsed) and proj_votes >= c.get("collapse_votes", 3)

    def _base(verdict, reason, flags):
        return {"verdict": verdict, "reason": reason,
                "raw_collapsed": bool(flags["raw_collapsed"]),
                "proj_collapsed": bool(flags["proj_collapsed"]),
                "physics_conditioning_failure": bool(
                    flags["physics_conditioning_failure"]),
                "target_gradient_leak": False,
                "raw_votes": raw_votes, "proj_votes": proj_votes,
                "raw_signals": raw_sig, "proj_signals": proj_sig,
                "physics_gap": physics_gap,
                "physics_shuffle_delta": physics_shuffle_delta}

    if raw_collapsed:
        return _base("RAW_COLLAPSE",
                     f"raw EMA-target space collapsed ({raw_votes} votes)",
                     {"raw_collapsed": True, "proj_collapsed": False,
                      "physics_conditioning_failure": False})
    if proj_collapsed:
        return _base("PROJECTOR_COLLAPSE",
                     f"raw healthy but projected space collapsed "
                     f"({proj_votes} proj votes)",
                     {"raw_collapsed": False, "proj_collapsed": True,
                      "physics_conditioning_failure": False})

    eps = 1e-6
    no_gap = physics_gap is not None and physics_gap <= eps
    no_shuffle = (physics_shuffle_delta is not None
                  and physics_shuffle_delta <= eps)
    if no_gap or no_shuffle:
        return _base("PHYSICS_CONDITIONING_FAILURE",
                     "representation healthy but the predictor does not use the "
                     "physics goal "
                     f"(physics_gap={physics_gap}, "
                     f"physics_shuffle_delta={physics_shuffle_delta})",
                     {"raw_collapsed": False, "proj_collapsed": False,
                      "physics_conditioning_failure": True})

    return _base("HEALTHY",
                 "raw and projected representations healthy and the predictor "
                 "uses the physics goal",
                 {"raw_collapsed": False, "proj_collapsed": False,
                  "physics_conditioning_failure": False})


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
