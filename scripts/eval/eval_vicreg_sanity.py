"""Shared Milestone-B offline evaluation — collapse diagnostics, physics controls,
and the short end-to-end training audit, for ALL three registered objectives
(jepa_vicreg, jepa_barlow, lejepa; architecture-repair spec §15/§25/§27/§28).

One objective per run (`--objective`), one interface `objective(model, G, S, M)`
(§24), projection always through the OBJECTIVE's projector — never a model
attribute (§17).

Two modes (standalone CLI, runnable on local dev CPU or cloud GPU):

1. Checkpoint validation (default):
       python scripts/eval/eval_vicreg_sanity.py \
           --checkpoint checkpoints/milestone_b/sweep_jepa_vicreg_latest.pt \
           --config configs/milestone_b.yaml
   Loads the model AND the objective from the §30 checkpoint (a checkpoint
   missing objective_state fails loudly, spec §12), then reports, on identical
   validation samples and masks:
     - raw target / raw predictor / projected target / projected predictor
       collapse diagnostics (effective rank, rank fraction, pairwise cosine,
       mean feature std, top-eigenvalue fraction, per-feature std stats),
     - geometry-level (masked-token mean-pooled per geometry) statistics,
     - projector input/output audit (singular values, condition number, ...),
     - the five-way collapse classification (spec §26: RAW_COLLAPSE /
       PROJECTOR_COLLAPSE / PHYSICS_CONDITIONING_FAILURE /
       TARGET_GRADIENT_LEAK / INVALID_IMPLEMENTATION / HEALTHY),
     - raw-vs-projected prediction (spec §27),
     - real / null / shuffled physics controls in raw and projected space (§28).

2. Short end-to-end audit (spec §15, before any long training):
       python scripts/eval/eval_vicreg_sanity.py --short-audit \
           --steps 200 --report-every 25 --subset 32 --batch-size 8 \
           --config configs/milestone_b.yaml
    100-300 optimizer steps on a fixed small subset. Every `--report-every`
    steps it reports loss components, raw/projected rank + pairwise cosine,
    feature-std stats, projector singular values, and per-term gradient norms
    (per-objective `term_names`). Aborts immediately on NaN/Inf loss or
    gradients, EMA-target gradients, raw/projected collapse, or persistent
    extreme term domination.

Run a tiny local crash test with `--smoke` (6 steps, 8 samples, CPU).
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
from diagnostics.representation_health import (
    classify_failure_mode, eff_ranks, pairwise_cos_stats,
)
from losses.objectives import build_objective
from train.engine import load_checkpoint

PIXEL_GRID = 16
EPS_SV = 1e-12


# ---------------------------------------------------------------------------
# math / stats helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rows(X):
    """Flatten (B, T, D) -> (N, D) or keep (N, D)."""
    return X.reshape(-1, X.shape[-1]) if X.ndim == 3 else X


def eff_rank(X):
    """Entropy effective rank (exp of entropy of normalized eigenvalues), and
    its fraction r_eff / D. Uses ONLY entropy effective rank (spec §14)."""
    Xr = _rows(X)
    Xc = Xr - Xr.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    e = (s ** 2).clamp(min=0.0)
    total = e.sum().item()
    if total <= 0.0:
        return 0.0, 0.0
    p = e / e.sum()
    ent = -(p * torch.log(p + 1e-12)).sum().item()
    r_eff = math.exp(ent)
    return r_eff, r_eff / Xr.shape[1]


def sv_stats(X):
    """Singular-value statistics of the centered (N, D) matrix (spec §16):
    effective rank, min/max/mean/median singular value, condition number."""
    Xr = _rows(X)
    Xc = Xr - Xr.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    r_eff, _ = eff_rank(Xc)
    return {
        "eff_rank": r_eff,
        "sv_min": s.min().item(),
        "sv_max": s.max().item(),
        "sv_mean": s.mean().item(),
        "sv_median": s.median().item(),
        "condition_number": (s.max() / s.min().clamp_min(EPS_SV)).item(),
    }


def feature_std_stats(X):
    """Per-feature std over samples (spec §13): min/median/mean and the
    fractions of dimensions with std < 0.5 and std < 0.1 (dimensional-collapse
    flags, §15)."""
    Xr = _rows(X).double()
    std = Xr.std(dim=0, unbiased=Xr.shape[0] >= 2)      # (D,)
    return {
        "min_std": std.min().item(),
        "median_std": std.median().item(),
        "mean_std": std.mean().item(),
        "frac_std_lt_0p5": (std < 0.5).double().mean().item(),
        "frac_std_lt_0p1": (std < 0.1).double().mean().item(),
    }


def space_diagnostics(X_tokens, X_pooled=None, tag="space"):
    """Full diagnostics for one space. X_tokens: (B, T, D) token embeddings;
    X_pooled: (B, D) mean-pooled per-geometry vectors (cross-geometry stats
    come from THIS — the geometry-level health axis, spec §6)."""
    if X_pooled is None:
        if X_tokens.ndim == 3:
            X_pooled = X_tokens.mean(1)
        else:
            X_pooled = X_tokens
    if X_pooled.shape[0] < 2:
        pcos = {"mean": float("nan"), "median": float("nan"),
                "p05": float("nan"), "p95": float("nan")}
    else:
        pcos = pairwise_cos_stats(X_pooled)
    r_eff, r_frac = eff_rank(X_pooled)
    er = eff_ranks(X_pooled)
    return {
        f"{tag}_eff_rank": r_eff,
        f"{tag}_rank_fraction": r_frac,
        f"{tag}_top_eig_frac": er["top_eig_frac"],
        f"{tag}_participation": er["participation"],
        f"{tag}_pairwise_cos": pcos,
        **{f"{tag}_{k}": v for k, v in feature_std_stats(X_tokens).items()},
        f"{tag}_n_geoms": int(X_pooled.shape[0]),
    }


def cosine_err(a, b, mask):
    """1 - cos(a[mask], b[mask]) mean (raw-space unless a/b are projected)."""
    d = (1.0 - F.cosine_similarity(
        F.normalize(a, dim=-1), F.normalize(b, dim=-1), dim=-1)).clamp(min=0)
    return d[mask].mean().item()


# ---------------------------------------------------------------------------
# model / objective loading
# ---------------------------------------------------------------------------

def load_model_and_objective(cfg, checkpoint, device, name):
    """Build model + objective; restore BOTH from the §30 checkpoint (missing
    objective_state fails loudly inside load_checkpoint, spec §12)."""
    model = build_model(
        cfg["model"],
        os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
        device=device,
        init_from_metadit=cfg["model"].get("init_from_metadit", True),
        metadit_weights=os.path.join(REPO_ROOT, cfg["weights"]["metadit"]),
    )
    objective = build_objective(
        name, cfg.get("objective_params", {}).get(name, {}),
        projector_input_dim=cfg["model"].get("hidden", 384))
    ckpt = load_checkpoint(checkpoint, model, objective, None, None, device)
    model.eval()
    model.ema.eval()
    objective.eval()
    return model, objective, ckpt


def _load_fixed_validation(cfg, device, n_samples, batch_size, ratio=0.5,
                           mask_seed=12345):
    val_ds = MetaDiTDataset(os.path.join(REPO_ROOT, cfg["data"]["val_split"]))
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_batch)
    batches, n = [], min(n_samples, len(val_ds))
    total = 0
    for G, S in loader:
        want = n - total
        if want <= 0:
            break
        batches.append((G[:want].to(device), S[:want].to(device)))
        total += G.shape[0]
    masker = BlockMasker(placement="random", grid=PIXEL_GRID, min_side=3,
                         k_range=(1, 4), seed=mask_seed)
    masks = [masker.sample(G, ratio).to(device) for G, _ in batches]
    return batches, masks


# ---------------------------------------------------------------------------
# checkpoint validation
# ---------------------------------------------------------------------------

def validate_checkpoint(cfg, args):
    device = torch.device(args.device)
    set_seed(args.seed)
    model, objective, ckpt = load_model_and_objective(
        cfg, args.checkpoint, device, args.objective)
    batches, masks = _load_fixed_validation(
        cfg, device, args.subset, args.batch_size, ratio=args.mask_ratio,
        mask_seed=args.mask_seed)
    P = objective.projector

    agg = {
        "raw_zy": [], "raw_zh": [],
        "proj_zy": [], "proj_zh": [],
        "raw_zy_pooled": [], "raw_zh_pooled": [],
        "proj_zy_pooled": [], "proj_zh_pooled": [],
        "proj_in": [], "proj_out": [],
        "proj_in_pooled": [], "proj_out_pooled": [],
    }
    phys = {k: [] for k in ("real_raw", "real_proj", "null_raw", "null_proj",
                            "shuf_raw", "shuf_proj")}
    pred = {"raw_cos": [], "proj_cos": []}
    with torch.no_grad():
        for (G, S), M in zip(batches, masks):
            out = model(G, S, M, goal_mode="real")
            out_n = model(G, S, M, goal_mode="null")
            perm = torch.randperm(G.shape[0], generator=torch.Generator(
                device=device).manual_seed(args.mask_seed))
            out_s = model(G, S[perm], M, goal_mode="real")

            mask = out["mask"]
            mw = mask.float()
zy, zh = out["z_y_raw"], out["z_hat"]
            p_zy, p_zh = P(zy), P(zh)

            def pool(x):
                return (x * mw.unsqueeze(-1)).sum(1) \
                    / mw.sum(1, keepdim=True).clamp(min=1)

            agg["raw_zy"].append(zy.cpu())
            agg["raw_zh"].append(zh.cpu())
            agg["proj_zy"].append(p_zy.cpu())
            agg["proj_zh"].append(p_zh.cpu())
            agg["raw_zy_pooled"].append(pool(zy).cpu())
            agg["raw_zh_pooled"].append(pool(zh).cpu())
            agg["proj_zy_pooled"].append(pool(p_zy).cpu())
            agg["proj_zh_pooled"].append(pool(p_zh).cpu())
            agg["proj_in"].append(torch.cat([_rows(zy), _rows(zh)], 0).cpu())
            agg["proj_out"].append(torch.cat([_rows(p_zy), _rows(p_zh)], 0).cpu())
            agg["proj_in_pooled"].append(torch.cat([pool(zy), pool(zh)], 0).cpu())
            agg["proj_out_pooled"].append(
                torch.cat([pool(p_zy), pool(p_zh)], 0).cpu())

            # physics controls (raw and projected)
            p_n_zy = P(out_n["z_y_raw"])
            p_s_zy = P(out_s["z_y_raw"])
            phys["real_raw"].append(cosine_err(zh, zy, mask))
            phys["null_raw"].append(cosine_err(out_n["z_hat"], zy, mask))
            phys["shuf_raw"].append(cosine_err(out_s["z_hat"], zy, mask))
            phys["real_proj"].append(cosine_err(p_zh, p_zy, mask))
            phys["null_proj"].append(cosine_err(P(out_n["z_hat"]), p_n_zy, mask))
            phys["shuf_proj"].append(cosine_err(P(out_s["z_hat"]), p_s_zy, mask))

            # raw-vs-projected prediction on identical samples/masks (§27)
            m = mask
            pred["raw_cos"].append(F.cosine_similarity(
                F.normalize(zh, dim=-1), F.normalize(zy, dim=-1), dim=-1
            )[m].mean().item())
            pred["proj_cos"].append(F.cosine_similarity(
                F.normalize(p_zh, dim=-1), F.normalize(p_zy, dim=-1), dim=-1
            )[m].mean().item())

    cat = lambda k: torch.cat(agg[k], dim=0)
    zy, zh = cat("raw_zy"), cat("raw_zh")
    p_zy, p_zh = cat("proj_zy"), cat("proj_zh")
    zy_p, zh_p = cat("raw_zy_pooled"), cat("raw_zh_pooled")
    p_zy_p, p_zh_p = cat("proj_zy_pooled"), cat("proj_zh_pooled")

    diag = {}
    diag.update(space_diagnostics(zy, zy_p, tag="raw_target"))
    diag.update(space_diagnostics(zh, zh_p, tag="raw_predictor"))
    diag.update(space_diagnostics(p_zy, p_zy_p, tag="proj_target"))
    diag.update(space_diagnostics(p_zh, p_zh_p, tag="proj_predictor"))
    diag["projector_input_audit"] = sv_stats(cat("proj_in"))
    diag["projector_input_audit"]["mean_feature_std"] = \
        feature_std_stats(cat("proj_in"))["mean_std"]
    diag["projector_input_audit"]["cross_sample_cosine"] = \
        pairwise_cos_stats(cat("proj_in_pooled")) if \
        cat("proj_in_pooled").shape[0] >= 2 else float("nan")
    diag["projector_output_audit"] = sv_stats(cat("proj_out"))
    diag["projector_output_audit"]["mean_feature_std"] = \
        feature_std_stats(cat("proj_out"))["mean_std"]
    diag["projector_output_audit"]["cross_sample_cosine"] = \
        pairwise_cos_stats(cat("proj_out_pooled")) if \
        cat("proj_out_pooled").shape[0] >= 2 else float("nan")

    diag["collapse_gates"] = classify_collapse(diag)

    physics = {k: float(np.mean(v)) for k, v in phys.items()}
    physics["raw_real_vs_null_improvement"] = \
        physics["null_raw"] - physics["real_raw"]
    physics["raw_real_vs_shuffled_improvement"] = \
        physics["shuf_raw"] - physics["real_raw"]
    physics["proj_real_vs_null_improvement"] = \
        physics["null_proj"] - physics["real_proj"]
    physics["proj_real_vs_shuffled_improvement"] = \
        physics["shuf_proj"] - physics["real_proj"]

    pred_metrics = {
        "raw_cosine_zhat_zy": float(np.mean(pred["raw_cos"])),
        "projected_cosine_pzhat_pzy": float(np.mean(pred["proj_cos"])),
    }

    # five-way classification (§26): physics gates use the RAW-space controls so
    # the verdict is independent of projector health. Healthy references come
    # from a deterministic released-init build on the SAME fixed validation set.
    from train.engine import (FixedValidation, build_deterministic_reference,
                              healthy_references)
    from diagnostics.representation_health import token_space_stats
    fv = FixedValidation(batches, ratio=args.mask_ratio, device=device,
                         mask_seed=args.mask_seed)
    refs_model = build_deterministic_reference(
        lambda: build_model(cfg["model"],
                            os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
                            device=device,
                            init_from_metadit=cfg["model"].get("init_from_metadit", True),
                            metadit_weights=os.path.join(REPO_ROOT,
                                                         cfg["weights"]["metadit"])))
    refs_model.eval()
    refs = healthy_references(refs_model, fv, objective=objective)
    raw_stats = token_space_stats(zy)
    proj_stats = token_space_stats(p_zy)
    failure = classify_failure_mode(
        raw_stats, proj_stats, refs["raw"], refs["proj"],
        physics_gap=physics["raw_real_vs_null_improvement"],
        physics_shuffle_delta=abs(physics["raw_real_vs_shuffled_improvement"]),
        target_gradient_leak=False,
        invalid_implementation=None)

    report = {
        "checkpoint": args.checkpoint,
        "objective": args.objective,
        "spaces": diag,
        "physics": physics,
        "prediction": pred_metrics,
        "failure_mode": failure,
        "mask_statistics": {
            "ratio": args.mask_ratio, "seed": args.mask_seed,
            "n_batches": len(batches),
            "n_samples": sum(G.shape[0] for G, _ in batches),
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_{args.objective}_sanity.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps(report, indent=2))
    print(f"\n-> {path}")


def classify_collapse(diag):
    """Collapse gates (spec §15). Dimensional collapse = large fraction of dims
    with std < 0.1; sample collapse = cross-geometry pairwise p05 > 0.98 with a
    collapsing effective rank. Projector collapse = raw representation healthy
    while the projected representation is collapsed -> PROJECTOR_COLLAPSE."""
    raw_p05 = diag["raw_target_pairwise_cos"]["p05"]
    raw_frac = diag["raw_target_rank_fraction"]
    proj_p05 = diag["proj_target_pairwise_cos"]["p05"]
    proj_frac = diag["proj_target_rank_fraction"]
    raw_dim = diag["raw_target_frac_std_lt_0p1"]
    proj_dim = diag["proj_target_frac_std_lt_0p1"]
    pred_dim = diag["proj_predictor_frac_std_lt_0p1"]

    raw_dimensional = raw_dim > 0.30
    proj_dimensional = (proj_dim > 0.30) or (pred_dim > 0.30)
    raw_sample = raw_p05 > 0.98 and raw_frac < 0.05
    proj_sample = proj_p05 > 0.98 and proj_frac < 0.05

    raw_collapsed = raw_dimensional or raw_sample
    proj_collapsed = proj_dimensional or proj_sample

    if raw_collapsed:
        verdict = "RAW_COLLAPSE"
    elif proj_collapsed:
        verdict = "PROJECTOR_COLLAPSE"
    else:
        verdict = "HEALTHY"

    return {
        "verdict": verdict,
        "raw_dimensional_collapse": raw_dimensional,
        "proj_dimensional_collapse": proj_dimensional,
        "raw_sample_collapse": raw_sample,
        "proj_sample_collapse": proj_sample,
        "raw_collapsed": raw_collapsed,
        "proj_collapsed": proj_collapsed,
        "raw_p05": raw_p05, "raw_eff_rank_frac": raw_frac,
        "proj_p05": proj_p05, "proj_eff_rank_frac": proj_frac,
    }


# ---------------------------------------------------------------------------
# short end-to-end audit (spec §15)
# ---------------------------------------------------------------------------

def _bn_snapshot(module):
    """Snapshot BatchNorm running stats (running_mean/var/num_batches_tracked)
    of every BN in `module`, so a diagnostics-only forward does not perturb
    the training statistics."""
    snaps = []
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            snaps.append((m, m.running_mean.clone(),
                          m.running_var.clone(), m.num_batches_tracked.clone()))
    return snaps


def _bn_restore(snaps):
    for m, rm, rv, nbt in snaps:
        m.running_mean.copy_(rm)
        m.running_var.copy_(rv)
        m.num_batches_tracked.copy_(nbt)


def _term_grad_norms(obj, model, G, S, M, params):
    """Unweighted per-term gradient norms (spec §15): forward + backward of each
    of the objective's `term_names` components separately (zeroing grads
    between), with the projector's BN statistics snapshotted so the diagnostic
    is measurement-only. Returns {term: grad_norm}."""
    snap = _bn_snapshot(obj.projector)
    norms = {}
    for term in obj.term_names:
        for p in params:
            if p.grad is not None:
                p.grad = None
        res = obj(model, G, S, M)
        res["components"][term].backward()
        total = sum(p.grad.norm().item() ** 2 for p in params
                    if p.grad is not None)
        norms[term] = math.sqrt(total)
    for p in params:
        if p.grad is not None:
            p.grad = None
    _bn_restore(snap)
    return norms


def run_short_audit(cfg, args):
    device = torch.device(args.device)
    set_seed(args.seed)

    steps = args.steps
    report_every = args.report_every
    if args.smoke:
        steps = 6
        report_every = 1
    subset_n = 8 if args.smoke else args.subset
    batch_size = 2 if args.smoke else args.batch_size

    objective = build_objective(
        args.objective, cfg.get("objective_params", {}).get(args.objective, {}),
        projector_input_dim=cfg["model"].get("hidden", 384))

    model = build_model(
        cfg["model"],
        os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
        device=device,
        init_from_metadit=cfg["model"].get("init_from_metadit", True),
        metadit_weights=os.path.join(REPO_ROOT, cfg["weights"]["metadit"]),
    )
    model.train()
    objective.train()

    params = [p for p in model.parameters() if p.requires_grad] \
        + [p for p in objective.parameters() if p.requires_grad]
    opt_ids = {id(p) for p in params}
    ema_ids = {id(p) for p in model.ema.parameters()}
    assert not (opt_ids & ema_ids), "EMA target parameters in optimizer"
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd,
                                  betas=(0.9, 0.999))

    ds = MetaDiTDataset(os.path.join(REPO_ROOT, cfg["data"]["train_split"]),
                        max_samples=subset_n, seed=args.seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True, collate_fn=collate_batch)
    masker = BlockMasker(placement="random", grid=PIXEL_GRID, min_side=3,
                         k_range=(1, 4), seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    step = 0
    abort_reason = None
    domination_streak = 0
    sigreg_info = None
    while step < steps:
        for G, S in loader:
            if step >= steps:
                break
            G, S = G.to(device), S.to(device)
            M = masker.sample(G, args.mask_ratio).to(device)

            res = objective(model, G, S, M)
            total = res["total_loss"]
            comps = res["components"]
            if isinstance(comps.get("sigreg_info"), dict):
                sigreg_info = comps["sigreg_info"]
            if not torch.isfinite(total):
                abort_reason = f"NaN/Inf total loss at step {step}"
                break
            total.backward()
            for n_, p in model.ema.named_parameters():
                if p.grad is not None:
                    abort_reason = (
                        f"EMA target encoder gradient on {n_} at step {step}")
                    break
            if abort_reason:
                break
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            objective.on_optimizer_step(model, step)

            if step % report_every == 0:
                row = _audit_row(step, model, objective, G, S, M, comps,
                                 params, args)
                rows.append(row)
                # Spec §15 aborts. Collapse is judged RELATIVE to the run's own
                # released-init start (first report), not by arbitrary absolute
                # thresholds that would fire on a healthy initialized model.
                if step == 0:
                    base = row
                else:
                    reason = _audit_collapse_abort(row, base)
                    if reason:
                        abort_reason = reason
                        break

            term_ratios = {k: comps[k].item() / max(total.item(), 1e-9)
                           for k in _weighted_terms(comps)}
            if term_ratios and max(term_ratios.values()) > 0.999:
                domination_streak += 1
            else:
                domination_streak = 0
            if domination_streak >= 5:
                abort_reason = (
                    "extreme term domination: single weighted term > 99.9% "
                    f"of total for {domination_streak} reports: {term_ratios}")
                break
            step += 1
        if abort_reason:
            break

    if abort_reason:
        print(f"[short-audit] ABORT: {abort_reason}")
    else:
        print(f"[short-audit] completed {step} optimizer steps")

    report = {"mode": "short_audit", "objective": args.objective,
              "steps": step, "abort_reason": abort_reason,
              "config": {"lr": args.lr, "wd": args.wd,
                         "mask_ratio": args.mask_ratio},
              "rows": rows, "sigreg_info": sigreg_info}
    path = out_dir / f"eval_{args.objective}_short_audit.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    for r in rows:
        print(json.dumps(r, default=float))
    print(f"\n-> {path}")
    if abort_reason:
        sys.exit(1)


def _weighted_terms(comps):
    """Component keys that are weighted loss terms (ratio candidates): keys
    ending in '_weighted' plus the unweighted L_J (LeJEPA)."""
    return [k for k in comps
            if isinstance(comps[k], torch.Tensor)
            and (k.endswith("_weighted") or k == "L_J")]


def _audit_collapse_abort(row, base):
    """Reference-relative abort verdict for one audit report (spec §15). All
    signals are measured against the run's own first report (released-init
    baseline): raw collapse = raw target rank fraction halved while the
    cross-geometry p05 cosine approaches identity (past both 0.98 and the
    baseline); projector collapse = projected rank fraction halved while the
    raw space stays healthy. Absolute thresholds alone would fire on a healthy
    released-init model (its raw tokens genuinely contain low-variance dims),
    so every gate is relative to the baseline."""
    step = row["step"]
    rank_floor = 0.5 * max(base["raw_target_rank_frac"], 1e-6)
    p05_floor = max(0.98, base["raw_pairwise_cos_p05"] + 0.02)
    raw_collapsed = (
        row["raw_target_rank_frac"] < rank_floor
        and row["raw_pairwise_cos_p05"] > p05_floor)
    if raw_collapsed:
        return (f"raw representation collapse at step {step} "
                f"(rank frac {row['raw_target_rank_frac']:.4f} vs baseline "
                f"{base['raw_target_rank_frac']:.4f}, p05 "
                f"{row['raw_pairwise_cos_p05']:.4f})")
    raw_healthy = row["raw_pairwise_cos_p05"] < base["raw_pairwise_cos_p05"] + 0.02
    p_floor = 0.5 * max(base["proj_target_rank_frac"], 1e-6)
    pp05_floor = max(0.98, base["proj_pairwise_cos_p05"] + 0.02)
    proj_collapsed = (
        row["proj_target_rank_frac"] < p_floor
        and row["proj_pairwise_cos_p05"] > pp05_floor)
    if proj_collapsed and raw_healthy:
        return (f"projector collapse at step {step} "
                f"(proj rank frac {row['proj_target_rank_frac']:.4f} vs "
                f"baseline {base['proj_target_rank_frac']:.4f}, p05 "
                f"{row['proj_pairwise_cos_p05']:.4f}) while raw stays healthy")
    return None


def _audit_row(step, model, objective, G, S, M, comps, params, args):
    """One per-report row of the short audit: loss components, raw/projected
    rank + pairwise cosine, feature stds, projector singular values, and
    per-term gradient norms. All measured on the last training batch."""
    P = objective.projector
    mask = (M.view(G.shape[0], -1) == 0)
    with torch.no_grad():
        out = model(G, S, M)
        zy, zh = out["z_y_raw"], out["z_hat"]
        p_zy, p_zh = P(zy), P(zh)
        mw = mask.float()
        zh_pool = (zh * mw.unsqueeze(-1)).sum(1) \
            / mw.sum(1, keepdim=True).clamp(min=1)          # (B, D)
        zy_pool = (zy * mw.unsqueeze(-1)).sum(1) \
            / mw.sum(1, keepdim=True).clamp(min=1)
        p_zh_pool = (p_zh * mw.unsqueeze(-1)).sum(1) \
            / mw.sum(1, keepdim=True).clamp(min=1)
        p_zy_pool = (p_zy * mw.unsqueeze(-1)).sum(1) \
            / mw.sum(1, keepdim=True).clamp(min=1)
        zh_m, zy_m = _rows(zh[mask]), _rows(zy[mask])
        p_zh_m, p_zy_m = _rows(p_zh[mask]), _rows(p_zy[mask])

        r_zh, f_zh = eff_rank(zh_m)
        r_zy, f_zy = eff_rank(zy_m)
        r_pzh, f_pzh = eff_rank(p_zh_m)
        r_pzy, f_pzy = eff_rank(p_zy_m)

        pcos = pairwise_cos_stats
        raw_cos = pcos(zh_pool) if zh_pool.shape[0] >= 2 else \
            {"mean": float("nan")}
        proj_cos = pcos(p_zh_pool) if p_zh_pool.shape[0] >= 2 else \
            {"mean": float("nan")}

    grads = _term_grad_norms(objective, model, G, S, M, params)
    row = {
        "step": step,
        "L_total": comps["total_loss"].item() if "total_loss" in comps
        else sum(v.item() for v in comps.values() if isinstance(v, torch.Tensor)),
        "raw_target_rank": r_zy, "raw_predictor_rank": r_zh,
        "raw_target_rank_frac": f_zy, "raw_predictor_rank_frac": f_zh,
        "proj_target_rank": r_pzy, "proj_predictor_rank": r_pzh,
        "proj_target_rank_frac": f_pzy, "proj_predictor_rank_frac": f_pzh,
        "raw_pairwise_cosine": raw_cos.get("mean", float("nan")),
        "raw_pairwise_cos_p05": raw_cos.get("p05", float("nan")),
        "proj_pairwise_cosine": proj_cos.get("mean", float("nan")),
        "proj_pairwise_cos_p05": proj_cos.get("p05", float("nan")),
        "raw_min_feature_std": feature_std_stats(zh_m)["min_std"],
        "raw_mean_feature_std": feature_std_stats(zh_m)["mean_std"],
        "raw_frac_std_lt_0p1": feature_std_stats(zh_m)["frac_std_lt_0p1"],
        "proj_min_feature_std": feature_std_stats(p_zh_m)["min_std"],
        "proj_mean_feature_std": feature_std_stats(p_zh_m)["mean_std"],
        "proj_frac_std_lt_0p1": feature_std_stats(p_zh_m)["frac_std_lt_0p1"],
        "projector_sv": sv_stats(p_zh_m),
    }
    for k, v in comps.items():
        if isinstance(v, torch.Tensor):
            row[f"comp_{k}"] = v.item()
    for term, norm in grads.items():
        row[f"grad_norm_{term}"] = norm
    return row


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Shared Milestone-B sanity audit")
    p.add_argument("--config", required=True)
    p.add_argument("--objective", default=None,
                   help="objective name (default: config objective)")
    p.add_argument("--checkpoint", default=None,
                   help="§30 checkpoint for validation mode")
    p.add_argument("--short-audit", action="store_true",
                   help="run the 100-300 step short end-to-end audit")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--report-every", type=int, default=25)
    p.add_argument("--subset", type=int, default=32,
                   help="fixed validation/train subset size")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--mask-seed", type=int, default=12345)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default="checkpoints/milestone_b/vicreg_sanity")
    p.add_argument("--smoke", action="store_true",
                   help="tiny local crash test (6 steps, 8 samples)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.objective is None:
        args.objective = cfg.get("objective", "jepa_vicreg")
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.device = "cpu"
    if args.short_audit:
        run_short_audit(cfg, args)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --short-audit")
        validate_checkpoint(cfg, args)


if __name__ == "__main__":
    main()