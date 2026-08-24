
#!/usr/bin/env python3
"""
Milestone-B representation / pipeline diagnostic suite.

Runs, from ONE accepted-or-rejected Milestone-B checkpoint:

1. Static architecture/pipeline contract checks.
2. Raw-token and geometry-pooled diversity diagnostics for:
      - trained student GeometryEncoder
      - trained EMA target encoder
      - released MetaDiT-initialized GeometryEncoder
      - random-init GeometryEncoder
3. Linear ridge probes from pooled representations for:
      - geometry scalars: r_atom, h_atom, l_lattice
      - spectrum summaries: mean |S|, max |S|, argmax |S|
4. Physics-conditioning controls:
      real target vs null goal vs shuffled goal.
5. Token-level vs geometry-level VICReg statistics, showing what the current
   loss actually regularizes versus what the strict health gate measures.
6. Optional 100-step continuation ablation, sequentially from the SAME checkpoint:
      A. baseline continuation
      B. baseline + small geometry-level VICReg term
   The two arms use identical fixed batches/masks and are evaluated before/after.
   This is intentionally sequential to avoid doubling GPU memory.

This script is diagnostic. It does NOT modify the checkpoint.

Recommended cloud usage:
python scripts/diagnostics/jpa_representation_root_cause.py \
  --checkpoint checkpoints/milestone_b/minimal_jepa_vicreg_latest.pt \
  --config configs/milestone_b.yaml \
  --device cuda:0 \
  --subset 512 \
  --batch-size 32 \
  --run-ablation
"""

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from assembly import build_model
from data.dataset import MetaDiTDataset, collate_batch
from data.mask import BlockMasker
from diagnostics.representation_health import (
    eff_ranks,
    pairwise_cos_stats,
    same_token_cos,
)
from losses.objectives import build_objective
from losses.vicreg import vicreg_branch_terms
from train.engine import load_checkpoint


DEFAULT_SEED = 0
PIXEL_GRID = 16


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_or_none(x):
    x = float(x)
    return x if math.isfinite(x) else None


def flatten_tokens(x):
    if x.ndim == 3:
        return x.reshape(-1, x.shape[-1])
    return x


def masked_pool(x, mask):
    m = mask.float()
    return (x * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)


def cosine_mean(a, b, mask=None):
    c = F.cosine_similarity(
        F.normalize(a, dim=-1),
        F.normalize(b, dim=-1),
        dim=-1,
    )
    if mask is not None:
        c = c[mask]
    return float(c.mean().item())


def stat_space(x_tokens, pooled=None):
    """
    Diagnostics for token embeddings x_tokens: [B,T,D].
    pooled is [B,D]; if omitted, mean-pool tokens.
    """
    x = x_tokens.detach().float().cpu()
    if pooled is None:
        pooled = x.mean(1)
    else:
        pooled = pooled.detach().float().cpu()

    out = {
        "n_geoms": int(pooled.shape[0]),
        "n_tokens": int(x.shape[1]),
        "dim": int(x.shape[-1]),
        "token_mean_std": float(x.std(dim=0, unbiased=x.shape[0] >= 2).mean().item()),
        "token_min_std": float(x.std(dim=0, unbiased=x.shape[0] >= 2).min().item()),
        "token_frac_std_lt_0p1": float(
            (x.std(dim=0, unbiased=x.shape[0] >= 2) < 0.1).float().mean().item()
        ),
        "token_frac_std_lt_0p5": float(
            (x.std(dim=0, unbiased=x.shape[0] >= 2) < 0.5).float().mean().item()
        ),
    }

    if pooled.shape[0] >= 2:
        er = eff_ranks(pooled)
        pc = pairwise_cos_stats(pooled)
        out.update({
            "pooled_eff_rank": float(er["eff_rank_unnorm"]),
            "pooled_eff_rank_frac": float(er["eff_rank_frac"]),
            "pooled_participation": float(er["participation"]),
            "pooled_top_eig_frac": float(er["top_eig_frac"]),
            "pooled_pairwise_cos_mean": float(pc["mean"]),
            "pooled_pairwise_cos_p05": float(pc["p05"]),
            "pooled_pairwise_cos_min": float(pc["min"]),
            "same_token_cos": float(same_token_cos(x)),
        })
    else:
        for k in (
            "pooled_eff_rank",
            "pooled_eff_rank_frac",
            "pooled_participation",
            "pooled_top_eig_frac",
            "pooled_pairwise_cos_mean",
            "pooled_pairwise_cos_p05",
            "pooled_pairwise_cos_min",
            "same_token_cos",
        ):
            out[k] = None

    return out


# ---------------------------------------------------------------------------
# Dataset / fixed evaluation batches
# ---------------------------------------------------------------------------

def load_fixed_batches(cfg, device, subset, batch_size, seed):
    val_path = os.path.join(REPO_ROOT, cfg["data"]["val_split"])
    ds = MetaDiTDataset(val_path)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_batch,
    )

    batches = []
    total = 0
    for G, S in loader:
        remaining = subset - total
        if remaining <= 0:
            break
        take = min(remaining, G.shape[0])
        batches.append((G[:take].to(device), S[:take].to(device)))
        total += take

    if total < 2:
        raise RuntimeError(f"Need at least 2 validation geometries, got {total}")

    masker = BlockMasker(
        placement="random",
        grid=PIXEL_GRID,
        min_side=cfg["mask"].get("min_side", 3),
        k_range=tuple(cfg["mask"].get("k_range", [1, 4])),
        seed=seed,
    )
    masks = [masker.sample(G, 0.5).to(device) for G, _ in batches]
    return batches, masks


# ---------------------------------------------------------------------------
# Model / objective loading
# ---------------------------------------------------------------------------

def load_model_objective(cfg, checkpoint, device, objective_name):
    model = build_model(
        cfg["model"],
        os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
        device=device,
        init_from_metadit=cfg["model"].get("init_from_metadit", True),
        metadit_weights=os.path.join(
            REPO_ROOT, cfg["weights"]["metadit"]
        ),
    )

    objective = build_objective(
        objective_name,
        cfg.get("objective_params", {}).get(objective_name, {}),
        projector_input_dim=cfg["model"].get("hidden", 384),
    ).to(device)

    ckpt = load_checkpoint(
        checkpoint,
        model,
        objective,
        None,
        None,
        device,
    )
    model.eval()
    model.ema.eval()
    objective.eval()
    return model, objective, ckpt


def build_reference_models(cfg, device):
    released = build_model(
        cfg["model"],
        os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
        device=device,
        init_from_metadit=True,
        metadit_weights=os.path.join(REPO_ROOT, cfg["weights"]["metadit"]),
    )
    released.eval()

    # Build a pure random geometry encoder by using the same architecture, then
    # replacing the geometry encoder with an uninitialized instance from the model.
    # We only need the geometry representation, so the complete model is unnecessary.
    random_model = build_model(
        cfg["model"],
        os.path.join(REPO_ROOT, cfg["weights"]["spectrum"]),
        device=device,
        init_from_metadit=False,
        metadit_weights=None,
    )
    random_model.eval()

    return released, random_model


# ---------------------------------------------------------------------------
# Pipeline contract checks
# ---------------------------------------------------------------------------

def run_contract_checks(model, objective, batches, masks):
    checks = {}

    checks["objective_device"] = str(next(objective.parameters()).device)
    checks["model_device"] = str(next(model.parameters()).device)

    checks["objective_and_model_same_device"] = (
        next(objective.parameters()).device
        == next(model.parameters()).device
    )

    checks["ema_parameters_frozen"] = all(
        not p.requires_grad for p in model.ema.parameters()
    )

    released = getattr(model.spectrum_path, "released", None)
    checks["released_spectrum_frozen"] = (
        released is not None
        and all(not p.requires_grad for p in released.parameters())
    )

    checks["context_shares_geometry_encoder"] = (
        model.context_encoder.geo is model.geometry_encoder
    )

    G, S = batches[0]
    M = masks[0]
    with torch.no_grad():
        out = model(G[:2], S[:2], M[:2], goal_mode="real")

    checks["z_hat_shape"] = list(out["z_hat"].shape)
    checks["z_y_raw_shape"] = list(out["z_y_raw"].shape)
    checks["mask_shape"] = list(out["mask"].shape)
    checks["physics_condition_shape"] = list(out["c_physics"].shape)
    checks["goal_token_shape"] = list(out["a_goal"].shape)

    checks["shape_contract"] = (
        out["z_hat"].shape == (min(2, G.shape[0]), 256, 384)
        and out["z_y_raw"].shape == (min(2, G.shape[0]), 256, 384)
        and out["mask"].shape == (min(2, G.shape[0]), 256)
        and out["c_physics"].shape == (min(2, G.shape[0]), 384)
        and out["a_goal"].shape == (min(2, G.shape[0]), 16, 384)
    )

    # Check EMA target receives no gradients from one real loss backward.
    model.train()
    objective.train()
    model.zero_grad(set_to_none=True)
    objective.zero_grad(set_to_none=True)

    out_train = model(G[:4], S[:4], M[:4], goal_mode="real")
    mask_bool = out_train["mask"]
    p_hat = objective.projector(out_train["z_hat"])
    p_y = objective.projector(out_train["z_y_raw"])
    p_hat_m = p_hat[mask_bool]
    p_y_m = p_y[mask_bool]
    terms = vicreg_branch_terms(p_hat_m, p_y_m, gamma=1.0, eps=1e-4)
    loss = 25.0 * terms[0] + 25.0 * terms[1] + terms[2]
    loss.backward()

    checks["ema_no_grad_after_backward"] = all(
        p.grad is None for p in model.ema.parameters()
    )

    model.zero_grad(set_to_none=True)
    objective.zero_grad(set_to_none=True)
    model.eval()
    objective.eval()

    checks["all_contracts_pass"] = all(
        v for k, v in checks.items()
        if k.endswith("_frozen")
        or k.endswith("_device")
        or k in {
            "objective_and_model_same_device",
            "context_shares_geometry_encoder",
            "shape_contract",
            "ema_no_grad_after_backward",
        }
    )
    return checks


# ---------------------------------------------------------------------------
# Representation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_representations(encoder, batches, max_geoms=None):
    tokens = []
    total = 0
    for G, _ in batches:
        x = encoder(G)
        take = x.shape[0]
        if max_geoms is not None:
            take = min(take, max_geoms - total)
        if take <= 0:
            break
        tokens.append(x[:take].detach().cpu())
        total += take
        if max_geoms is not None and total >= max_geoms:
            break
    X = torch.cat(tokens, dim=0)
    return X


# ---------------------------------------------------------------------------
# Geometry/spectrum probe targets
# ---------------------------------------------------------------------------

def geometry_labels(G):
    """
    Continuous geometry labels derived directly from the encoded geometry.

    r_atom and h_atom are averaged over occupied pixels. l_lattice is averaged
    over the full lattice map. This is deliberately simple: the probe asks
    whether the embedding retains measurable geometry information.
    """
    G = G.float()
    ch0 = G[:, 0]
    ch1 = G[:, 1]
    ch2 = G[:, 2]

    occupied = ch0.abs() > 1e-8
    occ_count = occupied.sum(dim=(1, 2)).clamp(min=1)

    r = (ch0 * occupied).sum(dim=(1, 2)) / occ_count
    h = (ch1 * occupied).sum(dim=(1, 2)) / occ_count
    lattice = ch2.mean(dim=(1, 2))

    return torch.stack([r, h, lattice], dim=1)


def spectrum_labels(S):
    """
    Continuous spectrum summaries from S=[real,imag,301].
    """
    S = S.float()
    mag = torch.sqrt(S[:, 0] ** 2 + S[:, 1] ** 2 + 1e-12)

    mean_mag = mag.mean(dim=1)
    max_mag = mag.max(dim=1).values

    idx = torch.arange(
        mag.shape[1],
        device=mag.device,
        dtype=mag.dtype,
    )
    total = mag.sum(dim=1).clamp(min=1e-8)
    centroid = (mag * idx.unsqueeze(0)).sum(dim=1) / total
    centroid = centroid / max(1, mag.shape[1] - 1)

    argmax = mag.argmax(dim=1).float() / max(1, mag.shape[1] - 1)

    return torch.stack([mean_mag, max_mag, centroid, argmax], dim=1)


def ridge_probe(X, Y, train_frac=0.6, ridge=1e-3):
    """
    Closed-form linear ridge probe with deterministic train/test split.
    Returns test MSE and R^2 per target plus aggregate mean R^2.

    X: [N,D]
    Y: [N,K]
    """
    X = X.float()
    Y = Y.float()

    n = X.shape[0]
    split = max(2, int(round(n * train_frac)))
    split = min(split, n - 1)

    # deterministic split: first train_frac samples for train, rest for test
    Xtr, Xte = X[:split], X[split:]
    Ytr, Yte = Y[:split], Y[split:]

    # standardize using training statistics
    xmu = Xtr.mean(0, keepdim=True)
    xstd = Xtr.std(0, unbiased=False, keepdim=True).clamp(min=1e-6)
    ymu = Ytr.mean(0, keepdim=True)
    ystd = Ytr.std(0, unbiased=False, keepdim=True).clamp(min=1e-6)

    Xtrz = (Xtr - xmu) / xstd
    Xtez = (Xte - xmu) / xstd
    Ytrz = (Ytr - ymu) / ystd

    ones_tr = torch.ones(Xtrz.shape[0], 1)
    ones_te = torch.ones(Xtez.shape[0], 1)
    A = torch.cat([Xtrz, ones_tr], dim=1)
    B = torch.cat([Xtez, ones_te], dim=1)

    I = torch.eye(A.shape[1])
    I[-1, -1] = 0.0

    W = torch.linalg.solve(
        A.T @ A + ridge * I,
        A.T @ Ytrz,
    )

    pred_z = B @ W
    pred = pred_z * ystd + ymu

    mse = ((pred - Yte) ** 2).mean(0)
    ybar = Yte.mean(0, keepdim=True)
    ss_res = ((pred - Yte) ** 2).sum(0)
    ss_tot = ((Yte - ybar) ** 2).sum(0).clamp(min=1e-12)
    r2 = 1.0 - ss_res / ss_tot

    return {
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "mse": [float(v) for v in mse],
        "r2": [float(v) for v in r2],
        "mean_r2": float(r2.mean()),
    }


# ---------------------------------------------------------------------------
# Physics conditioning
# ---------------------------------------------------------------------------

@torch.no_grad()
def physics_controls(model, batches, masks, max_batches=16):
    model.eval()
    real_raw = []
    null_raw = []
    shuf_raw = []
    real_proj = []
    null_proj = []
    shuf_proj = []

    objective = model._diagnostic_objective
    P = objective.projector

    for bi, ((G, S), M) in enumerate(zip(batches, masks)):
        if bi >= max_batches:
            break

        out_real = model(G, S, M, goal_mode="real")
        out_null = model(G, S, M, goal_mode="null")

        gen = torch.Generator(device="cpu").manual_seed(12345 + bi)
        perm = torch.randperm(G.shape[0], generator=gen).to(G.device)
        out_shuf = model(G, S[perm], M, goal_mode="real")

        mask = out_real["mask"]

        real_raw.append(cosine_mean(out_real["z_hat"], out_real["z_y_raw"], mask))
        null_raw.append(cosine_mean(out_null["z_hat"], out_real["z_y_raw"], mask))
        shuf_raw.append(cosine_mean(out_shuf["z_hat"], out_real["z_y_raw"], mask))

        p_real_h = P(out_real["z_hat"])
        p_real_y = P(out_real["z_y_raw"])
        p_null_h = P(out_null["z_hat"])
        p_null_y = P(out_null["z_y_raw"])
        p_shuf_h = P(out_shuf["z_hat"])
        p_shuf_y = P(out_shuf["z_y_raw"])

        real_proj.append(cosine_mean(p_real_h, p_real_y, mask))
        null_proj.append(cosine_mean(p_null_h, p_null_y, mask))
        shuf_proj.append(cosine_mean(p_shuf_h, p_shuf_y, mask))

    result = {
        "real_raw": float(np.mean(real_raw)),
        "null_raw": float(np.mean(null_raw)),
        "shuffled_raw": float(np.mean(shuf_raw)),
        "real_projected": float(np.mean(real_proj)),
        "null_projected": float(np.mean(null_proj)),
        "shuffled_projected": float(np.mean(shuf_proj)),
    }

    result["real_vs_null_raw_improvement"] = (
        result["null_raw"] - result["real_raw"]
    )
    result["real_vs_shuffle_raw_improvement"] = (
        result["shuffled_raw"] - result["real_raw"]
    )
    result["real_vs_null_projected_improvement"] = (
        result["null_projected"] - result["real_projected"]
    )
    result["real_vs_shuffle_projected_improvement"] = (
        result["shuffled_projected"] - result["real_projected"]
    )
    return result


# ---------------------------------------------------------------------------
# Geometry-level vs token-level VICReg statistics
# ---------------------------------------------------------------------------

@torch.no_grad()
def vicreg_scale_audit(model, objective, batches, masks, max_batches=8):
    model.eval()
    objective.eval()

    token_terms = []
    pooled_terms = []

    for bi, ((G, S), M) in enumerate(zip(batches, masks)):
        if bi >= max_batches:
            break

        out = model(G, S, M, goal_mode="real")
        mask = out["mask"]

        ph = objective.projector(out["z_hat"])
        py = objective.projector(out["z_y_raw"])

        ph_m = ph[mask]
        py_m = py[mask]

        token = vicreg_branch_terms(ph_m, py_m, gamma=1.0, eps=1e-4)
        token_terms.append([float(t.item()) for t in token])

        ph_g = masked_pool(ph, mask)
        py_g = masked_pool(py, mask)

        pooled = vicreg_branch_terms(ph_g, py_g, gamma=1.0, eps=1e-4)
        pooled_terms.append([float(t.item()) for t in pooled])

    token_mean = np.mean(token_terms, axis=0)
    pooled_mean = np.mean(pooled_terms, axis=0)

    return {
        "token_level": {
            "L_inv": float(token_mean[0]),
            "L_var": float(token_mean[1]),
            "L_cov": float(token_mean[2]),
            "weighted_total": float(25 * token_mean[0] + 25 * token_mean[1] + token_mean[2]),
        },
        "geometry_level_pooled": {
            "L_inv": float(pooled_mean[0]),
            "L_var": float(pooled_mean[1]),
            "L_cov": float(pooled_mean[2]),
            "weighted_same_coeff_total": float(25 * pooled_mean[0] + 25 * pooled_mean[1] + pooled_mean[2]),
        },
    }


# ---------------------------------------------------------------------------
# Optional sequential ablation
# ---------------------------------------------------------------------------

def _fresh_from_checkpoint(cfg, checkpoint, device, objective_name):
    model, objective, ckpt = load_model_objective(
        cfg, checkpoint, device, objective_name
    )
    return model, objective, ckpt


def _make_optimizer(model, objective, lr):
    params = [
        p for p in model.parameters() if p.requires_grad
    ] + [
        p for p in objective.parameters() if p.requires_grad
    ]

    ema_ids = {id(p) for p in model.ema.parameters()}
    if ema_ids.intersection({id(p) for p in params}):
        raise RuntimeError("EMA parameters leaked into ablation optimizer")

    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0), params


def _train_continuation(
    cfg,
    checkpoint,
    batches,
    masks,
    device,
    objective_name,
    steps,
    lr,
    geo_lambda,
):
    model, objective, ckpt = _fresh_from_checkpoint(
        cfg, checkpoint, device, objective_name
    )

    # Continue EMA in the checkpoint's global step coordinate. Do not restart
    # the EMA momentum schedule from step 0.
    start_step = int(ckpt.get("step", 0)) + 1 if isinstance(ckpt, dict) else 0
    model.train()
    objective.train()

    optimizer, trainable = _make_optimizer(model, objective, lr)

    fixed = list(zip(batches, masks))
    start = time.time()

    for step in range(steps):
        (G, S), M = fixed[step % len(fixed)]

        optimizer.zero_grad(set_to_none=True)

        out = model(G, S, M, goal_mode="real")
        mask = out["mask"]

        ph = objective.projector(out["z_hat"])
        py = objective.projector(out["z_y_raw"])

        ph_m = ph[mask]
        py_m = py[mask]

        token_terms = vicreg_branch_terms(ph_m, py_m, gamma=1.0, eps=1e-4)
        total = (
            25.0 * token_terms[0]
            + 25.0 * token_terms[1]
            + 1.0 * token_terms[2]
        )

        geo_terms = vicreg_branch_terms(
            masked_pool(ph, mask),
            masked_pool(py, mask),
            gamma=1.0,
            eps=1e-4,
        )

        if geo_lambda != 0.0:
            total = total + geo_lambda * (
                25.0 * geo_terms[0]
                + 25.0 * geo_terms[1]
                + 1.0 * geo_terms[2]
            )

        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite ablation loss at step {step}")

        total.backward()

        ema_grad = [
            p for p in model.ema.parameters()
            if p.grad is not None
        ]
        if ema_grad:
            raise RuntimeError("EMA target received gradient during ablation")

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        # Match the production objective behavior.
        if objective_name == "jepa_vicreg":
            model.ema.update(model.geometry_encoder, start_step + step)

    elapsed = time.time() - start

    # Freeze for evaluation
    model.eval()
    objective.eval()

    repr_tokens = []
    repr_pooled = []

    with torch.no_grad():
        for G, S in batches:
            x = model.ema(G)
            repr_tokens.append(x.cpu())
            repr_pooled.append(x.mean(1).cpu())

    X = torch.cat(repr_tokens)
    XP = torch.cat(repr_pooled)

    er = eff_ranks(XP)
    pc = pairwise_cos_stats(XP)

    return {
        "steps": steps,
        "geo_lambda": geo_lambda,
        "lr": lr,
        "elapsed_sec": elapsed,
        "final_loss": float(total.item()),
        "ema_raw_repr": stat_space(X, XP),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=os.path.join(
        REPO_ROOT, "configs", "milestone_b.yaml"
    ))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--subset", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objective", default="jepa_vicreg")
    parser.add_argument("--probe-train-frac", type=float, default=0.6)
    parser.add_argument("--probe-ridge", type=float, default=1e-3)
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--ablation-steps", type=int, default=100)
    parser.add_argument("--ablation-lr", type=float, default=1e-5)
    parser.add_argument("--ablation-geo-lambda", type=float, default=0.25)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        device = torch.device(args.device)
    else:
        device = torch.device(args.device)

    print(f"[suite] device={device}")
    print(f"[suite] checkpoint={args.checkpoint}")

    # Fixed validation data.
    batches, masks = load_fixed_batches(
        cfg, device, args.subset, args.batch_size, args.seed
    )

    model, objective, ckpt = load_model_objective(
        cfg, args.checkpoint, device, args.objective
    )

    # Used by physics_controls; keep object explicit and never save it.
    model._diagnostic_objective = objective

    results = {
        "meta": {
            "checkpoint": args.checkpoint,
            "objective": args.objective,
            "device": str(device),
            "subset": sum(G.shape[0] for G, _ in batches),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "checkpoint_step": ckpt.get("step") if isinstance(ckpt, dict) else None,
            "checkpoint_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        }
    }

    # ------------------------------------------------------------------
    # A. Contract checks
    # ------------------------------------------------------------------
    print("\n[A] PIPELINE CONTRACT CHECKS")
    results["pipeline_contracts"] = run_contract_checks(
        model, objective, batches, masks
    )
    print(json.dumps(results["pipeline_contracts"], indent=2))

    # ------------------------------------------------------------------
    # B. Representation comparison
    # ------------------------------------------------------------------
    print("\n[B] REPRESENTATION COMPARISON")

    released, random_model = build_reference_models(cfg, device)

    student_X = extract_representations(model.geometry_encoder, batches)
    ema_X = extract_representations(model.ema, batches)
    released_X = extract_representations(released.geometry_encoder, batches)
    random_X = extract_representations(random_model.geometry_encoder, batches)

    representation_spaces = {
        "student_geometry_encoder": student_X,
        "ema_target_encoder": ema_X,
        "released_metadit_geometry_encoder": released_X,
        "random_geometry_encoder": random_X,
    }

    results["representation_stats"] = {}
    for name, X in representation_spaces.items():
        stats = stat_space(X)
        results["representation_stats"][name] = stats
        print(f"\n{name}")
        print(json.dumps(stats, indent=2))

    # ------------------------------------------------------------------
    # C. Probe targets
    # ------------------------------------------------------------------
    print("\n[C] LINEAR INFORMATION PROBES")

    G_all = torch.cat([G.cpu() for G, _ in batches], dim=0)
    S_all = torch.cat([S.cpu() for _, S in batches], dim=0)

    Y_geo = geometry_labels(G_all)
    Y_spec = spectrum_labels(S_all)

    results["probes"] = {}

    for name, X in representation_spaces.items():
        XP = X.mean(1)

        geo_probe = ridge_probe(
            XP, Y_geo,
            train_frac=args.probe_train_frac,
            ridge=args.probe_ridge,
        )
        spec_probe = ridge_probe(
            XP, Y_spec,
            train_frac=args.probe_train_frac,
            ridge=args.probe_ridge,
        )

        results["probes"][name] = {
            "geometry": geo_probe,
            "spectrum": spec_probe,
        }

        print(f"\n{name} geometry mean-R2 = {geo_probe['mean_r2']:.4f}")
        print(f"{name} spectrum mean-R2 = {spec_probe['mean_r2']:.4f}")

    # ------------------------------------------------------------------
    # D. Physics conditioning
    # ------------------------------------------------------------------
    print("\n[D] PHYSICS CONDITIONING CONTROLS")
    results["physics_controls"] = physics_controls(
        model, batches, masks
    )
    print(json.dumps(results["physics_controls"], indent=2))

    # ------------------------------------------------------------------
    # E. Token-level vs geometry-level VICReg
    # ------------------------------------------------------------------
    print("\n[E] TOKEN VS GEOMETRY LEVEL VICREG")
    # Ensure normal eval behavior during diagnostics.
    results["vicreg_scale_audit"] = vicreg_scale_audit(
        model, objective, batches, masks
    )
    print(json.dumps(results["vicreg_scale_audit"], indent=2))

    # ------------------------------------------------------------------
    # F. Optional sequential continuation ablation
    # ------------------------------------------------------------------
    if args.run_ablation:
        print("\n[F] SEQUENTIAL 100-STEP ABLATION")
        print(
            "Arm 1: baseline continuation from the SAME checkpoint.\n"
            "Arm 2: same continuation + geometry-level VICReg regularizer."
        )

        baseline = _train_continuation(
            cfg=cfg,
            checkpoint=args.checkpoint,
            batches=batches,
            masks=masks,
            device=device,
            objective_name=args.objective,
            steps=args.ablation_steps,
            lr=args.ablation_lr,
            geo_lambda=0.0,
        )
        print("\nBaseline continuation result:")
        print(json.dumps(baseline, indent=2))

        geo = _train_continuation(
            cfg=cfg,
            checkpoint=args.checkpoint,
            batches=batches,
            masks=masks,
            device=device,
            objective_name=args.objective,
            steps=args.ablation_steps,
            lr=args.ablation_lr,
            geo_lambda=args.ablation_geo_lambda,
        )
        print("\nGeometry-level VICReg result:")
        print(json.dumps(geo, indent=2))

        results["ablation"] = {
            "baseline": baseline,
            "geometry_vicreg": geo,
            "delta_eff_rank": (
                geo["ema_raw_repr"]["pooled_eff_rank"]
                - baseline["ema_raw_repr"]["pooled_eff_rank"]
            ),
            "delta_pairwise_cos_mean": (
                geo["ema_raw_repr"]["pooled_pairwise_cos_mean"]
                - baseline["ema_raw_repr"]["pooled_pairwise_cos_mean"]
            ),
            "delta_same_token_cos": (
                geo["ema_raw_repr"]["same_token_cos"]
                - baseline["ema_raw_repr"]["same_token_cos"]
            ),
        }

    # ------------------------------------------------------------------
    # Final concise interpretation flags
    # ------------------------------------------------------------------
    results["interpretation_flags"] = {
        "trained_ema_better_rank_than_released": (
            results["representation_stats"]["ema_target_encoder"]["pooled_eff_rank"]
            > results["representation_stats"]["released_metadit_geometry_encoder"]["pooled_eff_rank"]
        ),
        "trained_ema_lower_cos_than_released": (
            results["representation_stats"]["ema_target_encoder"]["pooled_pairwise_cos_mean"]
            < results["representation_stats"]["released_metadit_geometry_encoder"]["pooled_pairwise_cos_mean"]
        ),
        "trained_ema_is_better_than_random_on_geometry_probe": (
            results["probes"]["ema_target_encoder"]["geometry"]["mean_r2"]
            > results["probes"]["random_geometry_encoder"]["geometry"]["mean_r2"]
        ),
        "trained_ema_is_better_than_random_on_spectrum_probe": (
            results["probes"]["ema_target_encoder"]["spectrum"]["mean_r2"]
            > results["probes"]["random_geometry_encoder"]["spectrum"]["mean_r2"]
        ),
        "physics_signal_positive_raw": (
            results["physics_controls"]["real_vs_null_raw_improvement"] > 0.0
        ),
        "geometry_level_vicreg_smaller_or_equal_than_token_level_total": (
            results["vicreg_scale_audit"]["geometry_level_pooled"]["weighted_same_coeff_total"]
            <= results["vicreg_scale_audit"]["token_level"]["weighted_total"]
        ),
    }

    print("\n[G] FINAL FLAGS")
    print(json.dumps(results["interpretation_flags"], indent=2))

    out_path = args.out
    if out_path is None:
        out_dir = os.path.join(REPO_ROOT, "checkpoints", "milestone_b", "root_cause")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "jpa_representation_root_cause.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[suite] JSON report -> {out_path}")


if __name__ == "__main__":
    main()
