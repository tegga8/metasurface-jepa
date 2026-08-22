#!/usr/bin/env python3
"""Diagnostic B — physics target selection (latent-selection spec §12-§20).

Question Q2: does the requested spectrum move the predictor toward the CORRECT
spatial target? For fixed visible geometry and fixed target latent, change ONLY
the spectrum:

    REAL     z_hat = P(G_masked, S_real)
    NULL     z_hat = P(G_masked, null)
    SHUFFLED z_hat = P(G_masked, S_j), j != i      (same mask, same target)

No training happens here; an existing checkpoint is loaded and evaluated frozen.
Primary metrics are RAW-space per-sample distances (never total VICReg loss):
    d_cos(a,b) = 1 - cos(a,b)      d_2(a,b) = ||a-b||_2
with margins  margin_null = d_null - d_real > 0  and  margin_shuffle = d_shuf - d_real > 0
meaning the correct physics improves target prediction.

Additional blocks (spec §16-§19):
  - exact spectrum retrieval matrix  D[i,j] = d(P_i(S_j), z_y_i); diagonal =
    correct spectrum; Recall@5 / rank stats / diagonal-vs-off-diagonal margin;
  - geometry-aware retrieval subset: anchors whose FULL occupancy differs from
    every other candidate (geometry-only selection, never latent-based);
  - predicted-latent spatial probe: the SAME frozen z_y_raw probe applied to
    z_hat under each condition (spec §18);
  - same-context/different-spectrum causal test (§19): mutual selection wins.

Evaluated at mask_ratio 1.00 (primary, spec §12) and 0.75 (§20 comparison).

Checkpoint-validity guards (validity-fix spec §6/§7/§10): a loaded checkpoint must
carry genuine training provenance (step, objective_name, cfg) and must NOT be a
smoke/near-init artifact (step <= 10 OR max_total_steps <= 20 OR
max_train_samples <= 64), otherwise the diagnostic FAILS LOUDLY instead of
producing a number that could be mistaken for a trained-model result. An explicit
operator override (--allow-smoke-reason "<why>") proceeds but labels the run
reference-only (`is_smoke_checkpoint: true`, never a trained-model result).
Runtime sanity checks on every batch: EMA target frozen, released spectrum
encoder frozen, z_y_raw shape [B, 256, 384], identical target across the three
conditions.

Run:
  python scripts/diagnostics/physics_target_selection.py \
      --config configs/milestone_b.yaml \
      --checkpoint <genuine_trained_checkpoint>.pt \
      --ratios 1.00 0.75

Output: checkpoints/milestone_b/physics_validation/physics_target_selection.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from assembly import build_model
from data.dataset import MetaDiTDataset
from data.mask import BlockMasker
from train.engine import build_deterministic_reference

sys.path.insert(0, str(REPO_ROOT / "scripts" / "diagnostics"))
from latent_geometry_probe import (  # noqa: E402
    build_probe, dist_stats, pr_iou,
)


# ---------------------------------------------------------------------------
# pure helpers (unit-tested in tests/test_physics_target_selection.py)
# ---------------------------------------------------------------------------

def shuffle_permutation(batch_size, generator=None):
    """Deterministic permutation with NO fixed points when batch_size > 1
    (cyclic roll — the repo's existing convention). batch_size == 1 returns
    [0] and the caller MUST treat shuffled conditions as INFEASIBLE."""
    idx = torch.arange(batch_size)
    if batch_size > 1:
        return torch.roll(idx, shifts=1, dims=0).tolist()
    return [0]


def per_sample_distances(z_hat, z_y, mask):
    """Per-sample raw-space distances to the fixed target.

    z_hat/z_y: (B, T, D); mask: (B, T) bool, True = masked (target) positions.
    Returns dict of per-sample arrays:
      cos : mean over masked tokens of 1 - cos(z_hat_t, z_y_t)
      l2_token : mean over masked tokens of ||z_hat_t - z_y_t||_2
      l2_pooled : ||mean_masked(z_hat) - mean_masked(z_y)||_2
    """
    b = z_hat.shape[0]
    cos, l2_tok, l2_pool = [], [], []
    zh_n = F.normalize(z_hat, dim=-1)
    zy_n = F.normalize(z_y, dim=-1)
    for i in range(b):
        m = mask[i]
        if m.sum() == 0:
            raise ValueError("sample has no masked/target positions")
        a, an = z_hat[i][m], zh_n[i][m]
        t, tn = z_y[i][m], zy_n[i][m]
        cos.append(float((1.0 - (an * tn).sum(-1)).clamp(min=0).mean()))
        l2_tok.append(float((a - t).norm(dim=-1).mean()))
        l2_pool.append(float((a.mean(0) - t.mean(0)).norm()))
    return {"cos": cos, "l2_token": l2_tok, "l2_pooled": l2_pool}


def margin_stats(real, other):
    """margin = other - real per sample (>0: correct physics closer to target),
    plus fraction positive/negative and full distribution stats."""
    real = np.asarray(real, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    m = other - real
    return {
        **dist_stats(m),
        "fraction_positive": float((m > 0).mean()),
        "fraction_negative": float((m < 0).mean()),
    }


def retrieval_matrix_metrics(D):
    """D: (B, B) distances, D[i, j] = d(P_i(S_j), z_y_i). Diagonal = correct
    spectrum. Row-wise ranks of the diagonal entry (1 = best)."""
    D = np.asarray(D, dtype=np.float64)
    b = D.shape[0]
    if b < 2:
        return {"feasible": False, "reason": "batch_size < 2", "batch_size": int(b)}
    ranks = []
    for i in range(b):
        # rank of diagonal among row i (1-indexed; ties -> average rank)
        order = np.argsort(np.argsort(D[i], kind="stable")) + 1
        ranks.append(int(order[i]))
    ranks = np.asarray(ranks)
    diag = np.diag(D)
    off = D[~np.eye(b, dtype=bool)]
    r5 = float((ranks <= 5).mean())
    return {
        "feasible": True,
        "batch_size": int(b),
        "recall_at_5": r5,
        "recall_at_1": float((ranks == 1).mean()),
        "mean_correct_rank": float(ranks.mean()),
        "median_correct_rank": float(np.median(ranks)),
        "mean_diagonal_distance": float(diag.mean()),
        "mean_offdiagonal_distance": float(off.mean()),
        "diagonal_minus_offdiagonal_margin": float(off.mean() - diag.mean()),
        "correct_rank_histogram": {str(r): int((ranks == r).sum())
                                   for r in range(1, b + 1)},
    }


def geometry_aware_subset(occ_flat, k, min_hamming=None):
    """Greedy farthest-min-Hamming anchor selection using GEOMETRY ONLY
    (occupancy Hamming distance; never latents/physics). Returns up to k indices
    whose minimum Hamming distance to the already-selected set is maximal."""
    occ = np.asarray(occ_flat).astype(np.uint8)
    n = occ.shape[0]
    packed = np.packbits(occ, axis=1)
    popcount = np.zeros(256, dtype=np.uint16)
    for i in range(256):
        popcount[i] = bin(i).count("1")

    def ham_to_set(idx, chosen):
        if not chosen:
            return None
        sel = packed[chosen]
        x = packed[idx]
        return popcount[sel ^ x].sum(axis=1)

    chosen = [int(np.argmax(occ.sum(axis=1)))]  # deterministic start: densest
    min_d = ham_to_set(np.arange(n), chosen)
    while len(chosen) < min(k, n):
        nxt = int(np.argmax(min_d))
        if min_d[nxt] == 0:
            break
        chosen.append(nxt)
        d_new = popcount[packed ^ packed[nxt]].sum(axis=1)
        min_d = np.minimum(min_d, d_new)
    sub = np.array(chosen, dtype=int)
    if len(sub) < 2:
        return sub, {"feasible": False, "reason": "no distinct geometries"}
    # pairwise min-hamming per selected anchor (selection-quality record)
    pd = popcount[packed[sub][:, None, :] ^ packed[sub][None, :, :]].sum(axis=-1)
    pair_h = pd.astype(np.int64)
    np.fill_diagonal(pair_h, 10 ** 9)
    info = {
        "feasible": True,
        "k": int(len(sub)),
        "indices": sub.tolist(),
        "min_pairwise_hamming": int(pair_h.min()),
        "mean_pairwise_hamming": float(pair_h[pair_h < 10 ** 9].mean()),
    }
    if min_hamming is not None:
        info["meets_min_hamming"] = bool(info["min_pairwise_hamming"] >= min_hamming)
    return sub, info


def same_context_causal_wins(D):
    """Mutual-selection test (§19) read off the retrieval matrix: for each
    ordered pair (i, j), i != j, count
        d(z_hat_i(S_i), z_y_i) < d(z_hat_i(S_j), z_y_i)   [row condition]
    and the mirrored column condition for j. 'mutual' requires both."""
    D = np.asarray(D, dtype=np.float64)
    b = D.shape[0]
    if b < 2:
        return {"feasible": False, "reason": "batch_size < 2"}
    rows = cols = mutual = total = 0
    for i in range(b):
        for j in range(b):
            if i == j:
                continue
            total += 1
            row_ok = D[i, i] < D[i, j]
            col_ok = D[j, j] < D[j, i]
            rows += row_ok
            cols += col_ok
            mutual += row_ok and col_ok
    return {
        "feasible": True,
        "n_pairs": total,
        "row_win_rate": float(rows / total),
        "col_win_rate": float(cols / total),
        "mutual_win_rate": float(mutual / total),
    }


# ---------------------------------------------------------------------------
# checkpoint-validity guards (validity-fix spec §6/§7/§10)
# ---------------------------------------------------------------------------

SMOKE_MAX_STEP = 10
SMOKE_MAX_TOTAL_STEPS = 20
SMOKE_MAX_TRAIN_SAMPLES = 64
ALLOWED_OBJECTIVES = ("jepa", "jepa_var", "jepa_vicreg", "jepa_vicreg2",
                      "jepa_barlow", "lejepa")


def validate_checkpoint_provenance(ckpt, path,
                                   allowed_objectives=ALLOWED_OBJECTIVES,
                                   allow_smoke_reason=None):
    """Metadata-based validity gate (Checks A + B). Returns the provenance dict
    required by spec §7; raises ValueError when the checkpoint cannot be shown
    to be genuinely trained — missing metadata, smoke-scale training signals, or
    an objective outside `allowed_objectives` (fail loudly rather than guessing).

    `allow_smoke_reason` is the explicit operator override: the run proceeds but
    the provenance is labeled reference-only (`is_smoke_checkpoint: true`,
    `genuinely_trained: false`) and must never be reported as a trained-model
    result.
    """
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(
            f"{path}: not a metadata checkpoint (raw state dict?) — refusing to "
            f"guess provenance; re-save via engine.save_checkpoint")
    step = ckpt.get("step")
    objective_name = ckpt.get("objective_name")
    cfg = ckpt.get("cfg") or {}
    if step is None:
        raise ValueError(f"{path}: missing 'step' metadata — cannot verify the "
                         f"checkpoint is genuinely trained")
    if objective_name is None:
        raise ValueError(f"{path}: missing 'objective_name' metadata — cannot "
                         f"verify which objective produced these weights")
    step = int(step)
    train_cfg = cfg.get("train") or {}
    data_cfg = cfg.get("data") or {}
    adapt_cfg = cfg.get("adaptive_training") or {}

    def _int(v):
        return int(v) if v is not None else None

    signals = {
        "step": step,
        "max_total_steps": _int(adapt_cfg.get("max_total_steps")),
        "max_train_samples": _int(data_cfg.get("max_train_samples")),
        "train_max_steps": _int(train_cfg.get("max_steps")),
        "batch_size": _int(train_cfg.get("batch_size")),
    }
    is_smoke = (
        signals["step"] <= SMOKE_MAX_STEP
        or (signals["max_total_steps"] is not None
            and signals["max_total_steps"] <= SMOKE_MAX_TOTAL_STEPS)
        or (signals["max_train_samples"] is not None
            and signals["max_train_samples"] <= SMOKE_MAX_TRAIN_SAMPLES)
    )
    if is_smoke and allow_smoke_reason is None:
        raise ValueError(
            f"{path}: SMOKE/near-init checkpoint refused "
            f"(signals={signals}; thresholds: step<={SMOKE_MAX_STEP} OR "
            f"max_total_steps<={SMOKE_MAX_TOTAL_STEPS} OR "
            f"max_train_samples<={SMOKE_MAX_TRAIN_SAMPLES}). A trained-model "
            f"result cannot be produced from this state. Pass "
            f"--allow-smoke-reason to label the run reference-only.")
    if objective_name not in allowed_objectives:
        raise ValueError(
            f"{path}: objective {objective_name!r} is not in the allowed set "
            f"{sorted(allowed_objectives)} — refusing to evaluate weights from "
            f"an unapproved objective")
    seed = train_cfg.get("seed")
    cfg_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    prov = {
        "path": str(path),
        "step": step,
        "epoch": int(ckpt.get("epoch") or 0),
        "objective_name": str(objective_name),
        "seed": _int(seed),
        "config_sha256_16": cfg_hash,
        "smoke_signals": signals,
        "is_smoke_checkpoint": bool(is_smoke),
        "genuinely_trained": not is_smoke,
    }
    if is_smoke:
        prov["override_reason"] = str(allow_smoke_reason)
        prov["run_status"] = "reference_only_not_a_trained_model_result"
    return prov


def assert_reference_modules_frozen(model):
    """Checks D + E: EMA target encoder and released spectrum encoder must have
    zero trainable parameters. Raises RuntimeError on any violation. Returns a
    record for the report (`released_spectrum` may be absent in unit-test
    stubs)."""
    trainable_ema = [n for n, p in model.ema.named_parameters() if p.requires_grad]
    if trainable_ema:
        raise RuntimeError(
            f"EMA target encoder has trainable parameters {trainable_ema[:5]}... "
            f"— target-leakage/protocol violation (Check D)")
    released = getattr(getattr(model, "spectrum_path", None), "released", None)
    record = {"ema_frozen": True,
              "ema_n_params": sum(1 for _ in model.ema.parameters())}
    if released is None:
        record["released_spectrum_frozen"] = None   # absent (stub/reference build)
    else:
        trainable_rel = [n for n, p in released.named_parameters()
                         if p.requires_grad]
        if trainable_rel:
            raise RuntimeError(
                f"released spectrum encoder has trainable parameters "
                f"{trainable_rel[:5]}... — frozen-component violation (Check E)")
        record["released_spectrum_frozen"] = True
        record["released_n_params"] = sum(1 for _ in released.parameters())
    return record


def check_target_shape(z_y_raw, n_tokens, dim):
    """Check C: z_y_raw must be (B, n_tokens, dim) — (B, 256, 384) at §11 sizes."""
    if z_y_raw.dim() != 3 or z_y_raw.shape[1] != n_tokens \
            or z_y_raw.shape[2] != dim:
        raise ValueError(
            f"z_y_raw shape {tuple(z_y_raw.shape)} != (B, {n_tokens}, {dim}) "
            f"(Check C)")


def check_same_target_across_conditions(out_r, out_n, out_s):
    """Check F: the target latent must be IDENTICAL across real/null/shuffled
    conditions (same G, same frozen EMA). Raises RuntimeError on any drift."""
    zr, zn, zs = out_r["z_y_raw"], out_n["z_y_raw"], out_s["z_y_raw"]
    if not (torch.equal(zr, zn) and torch.equal(zr, zs)):
        raise RuntimeError(
            "target latent differs across real/null/shuffled conditions "
            "(Check F) — protocol broken: the three conditions would no longer "
            "be comparable")


# ---------------------------------------------------------------------------
# evaluation core
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ratio(model, batches, masks, device, probe_payload=None,
                   expected_target_shape=None):
    """All conditions on IDENTICAL (G, S, M, z_y_raw). Returns the per-ratio report
    dict. batches: list of (G, S); masks: list of M aligned with batches.
    expected_target_shape: optional (n_tokens, dim) — Check C enforced per batch."""
    was_training = model.training
    model.eval()
    per_cond = {"real": {"cos": [], "l2_token": [], "l2_pooled": []},
                "null": {"cos": [], "l2_token": [], "l2_pooled": []},
                "shuffled": {"cos": [], "l2_token": [], "l2_pooled": []}}
    sens = {"real_vs_null": [], "real_vs_shuffled": []}
    probe_rows = {c: {"iou": [], "pixel_acc": []}
                  for c in ("real", "null", "shuffled")}
    probe = None
    mu = sd = None
    if probe_payload is not None:
        kind = probe_payload.get("kind", "linear")
        in_dim = len(probe_payload["input_stats"]["pooled"][0])
        out_dim = 64 * 64
        probe = build_probe(kind, in_dim, out_dim)
        probe.load_state_dict(probe_payload["state_dicts"]["pooled"])
        probe.eval()
        mu = torch.tensor(probe_payload["input_stats"]["pooled"][0])
        sd = torch.tensor(probe_payload["input_stats"]["pooled"][1])
    shuffled_feasible = True

    for (G, S), M in zip(batches, masks):
        G, S, M = G.to(device), S.to(device), M.to(device)
        b = G.shape[0]
        perm = shuffle_permutation(b)
        if b < 2:
            shuffled_feasible = False
        S_shuf = S[perm]

        out_r = model(G, S, M, goal_mode="real")
        out_n = model(G, S, M, goal_mode="null")
        out_s = model(G, S_shuf, M, goal_mode="real")

        check_same_target_across_conditions(out_r, out_n, out_s)
        if expected_target_shape is not None:
            check_target_shape(out_r["z_y_raw"], *expected_target_shape)

        z_y = out_r["z_y_raw"]
        mask = out_r["mask"]

        for name, out in (("real", out_r), ("null", out_n), ("shuffled", out_s)):
            d = per_sample_distances(out["z_hat"], z_y, mask)
            for k in per_cond[name]:
                per_cond[name][k].extend(d[k])

        sens["real_vs_null"].extend(
            (out_r["z_hat"] - out_n["z_hat"]).norm(dim=-1)[mask].cpu().tolist())
        sens["real_vs_shuffled"].extend(
            (out_r["z_hat"] - out_s["z_hat"]).norm(dim=-1)[mask].cpu().tolist())

        if probe is not None:
            occ_true = (G[:, 1] > 0).float().cpu().numpy().reshape(G.shape[0], -1)
            for name, out in (("real", out_r), ("null", out_n),
                              ("shuffled", out_s)):
                zp = (out["z_hat"].mean(dim=1).cpu() - mu) / sd
                logits = probe_logits_safe(probe, zp)
                pred = (torch.sigmoid(logits) >= 0.5).numpy().astype(np.float32)
                iou, _, _ = pr_iou(pred, occ_true)
                probe_rows[name]["iou"].extend(iou.tolist())
                probe_rows[name]["pixel_acc"].extend(
                    (pred == occ_true).mean(axis=-1).tolist())

    if was_training:
        model.train()

    real = per_cond["real"]
    report = {
        "n_samples": len(real["cos"]),
        "shuffled_condition_feasible": bool(shuffled_feasible),
        "distances": {c: {k: dist_stats(v) for k, v in per_cond[c].items()}
                      for c in per_cond},
        "margins": {
            "cos_null_minus_real": margin_stats(real["cos"], per_cond["null"]["cos"]),
            "cos_shuffle_minus_real": margin_stats(real["cos"],
                                                   per_cond["shuffled"]["cos"]),
            "l2_token_null_minus_real": margin_stats(real["l2_token"],
                                                     per_cond["null"]["l2_token"]),
            "l2_token_shuffle_minus_real": margin_stats(
                real["l2_token"], per_cond["shuffled"]["l2_token"]),
            "l2_pooled_null_minus_real": margin_stats(real["l2_pooled"],
                                                      per_cond["null"]["l2_pooled"]),
            "l2_pooled_shuffle_minus_real": margin_stats(
                real["l2_pooled"], per_cond["shuffled"]["l2_pooled"]),
        },
        "predictor_sensitivity": {
            "real_vs_null_l2": dist_stats(sens["real_vs_null"]),
            "real_vs_shuffled_l2": dist_stats(sens["real_vs_shuffled"]),
        },
    }
    if probe is not None:
        report["predicted_latent_probe"] = {
            c: {"iou": dist_stats(v["iou"]), "pixel_accuracy": dist_stats(v["pixel_acc"])}
            for c, v in probe_rows.items()
        }
    return report


def probe_logits_safe(probe, x):
    probe.eval()
    with torch.no_grad():
        return probe(x)


@torch.no_grad()
def retrieval_block(model, G, S, M, occ_flat, device, subset_info=None,
                    expected_target_shape=None):
    """Exact spectrum retrieval matrix (§16) on one fixed batch, optionally
    restricted to a geometry-aware subset (§17)."""
    model.eval()
    b = G.shape[0]
    idx = np.arange(b) if subset_info is None else np.asarray(subset_info["indices"])
    k = len(idx)
    if k < 2:
        return {"retrieval_matrix": retrieval_matrix_metrics(np.full((1, 1), np.nan)),
                "same_context_causal": same_context_causal_wins(
                    np.full((1, 1), np.nan))}
    G_sub, S_sub, M_sub = G[idx].to(device), S[idx].to(device), M[idx].to(device)
    z_y = model.ema(G_sub)                       # fixed target per row
    if expected_target_shape is not None:
        check_target_shape(z_y, *expected_target_shape)
    mask = (M_sub.view(k, -1) == 0)
    D = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        Gi = G_sub[i:i + 1].expand(k, -1, -1, -1).contiguous()
        Mi = M_sub[i:i + 1].expand(k, -1, -1).contiguous()
        out = model(Gi, S_sub, Mi, goal_mode="real")   # context i, spectra j
        d = per_sample_distances(out["z_hat"], z_y, mask)
        D[i, :] = d["cos"]
    return {
        "retrieval_matrix": retrieval_matrix_metrics(D),
        "same_context_causal": same_context_causal_wins(D),
        "distance_matrix_cos": D.tolist(),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnostic B: physics target selection (frozen checkpoint)")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="model checkpoint (weights only). Omit for released-init "
                        "reference.")
    p.add_argument("--ratios", nargs="+", type=float, default=[1.00, 0.75])
    p.add_argument("--n-samples", type=int, default=64,
                   help="fixed validation samples for the real/null/shuffle stats")
    p.add_argument("--retrieval-batch", type=int, default=24,
                   help="batch size B of the exact retrieval matrix (B^2 forwards)")
    p.add_argument("--subset-k", type=int, default=8,
                   help="anchors in the geometry-aware retrieval subset")
    p.add_argument("--min-hamming", type=int, default=300,
                   help="record whether the subset clears this pairwise Hamming bar")
    p.add_argument("--probe-file",
                   default="checkpoints/milestone_b/physics_validation/"
                           "latent_geometry_probe_weights.pt")
    p.add_argument("--no-probe", action="store_true",
                   help="skip the §18 predicted-latent probe block")
    p.add_argument("--mask-seed", type=int, default=12345)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-smoke-reason", default=None,
                   help="EXPLICIT operator override to evaluate a smoke/near-init "
                        "checkpoint anyway; the run is labeled reference-only in "
                        "the JSON and must never be reported as a trained-model "
                        "result")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir",
                   default="checkpoints/milestone_b/physics_validation")
    return p.parse_args()


def load_model(cfg, args, device):
    spec_w = REPO_ROOT / cfg["weights"]["spectrum"]
    metadit_w = REPO_ROOT / cfg["weights"]["metadit"]
    if args.checkpoint:
        model = build_model(cfg["model"], str(spec_w), device=device,
                            init_from_metadit=True, metadit_weights=str(metadit_w))
        ckpt_path = REPO_ROOT / args.checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # validity gate FIRST (Checks A+B): refuse smoke/unlabeled checkpoints
        # before any weights touch the model (validity-fix spec §6/§7/§10)
        provenance = validate_checkpoint_provenance(
            ckpt, args.checkpoint, allow_smoke_reason=args.allow_smoke_reason)
        from assembly import load_into_model
        load_into_model(model, ckpt["model"], device)
    else:
        model = build_deterministic_reference(
            lambda: build_model(cfg["model"], str(spec_w), device=device,
                                init_from_metadit=True,
                                metadit_weights=str(metadit_w)))
        provenance = {"checkpoint": None, "reference": "released_init_seed2026"}
    model.eval()
    model.ema.eval()
    return model, provenance


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    val_path = REPO_ROOT / cfg["data"]["val_split"]

    probe_payload = None
    if not args.no_probe:
        pf = REPO_ROOT / args.probe_file
        if pf.exists():
            probe_payload = torch.load(pf, map_location="cpu", weights_only=False)
            print(f"loaded frozen probe: {pf}")
        else:
            print(f"WARNING: probe file missing ({pf}); skipping §18 block")

    dataset = MetaDiTDataset(str(val_path))

    # Check C expectation: 16x16 token grid x model hidden dim (§11 sizes)
    expected_target_shape = (16 * 16, int(cfg["model"]["hidden"]))

    report = {"provenance": {}, "config": vars(args), "ratios": {}}

    for tag, label in (("trained", "checkpoint"), ("init_reference", "reference")):
        if tag == "trained" and not args.checkpoint:
            continue
        model, prov = load_model(cfg, args, device)
        prov["freeze_checks"] = assert_reference_modules_frozen(model)
        report["provenance"][tag] = prov
        if tag == "trained":
            # spec §7 literal key: top-level checkpoint_provenance block with
            # path/step/objective_name/seed/is_smoke_checkpoint/config hash
            report["checkpoint_provenance"] = prov

        for ratio in args.ratios:
            # fixed validation subset + fixed masks (repo convention: first n
            # samples, BlockMasker seed = mask_seed + round(ratio*1000)).
            # Masks are generated ONCE for all n samples and sliced for both the
            # real/null/shuffle stats AND the retrieval matrix, so every
            # intervention on a sample shares one identical mask.
            n = min(args.n_samples, len(dataset))
            G_all = torch.stack([dataset[i][0] for i in range(n)])
            S_all = torch.stack([dataset[i][1] for i in range(n)])
            masker = BlockMasker(placement="random", grid=16, min_side=3,
                                 k_range=(1, 4),
                                 seed=args.mask_seed + int(round(ratio * 1000)))
            M_all = masker.sample(G_all, ratio).cpu()
            rb = 8
            batches = [(G_all[i:i + rb], S_all[i:i + rb])
                       for i in range(0, n, rb)]
            masks = [M_all[i:i + rb] for i in range(0, n, rb)]

            print(f"[{tag}] ratio {ratio}: real/null/shuffle on {n} samples ...")
            res = evaluate_ratio(model, batches, masks, device,
                                 probe_payload=probe_payload,
                                 expected_target_shape=expected_target_shape)

            # retrieval matrix on the FIRST retrieval-batch samples (same fixed
            # set, same per-sample masks as the stats phase)
            nb = min(args.retrieval_batch, n)
            Gm, Sm, Mm = G_all[:nb], S_all[:nb], M_all[:nb]
            occ_flat = (Gm[:, 1] > 0).reshape(nb, -1).numpy().astype(np.uint8)
            print(f"[{tag}] ratio {ratio}: retrieval matrix B={nb} "
                  f"({nb ** 2} forwards) ...")
            res["retrieval_full"] = retrieval_block(model, Gm, Sm, Mm, occ_flat,
                                                    device,
                                                    expected_target_shape=expected_target_shape)
            sub, sub_info = geometry_aware_subset(occ_flat, args.subset_k,
                                                  min_hamming=args.min_hamming)
            res["geometry_aware_subset_selection"] = sub_info
            if sub_info.get("feasible"):
                print(f"[{tag}] ratio {ratio}: geometry-aware subset "
                      f"k={len(sub)} min_hamming={sub_info['min_pairwise_hamming']}")
                res["retrieval_geometry_aware"] = retrieval_block(
                    model, Gm, Sm, Mm, occ_flat, device, subset_info=sub_info,
                    expected_target_shape=expected_target_shape)
            report["ratios"][f"{tag}_r{ratio:.2f}"] = res

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "physics_target_selection.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n-> {path}")
    for key, res in report["ratios"].items():
        print(f"\n=== {key} ===")
        print(json.dumps(res["margins"], indent=2, default=float))


if __name__ == "__main__":
    main()
