#!/usr/bin/env python3
"""Diagnostic A — target-latent spatial geometry probe (latent-selection spec §7-§11).

Question Q1: does the EMA target latent z_y_raw contain information about the
INDEPENDENT spatial occupancy/layout, beyond the three scalar parameters
(l_lattice, h_atom, r_atom)?

Inference-side representation evaluation ONLY:
  - the EMA target encoder stays frozen (no training of any model component);
  - a linear probe (optionally a tiny 2-layer MLP) is fit on FROZEN features;
  - probe-validation / probe-test splits are disjoint; the reported scientific
    number comes from held-out probe-test data only;
  - a (l,h,r)-scalar control probe runs under the IDENTICAL protocol — if the
    latent probe does not substantially beat the scalar control, the latent adds
    no spatial information beyond the parameters.

Probe input modes:
  pooled : mean over the 256 target tokens -> 384-d -> whole 64x64 occupancy
  token  : per-token 384-d embedding -> shared head -> its own 4x4 patch (16 px)
  scalars: (l,h,r) control, same protocol/architecture as `pooled`

Pairwise diagnostic (§11): geometry-defined pairs ONLY (Hamming distance on
occupancy; never selected via latents or physics) -> correlation/rank statistics
between geometry Hamming distance and target-latent L2 distance, with the
scalar-parameter distance as control.

Run:
  python scripts/diagnostics/latent_geometry_probe.py \
      --config configs/milestone_b.yaml \
      --checkpoint checkpoints/milestone_b/minimal_jepa_vicreg_smoke_latest.pt

Outputs:
  checkpoints/milestone_b/physics_validation/latent_geometry_probe.json
  checkpoints/milestone_b/physics_validation/latent_geometry_probe_weights.pt
  (weights reused by scripts/diagnostics/physics_target_selection.py, spec §18)
"""

import argparse
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
from torch import nn

from assembly import build_model
from data.dataset import MetaDiTDataset
from train.engine import build_deterministic_reference

PIXEL_GRID = 16
PATCH = 4


# ---------------------------------------------------------------------------
# probe models / split discipline
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class MLPProbe(nn.Module):
    """Tiny 2-layer probe (spec §8 'optionally'), used when linear is insufficient."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_probe(kind, in_dim, out_dim, hidden_dim=256):
    if kind == "linear":
        return LinearProbe(in_dim, out_dim)
    if kind == "mlp":
        return MLPProbe(in_dim, hidden_dim, out_dim)
    raise ValueError(f"unknown probe kind {kind!r}")


def split_indices(n, seed=0, fractions=(0.6, 0.2, 0.2)):
    """Deterministic disjoint probe-train/probe-val/probe-test index arrays."""
    assert abs(sum(fractions) - 1.0) < 1e-9
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def fit_input_stats(x_train):
    """Per-feature mean/std from probe-TRAIN only (no test leakage)."""
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp_min(1e-8)
    return mean, std


def train_probe(probe, x_train, y_train, x_val, y_val, epochs=300, lr=1e-3,
                batch_size=256, device="cpu", log_every=0):
    """Fit one probe with BCE + best-probe-val-state restore. Returns best state dict
    and the best probe-val BCE. The probe is selected on probe-val, reported on
    probe-test — never fitted on the reporting split."""
    probe.to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    ds = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                         generator=torch.Generator().manual_seed(0))
    best_val, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(epochs):
        probe.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = probe(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            val_loss = F.binary_cross_entropy_with_logits(
                probe(x_val.to(device)), y_val.to(device)).item()
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in probe.state_dict().items()}
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:4d} val_bce {val_loss:.6f}"
                  f"{' *' if epoch == best_epoch else ''}")
    probe.load_state_dict(best_state)
    return best_state, best_val, best_epoch


@torch.no_grad()
def probe_logits(probe, x, device="cpu"):
    probe.eval()
    return probe(x.to(device)).cpu()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def dist_stats(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {k: float("nan") for k in ("mean", "median", "std", "p10", "p25",
                                          "p75", "p90")}
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std()),
        "p10": float(np.percentile(v, 10)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
        "p90": float(np.percentile(v, 90)),
    }


def pr_iou(pred, target, eps=1e-8):
    """pred/target: (..., H*W) binarized. Returns per-sample (iou, precision, recall)."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = (pred & target).sum(axis=-1)
    union = (pred | target).sum(axis=-1)
    psum = pred.sum(axis=-1)
    tsum = target.sum(axis=-1)
    iou = inter / np.maximum(union, eps)
    prec = inter / np.maximum(psum, eps)
    rec = inter / np.maximum(tsum, eps)
    return iou, prec, rec


def evaluate_binary_predictions(logits, target, threshold=0.5):
    """logits/target: (N, 4096). Per-sample IoU/precision/recall + distribution stats."""
    prob = torch.sigmoid(torch.from_numpy(logits)).numpy() if isinstance(
        logits, np.ndarray) else torch.sigmoid(logits).numpy()
    pred = (prob >= threshold).astype(np.float32)
    iou, prec, rec = pr_iou(pred, target)
    return {
        "iou": dist_stats(iou),
        "precision": dist_stats(prec),
        "recall": dist_stats(rec),
        "pixel_accuracy": dist_stats((pred == target).mean(axis=-1)),
        "n_samples": int(target.shape[0]),
    }


# ---------------------------------------------------------------------------
# feature extraction (frozen encoder)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_latents(model, dataset, n_samples, batch_size, device):
    """Frozen EMA target latents + occupancy targets + raw params for the first
    n_samples of the dataset (fixed order, repo fixed-validation convention)."""
    n = min(n_samples, len(dataset))
    pooled, tokens, occ, params = [], [], [], []
    done = 0
    while done < n:
        take = min(batch_size, n - done)
        G = torch.stack([dataset[i][0] for i in range(done, done + take)]).to(device)
        z_y = model.ema(G)                       # (B, 256, 384), frozen
        tokens.append(z_y.cpu())
        pooled.append(z_y.mean(dim=1).cpu())     # (B, 384)
        occ.append((G[:, 1] > 0).float().cpu())  # h_atom channel nonzero <=> occupied
        for i in range(done, done + take):
            p = dataset.data["parameter"][int(dataset.indices[i])]
            params.append(torch.tensor([float(p[0]), float(p[1]), float(p[2])]))
        done += take
    return (torch.cat(pooled), torch.cat(tokens), torch.cat(occ),
            torch.stack(params))


def occupancy_to_patches(occ):
    """(N, 64, 64) -> (N, 256, 16) per-token 4x4 patch occupancy (token-major,
    matching the geometry encoder's row-major token order)."""
    n = occ.shape[0]
    o = occ.reshape(n, PIXEL_GRID, PATCH, PIXEL_GRID, PATCH)
    return o.permute(0, 1, 3, 2, 4).reshape(n, PIXEL_GRID * PIXEL_GRID, PATCH * PATCH)


def patches_to_occupancy(patches):
    """Inverse of occupancy_to_patches: (N, 256, 16) -> (N, 4096)."""
    n = patches.shape[0]
    p = patches.reshape(n, PIXEL_GRID, PIXEL_GRID, PATCH, PATCH)
    return p.permute(0, 1, 3, 2, 4).reshape(n, 64, 64).reshape(n, -1)


# ---------------------------------------------------------------------------
# pairwise geometry-vs-latent diagnostic (§11) — geometry-only pair selection
# ---------------------------------------------------------------------------

def pairwise_geometry_latent(occ_flat, pooled_latent, params, n_pairs=4000,
                             seed=0, n_buckets=5):
    """Pairs are chosen by GEOMETRY criteria only (occupancy Hamming distance),
    never by latent or physics quantities. Returns Spearman rank correlation of
    Hamming vs latent L2, the scalar-parameter-distance control, and per-bucket
    means. occ_flat: (N, 4096) binary; pooled_latent: (N, D); params: (N, 3)."""
    rng = np.random.RandomState(seed)
    n = occ_flat.shape[0]
    bits = occ_flat.astype(np.uint8)
    packed = np.packbits(bits, axis=1)
    popcount = np.zeros(256, dtype=np.uint16)
    for i in range(256):
        popcount[i] = bin(i).count("1")

    def hamming(a, b):
        return int(popcount[packed[a] ^ packed[b]].sum())

    # z-score params so the control distance is scale-free
    pm = params.mean(dim=0, keepdim=True)
    ps = params.std(dim=0, keepdim=True).clamp_min(1e-8)
    pz = (params - pm) / ps

    idx_a = rng.randint(0, n, size=n_pairs)
    idx_b = rng.randint(0, n, size=n_pairs)
    keep = idx_a != idx_b
    idx_a, idx_b = idx_a[keep], idx_b[keep]

    ham = np.array([hamming(a, b) for a, b in zip(idx_a, idx_b)], dtype=np.float64)
    lat_l2 = torch.cdist(pooled_latent[idx_a], pooled_latent[idx_b]).diagonal().numpy()
    par_l2 = torch.cdist(pz[idx_a], pz[idx_b]).diagonal().numpy()

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        rx -= rx.mean()
        ry -= ry.mean()
        denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        return float((rx * ry).sum() / denom) if denom > 0 else float("nan")

    edges = np.quantile(ham, np.linspace(0, 1, n_buckets + 1))
    edges[-1] += 1e-6
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ham >= lo) & (ham < hi)
        if m.sum() == 0:
            continue
        buckets.append({
            "hamming_range": [float(lo), float(hi)],
            "n_pairs": int(m.sum()),
            "latent_l2_mean": float(lat_l2[m].mean()),
            "param_l2_mean": float(par_l2[m].mean()),
        })
    return {
        "spearman_hamming_vs_latent_l2": spearman(ham, lat_l2),
        "spearman_hamming_vs_param_l2": spearman(ham, par_l2),
        "pearson_hamming_vs_latent_l2": float(np.corrcoef(ham, lat_l2)[0, 1]),
        "pearson_hamming_vs_param_l2": float(np.corrcoef(ham, par_l2)[0, 1]),
        "buckets": buckets,
        "n_pairs": int(ham.size),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Diagnostic A: target-latent spatial probe")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="model checkpoint (model weights only; objective not needed). "
                        "Omit to evaluate the released-init reference build.")
    p.add_argument("--n-samples", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--probe-kind", choices=["linear", "mlp"], default="linear")
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--n-pairs", type=int, default=4000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir",
                   default="checkpoints/milestone_b/physics_validation")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    spec_w = REPO_ROOT / cfg["weights"]["spectrum"]
    metadit_w = REPO_ROOT / cfg["weights"]["metadit"]
    val_path = REPO_ROOT / cfg["data"]["val_split"]

    if args.checkpoint:
        model = build_model(cfg["model"], str(spec_w), device=device,
                            init_from_metadit=True, metadit_weights=str(metadit_w))
        ckpt = torch.load(REPO_ROOT / args.checkpoint, map_location="cpu",
                          weights_only=False)
        from assembly import load_into_model
        load_into_model(model, ckpt["model"], device)
        provenance = {"checkpoint": str(args.checkpoint),
                      "checkpoint_step": ckpt.get("step"),
                      "objective_name": ckpt.get("objective_name")}
    else:
        model = build_deterministic_reference(
            lambda: build_model(cfg["model"], str(spec_w), device=device,
                                init_from_metadit=True,
                                metadit_weights=str(metadit_w)))
        provenance = {"checkpoint": None, "reference": "released_init_seed2026"}
    model.eval()
    model.ema.eval()

    dataset = MetaDiTDataset(str(val_path))
    print(f"collecting {args.n_samples} frozen target latents ...")
    pooled, tokens, occ, params = collect_latents(
        model, dataset, args.n_samples, args.batch_size, device)
    occ_flat = occ.reshape(occ.shape[0], -1)                    # (N, 4096)
    n = pooled.shape[0]

    tr, va, te = split_indices(n, seed=args.split_seed)
    print(f"probe splits: train={len(tr)} val={len(va)} test={len(te)}")

    report = {
        "provenance": provenance,
        "config": {"n_samples": int(n), "probe_kind": args.probe_kind,
                   "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
                   "split_seed": args.split_seed,
                   "fractions": [len(tr), len(va), len(te)],
                   "device": str(device)},
        "probes": {},
    }
    weights_payload = {"input_stats": {}, "state_dicts": {}, "kind": args.probe_kind}

    # ---- pooled probe: z_y_raw (mean-pooled) -> 64x64 occupancy ----
    x_pooled = pooled
    y_img = occ_flat
    mu, sd = fit_input_stats(x_pooled[tr])
    zp = (x_pooled - mu) / sd
    probe = build_probe(args.probe_kind, x_pooled.shape[1], y_img.shape[1],
                        args.mlp_hidden)
    print("training POOLED probe (z_y_raw -> occupancy) ...")
    state, bce, ep = train_probe(probe, zp[tr], y_img[tr], zp[va], y_img[va],
                                 epochs=args.epochs, lr=args.lr, device=device)
    test_logits = probe_logits(probe, zp[te], device).numpy()
    pooled_metrics = evaluate_binary_predictions(test_logits, y_img[te].numpy())
    pooled_metrics["best_val_bce"] = float(bce)
    pooled_metrics["best_epoch"] = int(ep)
    report["probes"]["pooled_z_y_raw"] = pooled_metrics
    weights_payload["input_stats"]["pooled"] = [mu.tolist(), sd.tolist()]
    weights_payload["state_dicts"]["pooled"] = state

    # ---- token probe: per-token embedding -> its own 4x4 patch ----
    tok = tokens.reshape(-1, tokens.shape[-1])                  # (N*256, 384)
    pat = occupancy_to_patches(occ).reshape(-1, PATCH * PATCH)  # (N*256, 16)
    stride = 8  # subsample token rows for tractability (same rows for all splits)
    sel = np.arange(0, tok.shape[0], stride)
    tok_sel = tok[sel]
    pat_sel = pat[sel]
    # split by SAMPLE id so no geometry leaks across splits through its tokens
    sample_id = sel // (PIXEL_GRID * PIXEL_GRID)
    sid_set = set(sample_id.tolist())
    tr_m = np.array([s in set(tr.tolist()) for s in sample_id])
    va_m = np.array([s in set(va.tolist()) for s in sample_id])
    te_m = np.array([s in set(te.tolist()) for s in sample_id])
    assert tr_m.sum() and va_m.sum() and te_m.sum()
    mu_t, sd_t = fit_input_stats(tok_sel[tr_m])
    zt = (tok_sel - mu_t) / sd_t
    tprobe = build_probe(args.probe_kind, tok_sel.shape[1], pat_sel.shape[1],
                         args.mlp_hidden)
    print("training TOKEN probe (per-token -> own patch) ...")
    tstate, tbce, tep = train_probe(tprobe, zt[tr_m], pat_sel[tr_m],
                                    zt[va_m], pat_sel[va_m],
                                    epochs=args.epochs, lr=args.lr, device=device)
    tlogits = probe_logits(tprobe, zt[te_m], device).numpy()
    tpred = (torch.sigmoid(torch.from_numpy(tlogits)).numpy() >= 0.5).astype(np.float32)
    ttar = pat_sel[te_m].numpy()
    te_ids = sample_id[te_m]
    uniq = np.unique(te_ids)
    ious, precs, recs = [], [], []
    for s in uniq:
        m = te_ids == s
        iou, prec, rec = pr_iou(tpred[m].reshape(-1), ttar[m].reshape(-1))
        ious.append(float(iou)), precs.append(float(prec)), recs.append(float(rec))
    token_metrics = {
        "iou": dist_stats(ious), "precision": dist_stats(precs),
        "recall": dist_stats(recs),
        "best_val_bce": float(tbce), "best_epoch": int(tep),
        "token_row_stride": stride, "n_test_geometries": int(uniq.size),
    }
    report["probes"]["token_z_y_raw"] = token_metrics
    weights_payload["input_stats"]["token"] = [mu_t.tolist(), sd_t.tolist()]
    weights_payload["state_dicts"]["token"] = tstate

    # ---- scalar control probe: (l,h,r) -> occupancy, IDENTICAL protocol ----
    xs = params
    mu_s, sd_s = fit_input_stats(xs[tr])
    zs = (xs - mu_s) / sd_s
    sprobe = build_probe(args.probe_kind, xs.shape[1], y_img.shape[1],
                         args.mlp_hidden)
    print("training SCALAR CONTROL probe ((l,h,r) -> occupancy) ...")
    sstate, sbce, sep = train_probe(sprobe, zs[tr], y_img[tr], zs[va], y_img[va],
                                    epochs=args.epochs, lr=args.lr, device=device)
    slogits = probe_logits(sprobe, zs[te], device).numpy()
    scalar_metrics = evaluate_binary_predictions(slogits, y_img[te].numpy())
    scalar_metrics["best_val_bce"] = float(sbce)
    scalar_metrics["best_epoch"] = int(sep)
    report["probes"]["scalar_control_lhr"] = scalar_metrics

    # ---- key comparison (§10) ----
    report["key_comparison"] = {
        "latent_iou_median": pooled_metrics["iou"]["median"],
        "scalar_control_iou_median": scalar_metrics["iou"]["median"],
        "latent_beats_scalar_control":
            pooled_metrics["iou"]["median"] > scalar_metrics["iou"]["median"],
        "token_probe_iou_median": token_metrics["iou"]["median"],
    }

    # ---- pairwise geometry-vs-latent (§11) ----
    print("pairwise geometry-vs-latent analysis ...")
    report["pairwise_geometry_vs_latent"] = pairwise_geometry_latent(
        occ_flat.numpy(), pooled, params, n_pairs=args.n_pairs, seed=args.seed)

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "latent_geometry_probe.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    torch.save(weights_payload, out_dir / "latent_geometry_probe_weights.pt")

    print(f"\n-> {json_path}")
    print("\nKEY COMPARISON")
    print(json.dumps(report["key_comparison"], indent=2))


if __name__ == "__main__":
    main()
