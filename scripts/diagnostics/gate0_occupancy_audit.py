#!/usr/bin/env python3
"""Gate 0 — Occupancy determinism audit (PHYSICS_GEOMETRY_VALIDATION_V2 spec §8).

Tests whether the binary 64x64 occupancy mask is a deterministic rasterization of the
three scalar physical parameters (l_lattice, h_atom, r_atom), WITHOUT training the JEPA.

    G0-A  occupied pixel count vs r_atom^2  (deterministic-rasterization diagnostic)
    G0-B  matched-parameter occupancy identity (exact (l,h,r) buckets -> IoU/Hamming)
    G0-C  tiny diagnostic baseline (l,h,r) -> 64x64 occupancy (no attention, no JEPA)
    G0-D  parameter visibility under the ACTUAL BlockMasker semantics, measured
          directly from tensors (never inferred from model performance)

Reads the raw .mat files (train/val/test) exactly as `src/data/dataset.py` does and
does not mutate them. Machine-readable results -> gate0_occupancy_audit.json.

Run:  python scripts/diagnostics/gate0_occupancy_audit.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import io
from scipy.stats import linregress
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.mask import BlockMasker  # noqa: E402

# ---------------------------------------------------------------------------
# Exact parameter quantization (data is natively quantized at 0.01 per axis:
# l_lattice in [2.5, 3.0] (51 steps), h_atom in [0.5, 1.0] (51 steps),
# r_atom in [3.5, 5.0] (151 steps)). Bucket keys are exact float values rounded
# to 2 decimals — the documented quantization of the released dataset.
# ---------------------------------------------------------------------------
PARAM_NDIGITS = 2

# Popcount lookup for packed-byte Hamming distances.
_POPCOUNT = np.zeros(256, dtype=np.uint16)
for _i in range(256):
    _POPCOUNT[_i] = bin(_i).count("1")


def quantize_params(params):
    """(N,3) float -> (N,3) exact 2-decimal bucket keys."""
    return np.round(params, PARAM_NDIGITS)


def pack_patterns(patterns):
    """(64,64,N) int8 binary masks -> (N, 512) uint8 packed bits (row-major).

    Each of the 4096 pixels becomes one bit; 4096 / 8 = 512 packed bytes per mask.
    """
    flat = patterns.astype(np.uint8).reshape(64 * 64, -1).T   # (N, 4096)
    return np.packbits(flat, axis=1)                          # (N, 512)


def popcount_hamming(a, b):
    """Packed (P,512) uint8 pairs -> (P,) Hamming distances."""
    return _POPCOUNT[a ^ b].sum(axis=1).astype(np.int64)


def popcount_intersection(a, b):
    return _POPCOUNT[a & b].sum(axis=1).astype(np.int64)


def popcount_union(a, b):
    return _POPCOUNT[a | b].sum(axis=1).astype(np.int64)


def mask_iou(a, b):
    inter = popcount_intersection(a, b).astype(np.float64)
    union = popcount_union(a, b).astype(np.float64)
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def load_mat(path):
    return io.loadmat(str(path))


def _stat_block(x):
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


# ---------------------------------------------------------------------------
# G0-C model: tiny diagnostic baseline (spec §8 G0-C). Small MLP, no attention,
# no JEPA, no physics conditioning. Measures whether the pixel representation is
# redundant given the scalar parameters.
# ---------------------------------------------------------------------------
class OccupancyMLP(nn.Module):
    def __init__(self, in_dim=3, hidden=256, out_dim=64 * 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def run_g0c(train_params, train_pat, val_params, val_pat, n_train, n_val,
            seed, epochs, batch_size, lr, device):
    """Train (l,h,r) -> occupancy and report IoU / precision / recall on held-out."""
    rng = np.random.RandomState(seed)
    ti = rng.choice(len(train_params), size=n_train, replace=False)
    vi = rng.choice(len(val_params), size=n_val, replace=False)

    Xtr = torch.from_numpy(train_params[ti].astype(np.float32))
    Ytr = torch.from_numpy(train_pat[:, :, ti].astype(np.float32)).permute(2, 0, 1)
    Xva = torch.from_numpy(val_params[vi].astype(np.float32))
    Yva = torch.from_numpy(val_pat[:, :, vi].astype(np.float32)).permute(2, 0, 1)

    model = OccupancyMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()

    n = len(Xtr)
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = Xtr[idx].to(device)
            y = Ytr[idx].to(device)
            opt.zero_grad()
            loss = lossf(model(x).reshape(-1), y.reshape(-1))
            loss.backward()
            opt.step()

    def metrics(X, Y):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                x = X[i:i + batch_size].to(device)
                preds.append(torch.sigmoid(model(x)))
        pred = torch.cat(preds, 0).cpu().numpy() > 0.5
        truth = Y.cpu().numpy().reshape(len(pred), -1) > 0.5
        inter = (pred & truth).sum(axis=1).astype(np.float64)
        union = (pred | truth).sum(axis=1).astype(np.float64)
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        tp = (pred & truth).sum(axis=1).astype(np.float64)
        fp = (pred & ~truth).sum(axis=1).astype(np.float64)
        fn = (~pred & truth).sum(axis=1).astype(np.float64)
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        return {
            "n": int(len(X)),
            "iou": _stat_block(iou),
            "precision": _stat_block(prec),
            "recall": _stat_block(rec),
        }

    result = {"train": metrics(Xtr, Ytr), "val": metrics(Xva, Yva)}
    return result


# ---------------------------------------------------------------------------
# G0-D: parameter visibility under the actual BlockMasker (spec §8 G0-D).
# ---------------------------------------------------------------------------
def run_g0d(patterns, params, n_samples, ratios, mask_seed, min_side, k_range):
    """Measure observability of l/h/r from masked geometry tensors directly.

    l_lattice is painted over the entire image (grid[2] = l/3 everywhere), so a
    single visible pixel recovers it exactly. h_atom / r_atom are painted only on
    occupied pixels (grid[1] = h, grid[0] = r/5 where occ), so they need at least
    one visible occupied pixel.
    """
    subj = np.arange(n_samples)
    results = {}
    for ratio in ratios:
        masker = BlockMasker(
            placement="random",
            grid=16,
            min_side=min_side,
            k_range=tuple(k_range),
            seed=mask_seed + int(round(ratio * 1000)),
        )
        l_obs = []
        h_obs = []
        r_obs = []
        vis_occ = []
        vis_total = []
        per_sample = []
        for j in subj:
            pat = patterns[:, :, j]
            occ = pat == 1
            G = torch.zeros(3, 64, 64)
            G[0][torch.from_numpy(occ)] = params[j, 2] / 5.0
            G[1][torch.from_numpy(occ)] = params[j, 1]
            G[2] = params[j, 0] / 3.0
            M = masker.sample(G.unsqueeze(0), ratio)[0]
            up = M.repeat_interleave(4, dim=0).repeat_interleave(4, dim=1).numpy()
            vis = up > 0.5
            v_occ = int((vis & occ).sum())
            l_flag = bool(vis.any())
            h_flag = bool(v_occ > 0)
            r_flag = h_flag          # same pixels carry h and r (both on occ)
            l_obs.append(l_flag)
            h_obs.append(h_flag)
            r_obs.append(r_flag)
            vis_occ.append(v_occ)
            vis_total.append(int(vis.sum()))
            per_sample.append({
                "sample": int(j),
                "l_observed": l_flag,
                "h_observed": h_flag,
                "r_observed": r_flag,
                "visible_occupied_pixel_count": v_occ,
                "total_visible_pixel_count": int(vis.sum()),
            })
        results[str(ratio)] = {
            "n": int(len(subj)),
            "fraction_l_observed": float(np.mean(l_obs)),
            "fraction_h_observed": float(np.mean(h_obs)),
            "fraction_r_observed": float(np.mean(r_obs)),
            "visible_occupied_pixel_count": _stat_block(np.array(vis_occ)),
            "total_visible_pixel_count": _stat_block(np.array(vis_total)),
            "per_sample": per_sample,
        }
    return results


def main():
    p = argparse.ArgumentParser(description="Gate 0 occupancy determinism audit")
    p.add_argument("--train", default="data/metadit/split_data/train_set.mat")
    p.add_argument("--val", default="data/metadit/split_data/val_set.mat")
    p.add_argument("--test", default="data/metadit/split_data/test_set.mat")
    p.add_argument("--max-samples", type=int, default=0,
                   help="0 = use full splits (deterministic subsample otherwise)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--g0c-train", type=int, default=8192)
    p.add_argument("--g0c-val", type=int, default=2048)
    p.add_argument("--g0c-epochs", type=int, default=30)
    p.add_argument("--g0c-batch", type=int, default=512)
    p.add_argument("--g0c-lr", type=float, default=1e-3)
    p.add_argument("--g0d-samples", type=int, default=256)
    p.add_argument("--mask-seed", type=int, default=12345,
                   help="same convention as scripts/eval/physics_task_utility_mask_sweep.py")
    p.add_argument("--g0d-ratios", nargs="+", type=float,
                   default=[0.25, 0.50, 0.75, 1.00])
    p.add_argument("--min-side", type=int, default=3)
    p.add_argument("--k-range", nargs=2, type=int, default=[1, 4])
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir",
                   default="checkpoints/milestone_b/physics_validation")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    report = {"gate": 0, "seed": args.seed, "param_ndigits": PARAM_NDIGITS,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ---- load data (same semantics as src/data/dataset.py, no mutation) ----
    mats = {}
    for name, path in [("train", args.train), ("val", args.val), ("test", args.test)]:
        if Path(path).exists():
            mats[name] = load_mat(path)
        else:
            print(f"WARN: {name} split not found at {path} — skipped")
    if "train" not in mats:
        raise FileNotFoundError("train split is required")

    all_params = [m["parameter"] for m in mats.values()]
    all_pat = [m["pattern"] for m in mats.values()]
    all_real = [m["real"] for m in mats.values()]
    all_imag = [m["imag"] for m in mats.values()]

    if args.max_samples:
        rng = np.random.RandomState(args.seed)
        for k in range(len(all_params)):
            n = len(all_params[k])
            if n > args.max_samples:
                idx = rng.choice(n, args.max_samples, replace=False)
                all_params[k] = all_params[k][idx]
                all_pat[k] = all_pat[k][:, :, idx]
                all_real[k] = all_real[k][idx]
                all_imag[k] = all_imag[k][idx]

    params = np.concatenate(all_params, axis=0)     # (N, 3) [l_lattice, h_atom, r_atom]
    pat = np.concatenate(all_pat, axis=-1)          # (64, 64, N) int8
    report["data"] = {
        "splits": {k: int(m["parameter"].shape[0]) for k, m in mats.items()},
        "total_samples": int(params.shape[0]),
        "parameter_ranges": {
            "l_lattice": {"min": float(params[:, 0].min()), "max": float(params[:, 0].max())},
            "h_atom": {"min": float(params[:, 1].min()), "max": float(params[:, 1].max())},
            "r_atom": {"min": float(params[:, 2].min()), "max": float(params[:, 2].max())},
        },
        "unique_parameter_values": {
            "l_lattice": int(np.unique(params[:, 0]).size),
            "h_atom": int(np.unique(params[:, 1]).size),
            "r_atom": int(np.unique(params[:, 2]).size),
        },
        "pattern_unique_values": [int(v) for v in np.unique(pat)],
    }

    occ_count = pat.sum(axis=(0, 1)).astype(np.float64)   # (N,) occupied pixels
    r = params[:, 2]
    r2 = r ** 2

    # ---- G0-A: occupied area vs radius ----
    print("\nG0-A: occupied pixel count vs r_atom^2 ...")
    slope, intercept, rv, pv, se = linregress(r2, occ_count)
    pred = slope * r2 + intercept
    resid = occ_count - pred
    g0a = {
        "fit": {
            "formula": "occupied_pixel_count ~= a*r_atom^2 + b",
            "a": float(slope),
            "b": float(intercept),
            "R2": float(rv ** 2),
            "RMSE": float(np.sqrt(np.mean(resid ** 2))),
            "mean_occupied_count": float(occ_count.mean()),
        },
        "residual_distribution": _stat_block(resid),
        "correlations": {
            "occupied_count_vs_l_lattice": float(np.corrcoef(params[:, 0], occ_count)[0, 1]),
            "occupied_count_vs_h_atom": float(np.corrcoef(params[:, 1], occ_count)[0, 1]),
            "occupied_count_vs_r_atom": float(np.corrcoef(r, occ_count)[0, 1]),
            "occupied_count_vs_r_atom_sq": float(np.corrcoef(r2, occ_count)[0, 1]),
        },
    }
    report["G0-A"] = g0a
    print(json.dumps(g0a, indent=2))

    # ---- G0-B: matched-parameter occupancy identity ----
    print("\nG0-B: exact (l,h,r) buckets -> occupancy identity ...")
    keys = quantize_params(params)
    ukeys, inv = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inv)
    packed = pack_patterns(pat)

    multi = np.where(counts >= 2)[0]
    total_pairs = sum(int(c * (c - 1) // 2) for c in counts)
    multi_pairs = sum(int(c * (c - 1) // 2) for c in counts[multi]) if len(multi) else 0

    ious = np.empty(0)
    hams = np.empty(0)
    distinct_per_bucket = {}
    for b in multi:
        members = np.where(inv == b)[0]
        n = len(members)
        distinct_per_bucket[int(b)] = len(np.unique(packed[members], axis=0))
        if n == 2:
            a, c = packed[members[0]], packed[members[1]]
            ious = np.append(ious, mask_iou(a[None], c[None]))
            hams = np.append(hams, popcount_hamming(a[None], c[None]))
        else:
            for i in range(n):
                a = packed[members[i]]
                rest = packed[members[i + 1:]]
                ious = np.append(ious, mask_iou(a[None], rest))
                hams = np.append(hams, popcount_hamming(a[None], rest))

    near_identical = ious >= 0.95
    fully_identical = ious >= 0.999999

    # fraction of multi-member buckets whose members are ALL identical
    identical_buckets = sum(1 for d in distinct_per_bucket.values() if d == 1)
    g0b = {
        "bucket_definition": (
            "exact (l_lattice, h_atom, r_atom) rounded to 2 decimals "
            "(the data's native 0.01 quantization)"
        ),
        "matching_criterion": "pairs compared only within buckets with >= 2 members; "
                              "near-identical defined as IoU >= 0.95",
        "n_buckets": int(len(ukeys)),
        "n_single_member_buckets": int((counts == 1).sum()),
        "n_multi_member_buckets": int(len(multi)),
        "bucket_size": _stat_block(counts[counts >= 1]),
        "pairwise_comparisons": {
            "total_possible_pairs": int(total_pairs),
            "multi_member_pairs": int(multi_pairs),
            "pairs_computed": int(len(ious)),
        },
        "iou": _stat_block(ious),
        "hamming": _stat_block(hams),
        "fraction_near_identical_iou_ge_0.95": float(near_identical.mean()),
        "fraction_fully_identical": float(fully_identical.mean()),
        "buckets_all_identical": {
            "n": int(identical_buckets),
            "of": int(len(multi)),
            "fraction": float(identical_buckets / max(1, len(multi))),
        },
    }
    report["G0-B"] = g0b
    print(json.dumps(g0b, indent=2))
    del packed

    # ---- G0-C: tiny diagnostic baseline ----
    print("\nG0-C: tiny (l,h,r) -> occupancy MLP ...")
    device = torch.device(args.device)
    g0c = run_g0c(
        mats["train"]["parameter"], mats["train"]["pattern"],
        mats["val"]["parameter"], mats["val"]["pattern"],
        n_train=args.g0c_train, n_val=args.g0c_val,
        seed=args.seed, epochs=args.g0c_epochs,
        batch_size=args.g0c_batch, lr=args.g0c_lr, device=device,
    )
    report["G0-C"] = g0c
    print(json.dumps(g0c, indent=2))

    # ---- G0-D: parameter visibility under BlockMasker ----
    print("\nG0-D: parameter visibility under actual BlockMasker ...")
    g0d = run_g0d(
        mats["val"]["pattern"], mats["val"]["parameter"],
        n_samples=args.g0d_samples, ratios=args.g0d_ratios,
        mask_seed=args.mask_seed, min_side=args.min_side, k_range=args.k_range,
    )
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
               for k, v in g0d.items()}
    report["G0-D"] = summary
    report["G0-D_full_per_sample"] = g0d
    print(json.dumps(summary, indent=2))

    report["elapsed_seconds"] = time.time() - t0

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gate0_occupancy_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
