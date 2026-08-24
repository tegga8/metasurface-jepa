#!/usr/bin/env python3
"""Retrofit Phase 1, Gate-0 — scalar-only vs full-shape forward physics comparison.

Question: does the full 3x64x64 geometry carry meaningful spectral information beyond
the global physical scalars (r_atom, h_atom, l_lattice, occupancy_fraction,
occupancy_centroid)? If NOT, the swapped-target retrofit route (which relies on the
frozen ConvSurrogate reacting to masked-region SHAPE) has nothing to exploit and the
phase must stop.

Arms evaluated on the same held-out validation subset, identical metric:

  A. mean_spectrum   : predicts the train-set mean spectrum (floor reference).
  B. scalar_knn      : 1-nearest-neighbour in standardized scalar space among train
                       samples, returns that sample's spectrum (nonparametric bound on
                       what scalars alone determine).
  C. scalar_mlp      : small deterministic MLP regressor trained on the train split,
                       scalars -> complex spectrum.
  D. surrogate_shape : the FROZEN released ConvSurrogate forward on the full
                       3x64x64 geometry (no training; the retrofit's physics conduit).

Metric: rel_L2 = mean_i ||pred_i - true_i||_2 / ||true_i||_2 over the stacked
(real, imag) 602-vector, plus raw MSE and global R^2.

Verdict rule (PROPOSED — operator-confirmable per Standing Rule 3): shape carries
meaningful additional physics iff rel_L2(surrogate_shape) < 0.8 * min(rel_L2 of all
scalar-only arms). The raw numbers, not the verdict flag, are the deliverable.

Run:
    python scripts/diagnostics/gate0_scalar_vs_shape.py

Output: checkpoints/physics_retrofit/preflight/gate0_scalar_vs_shape.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import io
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
METADIT_SRC = REPO_ROOT / "external" / "metadit"
sys.path.insert(0, str(METADIT_SRC))


def build_scalars(parameter, pattern):
    """Per-sample scalar summary vector.

    parameter: (N, 3) [l_lattice, h_atom, r_atom]; pattern: (64, 64, N) in {0, 1}.
    Returns (N, 6) float64: [r_atom, h_atom, l_lattice, occ_frac, centroid_y,
    centroid_x] (centroids in [0, 1] grid units; row-major y first, documented).
    """
    n = parameter.shape[0]
    occ = (pattern == 1).astype(np.float64)                    # (64, 64, N)
    occ_frac = occ.sum(axis=(0, 1)) / (64.0 * 64.0)
    yy, xx = np.mgrid[0:64, 0:64]
    tot = np.maximum(occ.sum(axis=(0, 1)), 1.0)
    cy = (occ * yy[:, :, None]).sum(axis=(0, 1)) / tot / 63.0
    cx = (occ * xx[:, :, None]).sum(axis=(0, 1)) / tot / 63.0
    out = np.stack([parameter[:, 2], parameter[:, 1], parameter[:, 0],
                    occ_frac, cy, cx], axis=1)
    assert out.shape == (n, 6)
    return out


def build_geometries(pattern, parameter):
    """Vectorized MetaDiTDataset convention: -> (N, 3, 64, 64) float32."""
    occ_t = (pattern == 1).transpose(2, 0, 1)                  # (N, 64, 64)
    n = parameter.shape[0]
    g = np.zeros((n, 3, 64, 64), dtype=np.float32)
    r_col = (parameter[:, 2] / 5.0).astype(np.float32)[:, None, None]   # (N,1,1)
    h_col = parameter[:, 1].astype(np.float32)[:, None, None]
    l_row = (parameter[:, 0] / 3.0).astype(np.float32)[:, None, None]
    g[:, 0] = np.where(occ_t, r_col, np.float32(0.0))
    g[:, 1] = np.where(occ_t, h_col, np.float32(0.0))
    g[:, 2] = np.broadcast_to(l_row, (n, 64, 64))
    return torch.from_numpy(g)


def rel_l2(pred, true):
    d = np.linalg.norm(pred - true, axis=1)
    t = np.maximum(np.linalg.norm(true, axis=1), 1e-12)
    return float(np.mean(d / t))


def r2_score(pred, true):
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def mse(pred, true):
    return float(((pred - true) ** 2).mean())


def score_arm(name, pred, true):
    return {"arm": name, "mse": mse(pred, true),
            "rel_l2": rel_l2(pred, true), "r2": r2_score(pred, true)}


class ScalarMLP(nn.Module):
    def __init__(self, in_dim=6, hidden=256, out_dim=602):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_scalar_mlp(x_tr, y_tr, x_va, y_va, epochs, batch_size, lr, seed, device):
    """Deterministic small regression run; returns best-val-state predictions."""
    torch.manual_seed(seed)
    x_tr = torch.from_numpy(np.ascontiguousarray(x_tr, dtype=np.float32))
    y_tr = torch.from_numpy(np.ascontiguousarray(y_tr, dtype=np.float32))
    model = ScalarMLP(x_tr.shape[1], 256, y_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, best_epoch = float("inf"), None, -1
    n = len(x_tr)
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = x_tr[idx].to(device)
            yb = y_tr[idx].to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = torch.from_numpy(x_va).to(device)
            vloss = float(nn.functional.mse_loss(model(va), torch.from_numpy(y_va).to(device)).item())
        if vloss < best_val:
            best_val, best_epoch = vloss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = []
        xa = torch.from_numpy(x_va).to(device)
        for i in range(0, len(xa), 1024):
            preds.append(model(xa[i:i + 1024]).cpu().numpy())
    print(f"  scalar_mlp best epoch {best_epoch} (val mse {best_val:.6f})")
    return np.concatenate(preds, axis=0)


def scalar_knn(x_tr, y_tr, x_va, chunk=256):
    """1-NN by Euclidean distance in standardized scalar space (train stats)."""
    mu = x_tr.mean(axis=0, keepdims=True)
    sd = np.maximum(x_tr.std(axis=0, keepdims=True), 1e-8)
    a = (x_va - mu) / sd
    b = (x_tr - mu) / sd
    b_t = torch.from_numpy(b.astype(np.float32))
    out = np.empty((len(a), y_tr.shape[1]), dtype=np.float64)
    for i in range(0, len(a), chunk):
        qa = torch.from_numpy(a[i:i + chunk].astype(np.float32))
        d = torch.cdist(qa, b_t)
        nn_idx = d.argmin(dim=1).numpy()
        out[i:i + chunk] = y_tr[nn_idx]
    return out


def main():
    p = argparse.ArgumentParser(description="Gate-0 scalar-vs-shape physics audit")
    p.add_argument("--train", default="data/metadit/split_data/train_set.mat")
    p.add_argument("--val", default="data/metadit/split_data/val_set.mat")
    p.add_argument("--surrogate", default="data/metadit/weights/surrogate_model.bin")
    p.add_argument("--n-train", type=int, default=16384)
    p.add_argument("--n-val", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="checkpoints/physics_retrofit/preflight")
    args = p.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    tr = io.loadmat(str(REPO_ROOT / args.train))
    va = io.loadmat(str(REPO_ROOT / args.val))

    def subsample(mat, n):
        idx = rng.choice(mat["parameter"].shape[0], size=min(n, mat["parameter"].shape[0]),
                         replace=False)
        return idx

    tr_idx, va_idx = subsample(tr, args.n_train), subsample(va, args.n_val)

    def grab(mat, idx):
        params = mat["parameter"][idx].astype(np.float64)
        pat = mat["pattern"][:, :, idx]
        spec = np.concatenate([mat["real"][idx], mat["imag"][idx]], axis=1).astype(np.float64)
        return params, pat, spec

    tr_p, tr_pat, tr_s = grab(tr, tr_idx)
    va_p, va_pat, va_s = grab(va, va_idx)
    print(f"data: train={len(tr_idx)} val={len(va_idx)}")

    # ---- arm C prep: standardized scalar features ----
    f_tr = build_scalars(tr_p, tr_pat)
    f_va = build_scalars(va_p, va_pat)
    mu, sd = f_tr.mean(0, keepdims=True), np.maximum(f_tr.std(0, keepdims=True), 1e-8)
    x_tr = ((f_tr - mu) / sd).astype(np.float32)
    x_va = ((f_va - mu) / sd).astype(np.float32)
    y_mu = tr_s.mean(axis=0, keepdims=True)
    y_tr_n = (tr_s - y_mu).astype(np.float32)     # target centering for MLP stability

    device = torch.device(args.device)
    results = {}

    # ---- arm A: mean spectrum ----
    results["mean_spectrum"] = score_arm("mean_spectrum",
                                         np.repeat(y_mu, len(va_s), axis=0), va_s)

    # ---- arm B: scalar 1-NN ----
    print("arm B: scalar 1-NN ...")
    results["scalar_knn"] = score_arm("scalar_knn",
                                      scalar_knn(x_tr, tr_s, x_va), va_s)

    # ---- arm C: scalar MLP ----
    print(f"arm C: scalar MLP ({args.epochs} epochs) ...")
    pred_mlp = train_scalar_mlp(x_tr, y_tr_n, x_va, (va_s - y_mu).astype(np.float32),
                                args.epochs, args.batch_size, args.lr, args.seed, device)
    results["scalar_mlp"] = score_arm("scalar_mlp", pred_mlp + y_mu, va_s)

    # ---- arm D: frozen full-shape surrogate ----
    print("arm D: frozen ConvSurrogate on full 3x64x64 ...")
    sys.path.insert(0, str(METADIT_SRC))
    from model.surrogate import surrogate_s3  # noqa: E402
    surr = surrogate_s3()
    sd_ckpt = torch.load(REPO_ROOT / args.surrogate, map_location="cpu")
    surr.load_state_dict(sd_ckpt, strict=True)
    surr.eval().to(device)
    for q in surr.parameters():
        q.requires_grad_(False)
    preds = []
    with torch.no_grad():
        for i in range(0, len(va_idx), 128):
            gb = build_geometries(va_pat[:, :, i:i + 128], va_p[i:i + 128]).to(device)
            preds.append(surr(gb).prediction.cpu().numpy().reshape(len(gb), -1))
    results["surrogate_shape"] = score_arm("surrogate_shape",
                                           np.concatenate(preds, axis=0), va_s)

    # ---- verdict (proposed rule; numbers are the deliverable) ----
    scalar_best = min(results[k]["rel_l2"]
                      for k in ("mean_spectrum", "scalar_knn", "scalar_mlp"))
    shape_rel = results["surrogate_shape"]["rel_l2"]
    shape_ratio = shape_rel / max(scalar_best, 1e-12)
    verdict = {
        "rule": "shape_informative iff rel_L2(surrogate_shape) < 0.8 * best scalar arm",
        "best_scalar_rel_l2": scalar_best,
        "surrogate_rel_l2": shape_rel,
        "ratio_shape_over_best_scalar": shape_ratio,
        "shape_informative_proposed_verdict": bool(shape_ratio < 0.8),
    }

    report = {
        "gate": "retrofit-phase1-gate0-scalar-vs-shape",
        "seed": args.seed,
        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
        "splits": {"train": args.train, "val": args.val,
                   "surrogate": args.surrogate},
        "scalar_feature_order": ["r_atom", "h_atom", "l_lattice",
                                 "occupancy_fraction", "centroid_y", "centroid_x"],
        "mlp": {"hidden": 256, "depth": 3, "epochs": args.epochs,
                "batch_size": args.batch_size, "lr": args.lr},
        "arms": results,
        "verdict": verdict,
        "elapsed_seconds": time.time() - t0,
    }

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gate0_scalar_vs_shape.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps({"arms": results, "verdict": verdict}, indent=2, default=float))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
