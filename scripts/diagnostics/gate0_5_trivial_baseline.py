#!/usr/bin/env python3
"""Gate 0.5 — Trivial end-to-end ceiling baseline (PHYSICS_GEOMETRY_VALIDATION_V2 spec §9-§10).

Tests whether the product problem (geometry -> scalar physical parameters, assisted
by the target spectrum) is nearly solvable by a small parameter-regression MLP that
sees ONLY information recoverable from the masked geometry under the SAME BlockMasker
semantics the JEPA sees, plus the raw spectrum (the established repo representation).

Input per sample+mask:   x_visible_params = [l_value, h_value, r_value,
                                             l_observed, h_observed, r_observed]
                         (values recovered from visible geometry when the flag is 1,
                          else 0) concatenated with the raw (2,301) spectrum.
Target:                  [l_lattice, h_atom, r_atom]

Pre-registered thresholds (§10), locked before results:
    near-solving <=> NRMSE <= 0.10 for EACH scalar on held-out validation,
                     NRMSE = RMSE / std(target)
    occupancy IoU >= 0.95 required ONLY if Gate 0 establishes deterministic
                     rendering from the scalar parameters.

Run:  python scripts/diagnostics/gate0_5_trivial_baseline.py
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

from data.mask import BlockMasker  # noqa: E402

PARAM_NDIGITS = 2
NRMSE_THRESHOLD = 0.10
IOU_THRESHOLD = 0.95


def load_mat(path):
    return io.loadmat(str(path))


def stat_block(x):
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p10": float(np.percentile(x, 10)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def visible_param_features(G, M):
    """Build [l_value, h_value, r_value, l_observed, h_observed, r_observed] from the
    masked geometry tensor G (3,64,64) and token mask M (16,16) (1 = visible).

    Values use the dataset's OWN grid encoding (src/data/dataset.py): l/3, h, r/5 —
    the same encoding the geometry encoder consumes — so no new normalization is
    invented for this baseline (spec §9.3). l is recoverable from ANY visible pixel;
    h/r need a visible OCCUPIED pixel.
    """
    up = M.repeat_interleave(4, dim=0).repeat_interleave(4, dim=1)
    vis = up > 0.5
    occ = G[1] > 0.0                       # h_atom painted only where occupied
    vis_occ = vis & occ

    l_observed = bool(vis.any())
    h_observed = bool(vis_occ.any())

    l_value = float(G[2][vis].mean()) if l_observed else 0.0
    if h_observed:
        h_value = float(G[1][vis_occ].mean())
        r_value = float(G[0][vis_occ].mean())
    else:
        h_value = 0.0
        r_value = 0.0

    return np.array([l_value, h_value, r_value,
                     float(l_observed), float(h_observed), float(h_observed)],
                    dtype=np.float32)


def per_sample_observability(G, M):
    """Raw counts for the report (visible_occupied_pixel_count, total_visible_pixel_count)."""
    up = M.repeat_interleave(4, dim=0).repeat_interleave(4, dim=1)
    vis = up > 0.5
    occ = G[1] > 0.0
    return int((vis & occ).sum()), int(vis.sum())


class TrivialMLP(nn.Module):
    """Small MLP: 3 hidden layers, no attention, no JEPA, no physics conditioning."""

    def __init__(self, in_dim=6 + 2 * 301, hidden=(256, 128, 64), out_dim=3):
        super().__init__()
        layers = []
        d_in = in_dim
        for d in hidden:
            layers += [nn.Linear(d_in, d), nn.ReLU()]
            d_in = d
        layers.append(nn.Linear(d_in, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_baseline(Xtr, Ytr, Xva, Yva, epochs, batch_size, lr, seed, device,
                   patience=60):
    """Train the trivial MLP; return the best-validating model (true ceiling).

    Standard MLP training: minibatch Adam with cosine LR annealing (fixed LR
    was empirically found to stall at ~0.003 MSE — a training artifact, not a
    ceiling property; the exact-feature copy solution requires the annealed
    schedule to actually be reached), validation tracked every epoch, early
    stopping with `patience`, best-val state restored (a ceiling check must not
    report a randomly-bad late epoch as the model's capability).
    """
    torch.manual_seed(seed)
    model = TrivialMLP(in_dim=Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    n = len(Xtr)
    best_mse = float("inf")
    best_state = None
    best_epoch = -1
    stalled = 0
    for epoch in range(epochs):
        lr_now = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        for g in opt.param_groups:
            g["lr"] = lr_now
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx].to(device)), Ytr[idx].to(device))
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = float(lossf(model(Xva.to(device)), Yva.to(device)).item())
        if v < best_mse:
            best_mse = v
            best_epoch = epoch + 1
            best_state = {k: val.clone() for k, val in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
            if patience > 0 and stalled >= patience:
                break
    model.load_state_dict(best_state)
    return model, (best_mse, best_epoch)


def build_lookup(keys, masks):
    """Exact-parameter -> occupancy lookup table for deterministic re-rendering."""
    lookup = {}
    for k, m in zip(keys, masks):
        key = tuple(np.round(k, PARAM_NDIGITS).tolist())
        if key not in lookup:
            lookup[key] = m
    return lookup


def render_occupancy_lookup(pred_params, lookup, train_keys):
    """Render occupancy for predicted params via the training-set lookup table.

    Exact 2-decimal match preferred; nearest-neighbor (L1 over the 3 params) fallback.
    Returns (rendered mask, exact_match_bool, nn_distance).
    """
    rendered = np.zeros((len(pred_params), 64, 64), dtype=np.uint8)
    exact = np.zeros(len(pred_params), dtype=bool)
    nn_dist = np.zeros(len(pred_params), dtype=np.float32)
    tk = np.round(train_keys, PARAM_NDIGITS)
    for j, pp in enumerate(pred_params):
        key = tuple(np.round(pp, PARAM_NDIGITS).tolist())
        if key in lookup:
            rendered[j] = lookup[key]
            exact[j] = True
            continue
        d = np.abs(tk - pp).sum(axis=1)
        k = int(np.argmin(d))
        rendered[j] = lookup[tuple(tk[k].tolist())]
        nn_dist[j] = d[k]
    return rendered, exact, nn_dist


def main():
    p = argparse.ArgumentParser(description="Gate 0.5 trivial ceiling baseline")
    p.add_argument("--train", default="data/metadit/split_data/train_set.mat")
    p.add_argument("--val", default="data/metadit/split_data/val_set.mat")
    p.add_argument("--gate0-json",
                   default="checkpoints/milestone_b/physics_validation/gate0_occupancy_audit.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-samples", type=int, default=8192)
    p.add_argument("--val-samples", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=2400)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=0,
                       help="early-stop patience in epochs; 0 = run full schedule "
                            "(default: full cosine schedule with best-state restore)")
    p.add_argument("--mask-seed", type=int, default=12345)
    p.add_argument("--ratios", nargs="+", type=float,
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
    report = {
        "gate": "0.5",
        "seed": args.seed,
        "nrmse_threshold": NRMSE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "input_definition": (
            "[l_value, h_value, r_value, l_observed, h_observed, r_observed] "
            "recovered from visible geometry under the same BlockMasker semantics "
            "as the JEPA, in the dataset's own grid encoding (l/3, h, r/5 — the "
            "same values src/data/dataset.py paints and the geometry encoder "
            "consumes), concatenated with the raw (2,301) spectrum "
            "(the established repo representation; no new normalization)"
        ),
        "target_definition": "[l_lattice, h_atom, r_atom] raw units",
        "model": "TrivialMLP 608 -> 256 -> 128 -> 64 -> 3 (no attention / JEPA / physics)",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    train_m = load_mat(args.train)
    val_m = load_mat(args.val)

    tr_params = train_m["parameter"]
    tr_pat = train_m["pattern"]
    va_params = val_m["parameter"]
    va_pat = val_m["pattern"]

    rng = np.random.RandomState(args.seed)
    ti = rng.choice(len(tr_params), args.train_samples, replace=False)
    vi = rng.choice(len(va_params), args.val_samples, replace=False)

    train_keys = tr_params[ti]
    train_masks = tr_pat[:, :, ti].transpose(2, 0, 1)
    val_params = va_params[vi]

    # ---- deterministic re-rendering lookup (Gate 0 IoU applicability) ----
    gate0 = None
    if Path(args.gate0_json).exists():
        with open(args.gate0_json, "r", encoding="utf-8") as f:
            gate0 = json.load(f)
    lookup = build_lookup(train_keys, train_masks)
    report["gate0_loaded"] = gate0 is not None
    if gate0 is not None:
        g0b = gate0["G0-B"]
        fully_identical = g0b.get("fraction_fully_identical")
        buckets_identical = g0b.get("buckets_all_identical", {}).get("fraction")
        # strict determinism: zero within-bucket variation observed
        occupancy_deterministic = bool(
            fully_identical is not None and buckets_identical is not None
            and fully_identical >= 1.0 - 1e-9
            and buckets_identical >= 1.0 - 1e-9
        )
        report["gate0_occupancy_deterministic"] = occupancy_deterministic
        report["gate0_evidence"] = {
            "fraction_fully_identical": fully_identical,
            "fraction_buckets_all_identical": buckets_identical,
        }
        report["iou_criterion_applicable"] = (
            occupancy_deterministic or gate0.get("gate0_conclusion") == "deterministic"
        )
    else:
        report["gate0_occupancy_deterministic"] = None
        report["iou_criterion_applicable"] = False
        report["iou_criterion_note"] = (
            "Gate 0 JSON not found; IoU criterion treated as not applicable "
            "(spec §10 requires Gate 0 determinism evidence first)"
        )

    # ---- per-ratio feature construction + training + held-out eval ----
    per_ratio = {}
    for ratio in args.ratios:
        masker = BlockMasker(
            placement="random",
            grid=16,
            min_side=args.min_side,
            k_range=tuple(args.k_range),
            seed=args.mask_seed + int(round(ratio * 1000)),
        )

        def build_set(mat, params_sub, pat_sub, param_idx, masker_):
            """Features/targets/observability for a set of samples.

            param_idx: global sample indices (e.g. ti[j]) into the split's mat arrays;
            params_sub: (n,3) already-sliced parameter rows (same order as param_idx).
            """
            X, Y, obs = [], [], []
            for j in range(len(param_idx)):
                idx = int(param_idx[j])
                pat = pat_sub[:, :, j]
                occ = pat == 1
                G = torch.zeros(3, 64, 64)
                G[0][torch.from_numpy(occ)] = params_sub[j, 2] / 5.0
                G[1][torch.from_numpy(occ)] = params_sub[j, 1]
                G[2] = params_sub[j, 0] / 3.0
                M = masker_.sample(G.unsqueeze(0), ratio)[0].cpu()
                feat = visible_param_features(G, M)
                spec = np.concatenate([mat["real"][idx], mat["imag"][idx]])
                X.append(np.concatenate([feat, spec]).astype(np.float32))
                Y.append(params_sub[j])
                obs.append(per_sample_observability(G, M))
            return np.stack(X), np.stack(Y), np.array(obs)

        Xtr, Ytr, _ = build_set(train_m, tr_params[ti], tr_pat[:, :, ti], ti, masker)
        Xva, Yva, obs_va = build_set(val_m, va_params[vi], va_pat[:, :, vi], vi, masker)

        device = torch.device(args.device)
        Xtr_t = torch.from_numpy(Xtr)
        Ytr_t = torch.from_numpy(Ytr).float()
        Xva_t = torch.from_numpy(Xva)
        Yva_t = torch.from_numpy(Yva).float()

        # Standard MLP training-conditioning: standardize inputs with TRAIN-set
        # statistics only (per-feature mean/std over the 608 dims). This does not
        # change the input representation (raw spectrum + grid-encoded params) —
        # it only rescales it for optimizer conditioning, and it is invertible.
        feat_mean = Xtr_t.mean(dim=0, keepdim=True)
        feat_std = Xtr_t.std(dim=0, keepdim=True).clamp_min(1e-8)
        Xtr_s = (Xtr_t - feat_mean) / feat_std
        Xva_s = (Xva_t - feat_mean) / feat_std
        std_stats = {
            "note": "train-set per-feature mean/std; invertible; representation unchanged",
            "train_mean_abs": float(feat_mean.abs().mean().item()),
            "train_std_mean": float(feat_std.mean().item()),
        }

        model, best = train_baseline(Xtr_s, Ytr_t, Xva_s, Yva_t,
                                     args.epochs, args.batch_size, args.lr,
                                     args.seed, device, patience=args.patience)
        with torch.no_grad():
            pred = model(Xva_s.to(device)).cpu().numpy()

        rmse = np.sqrt(((pred - Yva) ** 2).mean(axis=0))
        std_target = Yva.std(axis=0)
        nrmse = rmse / std_target

        # ---- occupancy IoU via deterministic re-render (spec §10) ----
        rendered, exact, nn_dist = render_occupancy_lookup(pred, lookup, train_keys)
        true_masks = va_pat[:, :, vi].transpose(2, 0, 1) > 0
        inter = (rendered & true_masks).sum(axis=(1, 2)).astype(np.float64)
        union = (rendered | true_masks).sum(axis=(1, 2)).astype(np.float64)
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

        obs_arr = np.array(obs_va)
        per_ratio[str(ratio)] = {
            "n_train": int(len(Xtr)),
            "n_val": int(len(Xva)),
            "rmse_raw_units": {
                "l_lattice": float(rmse[0]),
                "h_atom": float(rmse[1]),
                "r_atom": float(rmse[2]),
            },
            "std_target_val": {
                "l_lattice": float(std_target[0]),
                "h_atom": float(std_target[1]),
                "r_atom": float(std_target[2]),
            },
            "nrmse": {
                "l_lattice": float(nrmse[0]),
                "h_atom": float(nrmse[1]),
                "r_atom": float(nrmse[2]),
            },
            "nrmse_meets_0.10": {
                "l_lattice": bool(nrmse[0] <= NRMSE_THRESHOLD),
                "h_atom": bool(nrmse[1] <= NRMSE_THRESHOLD),
                "r_atom": bool(nrmse[2] <= NRMSE_THRESHOLD),
            },
            "nrmse_all_meet": bool((nrmse <= NRMSE_THRESHOLD).all()),
            "occupancy_lookup_render": {
                "iou": stat_block(iou),
                "fraction_exact_param_match": float(exact.mean()),
                "nn_distance": stat_block(nn_dist),
            },
            "iou_meets_0.95": bool(iou.mean() >= IOU_THRESHOLD),
            "observability": {
                "fraction_l_observed": float((obs_arr[:, 1] > 0).mean()),
                "fraction_h_observed": float((obs_arr[:, 0] > 0).mean()),
                "fraction_r_observed": float((obs_arr[:, 0] > 0).mean()),
                "visible_occupied_pixel_count": stat_block(obs_arr[:, 0]),
                "total_visible_pixel_count": stat_block(obs_arr[:, 1]),
            },
            "best_val_mse": float(best[0]) if best else None,
            "best_epoch": int(best[1]) if best else None,
            "input_standardization": std_stats,
        }
        print(f"\nMASK {ratio:.2f}")
        print(json.dumps(per_ratio[str(ratio)], indent=2))

    report["per_ratio"] = per_ratio
    report["elapsed_seconds"] = time.time() - t0

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gate0_5_trivial_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
