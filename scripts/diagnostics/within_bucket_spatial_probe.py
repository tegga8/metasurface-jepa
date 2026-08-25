#!/usr/bin/env python3
"""
Within-bucket spatial-structure probe.

Question:
    At matched (l_lattice, h_atom, r_atom), does a representation preserve
    differences in the 64x64 occupancy pattern?

Protocol:
  1. Check bucket viability using quantized physical parameters.
  2. Adaptively widen quantization bins until enough buckets/pairs exist.
  3. Freeze all buckets and pair indices using ground-truth parameters/occupancy.
  4. For the SAME pairs, compare:
       - occupancy Hamming distance (ground-truth shape distance)
       - trivial shape-aware feature distance
       - trained EMA representation distance
       - released ViT representation distance
       - random-init representation distance
  5. Compute Spearman rho separately inside each bucket.
  6. Pool only AFTER within-bucket correlations are computed:
       - median rho
       - weighted mean rho by pair count
       - number of usable buckets
  7. Verify checkpoint parameters are unchanged.

No training, no backward, no optimizer, no gate verdict.

Trivial shape-aware baseline:
    occupancy fraction + 4x4 coarse occupancy grid (16 values)

Representation distance:
    mean token-wise cosine distance over the 256 aligned tokens.
    This preserves spatial token alignment and avoids the mean-pooling issue
    that motivated this experiment.

Usage:
  python scripts/diagnostics/within_bucket_spatial_probe.py \
      --checkpoint checkpoints/milestone_b/minimal_jepa_vicreg_latest.pt \
      --config configs/milestone_b.yaml \
      --data-root data/metadit \
      --device cuda:0 \
      --max-geoms 4096 \
      --probe-seed 0 \
      --out checkpoints/milestone_b/within_bucket_spatial_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from assembly import build_model, load_into_model  # noqa: E402
from data.dataset import MetaDiTDataset, collate_batch  # noqa: E402
from encoders.geometry_encoder import GeometryEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "milestone_b.yaml"),
    )
    p.add_argument("--data-root", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-geoms", type=int, default=4096)
    p.add_argument("--probe-seed", type=int, default=0)
    p.add_argument("--min-bucket-size", type=int, default=2)
    p.add_argument("--min-usable-buckets", type=int, default=10)
    p.add_argument("--min-total-pairs", type=int, default=100)
    p.add_argument("--max-pairs-per-bucket", type=int, default=200)
    p.add_argument("--min-pairs-for-rho", type=int, default=3)
    p.add_argument("--out", default=str(
        REPO_ROOT
        / "checkpoints"
        / "milestone_b"
        / "within_bucket_spatial_probe.json"
    ))
    return p.parse_args()


def resolve_val_mat(cfg: dict, data_root: str | None) -> Path:
    if data_root:
        root = Path(data_root)
        if root.is_dir():
            candidate = root / "split_data" / "val_set.mat"
            if candidate.exists():
                return candidate
        if root.suffix == ".mat" and root.exists():
            return root
        raise FileNotFoundError(
            f"Cannot resolve val_set.mat from --data-root={data_root}"
        )
    raw = Path(cfg["data"]["val_split"])
    return raw if raw.is_absolute() else REPO_ROOT / raw


def load_geometries(
    mat_path: Path,
    max_geoms: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    ds = MetaDiTDataset(str(mat_path))
    n = min(int(max_geoms), len(ds))
    if n < 8:
        raise RuntimeError(f"Need at least 8 geometries, got {n}")

    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_batch,
    )

    chunks: List[torch.Tensor] = []
    remaining = n
    for G, _S in loader:
        take = min(remaining, G.shape[0])
        chunks.append(G[:take].clone())
        remaining -= take
        if remaining == 0:
            break

    geometries = torch.cat(chunks, dim=0)
    if geometries.shape[0] != n:
        raise RuntimeError(
            f"Collected {geometries.shape[0]} geometries, expected {n}"
        )

    params = np.asarray(ds.parameter[:n], dtype=np.float64)
    if params.shape != (n, 3):
        raise RuntimeError(
            f"Expected parameter array [{n},3], got {params.shape}"
        )

    return geometries, params


def occupancy_binary(geometries: torch.Tensor) -> np.ndarray:
    # The dataset uses occupancy implicitly through nonzero r/h channels.
    occ = ((geometries[:, 0] != 0) | (geometries[:, 1] != 0))
    return occ.cpu().numpy().astype(np.uint8)


def coarse_shape_features(occ: np.ndarray) -> np.ndarray:
    """
    Shape-aware trivial baseline:
      occupancy fraction
      4x4 block occupancy fractions

    This intentionally excludes r/h/l scalar values.
    """
    n, h, w = occ.shape
    if h != 64 or w != 64:
        raise ValueError(f"Expected 64x64 occupancy, got {occ.shape}")

    x = occ.reshape(n, 16, 4, 16, 4).mean(axis=(2, 4))
    coarse = x.reshape(n, 16)
    frac = occ.mean(axis=(1, 2), keepdims=True)
    return np.concatenate([frac, coarse], axis=1).astype(np.float64)


def robust_quantize(
    params: np.ndarray,
    fractions: List[float],
) -> Tuple[float, np.ndarray, Dict[str, float]]:
    """
    Try relative bin widths as fractions of each parameter's data range.
    Exact/very-fine bins are tried first; widths are widened until viability.
    """
    spans = params.max(axis=0) - params.min(axis=0)
    scales = np.maximum(spans, 1e-12)

    candidates = []
    for f in fractions:
        widths = scales * f
        mins = params.min(axis=0)
        keys = np.floor((params - mins) / widths).astype(np.int64)
        candidates.append((f, widths, mins, keys))
    return candidates


def bucket_map(
    keys: np.ndarray,
    min_bucket_size: int,
) -> Dict[Tuple[int, int, int], np.ndarray]:
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for i, k in enumerate(keys):
        key = (int(k[0]), int(k[1]), int(k[2]))
        buckets.setdefault(key, []).append(i)
    return {
        k: np.asarray(v, dtype=np.int64)
        for k, v in buckets.items()
        if len(v) >= min_bucket_size
    }


def pair_indices(
    buckets: Dict[Tuple[int, int, int], np.ndarray],
    seed: int,
    max_pairs_per_bucket: int,
) -> Dict[Tuple[int, int, int], np.ndarray]:
    """
    Freeze the exact pair set using only bucket membership.
    Deterministic and representation-independent.
    """
    rng = np.random.RandomState(seed)
    out = {}

    for key in sorted(buckets):
        idx = buckets[key]
        pairs = []
        for a in range(len(idx) - 1):
            for b in range(a + 1, len(idx)):
                pairs.append((int(idx[a]), int(idx[b])))

        if len(pairs) > max_pairs_per_bucket:
            choose = rng.choice(
                len(pairs),
                size=max_pairs_per_bucket,
                replace=False,
            )
            pairs = [pairs[int(i)] for i in np.sort(choose)]

        if len(pairs) >= 1:
            out[key] = np.asarray(pairs, dtype=np.int64)

    return out


def viability(
    buckets: Dict[Tuple[int, int, int], np.ndarray],
    min_total_pairs: int,
) -> Dict[str, int]:
    total_pairs = sum(len(v) * (len(v) - 1) // 2 for v in buckets.values())
    return {
        "usable_buckets": int(len(buckets)),
        "total_possible_pairs": int(total_pairs),
        "max_bucket_size": int(max((len(v) for v in buckets.values()), default=0)),
        "n_geometries_in_usable_buckets": int(
            sum(len(v) for v in buckets.values())
        ),
        "viable": int(
            len(buckets) >= 1 and total_pairs >= min_total_pairs
        ),
    }


def select_bucket_width(
    params: np.ndarray,
    min_bucket_size: int,
    min_usable_buckets: int,
    min_total_pairs: int,
):
    # Begin strict; widen only as needed.
    fractions = [
        0.001,
        0.0025,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
    ]

    candidates = robust_quantize(params, fractions)
    report = []

    for fraction, widths, mins, keys in candidates:
        buckets = bucket_map(keys, min_bucket_size)
        v = viability(buckets, min_total_pairs)
        v.update({
            "relative_bin_fraction": float(fraction),
            "bin_width_l_lattice": float(widths[0]),
            "bin_width_h_atom": float(widths[1]),
            "bin_width_r_atom": float(widths[2]),
        })
        report.append(v)

        if (
            v["usable_buckets"] >= min_usable_buckets
            and v["total_possible_pairs"] >= min_total_pairs
        ):
            return (
                fraction,
                widths,
                mins,
                keys,
                buckets,
                report,
            )

    raise RuntimeError(
        "Bucket viability failed. No tested quantization width produced "
        f">={min_usable_buckets} usable buckets and "
        f">={min_total_pairs} possible pairs. "
        "Increase --max-geoms or explicitly widen the candidate fractions."
    )


def cosine_token_distance(
    X: torch.Tensor,
    pairs: np.ndarray,
) -> np.ndarray:
    """
    X: [N,T,D] on CPU.
    Distance = 1 - mean cosine over aligned spatial token positions.
    """
    out = np.empty(len(pairs), dtype=np.float64)

    for j, (a, b) in enumerate(pairs):
        xa = X[a].float()
        xb = X[b].float()
        cos = torch.nn.functional.cosine_similarity(
            xa,
            xb,
            dim=-1,
        ).mean()
        out[j] = float(1.0 - cos.item())

    return out


def euclidean_distance(
    X: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    out = np.empty(len(pairs), dtype=np.float64)
    for j, (a, b) in enumerate(pairs):
        d = X[a] - X[b]
        out[j] = float(np.sqrt(np.dot(d, d)))
    return out


def hamming_distance(
    occ: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    out = np.empty(len(pairs), dtype=np.float64)
    for j, (a, b) in enumerate(pairs):
        out[j] = float(np.mean(occ[a] != occ[b]))
    return out


def spearman_for_pairs(
    shape_dist: np.ndarray,
    rep_dist: np.ndarray,
) -> float:
    if len(shape_dist) < 3:
        return float("nan")
    if np.allclose(shape_dist, shape_dist[0]):
        return float("nan")
    if np.allclose(rep_dist, rep_dist[0]):
        return float("nan")
    rho = spearmanr(shape_dist, rep_dist).statistic
    return float(rho) if rho is not None and np.isfinite(rho) else float("nan")


def checksum(module: torch.nn.Module) -> float:
    return sum(
        p.detach().double().sum().item()
        for p in module.parameters()
    )


def collect_embeddings(
    encoder: torch.nn.Module,
    geometry_batches: List[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    actual = next(encoder.parameters()).device
    if actual != device:
        raise RuntimeError(
            f"Encoder is on {actual}, requested {device}"
        )
    was_training = encoder.training
    encoder.eval()
    try:
        with torch.no_grad():
            chunks = [
                encoder(g.to(device)).detach().cpu()
                for g in geometry_batches
            ]
        return torch.cat(chunks, dim=0)
    finally:
        encoder.train(was_training)


def build_random_encoder(
    hidden: int,
    heads: int,
    depth: int,
    seed: int,
    device: torch.device,
) -> GeometryEncoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return GeometryEncoder(
            hidden=hidden,
            num_heads=heads,
            depth=depth,
        ).to(device)


def main() -> None:
    a = parse_args()
    device = torch.device(a.device)

    with open(a.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt_path = Path(a.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    ckpt_path = ckpt_path.resolve()

    ck = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    ck_cfg = ck.get("cfg", cfg)

    val_mat = resolve_val_mat(cfg, a.data_root)
    geometries, params = load_geometries(val_mat, a.max_geoms)
    occ = occupancy_binary(geometries)
    shape_features = coarse_shape_features(occ)

    # ------------------------------------------------------------------
    # PHASE 1: data availability check BEFORE any representation work
    # ------------------------------------------------------------------
    (
        selected_fraction,
        widths,
        mins,
        bucket_keys,
        buckets,
        viability_report,
    ) = select_bucket_width(
        params,
        a.min_bucket_size,
        a.min_usable_buckets,
        a.min_total_pairs,
    )

    fixed_pairs = pair_indices(
        buckets,
        seed=a.probe_seed,
        max_pairs_per_bucket=a.max_pairs_per_bucket,
    )

    usable_pair_buckets = {
        k: v for k, v in fixed_pairs.items()
        if len(v) >= a.min_pairs_for_rho
    }

    total_fixed_pairs = sum(len(v) for v in usable_pair_buckets.values())

    if (
        len(usable_pair_buckets) < a.min_usable_buckets
        or total_fixed_pairs < a.min_total_pairs
    ):
        raise RuntimeError(
            "After pair capping, the frozen pair set is still too sparse: "
            f"buckets={len(usable_pair_buckets)}, pairs={total_fixed_pairs}. "
            "Increase --max-geoms or --max-pairs-per-bucket."
        )

    print("\nBUCKET VIABILITY")
    print("=" * 78)
    print(f"validation geometries used : {len(geometries)}")
    print(
        "selected relative bin width: "
        f"{selected_fraction:.4g} × parameter range"
    )
    print(
        "bin widths [l,h,r]         : "
        f"{widths[0]:.8g}, {widths[1]:.8g}, {widths[2]:.8g}"
    )
    print(f"usable buckets             : {len(buckets)}")
    print(f"usable pair buckets        : {len(usable_pair_buckets)}")
    print(f"frozen pair count          : {total_fixed_pairs}")
    print("[OK] bucket/pair set is viable")

    # Save the bucket decision BEFORE loading/encoding any representation.
    bucket_report = {
        "selected_relative_bin_fraction": float(selected_fraction),
        "bin_widths": {
            "l_lattice": float(widths[0]),
            "h_atom": float(widths[1]),
            "r_atom": float(widths[2]),
        },
        "parameter_min": mins.tolist(),
        "parameter_max": params.max(axis=0).tolist(),
        "viability_trials": viability_report,
        "usable_buckets": int(len(buckets)),
        "usable_pair_buckets": int(len(usable_pair_buckets)),
        "frozen_pair_count": int(total_fixed_pairs),
    }

    # ------------------------------------------------------------------
    # PHASE 2: only now load representations
    # ------------------------------------------------------------------
    spec_path = REPO_ROOT / ck_cfg["weights"]["spectrum"]
    metadit_path = REPO_ROOT / ck_cfg["weights"]["metadit"]

    model = build_model(
        ck_cfg["model"],
        str(spec_path),
        device=device,
        init_from_metadit=False,
        metadit_weights=str(metadit_path),
    )
    load_into_model(model, ck["model"], device)

    from train.engine import restore_ema_state
    restore_ema_state(model, ck.get("ema_state"))

    before = checksum(model)

    # Preserve deterministic geometry batch order.
    geometry_batches = []
    for start in range(0, len(geometries), 64):
        geometry_batches.append(geometries[start:start + 64])

    X_trained = collect_embeddings(
        model.ema,
        geometry_batches,
        device,
    )

    hidden = int(ck_cfg["model"].get("hidden", 384))
    heads = int(ck_cfg["model"].get("num_heads", 6))
    depth = int(ck_cfg["model"].get("geo_depth", 6))

    released = GeometryEncoder(
        hidden=hidden,
        num_heads=heads,
        depth=depth,
    ).to(device)
    released_payload = torch.load(
        metadit_path,
        map_location="cpu",
        weights_only=False,
    )
    released.init_from_metadit(
        released_payload,
        blocks_to_take=depth,
    )

    X_released = collect_embeddings(
        released,
        geometry_batches,
        device,
    )

    random_encoder = build_random_encoder(
        hidden,
        heads,
        depth,
        a.probe_seed,
        device,
    )
    X_random = collect_embeddings(
        random_encoder,
        geometry_batches,
        device,
    )

    # ------------------------------------------------------------------
    # PHASE 3: same fixed pairs, all representations
    # ------------------------------------------------------------------
    all_pair_results = {
        "trained_ema": [],
        "released_vit": [],
        "random_init": [],
        "trivial_shape": [],
    }

    bucket_results = []

    for bucket_id, pairs in sorted(usable_pair_buckets.items()):
        shape_dist = hamming_distance(occ, pairs)

        # Shape-aware trivial baseline uses only coarse occupancy features.
        trivial_dist = euclidean_distance(
            shape_features,
            pairs,
        )

        trained_dist = cosine_token_distance(
            X_trained,
            pairs,
        )
        released_dist = cosine_token_distance(
            X_released,
            pairs,
        )
        random_dist = cosine_token_distance(
            X_random,
            pairs,
        )

        rho_trained = spearman_for_pairs(shape_dist, trained_dist)
        rho_released = spearman_for_pairs(shape_dist, released_dist)
        rho_random = spearman_for_pairs(shape_dist, random_dist)
        rho_trivial = spearman_for_pairs(shape_dist, trivial_dist)

        bucket_results.append({
            "bucket": list(bucket_id),
            "n_samples": int(len(buckets[bucket_id])),
            "n_pairs": int(len(pairs)),
            "rho_trained_ema": rho_trained,
            "rho_released_vit": rho_released,
            "rho_random_init": rho_random,
            "rho_trivial_shape": rho_trivial,
        })

        all_pair_results["trained_ema"].append(rho_trained)
        all_pair_results["released_vit"].append(rho_released)
        all_pair_results["random_init"].append(rho_random)
        all_pair_results["trivial_shape"].append(rho_trivial)

    def aggregate(values: List[float]) -> dict:
        x = np.asarray(
            [v for v in values if np.isfinite(v)],
            dtype=np.float64,
        )
        if len(x) == 0:
            return {
                "n_buckets": 0,
                "median_rho": float("nan"),
                "mean_rho": float("nan"),
            }
        return {
            "n_buckets": int(len(x)),
            "median_rho": float(np.median(x)),
            "mean_rho": float(np.mean(x)),
        }

    aggregate_results = {
        k: aggregate(v)
        for k, v in all_pair_results.items()
    }

    after = checksum(model)
    if after != before:
        raise RuntimeError(
            "Checkpoint model parameters changed during spatial probe"
        )

    print("\nWITHIN-BUCKET SPATIAL-STRUCTURE RESULT")
    print("=" * 78)
    print(
        f"{'representation':<22}"
        f"{'median rho':>14}"
        f"{'mean rho':>14}"
        f"{'buckets':>12}"
    )
    print("-" * 78)

    for name, label in [
        ("trained_ema", "trained EMA"),
        ("released_vit", "released ViT"),
        ("random_init", "random init"),
        ("trivial_shape", "trivial shape"),
    ]:
        r = aggregate_results[name]
        print(
            f"{label:<22}"
            f"{r['median_rho']:>14.6f}"
            f"{r['mean_rho']:>14.6f}"
            f"{r['n_buckets']:>12d}"
        )

    trained_mean = aggregate_results["trained_ema"]["mean_rho"]
    trivial_mean = aggregate_results["trivial_shape"]["mean_rho"]
    random_mean = aggregate_results["random_init"]["mean_rho"]
    released_mean = aggregate_results["released_vit"]["mean_rho"]

    print("\nCOMPARISONS")
    print("-" * 78)
    print(f"trained - trivial shape : {trained_mean - trivial_mean:+.6f}")
    print(f"trained - random       : {trained_mean - random_mean:+.6f}")
    print(f"trained - released     : {trained_mean - released_mean:+.6f}")

    print(
        "\nInterpretation must be based on the within-bucket result. "
        "Do not use scalar-parameter variation or the earlier mean-pooled "
        "collapse gate to infer spatial geometry from this test."
    )

    out = Path(a.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint": str(ckpt_path),
        "step": ck.get("step"),
        "epoch": ck.get("epoch"),
        "n_geometries": int(len(geometries)),
        "probe_seed": int(a.probe_seed),
        "bucket_selection": bucket_report,
        "protocol": {
            "shape_distance": "occupancy_hamming",
            "representation_distance": (
                "mean aligned-token cosine distance"
            ),
            "trivial_baseline": (
                "occupancy_fraction + 4x4 coarse occupancy"
            ),
            "correlation": (
                "Spearman within each bucket; aggregate only afterward"
            ),
            "pair_set_locked_from_ground_truth": True,
        },
        "aggregate_results": aggregate_results,
        "bucket_results": bucket_results,
        "read_only_verified": True,
    }

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
