#!/usr/bin/env python3
"""
Rigorous within-bucket spatial-structure probe.

Question
--------
At approximately matched (l_lattice, h_atom, r_atom), does a representation
preserve differences in the 64x64 occupancy pattern?

Why this version exists
-----------------------
The first probe had only 60 matched pairs. Its result was hypothesis-generating
but not statistically strong. This version:

1. Uses a larger held-out pool (validation + test by default).
2. Performs bucket selection using ONLY ground-truth physical parameters.
3. Builds and freezes the exact pair set BEFORE loading representations.
4. Uses all valid pairs unless an explicit cap is requested.
5. Computes Spearman correlation INSIDE each bucket.
6. Aggregates only after the within-bucket step.
7. Reports actual parameter spread inside selected pairs.
8. Compares:
      - trained EMA
      - released ViT
      - random-init
      - shape-aware trivial baseline
9. Adds a random-token-order control for the random-init encoder.
   This tests whether the high random-init correlation is mainly due to
   aligned spatial/token positions rather than learned shape semantics.
10. Does not train or modify the checkpoint.

Shape-aware trivial baseline
----------------------------
occupancy fraction + 4x4 coarse occupancy grid.

Representation distance
------------------------
1 - mean cosine similarity over aligned spatial tokens.

Random-token control
--------------------
For the random-init encoder only, independently permute token positions
for each sample before computing pairwise distance. This destroys consistent
spatial token alignment while preserving each sample's token content.

Usage
-----
python scripts/diagnostics/within_bucket_spatial_probe.py \
    --checkpoint checkpoints/milestone_b/minimal_jepa_vicreg_latest.pt \
    --config configs/milestone_b.yaml \
    --data-root data/metadit \
    --device cuda:0 \
    --max-geoms 0 \
    --probe-seed 0 \
    --out checkpoints/milestone_b/within_bucket_spatial_probe.json

max-geoms=0 means use the complete requested held-out pool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, ConcatDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from assembly import build_model, load_into_model  # noqa: E402
from data.dataset import MetaDiTDataset, collate_batch  # noqa: E402
from encoders.geometry_encoder import GeometryEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/milestone_b.yaml"),
    )
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cpu")

    # 0 = use complete held-out pool.
    ap.add_argument("--max-geoms", type=int, default=0)

    ap.add_argument("--probe-seed", type=int, default=0)

    ap.add_argument(
        "--splits",
        default="val,test",
        help="Comma-separated held-out splits.",
    )

    ap.add_argument("--min-bucket-size", type=int, default=2)
    ap.add_argument("--min-rho-buckets", type=int, default=20)
    ap.add_argument("--min-pairs-total", type=int, default=500)

    ap.add_argument(
        "--min-pairs-for-rho",
        type=int,
        default=3,
    )

    # 0 = no per-bucket cap.
    ap.add_argument(
        "--max-pairs-per-bucket",
        type=int,
        default=0,
    )

    # Parameter-bin widths are relative to observed parameter ranges.
    ap.add_argument(
        "--bin-fractions",
        type=str,
        default=(
            "0.001,0.0025,0.005,0.01,0.02,0.05,0.10,0.20,0.30,0.50"
        ),
    )

    # Extra anti-confounding check inside a bucket.
    ap.add_argument(
        "--max-normalized-param-distance",
        type=float,
        default=0.25,
    )

    ap.add_argument(
        "--out",
        default=str(
            REPO_ROOT
            / "checkpoints/milestone_b/within_bucket_spatial_probe.json"
        ),
    )

    return ap.parse_args()


def resolve_root(data_root: str | None) -> Path:
    if data_root is not None:
        root = Path(data_root)
        if not root.exists():
            raise FileNotFoundError(root)
        return root

    return REPO_ROOT / "data/metadit"


def load_split_dataset(
    data_root: Path,
    split_name: str,
) -> MetaDiTDataset:
    path = data_root / "split_data" / f"{split_name}_set.mat"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing split file: {path}"
        )

    return MetaDiTDataset(str(path))


def load_heldout_pool(
    data_root: Path,
    split_names: List[str],
    max_geoms: int,
):
    datasets = [
        load_split_dataset(
            data_root,
            name,
        )
        for name in split_names
    ]

    # Load the complete split contents first, then concatenate.
    all_geometry = []
    all_params = []

    for split_name, ds in zip(
        split_names,
        datasets,
    ):
        loader = DataLoader(
            ds,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=collate_batch,
        )

        for geometry, _spectrum in loader:
            all_geometry.append(
                geometry.cpu()
            )

        all_params.append(
            np.asarray(
                ds.parameter,
                dtype=np.float64,
            )
        )

    geometries = torch.cat(
        all_geometry,
        dim=0,
    )

    params = np.concatenate(
        all_params,
        axis=0,
    )

    # The pool may be large; keep deterministic prefix.
    if max_geoms > 0:
        n = min(
            max_geoms,
            len(geometries),
        )
        geometries = geometries[:n]
        params = params[:n]

    if len(geometries) < 32:
        raise RuntimeError(
            f"Held-out pool is too small: {len(geometries)}"
        )

    return geometries, params


def occupancy_binary(
    geometries: torch.Tensor,
) -> np.ndarray:
    occ = (
        (geometries[:, 0] != 0)
        | (geometries[:, 1] != 0)
    )

    return occ.cpu().numpy().astype(
        np.uint8
    )


def coarse_shape_features(
    occupancy: np.ndarray,
) -> np.ndarray:
    """
    [N,64,64] ->
       occupancy fraction [N,1]
       4x4 coarse occupancy [N,16]

    Final shape = [N,17].
    """

    if occupancy.ndim != 3:
        raise ValueError(
            f"Expected [N,H,W], got {occupancy.shape}"
        )

    n, h, w = occupancy.shape

    if (h, w) != (64, 64):
        raise ValueError(
            f"Expected 64x64 occupancy, got {occupancy.shape}"
        )

    coarse = (
        occupancy
        .reshape(n, 4, 16, 4, 16)
        .mean(axis=(2, 4))
        .reshape(n, 16)
    )

    occupancy_fraction = (
        occupancy
        .mean(axis=(1, 2))
        [:, None]
    )

    features = np.concatenate(
        [
            occupancy_fraction,
            coarse,
        ],
        axis=1,
    ).astype(np.float64)

    if features.shape != (n, 17):
        raise RuntimeError(
            f"Wrong trivial-feature shape: {features.shape}"
        )

    return features


def quantize_parameters(
    params: np.ndarray,
    fraction: float,
):
    mins = params.min(axis=0)
    maxs = params.max(axis=0)

    spans = np.maximum(
        maxs - mins,
        1e-12,
    )

    widths = spans * fraction

    keys = np.floor(
        (params - mins) / widths
    ).astype(np.int64)

    return (
        keys,
        mins,
        maxs,
        spans,
        widths,
    )


def build_buckets(
    keys: np.ndarray,
    min_bucket_size: int,
):
    buckets: Dict[
        Tuple[int, int, int],
        List[int],
    ] = {}

    for i, key in enumerate(keys):

        bucket = (
            int(key[0]),
            int(key[1]),
            int(key[2]),
        )

        buckets.setdefault(
            bucket,
            [],
        ).append(i)

    return {
        k: np.asarray(
            v,
            dtype=np.int64,
        )
        for k, v in buckets.items()
        if len(v) >= min_bucket_size
    }


def build_pairs(
    buckets,
    params,
    spans,
    seed,
    max_pairs_per_bucket,
    max_normalized_distance,
):
    """
    Pair selection is entirely ground-truth based.

    A pair must:
      1. be in the same quantized bucket;
      2. satisfy the normalized parameter-distance constraint.
    """

    rng = np.random.RandomState(
        seed
    )

    normalization = np.maximum(
        spans,
        1e-12,
    )

    result = {}

    for bucket in sorted(buckets):

        members = buckets[bucket]
        pairs = []

        for i in range(len(members) - 1):

            a = int(members[i])

            for j in range(i + 1, len(members)):

                b = int(members[j])

                delta = (
                    params[a] - params[b]
                ) / normalization

                dist = float(
                    np.linalg.norm(delta)
                )

                if (
                    max_normalized_distance > 0
                    and dist > max_normalized_distance
                ):
                    continue

                pairs.append(
                    (a, b)
                )

        if (
            max_pairs_per_bucket > 0
            and len(pairs) > max_pairs_per_bucket
        ):
            chosen = rng.choice(
                len(pairs),
                size=max_pairs_per_bucket,
                replace=False,
            )

            pairs = [
                pairs[int(i)]
                for i in np.sort(chosen)
            ]

        if len(pairs) >= 1:
            result[bucket] = np.asarray(
                pairs,
                dtype=np.int64,
            )

    return result


def evaluate_bin(
    params,
    fraction,
    args,
):
    (
        keys,
        mins,
        maxs,
        spans,
        widths,
    ) = quantize_parameters(
        params,
        fraction,
    )

    buckets = build_buckets(
        keys,
        args.min_bucket_size,
    )

    pairs = build_pairs(
        buckets,
        params,
        spans,
        args.probe_seed,
        args.max_pairs_per_bucket,
        args.max_normalized_param_distance,
    )

    usable = {
        k: v
        for k, v in pairs.items()
        if len(v) >= args.min_pairs_for_rho
    }

    pair_count = sum(
        len(v)
        for v in usable.values()
    )

    return {
        "fraction": float(fraction),
        "mins": mins,
        "maxs": maxs,
        "spans": spans,
        "widths": widths,
        "buckets": buckets,
        "pairs": usable,
        "usable_buckets": len(usable),
        "pair_count": pair_count,
    }


def choose_bin(
    params,
    args,
):
    fractions = sorted(
        set(
            float(x.strip())
            for x in args.bin_fractions.split(",")
        )
    )

    trials = []

    for fraction in fractions:

        result = evaluate_bin(
            params,
            fraction,
            args,
        )

        trials.append(
            {
                "fraction": result["fraction"],
                "bin_widths": result["widths"].tolist(),
                "usable_buckets": result["usable_buckets"],
                "pair_count": result["pair_count"],
            }
        )

        if (
            result["usable_buckets"]
            >= args.min_rho_buckets
            and result["pair_count"]
            >= args.min_pairs_total
        ):
            return result, trials

    best = max(
        trials,
        key=lambda x: (
            x["usable_buckets"],
            x["pair_count"],
        ),
        default=None,
    )

    raise RuntimeError(
        "No bin width produced the required matched-pair pool.\n"
        f"Best candidate: {best}\n"
        "Do NOT silently lower thresholds. Increase the held-out pool or "
        "explicitly change the matching criteria."
    )


def hamming_distance(
    occupancy,
    pairs,
):
    out = np.empty(
        len(pairs),
        dtype=np.float64,
    )

    for i, (a, b) in enumerate(pairs):
        out[i] = np.mean(
            occupancy[a] != occupancy[b]
        )

    return out


def feature_distance(
    features,
    pairs,
):
    out = np.empty(
        len(pairs),
        dtype=np.float64,
    )

    for i, (a, b) in enumerate(pairs):
        out[i] = np.linalg.norm(
            features[a] - features[b]
        )

    return out


def token_cosine_distance(
    embeddings,
    pairs,
):
    out = np.empty(
        len(pairs),
        dtype=np.float64,
    )

    for i, (a, b) in enumerate(pairs):

        za = embeddings[a].float()
        zb = embeddings[b].float()

        similarity = (
            torch.nn.functional
            .cosine_similarity(
                za,
                zb,
                dim=-1,
            )
            .mean()
        )

        out[i] = (
            1.0
            - float(similarity.item())
        )

    return out


def random_token_order(
    embeddings: torch.Tensor,
    seed: int,
):
    """
    Independently shuffle token positions within each sample.

    This destroys cross-sample spatial-token alignment while preserving the
    set of token vectors in each sample.
    """

    rng = np.random.RandomState(
        seed
    )

    output = embeddings.clone()

    n, t, _d = output.shape

    for i in range(n):
        permutation = torch.from_numpy(
            rng.permutation(t)
        ).long()

        output[i] = output[i][permutation]

    return output


def spearman(
    ground_truth_distance,
    representation_distance,
):
    if len(ground_truth_distance) < 3:
        return None

    if (
        np.allclose(
            ground_truth_distance,
            ground_truth_distance[0],
        )
        or np.allclose(
            representation_distance,
            representation_distance[0],
        )
    ):
        return None

    rho = spearmanr(
        ground_truth_distance,
        representation_distance,
    ).statistic

    if rho is None or not np.isfinite(rho):
        return None

    return float(rho)


def aggregate(
    records,
    key,
):
    values = []
    weights = []

    for record in records:

        rho = record[key]

        if rho is None or not np.isfinite(rho):
            continue

        values.append(
            float(rho)
        )

        weights.append(
            int(record["n_pairs"])
        )

    if not values:
        return {
            "valid_buckets": 0,
            "pairs": 0,
            "median": None,
            "mean": None,
            "weighted_mean": None,
        }

    return {
        "valid_buckets": len(values),
        "pairs": int(sum(weights)),
        "median": float(
            np.median(values)
        ),
        "mean": float(
            np.mean(values)
        ),
        "weighted_mean": float(
            np.average(
                values,
                weights=weights,
            )
        ),
    }


def checksum(
    module,
):
    return sum(
        p.detach()
        .double()
        .sum()
        .item()
        for p in module.parameters()
    )


def build_random_encoder(
    hidden,
    heads,
    depth,
    seed,
    device,
):
    with torch.random.fork_rng(
        devices=[]
    ):
        torch.manual_seed(seed)

        return GeometryEncoder(
            hidden=hidden,
            num_heads=heads,
            depth=depth,
        ).to(device)


def collect_embeddings(
    encoder,
    geometry_batches,
    device,
):
    actual = next(
        encoder.parameters()
    ).device

    if actual != device:
        raise RuntimeError(
            f"Encoder on {actual}, requested {device}"
        )

    was_training = encoder.training
    encoder.eval()

    try:
        with torch.no_grad():

            return torch.cat(
                [
                    encoder(
                        batch.to(device)
                    )
                    .detach()
                    .cpu()
                    for batch in geometry_batches
                ],
                dim=0,
            )

    finally:
        encoder.train(
            was_training
        )


def main():
    args = parse_args()

    device = torch.device(
        args.device
    )

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.is_absolute():
        checkpoint_path = (
            REPO_ROOT
            / checkpoint_path
        )

    checkpoint_path = (
        checkpoint_path
        .resolve()
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_cfg = checkpoint.get(
        "cfg",
        cfg,
    )

    # ------------------------------------------------------------
    # Ground-truth held-out pool
    # ------------------------------------------------------------

    data_root = resolve_root(
        args.data_root
    )

    split_names = [
        x.strip()
        for x in args.splits.split(",")
        if x.strip()
    ]

    geometries, params = load_heldout_pool(
        data_root,
        split_names,
        args.max_geoms,
    )

    occupancy = occupancy_binary(
        geometries
    )

    shape_features = coarse_shape_features(
        occupancy
    )

    # ------------------------------------------------------------
    # Bucket search BEFORE representations
    # ------------------------------------------------------------

    selected, bin_trials = choose_bin(
        params,
        args,
    )

    frozen_pairs = selected[
        "pairs"
    ]

    total_pairs = sum(
        len(v)
        for v in frozen_pairs.values()
    )

    print()
    print("=" * 80)
    print("HELD-OUT DATA / BUCKET VIABILITY")
    print("=" * 80)

    print(
        f"splits                  : "
        f"{','.join(split_names)}"
    )

    print(
        f"geometries              : "
        f"{len(geometries)}"
    )

    print(
        f"selected bin fraction   : "
        f"{selected['fraction']:.6g}"
    )

    print(
        "bin widths [l,h,r]      : "
        f"{selected['widths'][0]:.8g}, "
        f"{selected['widths'][1]:.8g}, "
        f"{selected['widths'][2]:.8g}"
    )

    print(
        f"usable buckets          : "
        f"{selected['usable_buckets']}"
    )

    print(
        f"frozen pairs            : "
        f"{total_pairs}"
    )

    print(
        "[OK] pair set frozen from ground truth only"
    )

    # ------------------------------------------------------------
    # Load trained model AFTER pair selection
    # ------------------------------------------------------------

    spectrum_path = (
        REPO_ROOT
        / checkpoint_cfg["weights"]["spectrum"]
    )

    metadit_path = (
        REPO_ROOT
        / checkpoint_cfg["weights"]["metadit"]
    )

    model = build_model(
        checkpoint_cfg["model"],
        str(spectrum_path),
        device=device,
        init_from_metadit=False,
        metadit_weights=str(
            metadit_path
        ),
    )

    load_into_model(
        model,
        checkpoint["model"],
        device,
    )

    from train.engine import restore_ema_state

    restore_ema_state(
        model,
        checkpoint.get(
            "ema_state"
        ),
    )

    checksum_before = checksum(
        model
    )

    geometry_batches = [
        geometries[i:i + 64]
        for i in range(
            0,
            len(geometries),
            64,
        )
    ]

    trained_embeddings = collect_embeddings(
        model.ema,
        geometry_batches,
        device,
    )

    hidden = int(
        checkpoint_cfg["model"].get(
            "hidden",
            384,
        )
    )

    heads = int(
        checkpoint_cfg["model"].get(
            "num_heads",
            6,
        )
    )

    depth = int(
        checkpoint_cfg["model"].get(
            "geo_depth",
            6,
        )
    )

    # Released encoder
    released_encoder = GeometryEncoder(
        hidden=hidden,
        num_heads=heads,
        depth=depth,
    ).to(device)

    released_payload = torch.load(
        metadit_path,
        map_location="cpu",
        weights_only=False,
    )

    released_encoder.init_from_metadit(
        released_payload,
        blocks_to_take=depth,
    )

    released_embeddings = collect_embeddings(
        released_encoder,
        geometry_batches,
        device,
    )

    # Random encoder
    random_encoder = build_random_encoder(
        hidden,
        heads,
        depth,
        args.probe_seed,
        device,
    )

    random_embeddings = collect_embeddings(
        random_encoder,
        geometry_batches,
        device,
    )

    # Random-token-order control
    random_shuffled_embeddings = (
        random_token_order(
            random_embeddings,
            seed=args.probe_seed + 1,
        )
    )

    # ------------------------------------------------------------
    # Same pairs for all methods
    # ------------------------------------------------------------

    records = []

    for bucket in sorted(
        frozen_pairs
    ):

        pairs = frozen_pairs[
            bucket
        ]

        shape_distance = hamming_distance(
            occupancy,
            pairs,
        )

        trivial_distance = feature_distance(
            shape_features,
            pairs,
        )

        trained_distance = token_cosine_distance(
            trained_embeddings,
            pairs,
        )

        released_distance = token_cosine_distance(
            released_embeddings,
            pairs,
        )

        random_distance = token_cosine_distance(
            random_embeddings,
            pairs,
        )

        random_shuffled_distance = token_cosine_distance(
            random_shuffled_embeddings,
            pairs,
        )

        records.append(
            {
                "bucket": list(bucket),
                "n_pairs": int(len(pairs)),
                "rho_trained": spearman(
                    shape_distance,
                    trained_distance,
                ),
                "rho_released": spearman(
                    shape_distance,
                    released_distance,
                ),
                "rho_random": spearman(
                    shape_distance,
                    random_distance,
                ),
                "rho_random_shuffled": spearman(
                    shape_distance,
                    random_shuffled_distance,
                ),
                "rho_trivial": spearman(
                    shape_distance,
                    trivial_distance,
                ),
            }
        )

    aggregates = {
        "trained": aggregate(
            records,
            "rho_trained",
        ),
        "released": aggregate(
            records,
            "rho_released",
        ),
        "random": aggregate(
            records,
            "rho_random",
        ),
        "random_shuffled": aggregate(
            records,
            "rho_random_shuffled",
        ),
        "trivial": aggregate(
            records,
            "rho_trivial",
        ),
    }

    checksum_after = checksum(
        model
    )

    if checksum_after != checksum_before:
        raise RuntimeError(
            "Checkpoint parameters changed during diagnostic."
        )

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("WITHIN-BUCKET SPATIAL-STRUCTURE RESULT")
    print("=" * 80)

    print(
        f"{'representation':<25}"
        f"{'median':>12}"
        f"{'mean':>12}"
        f"{'weighted':>14}"
        f"{'buckets':>12}"
        f"{'pairs':>10}"
    )

    print("-" * 80)

    labels = [
        ("trained", "trained EMA"),
        ("released", "released ViT"),
        ("random", "random init"),
        ("random_shuffled", "random shuffled"),
        ("trivial", "trivial shape"),
    ]

    for key, label in labels:

        r = aggregates[key]

        print(
            f"{label:<25}"
            f"{r['median'] if r['median'] is not None else float('nan'):>12.6f}"
            f"{r['mean'] if r['mean'] is not None else float('nan'):>12.6f}"
            f"{r['weighted_mean'] if r['weighted_mean'] is not None else float('nan'):>14.6f}"
            f"{r['valid_buckets']:>12d}"
            f"{r['pairs']:>10d}"
        )

    print()
    print("KEY DIFFERENCES")
    print("-" * 80)

    if (
        aggregates["trained"]["weighted_mean"] is not None
        and aggregates["trivial"]["weighted_mean"] is not None
    ):
        print(
            "trained - trivial        : "
            f"{aggregates['trained']['weighted_mean'] - aggregates['trivial']['weighted_mean']:+.6f}"
        )

    if (
        aggregates["trained"]["weighted_mean"] is not None
        and aggregates["random"]["weighted_mean"] is not None
    ):
        print(
            "trained - random         : "
            f"{aggregates['trained']['weighted_mean'] - aggregates['random']['weighted_mean']:+.6f}"
        )

    if (
        aggregates["trained"]["weighted_mean"] is not None
        and aggregates["released"]["weighted_mean"] is not None
    ):
        print(
            "trained - released       : "
            f"{aggregates['trained']['weighted_mean'] - aggregates['released']['weighted_mean']:+.6f}"
        )

    if (
        aggregates["random"]["weighted_mean"] is not None
        and aggregates["random_shuffled"]["weighted_mean"] is not None
    ):
        print(
            "random aligned-shuffled  : "
            f"{aggregates['random']['weighted_mean'] - aggregates['random_shuffled']['weighted_mean']:+.6f}"
        )

    print()
    print(
        "No automatic architecture or collapse verdict is issued."
    )

    # ------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------

    output_path = Path(
        args.out
    )

    if not output_path.is_absolute():
        output_path = (
            REPO_ROOT
            / output_path
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "checkpoint": str(
            checkpoint_path
        ),
        "step": checkpoint.get(
            "step"
        ),
        "epoch": checkpoint.get(
            "epoch"
        ),
        "splits": split_names,
        "n_geometries": int(
            len(geometries)
        ),
        "configuration": {
            "probe_seed": args.probe_seed,
            "min_bucket_size": args.min_bucket_size,
            "min_rho_buckets": args.min_rho_buckets,
            "min_pairs_total": args.min_pairs_total,
            "min_pairs_for_rho": args.min_pairs_for_rho,
            "max_pairs_per_bucket": args.max_pairs_per_bucket,
            "max_normalized_param_distance": (
                args.max_normalized_param_distance
            ),
            "bin_fractions": [
                float(x.strip())
                for x in args.bin_fractions.split(",")
            ],
        },
        "bin_trials": bin_trials,
        "selected_bucket": {
            "fraction": float(
                selected["fraction"]
            ),
            "widths": selected[
                "widths"
            ].tolist(),
            "mins": selected[
                "mins"
            ].tolist(),
            "maxs": selected[
                "maxs"
            ].tolist(),
            "spans": selected[
                "spans"
            ].tolist(),
            "usable_buckets": int(
                selected["usable_buckets"]
            ),
            "pair_count": int(
                total_pairs
            ),
        },
        "protocol": {
            "ground_truth_shape_distance": (
                "binary occupancy Hamming distance"
            ),
            "representation_distance": (
                "mean aligned-token cosine distance"
            ),
            "trivial_shape_baseline": (
                "occupancy fraction + 4x4 coarse occupancy"
            ),
            "random_token_control": (
                "independent token permutation per sample"
            ),
            "within_bucket_only": True,
            "pair_set_locked_before_representation_loading": True,
        },
        "aggregate_results": aggregates,
        "bucket_results": records,
        "read_only_verified": True,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print(
        f"JSON: {output_path}"
    )


if __name__ == "__main__":
    main()