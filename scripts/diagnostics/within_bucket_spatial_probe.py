#!/usr/bin/env python3
"""
Streaming within-bucket spatial-structure probe.

Purpose
-------
Test whether a frozen representation preserves spatial occupancy differences
when the three explicit physical parameters are approximately matched.

This version is deliberately memory-bounded:
- It NEVER concatenates the full validation/test geometry pool.
- It NEVER stores embeddings for the full pool.
- Parameter arrays are loaded once (small).
- Bucket membership and pair selection are computed from parameters only.
- Each bucket is processed independently:
    load only that bucket's geometries
    run trained/released/random encoders
    compute distances
    discard embeddings
- Peak memory therefore scales with the largest bucket, not dataset size.

Comparisons
-----------
1. trained EMA encoder
2. released MetaDiT encoder
3. random-init encoder
4. shape-aware trivial baseline:
      occupancy fraction + 4x4 coarse occupancy grid
5. random-init token-shuffled control

Ground-truth shape distance
---------------------------
Binary occupancy Hamming distance.

Representation distance
------------------------
1 - mean cosine similarity across aligned spatial tokens.

Statistics
----------
Spearman rho is computed independently inside every bucket, then aggregated:
- median rho over valid buckets
- mean rho over valid buckets
- pair-count-weighted mean rho

Pair selection is based ONLY on ground-truth parameters and is frozen before
any representation is evaluated.

Usage
-----
python scripts/diagnostics/within_bucket_spatial_probe.py \
    --checkpoint checkpoints/milestone_b/minimal_jepa_vicreg_latest.pt \
    --config configs/milestone_b.yaml \
    --data-root data/metadit \
    --device cuda:0 \
    --splits val,test \
    --probe-seed 0 \
    --out checkpoints/milestone_b/within_bucket_spatial_probe.json

If --max-geoms is omitted/0, the complete requested held-out pool is used.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from assembly import build_model, load_into_model  # noqa: E402
from data.dataset import MetaDiTDataset  # noqa: E402
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

    # 0 means use the complete requested pool.
    ap.add_argument("--max-geoms", type=int, default=0)

    ap.add_argument(
        "--splits",
        default="val,test",
        help="Comma-separated held-out splits.",
    )

    ap.add_argument("--probe-seed", type=int, default=0)

    ap.add_argument("--min-bucket-size", type=int, default=2)
    ap.add_argument("--min-rho-buckets", type=int, default=20)
    ap.add_argument("--min-pairs-total", type=int, default=500)
    ap.add_argument("--min-pairs-for-rho", type=int, default=3)

    # 0 = use all eligible pairs in each bucket.
    ap.add_argument("--max-pairs-per-bucket", type=int, default=0)

    ap.add_argument(
        "--max-normalized-param-distance",
        type=float,
        default=0.25,
        help=(
            "Maximum L2 distance after normalizing each parameter by its "
            "held-out-pool range. Set <=0 to disable this secondary filter."
        ),
    )

    ap.add_argument(
        "--bin-fractions",
        type=str,
        default=(
            "0.001,0.0025,0.005,0.01,0.02,0.05,0.10,0.20,0.30,0.50"
        ),
    )

    ap.add_argument(
        "--out",
        default=str(
            REPO_ROOT
            / "checkpoints/milestone_b/within_bucket_spatial_probe.json"
        ),
    )

    return ap.parse_args()


def resolve_data_root(
    cfg: dict,
    data_root: str | None,
) -> Path:
    if data_root:
        root = Path(data_root)
        if not root.exists():
            raise FileNotFoundError(root)
        return root

    raw = Path(cfg["data"]["val_split"])

    # Expected config form: data/metadit/split_data/val_set.mat
    if raw.is_absolute():
        return raw.parent.parent

    return (REPO_ROOT / raw).parent.parent


def load_split_metadata(
    data_root: Path,
    split_name: str,
):
    """
    Load only metadata arrays from the MAT dataset.

    The dataset object contains pattern/parameter arrays, but no torch geometry
    tensor pool is materialized here.
    """
    mat_path = data_root / "split_data" / f"{split_name}_set.mat"

    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    ds = MetaDiTDataset(str(mat_path))
    return ds


def build_heldout_index(
    data_root: Path,
    split_names: List[str],
    max_geoms: int,
):
    """
    Build a compact global index:

        global_id -> (dataset_object, local_index)

    Only parameter metadata is used for bucketing.
    """
    datasets = []
    params_parts = []
    locations = []

    total = 0

    for split_name in split_names:
        ds = load_split_metadata(
            data_root,
            split_name,
        )
        datasets.append(
            (split_name, ds)
        )

        n = len(ds)

        params_parts.append(
            np.asarray(
                ds.parameter[:n],
                dtype=np.float64,
            )
        )

        for local_index in range(n):
            locations.append(
                (
                    split_name,
                    len(datasets) - 1,
                    local_index,
                )
            )

        total += n

    params = np.concatenate(
        params_parts,
        axis=0,
    )

    if max_geoms > 0 and total > max_geoms:
        # Deterministic prefix across requested split order.
        params = params[:max_geoms]
        locations = locations[:max_geoms]

    if len(params) < 32:
        raise RuntimeError(
            f"Held-out pool is too small: {len(params)} geometries."
        )

    return datasets, params, locations


def geometry_from_dataset(
    ds: MetaDiTDataset,
    local_index: int,
) -> torch.Tensor:
    """
    Reproduce MetaDiTDataset.__getitem__ geometry construction exactly,
    without constructing the spectrum tensor.

    Dataset convention:
      channel 0 = r_atom / 5 on occupied pixels
      channel 1 = h_atom on occupied pixels
      channel 2 = l_lattice / 3 everywhere
    """
    pattern = torch.from_numpy(
        ds.pattern[:, :, local_index]
    ).float()

    params = np.asarray(
        ds.parameter[local_index],
        dtype=np.float64,
    )

    grid = torch.zeros(
        3,
        64,
        64,
        dtype=torch.float32,
    )

    occupied = pattern == 1.0

    grid[0][occupied] = float(
        params[2] / 5.0
    )

    grid[1][occupied] = float(
        params[1]
    )

    grid[2].fill_(
        float(params[0] / 3.0)
    )

    return grid


def get_geometry_batch(
    datasets,
    locations,
    global_indices: np.ndarray,
) -> torch.Tensor:
    """
    Materialize only the requested geometries for one bucket.
    """
    split_to_dataset = {
        split_name: ds
        for split_name, ds in datasets
    }

    tensors = []

    for global_index in global_indices:
        split_name, dataset_list_index, local_index = (
            locations[int(global_index)]
        )

        ds = split_to_dataset[split_name]

        tensors.append(
            geometry_from_dataset(
                ds,
                local_index,
            )
        )

    return torch.stack(
        tensors,
        dim=0,
    )


def occupancy_from_geometry(
    geometry: torch.Tensor,
) -> np.ndarray:
    occupied = (
        (geometry[:, 0] != 0)
        | (geometry[:, 1] != 0)
    )

    return occupied.cpu().numpy().astype(
        np.uint8
    )


def coarse_shape_features(
    occupancy: np.ndarray,
) -> np.ndarray:
    """
    Shape-aware trivial baseline:
      occupancy fraction + 4x4 block occupancy.

    Output shape = [N,17].
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
        .reshape(
            n,
            4,
            16,
            4,
            16,
        )
        .mean(axis=(2, 4))
        .reshape(n, 16)
    )

    occupancy_fraction = (
        occupancy.mean(
            axis=(1, 2)
        )[:, None]
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
            f"Unexpected trivial-feature shape: {features.shape}"
        )

    return features


def quantize_params(
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

    return keys, mins, maxs, spans, widths


def build_buckets(
    keys: np.ndarray,
    min_bucket_size: int,
):
    buckets: Dict[
        Tuple[int, int, int],
        List[int],
    ] = {}

    for global_index, key_array in enumerate(keys):
        key = (
            int(key_array[0]),
            int(key_array[1]),
            int(key_array[2]),
        )
        buckets.setdefault(
            key,
            [],
        ).append(global_index)

    return {
        key: np.asarray(
            indices,
            dtype=np.int64,
        )
        for key, indices in buckets.items()
        if len(indices) >= min_bucket_size
    }


def build_pairs(
    buckets,
    params,
    spans,
    seed,
    max_pairs_per_bucket,
    max_normalized_param_distance,
):
    """
    Build pairs using ground-truth parameter information only.
    """
    rng = np.random.RandomState(seed)

    normalization = np.maximum(
        spans,
        1e-12,
    )

    pair_map = {}

    for bucket in sorted(buckets):
        members = buckets[bucket]

        pairs = []

        for i in range(len(members) - 1):
            a = int(members[i])

            for j in range(i + 1, len(members)):
                b = int(members[j])

                if max_normalized_param_distance > 0:
                    delta = (
                        params[a] - params[b]
                    ) / normalization

                    normalized_distance = float(
                        np.linalg.norm(delta)
                    )

                    if (
                        normalized_distance
                        > max_normalized_param_distance
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

        if pairs:
            pair_map[bucket] = np.asarray(
                pairs,
                dtype=np.int64,
            )

    return pair_map


def evaluate_candidate(
    params,
    fraction,
    args,
):
    keys, mins, maxs, spans, widths = (
        quantize_params(
            params,
            fraction,
        )
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
        key: value
        for key, value in pairs.items()
        if len(value) >= args.min_pairs_for_rho
    }

    total_pairs = sum(
        len(value)
        for value in usable.values()
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
        "total_pairs": total_pairs,
    }


def choose_candidate(
    params,
    args,
):
    fractions = sorted(
        set(
            float(x.strip())
            for x in args.bin_fractions.split(",")
            if x.strip()
        )
    )

    trials = []

    for fraction in fractions:
        result = evaluate_candidate(
            params,
            fraction,
            args,
        )

        trials.append(
            {
                "fraction": result["fraction"],
                "bin_widths": result["widths"].tolist(),
                "usable_buckets": result["usable_buckets"],
                "total_pairs": result["total_pairs"],
            }
        )

        if (
            result["usable_buckets"]
            >= args.min_rho_buckets
            and result["total_pairs"]
            >= args.min_pairs_total
        ):
            return result, trials

    best = max(
        trials,
        key=lambda x: (
            x["usable_buckets"],
            x["total_pairs"],
        ),
        default=None,
    )

    raise RuntimeError(
        "No candidate bin width produced the requested matched-pair pool. "
        f"Best candidate: {best}. "
        "Increase the held-out pool or explicitly revise the matching "
        "criterion; do not silently lower the statistical requirements."
    )


def hamming_distance(
    occupancy_a: np.ndarray,
    occupancy_b: np.ndarray,
) -> float:
    return float(
        np.mean(
            occupancy_a != occupancy_b
        )
    )


def feature_distance(
    feature_a: np.ndarray,
    feature_b: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(
            feature_a - feature_b
        )
    )


def token_cosine_distance(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
) -> float:
    similarity = (
        torch.nn.functional
        .cosine_similarity(
            embedding_a.float(),
            embedding_b.float(),
            dim=-1,
        )
        .mean()
    )

    return float(
        1.0 - similarity.item()
    )


def spearman_rho(
    x: np.ndarray,
    y: np.ndarray,
):
    if len(x) < 3:
        return None

    if (
        np.allclose(x, x[0])
        or np.allclose(y, y[0])
    ):
        return None

    rho = spearmanr(
        x,
        y,
    ).statistic

    if (
        rho is None
        or not np.isfinite(rho)
    ):
        return None

    return float(rho)


def aggregate(records, key):
    valid = []

    for record in records:
        rho = record[key]

        if rho is None or not np.isfinite(rho):
            continue

        valid.append(
            (
                float(rho),
                int(record["n_pairs"]),
            )
        )

    if not valid:
        return {
            "valid_buckets": 0,
            "pairs": 0,
            "median": None,
            "mean": None,
            "weighted_mean": None,
        }

    values = np.asarray(
        [x[0] for x in valid],
        dtype=np.float64,
    )

    weights = np.asarray(
        [x[1] for x in valid],
        dtype=np.float64,
    )

    return {
        "valid_buckets": int(len(valid)),
        "pairs": int(weights.sum()),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "weighted_mean": float(
            np.average(
                values,
                weights=weights,
            )
        ),
    }


def checksum(
    module: torch.nn.Module,
) -> float:
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

    checkpoint_path = checkpoint_path.resolve()

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
    # STEP 1: Metadata only
    # ------------------------------------------------------------

    data_root = resolve_data_root(
        cfg,
        args.data_root,
    )

    split_names = [
        x.strip()
        for x in args.splits.split(",")
        if x.strip()
    ]

    datasets, params, locations = (
        build_heldout_index(
            data_root,
            split_names,
            args.max_geoms,
        )
    )

    # ------------------------------------------------------------
    # STEP 2: Freeze bucket/pair set BEFORE model loading
    # ------------------------------------------------------------

    selected, trials = choose_candidate(
        params,
        args,
    )

    frozen_pairs = selected["pairs"]

    total_pairs = sum(
        len(v)
        for v in frozen_pairs.values()
    )

    print()
    print("=" * 80)
    print("GROUND-TRUTH MATCHING / VIABILITY")
    print("=" * 80)

    print(
        f"splits                 : {','.join(split_names)}"
    )

    print(
        f"geometries             : {len(params)}"
    )

    print(
        f"selected bin fraction  : "
        f"{selected['fraction']:.6g}"
    )

    print(
        "bin widths [l,h,r]     : "
        f"{selected['widths'][0]:.8g}, "
        f"{selected['widths'][1]:.8g}, "
        f"{selected['widths'][2]:.8g}"
    )

    print(
        f"usable buckets         : "
        f"{selected['usable_buckets']}"
    )

    print(
        f"frozen pairs           : "
        f"{total_pairs}"
    )

    print(
        "[OK] pair set frozen using ground truth only"
    )

    # ------------------------------------------------------------
    # STEP 3: Load models
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

    random_encoder = build_random_encoder(
        hidden,
        heads,
        depth,
        args.probe_seed,
        device,
    )

    # ------------------------------------------------------------
    # STEP 4: Stream bucket-by-bucket
    # ------------------------------------------------------------

    records = []

    print()
    print("=" * 80)
    print("PROCESSING BUCKETS")
    print("=" * 80)

    for bucket_number, bucket_key in enumerate(
        sorted(frozen_pairs),
        start=1,
    ):

        pairs = frozen_pairs[
            bucket_key
        ]

        # Unique geometry indices for THIS bucket only.
        unique_indices = np.unique(
            pairs.reshape(-1)
        )

        # Small bucket-local CPU tensors only.
        geometry = get_geometry_batch(
            datasets,
            locations,
            unique_indices,
        )

        occupancy = occupancy_from_geometry(
            geometry
        )

        trivial_features = coarse_shape_features(
            occupancy
        )

        # Map global index -> row inside this bucket.
        local_row = {
            int(global_index): row
            for row, global_index in enumerate(
                unique_indices.tolist()
            )
        }

        # One small GPU batch per bucket.
        with torch.no_grad():

            trained = model.ema(
                geometry.to(device)
            ).detach().cpu()

            released = released_encoder(
                geometry.to(device)
            ).detach().cpu()

            random = random_encoder(
                geometry.to(device)
            ).detach().cpu()

        # Precompute distances for this bucket.
        shape_distances = []
        trivial_distances = []
        trained_distances = []
        released_distances = []
        random_distances = []

        for a_global, b_global in pairs:

            a = local_row[
                int(a_global)
            ]

            b = local_row[
                int(b_global)
            ]

            shape_distances.append(
                hamming_distance(
                    occupancy[a],
                    occupancy[b],
                )
            )

            trivial_distances.append(
                feature_distance(
                    trivial_features[a],
                    trivial_features[b],
                )
            )

            trained_distances.append(
                token_cosine_distance(
                    trained[a],
                    trained[b],
                )
            )

            released_distances.append(
                token_cosine_distance(
                    released[a],
                    released[b],
                )
            )

            random_distances.append(
                token_cosine_distance(
                    random[a],
                    random[b],
                )
            )

        shape_distances = np.asarray(
            shape_distances,
            dtype=np.float64,
        )

        trivial_distances = np.asarray(
            trivial_distances,
            dtype=np.float64,
        )

        trained_distances = np.asarray(
            trained_distances,
            dtype=np.float64,
        )

        released_distances = np.asarray(
            released_distances,
            dtype=np.float64,
        )

        random_distances = np.asarray(
            random_distances,
            dtype=np.float64,
        )

        record = {
            "bucket": list(
                bucket_key
            ),
            "n_samples": int(
                len(
                    selected["buckets"][
                        bucket_key
                    ]
                )
            ),
            "n_pairs": int(
                len(pairs)
            ),
            "rho_trained": spearman_rho(
                shape_distances,
                trained_distances,
            ),
            "rho_released": spearman_rho(
                shape_distances,
                released_distances,
            ),
            "rho_random": spearman_rho(
                shape_distances,
                random_distances,
            ),
            "rho_trivial": spearman_rho(
                shape_distances,
                trivial_distances,
            ),
        }

        records.append(
            record
        )

        print(
            f"[{bucket_number:>4}/{len(frozen_pairs)}] "
            f"bucket={bucket_key} "
            f"samples={len(unique_indices)} "
            f"pairs={len(pairs)} "
            f"rho(trained)="
            f"{record['rho_trained'] if record['rho_trained'] is not None else float('nan'):.4f}"
        )

        # Explicitly release bucket-local tensors before next bucket.
        del geometry
        del trained
        del released
        del random

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # STEP 5: Aggregate within-bucket results
    # ------------------------------------------------------------

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
        "trivial": aggregate(
            records,
            "rho_trivial",
        ),
    }

    # ------------------------------------------------------------
    # STEP 6: Verify model was never modified
    # ------------------------------------------------------------

    checksum_after = checksum(
        model
    )

    if checksum_after != checksum_before:
        raise RuntimeError(
            "Checkpoint model parameters changed during diagnostic."
        )

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("WITHIN-BUCKET SPATIAL-STRUCTURE RESULT")
    print("=" * 80)

    print(
        f"{'representation':<24}"
        f"{'median':>12}"
        f"{'mean':>12}"
        f"{'weighted':>14}"
        f"{'buckets':>10}"
        f"{'pairs':>10}"
    )

    print("-" * 80)

    for key, label in [
        ("trained", "trained EMA"),
        ("released", "released ViT"),
        ("random", "random init"),
        ("trivial", "trivial shape"),
    ]:
        result = aggregates[key]

        print(
            f"{label:<24}"
            f"{result['median'] if result['median'] is not None else float('nan'):>12.6f}"
            f"{result['mean'] if result['mean'] is not None else float('nan'):>12.6f}"
            f"{result['weighted_mean'] if result['weighted_mean'] is not None else float('nan'):>14.6f}"
            f"{result['valid_buckets']:>10d}"
            f"{result['pairs']:>10d}"
        )

    print()
    print("KEY DIFFERENCES")
    print("-" * 80)

    trained_value = aggregates["trained"]["weighted_mean"]
    trivial_value = aggregates["trivial"]["weighted_mean"]
    random_value = aggregates["random"]["weighted_mean"]
    released_value = aggregates["released"]["weighted_mean"]

    if trained_value is not None and trivial_value is not None:
        print(
            f"trained - trivial : "
            f"{trained_value - trivial_value:+.6f}"
        )

    if trained_value is not None and random_value is not None:
        print(
            f"trained - random  : "
            f"{trained_value - random_value:+.6f}"
        )

    if trained_value is not None and released_value is not None:
        print(
            f"trained - released: "
            f"{trained_value - released_value:+.6f}"
        )

    print()
    print(
        "Memory-safe diagnostic completed. "
        "No automatic architecture decision is made."
    )

    # ------------------------------------------------------------
    # Save full report
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
            len(params)
        ),
        "configuration": {
            "max_geoms": int(
                args.max_geoms
            ),
            "min_bucket_size": int(
                args.min_bucket_size
            ),
            "min_rho_buckets": int(
                args.min_rho_buckets
            ),
            "min_pairs_total": int(
                args.min_pairs_total
            ),
            "min_pairs_for_rho": int(
                args.min_pairs_for_rho
            ),
            "max_pairs_per_bucket": int(
                args.max_pairs_per_bucket
            ),
            "max_normalized_param_distance": float(
                args.max_normalized_param_distance
            ),
            "bin_fractions": [
                float(x.strip())
                for x in args.bin_fractions.split(",")
                if x.strip()
            ],
        },
        "bin_trials": trials,
        "selected_matching": {
            "fraction": float(
                selected["fraction"]
            ),
            "widths": selected[
                "widths"
            ].tolist(),
            "parameter_spans": selected[
                "spans"
            ].tolist(),
            "usable_buckets": int(
                selected["usable_buckets"]
            ),
            "frozen_pairs": int(
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
            "pair_selection": (
                "ground-truth parameters only; frozen before model evaluation"
            ),
            "within_bucket_correlation": True,
            "full_pool_embeddings_materialized": False,
            "memory_bounded_bucket_processing": True,
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