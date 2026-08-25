#!/usr/bin/env python3
"""
Rigorous within-bucket spatial-structure probe.

Scientific question
-------------------
At approximately matched (l_lattice, h_atom, r_atom), does a representation
preserve differences in the 64x64 occupancy pattern?

This diagnostic is intentionally independent of the earlier mean-pooled
collapse gate and scalar-parameter R2 probe.

Protocol
--------
A. Ground-truth-only viability analysis:
   - Load a deterministic validation prefix.
   - Quantize each physical parameter into bins.
   - Try progressively wider bins.
   - For every candidate bin width, compute the FINAL bucket/pair set exactly
     as it will be used for the correlation analysis.
   - Do not select a width using "possible pairs" and then discover later that
     the final pair set is too small.
   - Report all candidate widths before choosing one.

B. Pair-set construction:
   - Buckets are formed only from ground-truth parameters.
   - Pairs are selected only from those buckets.
   - The pair set is frozen before loading any representation.
   - No representation-dependent filtering is permitted.

C. Shape distance:
   - Binary occupancy Hamming distance.

D. Representation distance:
   - Mean aligned-token cosine distance over all spatial tokens.
   - No mean-pooling of tokens before the distance.

E. Trivial shape baseline:
   - occupancy fraction
   - 4x4 coarse occupancy grid
   - no physical parameter channels.

F. Statistics:
   - Spearman rho is computed independently inside each bucket.
   - Aggregate only the valid within-bucket rho values afterward.
   - Report median and pair-count-weighted mean rho.
   - Report bucket-level sample/pair counts and parameter spread.

Important:
This script never trains anything, never backpropagates, never changes the
checkpoint, and never issues an automatic HEALTHY/COLLAPSED verdict.

A dataset with sparse matched buckets is a DATA LIMITATION, not a reason to
silently loosen the matching criterion. The script therefore reports the
selected bin widths and the actual physical spread inside matched buckets.

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

for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
    ap.add_argument("--max-geoms", type=int, default=4096)
    ap.add_argument("--probe-seed", type=int, default=0)

    # Matching controls.
    ap.add_argument(
        "--min-bucket-size",
        type=int,
        default=2,
    )
    ap.add_argument(
        "--min-usable-buckets",
        type=int,
        default=10,
    )
    ap.add_argument(
        "--min-rho-buckets",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--min-pairs-total",
        type=int,
        default=50,
    )
    ap.add_argument(
        "--max-pairs-per-bucket",
        type=int,
        default=0,
        help="0 means use all pairs; otherwise deterministic cap.",
    )
    ap.add_argument(
        "--min-pairs-for-rho",
        type=int,
        default=3,
    )

    # Bins are fractions of each parameter's observed range.
    ap.add_argument(
        "--bin-fractions",
        type=str,
        default=(
            "0.001,0.0025,0.005,0.01,0.02,0.05,0.10,0.20,0.30,0.50,0.75,1.0"
        ),
    )

    # To keep bins genuinely "matched", reject pairs whose actual normalized
    # parameter difference is larger than this. This is a stricter criterion
    # than merely sharing the same quantized bucket.
    ap.add_argument(
        "--max-normalized-param-distance",
        type=float,
        default=0.25,
        help=(
            "Maximum Euclidean distance between the three normalized physical "
            "parameters for a pair. 0 disables this secondary check."
        ),
    )

    ap.add_argument(
        "--out",
        default=str(
            REPO_ROOT
            / "checkpoints"
            / "milestone_b"
            / "within_bucket_spatial_probe.json"
        ),
    )

    return ap.parse_args()


def resolve_val_mat(
    cfg: dict,
    data_root: str | None,
) -> Path:
    if data_root:
        root = Path(data_root)

        if root.is_dir():
            candidate = root / "split_data" / "val_set.mat"
            if candidate.exists():
                return candidate

        if root.suffix == ".mat" and root.exists():
            return root

        raise FileNotFoundError(
            f"Cannot resolve validation MAT from --data-root={data_root}"
        )

    raw = Path(cfg["data"]["val_split"])

    return (
        raw
        if raw.is_absolute()
        else REPO_ROOT / raw
    )


def load_geometries(
    mat_path: Path,
    max_geoms: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    ds = MetaDiTDataset(str(mat_path))

    n = min(
        int(max_geoms),
        len(ds),
    )

    if n < 8:
        raise RuntimeError(
            f"Need at least 8 geometries, got {n}"
        )

    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_batch,
    )

    geometry_chunks: List[torch.Tensor] = []
    remaining = n

    for geometry, _spectrum in loader:
        take = min(
            remaining,
            geometry.shape[0],
        )

        geometry_chunks.append(
            geometry[:take].clone()
        )

        remaining -= take

        if remaining == 0:
            break

    geometries = torch.cat(
        geometry_chunks,
        dim=0,
    )

    if geometries.shape[0] != n:
        raise RuntimeError(
            f"Collected {geometries.shape[0]} geometries, expected {n}"
        )

    params = np.asarray(
        ds.parameter[:n],
        dtype=np.float64,
    )

    if params.shape != (n, 3):
        raise RuntimeError(
            f"Expected params shape {(n, 3)}, got {params.shape}"
        )

    return geometries, params


def occupancy_binary(
    geometries: torch.Tensor,
) -> np.ndarray:
    """
    Recover occupancy from the geometry tensor.

    The dataset writes r_atom/h_atom only at occupied positions.
    Channel 2 is global and must NOT be used for occupancy.
    """

    occupied = (
        (geometries[:, 0] != 0)
        | (geometries[:, 1] != 0)
    )

    return occupied.cpu().numpy().astype(
        np.uint8
    )


def coarse_shape_features(
    occupancy: np.ndarray,
) -> np.ndarray:
    """
    Shape-only trivial baseline.

    Features:
      - occupancy fraction: 1 feature
      - 4x4 coarse occupancy grid: 16 features

    Total = 17 features.

    No physical parameter values are included.
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

    # [N,64,64]
    # -> [N,4,16,4,16]
    # -> [N,4,4]
    coarse_grid = occupancy.reshape(
        n,
        4,
        16,
        4,
        16,
    ).mean(axis=(2, 4))

    # [N,4,4] -> [N,16]
    coarse_features = coarse_grid.reshape(
        n,
        16,
    )

    # IMPORTANT: reduce both spatial dimensions before adding singleton dim.
    occupancy_fraction = occupancy.mean(
        axis=(1, 2)
    )[:, None]

    assert occupancy_fraction.shape == (
        n,
        1,
    )
    assert coarse_features.shape == (
        n,
        16,
    )

    features = np.concatenate(
        [
            occupancy_fraction,
            coarse_features,
        ],
        axis=1,
    ).astype(np.float64)

    assert features.shape == (
        n,
        17,
    )

    return features


def parse_bin_fractions(
    value: str,
) -> List[float]:
    fractions = []

    for token in value.split(","):
        x = float(token.strip())

        if x <= 0:
            raise ValueError(
                "Bin fractions must be > 0"
            )

        fractions.append(x)

    fractions = sorted(
        set(fractions)
    )

    return fractions


def quantize_parameters(
    params: np.ndarray,
    fraction: float,
):
    """
    Quantize each physical parameter independently.

    Bins are fractions of the observed range of that parameter.
    """

    mins = params.min(
        axis=0
    )

    maxs = params.max(
        axis=0
    )

    spans = np.maximum(
        maxs - mins,
        1e-12,
    )

    widths = spans * fraction

    keys = np.floor(
        (params - mins)
        / widths
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

    for index, key_array in enumerate(keys):

        key = (
            int(key_array[0]),
            int(key_array[1]),
            int(key_array[2]),
        )

        buckets.setdefault(
            key,
            [],
        ).append(index)

    return {
        key: np.asarray(
            indices,
            dtype=np.int64,
        )
        for key, indices in buckets.items()
        if len(indices) >= min_bucket_size
    }


def build_frozen_pairs(
    buckets: Dict[
        Tuple[int, int, int],
        np.ndarray,
    ],
    params: np.ndarray,
    parameter_spans: np.ndarray,
    seed: int,
    max_pairs_per_bucket: int,
    max_normalized_param_distance: float,
):
    """
    Create the exact representation-independent pair set.

    Pair eligibility:
      1. both samples belong to the same quantized bucket;
      2. if max_normalized_param_distance > 0, their normalized
         [l,h,r] distance must also be <= that threshold.

    This second check ensures "same bucket" cannot hide a large physical
    parameter difference.
    """

    rng = np.random.RandomState(
        seed
    )

    pairs_by_bucket = {}

    normalization = np.maximum(
        parameter_spans,
        1e-12,
    )

    for bucket_key in sorted(
        buckets
    ):

        members = buckets[
            bucket_key
        ]

        candidates = []

        for ai in range(
            len(members) - 1
        ):

            a = int(
                members[ai]
            )

            for bi in range(
                ai + 1,
                len(members),
            ):

                b = int(
                    members[bi]
                )

                if (
                    max_normalized_param_distance
                    > 0
                ):

                    delta = (
                        params[a]
                        - params[b]
                    ) / normalization

                    normalized_distance = float(
                        np.linalg.norm(delta)
                    )

                    if (
                        normalized_distance
                        > max_normalized_param_distance
                    ):
                        continue

                candidates.append(
                    (a, b)
                )

        if (
            max_pairs_per_bucket > 0
            and len(candidates)
            > max_pairs_per_bucket
        ):

            chosen = rng.choice(
                len(candidates),
                size=max_pairs_per_bucket,
                replace=False,
            )

            candidates = [
                candidates[int(i)]
                for i in np.sort(chosen)
            ]

        if candidates:

            pairs_by_bucket[
                bucket_key
            ] = np.asarray(
                candidates,
                dtype=np.int64,
            )

    return pairs_by_bucket


def evaluate_candidate_bins(
    params: np.ndarray,
    fraction: float,
    min_bucket_size: int,
    min_pairs_for_rho: int,
    min_normalized_distance: float,
    seed: int,
    max_pairs_per_bucket: int,
):
    """
    Fully evaluate a candidate bin width using the EXACT final pair-selection
    procedure. This avoids the previous bug where "possible pair" counts were
    checked before later pair filtering/capping.
    """

    keys, mins, maxs, spans, widths = (
        quantize_parameters(
            params,
            fraction,
        )
    )

    buckets = build_buckets(
        keys,
        min_bucket_size,
    )

    pairs = build_frozen_pairs(
        buckets,
        params,
        spans,
        seed,
        max_pairs_per_bucket,
        min_normalized_distance,
    )

    valid_pair_buckets = {
        key: value
        for key, value in pairs.items()
        if len(value) >= min_pairs_for_rho
    }

    total_pairs = sum(
        len(value)
        for value in valid_pair_buckets.values()
    )

    return {
        "fraction": float(fraction),
        "mins": mins,
        "maxs": maxs,
        "spans": spans,
        "widths": widths,
        "keys": keys,
        "buckets": buckets,
        "pairs": valid_pair_buckets,
        "usable_buckets": int(
            len(valid_pair_buckets)
        ),
        "total_pairs": int(
            total_pairs
        ),
        "max_bucket_size": int(
            max(
                (
                    len(v)
                    for v in buckets.values()
                ),
                default=0,
            )
        ),
    }


def select_bucket_candidate(
    params: np.ndarray,
    fractions: List[float],
    min_bucket_size: int,
    min_usable_buckets: int,
    min_rho_buckets: int,
    min_pairs_total: int,
    min_pairs_for_rho: int,
    max_normalized_distance: float,
    seed: int,
    max_pairs_per_bucket: int,
):
    trials = []

    best = None

    for fraction in fractions:

        result = evaluate_candidate_bins(
            params=params,
            fraction=fraction,
            min_bucket_size=min_bucket_size,
            min_pairs_for_rho=min_pairs_for_rho,
            min_normalized_distance=max_normalized_distance,
            seed=seed,
            max_pairs_per_bucket=max_pairs_per_bucket,
        )

        report = {
            "fraction": result["fraction"],
            "bin_widths": result["widths"].tolist(),
            "usable_buckets": result["usable_buckets"],
            "total_pairs": result["total_pairs"],
            "max_bucket_size": result["max_bucket_size"],
        }

        trials.append(
            report
        )

        if (
            result["usable_buckets"]
            >= min_usable_buckets
            and result["total_pairs"]
            >= min_pairs_total
            and result["usable_buckets"]
            >= min_rho_buckets
        ):
            best = result
            break

    # If no candidate reaches the requested threshold, choose the candidate
    # with the largest number of valid buckets/pairs and STOP with a clear
    # message rather than silently running an underpowered test.
    if best is None:

        ranked = sorted(
            trials,
            key=lambda x: (
                x["usable_buckets"],
                x["total_pairs"],
            ),
            reverse=True,
        )

        raise RuntimeError(
            "No quantization produced the requested statistical viability.\n"
            f"Best candidate: {ranked[0] if ranked else 'none'}\n"
            "This is a DATA-AVAILABILITY limitation. Do not loosen the "
            "thresholds silently. Either increase --max-geoms if more "
            "validation geometries exist, or explicitly adjust the matching "
            "criteria and document the change."
        )

    return best, trials


def hamming_distance(
    occupancy: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:

    result = np.empty(
        len(pairs),
        dtype=np.float64,
    )

    for j, (
        a,
        b,
    ) in enumerate(pairs):

        result[j] = float(
            np.mean(
                occupancy[a]
                != occupancy[b]
            )
        )

    return result


def euclidean_distance(
    features: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:

    result = np.empty(
        len(pairs),
        dtype=np.float64,
    )

    for j, (
        a,
        b,
    ) in enumerate(pairs):

        delta = (
            features[a]
            - features[b]
        )

        result[j] = float(
            np.linalg.norm(
                delta
            )
        )

    return result


def aligned_token_cosine_distance(
    embeddings: torch.Tensor,
    pairs: np.ndarray,
) -> np.ndarray:
    """
    No token mean-pooling.

    Distance for pair (a,b):
        1 - mean_t cosine(z_a,t, z_b,t)
    """

    result = np.empty(
        len(pairs),
        dtype=np.float64,
    )

    for j, (
        a,
        b,
    ) in enumerate(pairs):

        za = embeddings[a].float()
        zb = embeddings[b].float()

        cosine = (
            torch.nn.functional
            .cosine_similarity(
                za,
                zb,
                dim=-1,
            )
            .mean()
        )

        result[j] = (
            1.0
            - float(cosine.item())
        )

    return result


def within_bucket_spearman(
    shape_distance: np.ndarray,
    representation_distance: np.ndarray,
):
    """
    Compute Spearman rho for ONE bucket only.

    No cross-bucket pooling occurs here.
    """

    if len(shape_distance) < 3:
        return None

    if (
        np.allclose(
            shape_distance,
            shape_distance[0],
        )
        or np.allclose(
            representation_distance,
            representation_distance[0],
        )
    ):
        return None

    rho = spearmanr(
        shape_distance,
        representation_distance,
    ).statistic

    if rho is None or not np.isfinite(rho):
        return None

    return float(rho)


def aggregate_rhos(
    bucket_results: List[dict],
    rho_key: str,
):
    """
    Aggregate per-bucket rho values only.

    Provides:
      - median rho
      - unweighted mean rho
      - pair-count-weighted mean rho
      - number of valid buckets
      - total pairs contributing to valid buckets
    """

    values = []
    weighted_values = []
    weights = []

    for record in bucket_results:

        rho = record[rho_key]

        if rho is None or not np.isfinite(rho):
            continue

        pair_count = int(
            record["n_pairs"]
        )

        values.append(
            float(rho)
        )

        weighted_values.append(
            float(rho)
            * pair_count
        )

        weights.append(
            pair_count
        )

    if not values:
        return {
            "n_valid_buckets": 0,
            "total_pairs": 0,
            "median_rho": None,
            "mean_rho": None,
            "weighted_mean_rho": None,
        }

    total_weight = sum(
        weights
    )

    return {
        "n_valid_buckets": int(
            len(values)
        ),
        "total_pairs": int(
            total_weight
        ),
        "median_rho": float(
            np.median(values)
        ),
        "mean_rho": float(
            np.mean(values)
        ),
        "weighted_mean_rho": float(
            np.average(
                values,
                weights=weights,
            )
        ),
    }


def parameter_spread_stats(
    params: np.ndarray,
    pairs: np.ndarray,
    spans: np.ndarray,
):
    if len(pairs) == 0:
        return {
            "max_normalized_distance": None,
            "mean_normalized_distance": None,
            "median_normalized_distance": None,
        }

    distances = []

    normalization = np.maximum(
        spans,
        1e-12,
    )

    for a, b in pairs:

        delta = (
            params[a]
            - params[b]
        ) / normalization

        distances.append(
            np.linalg.norm(
                delta
            )
        )

    values = np.asarray(
        distances,
        dtype=np.float64,
    )

    return {
        "max_normalized_distance": float(
            values.max()
        ),
        "mean_normalized_distance": float(
            values.mean()
        ),
        "median_normalized_distance": float(
            np.median(values)
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
    hidden: int,
    heads: int,
    depth: int,
    seed: int,
    device: torch.device,
):
    with torch.random.fork_rng(
        devices=[]
    ):
        torch.manual_seed(
            seed
        )

        return GeometryEncoder(
            hidden=hidden,
            num_heads=heads,
            depth=depth,
        ).to(device)


def collect_embeddings(
    encoder: torch.nn.Module,
    geometry_batches: List[torch.Tensor],
    device: torch.device,
):
    actual_device = next(
        encoder.parameters()
    ).device

    if actual_device != device:
        raise RuntimeError(
            f"Encoder is on {actual_device}, requested {device}"
        )

    was_training = encoder.training
    encoder.eval()

    try:

        with torch.no_grad():

            chunks = [
                encoder(
                    batch.to(device)
                )
                .detach()
                .cpu()
                for batch in geometry_batches
            ]

        return torch.cat(
            chunks,
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
    # 1. Ground-truth data
    # ------------------------------------------------------------

    val_mat = resolve_val_mat(
        cfg,
        args.data_root,
    )

    geometries, params = load_geometries(
        val_mat,
        args.max_geoms,
    )

    occupancy = occupancy_binary(
        geometries
    )

    shape_features = coarse_shape_features(
        occupancy
    )

    if shape_features.shape != (
        len(geometries),
        17,
    ):
        raise RuntimeError(
            f"Shape baseline has wrong shape: "
            f"{shape_features.shape}"
        )

    # ------------------------------------------------------------
    # 2. Select and FREEZE bucket/pair set using only ground truth
    # ------------------------------------------------------------

    fractions = parse_bin_fractions(
        args.bin_fractions
    )

    selected, viability_trials = (
        select_bucket_candidate(
            params=params,
            fractions=fractions,
            min_bucket_size=args.min_bucket_size,
            min_usable_buckets=args.min_usable_buckets,
            min_rho_buckets=args.min_rho_buckets,
            min_pairs_total=args.min_pairs_total,
            min_pairs_for_rho=args.min_pairs_for_rho,
            max_normalized_distance=(
                args.max_normalized_param_distance
            ),
            seed=args.probe_seed,
            max_pairs_per_bucket=(
                args.max_pairs_per_bucket
            ),
        )
    )

    frozen_pairs = selected[
        "pairs"
    ]

    selected_spans = selected[
        "spans"
    ]

    total_frozen_pairs = sum(
        len(v)
        for v in frozen_pairs.values()
    )

    print()
    print("=" * 78)
    print("GROUND-TRUTH BUCKET VIABILITY")
    print("=" * 78)

    print(
        f"geometries used          : "
        f"{len(geometries)}"
    )

    print(
        f"selected bin fraction    : "
        f"{selected['fraction']:.6g}"
    )

    print(
        "bin widths [l,h,r]       : "
        f"{selected['widths'][0]:.8g}, "
        f"{selected['widths'][1]:.8g}, "
        f"{selected['widths'][2]:.8g}"
    )

    print(
        f"usable rho buckets      : "
        f"{selected['usable_buckets']}"
    )

    print(
        f"frozen pair count       : "
        f"{total_frozen_pairs}"
    )

    print(
        "[OK] pair set frozen before representation loading"
    )

    # ------------------------------------------------------------
    # 3. Load representations only after freezing pairs
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
        geometries[start:start + 64]
        for start in range(
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

    # ------------------------------------------------------------
    # 4. Same frozen pairs for every method
    # ------------------------------------------------------------

    bucket_records = []

    for bucket_key in sorted(
        frozen_pairs
    ):

        pairs = frozen_pairs[
            bucket_key
        ]

        shape_distance = hamming_distance(
            occupancy,
            pairs,
        )

        trivial_distance = euclidean_distance(
            shape_features,
            pairs,
        )

        trained_distance = aligned_token_cosine_distance(
            trained_embeddings,
            pairs,
        )

        released_distance = aligned_token_cosine_distance(
            released_embeddings,
            pairs,
        )

        random_distance = aligned_token_cosine_distance(
            random_embeddings,
            pairs,
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
            **parameter_spread_stats(
                params,
                pairs,
                selected_spans,
            ),
            "rho_trained_ema": (
                within_bucket_spearman(
                    shape_distance,
                    trained_distance,
                )
            ),
            "rho_released_vit": (
                within_bucket_spearman(
                    shape_distance,
                    released_distance,
                )
            ),
            "rho_random_init": (
                within_bucket_spearman(
                    shape_distance,
                    random_distance,
                )
            ),
            "rho_trivial_shape": (
                within_bucket_spearman(
                    shape_distance,
                    trivial_distance,
                )
            ),
        }

        bucket_records.append(
            record
        )

    # ------------------------------------------------------------
    # 5. Aggregate only within-bucket correlations
    # ------------------------------------------------------------

    aggregates = {
        "trained_ema": aggregate_rhos(
            bucket_records,
            "rho_trained_ema",
        ),
        "released_vit": aggregate_rhos(
            bucket_records,
            "rho_released_vit",
        ),
        "random_init": aggregate_rhos(
            bucket_records,
            "rho_random_init",
        ),
        "trivial_shape": aggregate_rhos(
            bucket_records,
            "rho_trivial_shape",
        ),
    }

    checksum_after = checksum(
        model
    )

    if checksum_after != checksum_before:
        raise RuntimeError(
            "Checkpoint parameters changed during this read-only diagnostic."
        )

    # ------------------------------------------------------------
    # 6. Human-readable output
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("WITHIN-BUCKET SPATIAL-STRUCTURE RESULT")
    print("=" * 78)

    print(
        f"{'representation':<22}"
        f"{'median rho':>14}"
        f"{'mean rho':>14}"
        f"{'weighted rho':>16}"
        f"{'valid buckets':>15}"
    )

    print("-" * 78)

    for key, label in [
        ("trained_ema", "trained EMA"),
        ("released_vit", "released ViT"),
        ("random_init", "random init"),
        ("trivial_shape", "trivial shape"),
    ]:

        result = aggregates[key]

        print(
            f"{label:<22}"
            f"{result['median_rho'] if result['median_rho'] is not None else float('nan'):>14.6f}"
            f"{result['mean_rho'] if result['mean_rho'] is not None else float('nan'):>14.6f}"
            f"{result['weighted_mean_rho'] if result['weighted_mean_rho'] is not None else float('nan'):>16.6f}"
            f"{result['n_valid_buckets']:>15d}"
        )

    trained = aggregates[
        "trained_ema"
    ]["weighted_mean_rho"]

    trivial = aggregates[
        "trivial_shape"
    ]["weighted_mean_rho"]

    random_value = aggregates[
        "random_init"
    ]["weighted_mean_rho"]

    released_value = aggregates[
        "released_vit"
    ]["weighted_mean_rho"]

    print()
    print("WEIGHTED WITHIN-BUCKET COMPARISONS")
    print("-" * 78)

    if (
        trained is not None
        and trivial is not None
    ):
        print(
            f"trained - trivial shape : "
            f"{trained - trivial:+.6f}"
        )

    if (
        trained is not None
        and random_value is not None
    ):
        print(
            f"trained - random       : "
            f"{trained - random_value:+.6f}"
        )

    if (
        trained is not None
        and released_value is not None
    ):
        print(
            f"trained - released     : "
            f"{trained - released_value:+.6f}"
        )

    print()
    print(
        "No automatic HEALTHY/COLLAPSED verdict is issued."
    )
    print(
        "Interpret only the within-bucket results."
    )

    # ------------------------------------------------------------
    # 7. Save complete audit report
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
        "n_geometries": int(
            len(geometries)
        ),
        "configuration": {
            "probe_seed": int(
                args.probe_seed
            ),
            "min_bucket_size": int(
                args.min_bucket_size
            ),
            "min_usable_buckets": int(
                args.min_usable_buckets
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
            "bin_fractions": fractions,
        },
        "bucket_selection": {
            "trials": viability_trials,
            "selected_fraction": float(
                selected["fraction"]
            ),
            "selected_bin_widths": (
                selected["widths"].tolist()
            ),
            "selected_parameter_min": (
                selected["mins"].tolist()
            ),
            "selected_parameter_max": (
                selected["maxs"].tolist()
            ),
            "selected_parameter_spans": (
                selected["spans"].tolist()
            ),
            "usable_buckets": int(
                selected["usable_buckets"]
            ),
            "total_frozen_pairs": int(
                total_frozen_pairs
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
                "occupancy fraction + 4x4 coarse occupancy grid"
            ),
            "correlation": (
                "Spearman within bucket"
            ),
            "aggregation": (
                "median, mean, and pair-count-weighted mean "
                "of valid within-bucket rho"
            ),
            "pair_set_locked_before_representation_loading": True,
            "representation_independent_pair_selection": True,
        },
        "aggregate_results": aggregates,
        "bucket_results": bucket_records,
        "read_only_verified": True,
    }

    with open(
        output_path,
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