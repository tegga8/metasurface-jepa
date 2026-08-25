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
  6. Aggregate only AFTER within-bucket correlations are computed:
       - median rho
       - mean rho
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
    parser = argparse.ArgumentParser(
        description="Within-bucket spatial-structure probe"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--config",
        default=str(
            REPO_ROOT / "configs" / "milestone_b.yaml"
        ),
    )

    parser.add_argument(
        "--data-root",
        default=None,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--max-geoms",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--probe-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--min-bucket-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--min-usable-buckets",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min-total-pairs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max-pairs-per-bucket",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--min-pairs-for-rho",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT
            / "checkpoints"
            / "milestone_b"
            / "within_bucket_spatial_probe.json"
        ),
    )

    return parser.parse_args()


def resolve_val_mat(
    cfg: dict,
    data_root: str | None,
) -> Path:
    """
    Resolve the validation MAT file.

    --data-root may point to:
      - data/metadit/
      - a direct .mat file
    """

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

    if raw.is_absolute():
        return raw

    return REPO_ROOT / raw


def load_geometries(
    mat_path: Path,
    max_geoms: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load a deterministic prefix of the validation set.

    Returns:
        geometries: [N, 3, 64, 64]
        params:     [N, 3] ordered [l_lattice, h_atom, r_atom]
    """

    ds = MetaDiTDataset(str(mat_path))

    n = min(int(max_geoms), len(ds))

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

    chunks: List[torch.Tensor] = []
    remaining = n

    for G, _S in loader:
        take = min(remaining, G.shape[0])

        chunks.append(
            G[:take].clone()
        )

        remaining -= take

        if remaining == 0:
            break

    geometries = torch.cat(
        chunks,
        dim=0,
    )

    if geometries.shape[0] != n:
        raise RuntimeError(
            f"Collected {geometries.shape[0]} geometries, "
            f"expected {n}"
        )

    params = np.asarray(
        ds.parameter[:n],
        dtype=np.float64,
    )

    if params.shape != (n, 3):
        raise RuntimeError(
            f"Expected parameter array [{n},3], "
            f"got {params.shape}"
        )

    return geometries, params


def occupancy_binary(
    geometries: torch.Tensor,
) -> np.ndarray:
    """
    Recover binary occupancy from the geometry tensor.

    The dataset loader writes nonzero r_atom/h_atom values only at occupied
    locations, while the third channel is global.
    """

    occ = (
        (geometries[:, 0] != 0)
        | (geometries[:, 1] != 0)
    )

    return occ.cpu().numpy().astype(np.uint8)


def coarse_shape_features(occ: np.ndarray) -> np.ndarray:
    """
    Shape-aware trivial baseline.

    Features:
      - occupancy fraction: [N,1]
      - 4x4 coarse occupancy grid: [N,16]

    Total: [N,17].
    """
    if occ.ndim != 3:
        raise ValueError(f"Expected [N,H,W], got {occ.shape}")

    n, h, w = occ.shape

    if (h, w) != (64, 64):
        raise ValueError(f"Expected 64x64 occupancy, got {occ.shape}")

    # [N,64,64] -> [N,4,16,4,16] -> [N,4,4]
    coarse_2d = occ.reshape(n, 4, 16, 4, 16).mean(axis=(2, 4))

    # [N,4,4] -> [N,16]
    coarse = coarse_2d.reshape(n, 16)

    # IMPORTANT: reduce BOTH spatial dimensions first.
    # Result is [N,1], not [N,1,1].
    occupancy_fraction = occ.mean(axis=(1, 2))[:, None]

    assert occupancy_fraction.shape == (n, 1)
    assert coarse.shape == (n, 16)

    features = np.concatenate(
        [occupancy_fraction, coarse],
        axis=1,
    ).astype(np.float64)

    assert features.shape == (n, 17)

    return features

def robust_quantize(
    params: np.ndarray,
    fractions: List[float],
) -> list:
    """
    Try relative bin widths as fractions of the observed parameter ranges.

    We start with narrow bins and widen only if necessary.
    """

    spans = (
        params.max(axis=0)
        - params.min(axis=0)
    )

    scales = np.maximum(
        spans,
        1e-12,
    )

    candidates = []

    for fraction in fractions:

        widths = (
            scales * fraction
        )

        mins = params.min(axis=0)

        keys = np.floor(
            (params - mins)
            / widths
        ).astype(np.int64)

        candidates.append(
            (
                fraction,
                widths,
                mins,
                keys,
            )
        )

    return candidates


def bucket_map(
    keys: np.ndarray,
    min_bucket_size: int,
) -> Dict[
    Tuple[int, int, int],
    np.ndarray,
]:
    """
    Convert quantized parameter keys to buckets.

    Only buckets with at least min_bucket_size members survive.
    """

    buckets: Dict[
        Tuple[int, int, int],
        List[int],
    ] = {}

    for i, key_array in enumerate(keys):

        key = (
            int(key_array[0]),
            int(key_array[1]),
            int(key_array[2]),
        )

        buckets.setdefault(
            key,
            [],
        ).append(i)

    return {
        key: np.asarray(
            indices,
            dtype=np.int64,
        )
        for key, indices in buckets.items()
        if len(indices) >= min_bucket_size
    }


def pair_indices(
    buckets: Dict[
        Tuple[int, int, int],
        np.ndarray,
    ],
    seed: int,
    max_pairs_per_bucket: int,
) -> Dict[
    Tuple[int, int, int],
    np.ndarray,
]:
    """
    Freeze the exact pair set using bucket membership only.

    Representation values are never consulted here.
    """

    rng = np.random.RandomState(
        seed
    )

    output = {}

    for key in sorted(buckets):

        indices = buckets[key]

        pairs = []

        for a in range(
            len(indices) - 1
        ):
            for b in range(
                a + 1,
                len(indices),
            ):
                pairs.append(
                    (
                        int(indices[a]),
                        int(indices[b]),
                    )
                )

        if len(pairs) > max_pairs_per_bucket:

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
            output[key] = np.asarray(
                pairs,
                dtype=np.int64,
            )

    return output


def viability(
    buckets: Dict[
        Tuple[int, int, int],
        np.ndarray,
    ],
    min_total_pairs: int,
) -> Dict[str, int]:

    total_pairs = sum(
        len(values)
        * (len(values) - 1)
        // 2
        for values in buckets.values()
    )

    return {
        "usable_buckets": int(
            len(buckets)
        ),
        "total_possible_pairs": int(
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
        "n_geometries_in_usable_buckets": int(
            sum(
                len(v)
                for v in buckets.values()
            )
        ),
        "viable": int(
            (
                len(buckets) >= 1
                and total_pairs >= min_total_pairs
            )
        ),
    }


def select_bucket_width(
    params: np.ndarray,
    min_bucket_size: int,
    min_usable_buckets: int,
    min_total_pairs: int,
):
    """
    Select the narrowest parameter quantization that gives enough
    within-bucket pairs.

    No representation is loaded during this stage.
    """

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

    candidates = robust_quantize(
        params,
        fractions,
    )

    report = []

    for (
        fraction,
        widths,
        mins,
        keys,
    ) in candidates:

        buckets = bucket_map(
            keys,
            min_bucket_size,
        )

        result = viability(
            buckets,
            min_total_pairs,
        )

        result.update(
            {
                "relative_bin_fraction": float(
                    fraction
                ),
                "bin_width_l_lattice": float(
                    widths[0]
                ),
                "bin_width_h_atom": float(
                    widths[1]
                ),
                "bin_width_r_atom": float(
                    widths[2]
                ),
            }
        )

        report.append(result)

        if (
            result["usable_buckets"]
            >= min_usable_buckets
            and result["total_possible_pairs"]
            >= min_total_pairs
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
        "Bucket viability failed. "
        "No tested quantization width produced "
        f">={min_usable_buckets} usable buckets and "
        f">={min_total_pairs} possible pairs. "
        "Increase --max-geoms or widen the candidate fractions."
    )


def cosine_token_distance(
    X: torch.Tensor,
    pairs: np.ndarray,
) -> np.ndarray:
    """
    Representation distance:

        1 - mean tokenwise cosine similarity

    over aligned 256 spatial tokens.

    X shape:
        [N, T, D]
    """

    output = np.empty(
        len(pairs),
        dtype=np.float64,
    )

    for j, (
        a,
        b,
    ) in enumerate(pairs):

        xa = X[a].float()
        xb = X[b].float()

        cosine = torch.nn.functional.cosine_similarity(
            xa,
            xb,
            dim=-1,
        ).mean()

        output[j] = float(
            1.0 - cosine.item()
        )

    return output


def euclidean_distance(
    X: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:

    output = np.empty(
        len(pairs),
        dtype=np.float64,
    )

    for j, (
        a,
        b,
    ) in enumerate(pairs):

        delta = X[a] - X[b]

        output[j] = float(
            np.sqrt(
                np.dot(
                    delta,
                    delta,
                )
            )
        )

    return output


def hamming_distance(
    occupancy: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:

    output = np.empty(
        len(pairs),
        dtype=np.float64,
    )

    for j, (
        a,
        b,
    ) in enumerate(pairs):

        output[j] = float(
            np.mean(
                occupancy[a]
                != occupancy[b]
            )
        )

    return output


def spearman_for_pairs(
    shape_distance: np.ndarray,
    representation_distance: np.ndarray,
) -> float:
    """
    Spearman rho for a single bucket.

    Returns NaN if either distance vector has insufficient variation.
    """

    if len(shape_distance) < 3:
        return float("nan")

    if np.allclose(
        shape_distance,
        shape_distance[0],
    ):
        return float("nan")

    if np.allclose(
        representation_distance,
        representation_distance[0],
    ):
        return float("nan")

    result = spearmanr(
        shape_distance,
        representation_distance,
    ).statistic

    if result is None or not np.isfinite(result):
        return float("nan")

    return float(result)


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


def collect_embeddings(
    encoder: torch.nn.Module,
    geometry_batches: List[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Run a frozen encoder over deterministic geometry batches.
    """

    actual_device = next(
        encoder.parameters()
    ).device

    if actual_device != device:
        raise RuntimeError(
            f"Encoder is on {actual_device}, "
            f"requested {device}"
        )

    was_training = encoder.training

    encoder.eval()

    try:

        with torch.no_grad():

            chunks = [
                encoder(
                    geometries.to(device)
                )
                .detach()
                .cpu()
                for geometries in geometry_batches
            ]

        return torch.cat(
            chunks,
            dim=0,
        )

    finally:

        encoder.train(
            was_training
        )


def build_random_encoder(
    hidden: int,
    heads: int,
    depth: int,
    seed: int,
    device: torch.device,
) -> GeometryEncoder:
    """
    Build random encoder without changing global RNG state.
    """

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


def main() -> None:

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
    # Load ground-truth data BEFORE representations.
    # ------------------------------------------------------------

    validation_mat = resolve_val_mat(
        cfg,
        args.data_root,
    )

    geometries, params = load_geometries(
        validation_mat,
        args.max_geoms,
    )

    occupancy = occupancy_binary(
        geometries
    )

    shape_features = coarse_shape_features(
        occupancy
    )

    # ------------------------------------------------------------
    # Phase 1: bucket viability
    #
    # Absolutely no learned representation is used here.
    # ------------------------------------------------------------

    (
        selected_fraction,
        bin_widths,
        parameter_mins,
        bucket_keys,
        buckets,
        viability_report,
    ) = select_bucket_width(
        params=params,
        min_bucket_size=args.min_bucket_size,
        min_usable_buckets=args.min_usable_buckets,
        min_total_pairs=args.min_total_pairs,
    )

    frozen_pairs = pair_indices(
        buckets=buckets,
        seed=args.probe_seed,
        max_pairs_per_bucket=args.max_pairs_per_bucket,
    )

    usable_pair_buckets = {
        key: pairs
        for key, pairs in frozen_pairs.items()
        if len(pairs)
        >= args.min_pairs_for_rho
    }

    total_frozen_pairs = sum(
        len(pairs)
        for pairs in usable_pair_buckets.values()
    )

    if (
        len(usable_pair_buckets)
        < args.min_usable_buckets
        or total_frozen_pairs
        < args.min_total_pairs
    ):
        raise RuntimeError(
            "After pair capping the frozen pair set is still too sparse: "
            f"buckets={len(usable_pair_buckets)}, "
            f"pairs={total_frozen_pairs}. "
            "Increase --max-geoms or --max-pairs-per-bucket."
        )

    print()
    print(
        "BUCKET VIABILITY"
    )
    print(
        "=" * 78
    )

    print(
        f"validation geometries used : "
        f"{len(geometries)}"
    )

    print(
        f"selected relative bin width: "
        f"{selected_fraction:.6g} × parameter range"
    )

    print(
        "bin widths [l,h,r]         : "
        f"{bin_widths[0]:.8g}, "
        f"{bin_widths[1]:.8g}, "
        f"{bin_widths[2]:.8g}"
    )

    print(
        f"usable buckets             : "
        f"{len(buckets)}"
    )

    print(
        f"usable pair buckets        : "
        f"{len(usable_pair_buckets)}"
    )

    print(
        f"frozen pair count          : "
        f"{total_frozen_pairs}"
    )

    print(
        "[OK] bucket/pair set is viable"
    )

    bucket_report = {
        "selected_relative_bin_fraction": float(
            selected_fraction
        ),
        "bin_widths": {
            "l_lattice": float(
                bin_widths[0]
            ),
            "h_atom": float(
                bin_widths[1]
            ),
            "r_atom": float(
                bin_widths[2]
            ),
        },
        "parameter_min": (
            parameter_mins
            .tolist()
        ),
        "parameter_max": (
            params.max(
                axis=0
            ).tolist()
        ),
        "viability_trials": viability_report,
        "usable_buckets": int(
            len(buckets)
        ),
        "usable_pair_buckets": int(
            len(
                usable_pair_buckets
            )
        ),
        "frozen_pair_count": int(
            total_frozen_pairs
        ),
    }

    # ------------------------------------------------------------
    # Phase 2: load trained / released / random representations.
    # The pair set is already frozen.
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

    # Trained EMA representation.
    trained_embeddings = collect_embeddings(
        model.ema,
        geometry_batches,
        device,
    )

    hidden = int(
        checkpoint_cfg["model"]
        .get(
            "hidden",
            384,
        )
    )

    heads = int(
        checkpoint_cfg["model"]
        .get(
            "num_heads",
            6,
        )
    )

    depth = int(
        checkpoint_cfg["model"]
        .get(
            "geo_depth",
            6,
        )
    )

    # Released MetaDiT encoder.
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

    # Random-init encoder.
    random_encoder = build_random_encoder(
        hidden=hidden,
        heads=heads,
        depth=depth,
        seed=args.probe_seed,
        device=device,
    )

    random_embeddings = collect_embeddings(
        random_encoder,
        geometry_batches,
        device,
    )

    # ------------------------------------------------------------
    # Phase 3: same frozen pairs, same ground-truth shape distance.
    # ------------------------------------------------------------

    bucket_results = []

    rho_lists = {
        "trained_ema": [],
        "released_vit": [],
        "random_init": [],
        "trivial_shape": [],
    }

    for bucket_id in sorted(
        usable_pair_buckets
    ):

        pairs = (
            usable_pair_buckets[
                bucket_id
            ]
        )

        # Ground-truth occupancy distance.
        shape_distance = hamming_distance(
            occupancy,
            pairs,
        )

        # Shape-only trivial baseline distance.
        trivial_distance = euclidean_distance(
            shape_features,
            pairs,
        )

        # Learned representation distances.
        trained_distance = cosine_token_distance(
            trained_embeddings,
            pairs,
        )

        released_distance = cosine_token_distance(
            released_embeddings,
            pairs,
        )

        random_distance = cosine_token_distance(
            random_embeddings,
            pairs,
        )

        rho_trained = spearman_for_pairs(
            shape_distance,
            trained_distance,
        )

        rho_released = spearman_for_pairs(
            shape_distance,
            released_distance,
        )

        rho_random = spearman_for_pairs(
            shape_distance,
            random_distance,
        )

        rho_trivial = spearman_for_pairs(
            shape_distance,
            trivial_distance,
        )

        bucket_results.append(
            {
                "bucket": list(
                    bucket_id
                ),
                "n_samples": int(
                    len(
                        buckets[
                            bucket_id
                        ]
                    )
                ),
                "n_pairs": int(
                    len(pairs)
                ),
                "rho_trained_ema": (
                    rho_trained
                ),
                "rho_released_vit": (
                    rho_released
                ),
                "rho_random_init": (
                    rho_random
                ),
                "rho_trivial_shape": (
                    rho_trivial
                ),
            }
        )

        rho_lists[
            "trained_ema"
        ].append(
            rho_trained
        )

        rho_lists[
            "released_vit"
        ].append(
            rho_released
        )

        rho_lists[
            "random_init"
        ].append(
            rho_random
        )

        rho_lists[
            "trivial_shape"
        ].append(
            rho_trivial
        )

    # ------------------------------------------------------------
    # Aggregate ONLY AFTER within-bucket correlations.
    # ------------------------------------------------------------

    def aggregate(
        values: List[float],
    ) -> dict:

        valid = np.asarray(
            [
                value
                for value in values
                if np.isfinite(value)
            ],
            dtype=np.float64,
        )

        if len(valid) == 0:
            return {
                "n_buckets": 0,
                "median_rho": float("nan"),
                "mean_rho": float("nan"),
            }

        return {
            "n_buckets": int(
                len(valid)
            ),
            "median_rho": float(
                np.median(valid)
            ),
            "mean_rho": float(
                np.mean(valid)
            ),
        }

    aggregate_results = {
        key: aggregate(values)
        for key, values in rho_lists.items()
    }

    # ------------------------------------------------------------
    # Read-only integrity check.
    # ------------------------------------------------------------

    checksum_after = checksum(
        model
    )

    if checksum_after != checksum_before:
        raise RuntimeError(
            "Checkpoint model parameters changed "
            "during the spatial probe."
        )

    # ------------------------------------------------------------
    # Print final result.
    # ------------------------------------------------------------

    print()
    print(
        "WITHIN-BUCKET SPATIAL-STRUCTURE RESULT"
    )
    print(
        "=" * 78
    )

    print(
        f"{'representation':<22}"
        f"{'median rho':>14}"
        f"{'mean rho':>14}"
        f"{'buckets':>12}"
    )

    print(
        "-" * 78
    )

    labels = [
        (
            "trained_ema",
            "trained EMA",
        ),
        (
            "released_vit",
            "released ViT",
        ),
        (
            "random_init",
            "random init",
        ),
        (
            "trivial_shape",
            "trivial shape",
        ),
    ]

    for key, label in labels:

        result = (
            aggregate_results[
                key
            ]
        )

        print(
            f"{label:<22}"
            f"{result['median_rho']:>14.6f}"
            f"{result['mean_rho']:>14.6f}"
            f"{result['n_buckets']:>12d}"
        )

    trained_mean = (
        aggregate_results[
            "trained_ema"
        ]["mean_rho"]
    )

    trivial_mean = (
        aggregate_results[
            "trivial_shape"
        ]["mean_rho"]
    )

    random_mean = (
        aggregate_results[
            "random_init"
        ]["mean_rho"]
    )

    released_mean = (
        aggregate_results[
            "released_vit"
        ]["mean_rho"]
    )

    print()
    print(
        "COMPARISONS"
    )
    print(
        "-" * 78
    )

    print(
        f"trained - trivial shape : "
        f"{trained_mean - trivial_mean:+.6f}"
    )

    print(
        f"trained - random       : "
        f"{trained_mean - random_mean:+.6f}"
    )

    print(
        f"trained - released     : "
        f"{trained_mean - released_mean:+.6f}"
    )

    print()
    print(
        "Interpretation must use only the within-bucket results."
    )

    print(
        "Do not use variation between parameter buckets or the "
        "earlier mean-pooled collapse gate to infer spatial geometry."
    )

    # ------------------------------------------------------------
    # JSON report.
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

    payload = {
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
        "probe_seed": int(
            args.probe_seed
        ),
        "bucket_selection": bucket_report,
        "protocol": {
            "shape_distance": (
                "binary occupancy Hamming distance"
            ),
            "representation_distance": (
                "mean aligned-token cosine distance"
            ),
            "trivial_baseline": (
                "occupancy fraction + 4x4 coarse occupancy"
            ),
            "correlation": (
                "Spearman within each bucket; "
                "aggregate only afterward"
            ),
            "pair_set_locked_from_ground_truth": True,
        },
        "aggregate_results": aggregate_results,
        "bucket_results": bucket_results,
        "read_only_verified": True,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    print()
    print(
        f"JSON: {output_path}"
    )


if __name__ == "__main__":
    main()