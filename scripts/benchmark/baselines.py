"""Baselines for the MetaDiT-comparable benchmark, all in MAE/AAE units.

Arms:
    AVG1        predict the dataset-mean spectrum (paper: 0.5860 / 352.7424).
    NN          L1-nearest real training spectrum -> its real geometry -> frozen
                surrogate (the retrofit baseline architecture_v5.md §10 names).
    surrogate   surrogate(G_true) vs S_true -- the achievable floor (paper ~0.0084).

MAE/AAE are imported from metadit_metrics so every arm sits in one table, in the
paper's units.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")
BENCH_DIR = os.path.join(REPO_ROOT, "scripts", "benchmark")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

import torch

import metadit_metrics as mm
from data.factorize import assemble_metadit_geometry


def mean_spectrum(specs):
    """(N, 2, 301) -> (2, 301) dataset-mean spectrum."""
    return specs.mean(dim=0)


@torch.no_grad()
def avg1_metrics(target_spec, mean_spec):
    """AVG1: the same constant spectrum predicted for every item."""
    pred = mean_spec.unsqueeze(0).expand_as(target_spec)
    return {"MAE": mm.mae(target_spec, pred), "AAE": mm.aae(target_spec, pred)}


@torch.no_grad()
def surrogate_floor_metrics(occ, sv, spec, surrogate):
    """Surrogate self-consistency error on the TRUE geometry, in MAE units.

    No geometry->spectrum method can beat this: it is the surrogate's own error
    against the target. In the paper the released surrogate error is ~0.0084.
    """
    geom = assemble_metadit_geometry(occ, sv[:, 0], sv[:, 1], sv[:, 2])
    pred = surrogate(geom).prediction
    return {"MAE": mm.mae(spec, pred), "AAE": mm.aae(spec, pred)}


@torch.no_grad()
def nn_metrics(target_spec, train_specs, train_occ, train_sv, surrogate):
    """L1 nearest real training spectrum -> its real geometry -> surrogate.

    Memory-safe: expects already-factorized, capped training tensors (the caller
    limits n_train; the full 139,906 x 2 x 301 tensor does not fit comfortably).
    Returns per-item MAE/AAE lists and the mean retrieval distance.
    """
    n = target_spec.shape[0]
    mae_vals, aae_vals, dists = [], [], []
    for i in range(n):
        target = target_spec[i:i + 1]
        d = (train_specs - target).abs().mean(dim=(-2, -1))
        best = int(d.argmin())
        dists.append(float(d[best].item()))
        nn_geom = assemble_metadit_geometry(
            train_occ[best:best + 1],
            train_sv[best:best + 1, 0],
            train_sv[best:best + 1, 1],
            train_sv[best:best + 1, 2])
        pred = surrogate(nn_geom).prediction
        mae_vals.append(float(mm.per_item_mae(target, pred).mean().item()))
        aae_vals.append(float(mm.per_item_aae(target, pred).mean().item()))
    import numpy as np
    return {
        "MAE": float(np.mean(mae_vals)),
        "AAE": float(np.mean(aae_vals)),
        "nn_mean_retrieval_distance": float(np.mean(dists)),
        "n": int(n),
    }
