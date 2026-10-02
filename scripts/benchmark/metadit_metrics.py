"""MetaDiT-comparable spectrum metrics: MAE, AAE, AAE&K.

architecture_v5.md §10 mandates benchmarking against the released MetaDiT
diffusion transformer on the pure-inverse-design scenario. MetaDiT reports MAE,
AAE and AAE&K (external/metadit/metric.py); our own evaluator headlines a
target-std-normalized L1, which is scale-free and therefore not in the paper's
units. These functions put our numbers in the paper's units.

Reference definition (do not diverge silently):
    external/metadit/metric.py::mean_absolute_error
    external/metadit/metric.py::accumulate_absolute_error
    external/metadit/metric.py::calculate_aaeandk

The batch implementation here is VECTORIZED (per-item, so AAE&K has an AAE per
item) rather than the reference's scalar-per-item `.item()` loop. Parity with the
reference is enforced by tests/test_benchmark_metadit_metrics.py, which imports
the reference module and compares both -- the guard against silent drift.

Convention: spectra are [B, 2, 301] (`[real, imag]` x 301), matching
src/data/dataset.py and external/metadit (repo tensor order is (2, 301)).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADIT_SRC = os.path.join(REPO_ROOT, "external", "metadit")


def reference_module():
    """Import external/metadit/metric.py (the authoritative metric definition).

    Lazy so that importing this module never depends on the external tree being
    staged, and so the pure-math code paths stay import-free of torch models.
    """
    if METADIT_SRC not in sys.path:
        sys.path.insert(0, METADIT_SRC)
    import importlib
    return importlib.import_module("metric")


# ---------------------------------------------------------------------------
# Batched implementation (per item)
# ---------------------------------------------------------------------------

def per_item_mae(y_true, y_pred):
    """(B, 2, 301) -> (B,) mean absolute error per item (mean over C=2, F=301)."""
    return (y_pred - y_true).abs().flatten(1).mean(dim=1)


def per_item_aae(y_true, y_pred):
    """(B, 2, 301) -> (B,) accumulated absolute error per item (sum over C*F)."""
    return (y_pred - y_true).abs().flatten(1).sum(dim=1)


def mae(y_true, y_pred):
    """Scalar MAE over a batch (mean of per_item_mae) -- MetaDiT 'MAE'."""
    return float(per_item_mae(y_true, y_pred).mean().item())


def aae(y_true, y_pred):
    """Scalar AAE over a batch (mean of per_item_aae) -- MetaDiT 'AAE'."""
    return float(per_item_aae(y_true, y_pred).mean().item())


def aae_and_k(aae_matrix, k=None):
    """MetaDiT 'AAE&K': mean over conditions of the max AAE across the first K
    candidates.

    Args:
        aae_matrix: (N, J) tensor -- AAE of candidate j for condition i.
        k:          use the first k columns (default: all J).

    Returns:
        float. ``k == 1`` reduces to the plain AAE (max over a single candidate).

    NOTE: MetaDiT draws the K candidates from independent diffusion seeds. This
    repo's model is deterministic; candidates come from a decode-time generator
    (scripts/benchmark/candidate_sampling.py). The number is a decode-robustness
    analogue, never diffusion-seed diversity -- callers must carry that flag.
    """
    if aae_matrix.dim() != 2:
        raise ValueError(f"aae_matrix must be (N, J), got {tuple(aae_matrix.shape)}")
    a = aae_matrix if k is None else aae_matrix[:, :int(k)]
    if a.shape[1] < 1:
        raise ValueError("AAE&K needs at least one candidate column")
    return float(a.max(dim=1).values.mean().item())


# ---------------------------------------------------------------------------
# Reference delegation (parity tests only -- keeps the definition single-sourced)
# ---------------------------------------------------------------------------

def reference_mae(y_true, y_pred):
    return float(reference_module().mean_absolute_error(y_true, y_pred))


def reference_aae(y_true, y_pred):
    """Reference AAE. NOTE: `accumulate_absolute_error` SUMS the whole tensor, so
    this is only comparable to `aae()` per single item (as metric.eval_loop uses
    it) -- averaging over a batch is the caller's job."""
    return float(reference_module().accumulate_absolute_error(y_true, y_pred))


def reference_aae_and_k(aae_matrix, k):
    """Same value as aae_and_k, computed by the reference implementation."""
    a = aae_matrix[:, :int(k)]
    n = a.shape[0]
    data_dict = {
        f"seed{j}": {"aaes": [float(a[i, j].item()) for i in range(n)]}
        for j in range(int(k))
    }
    return float(reference_module().calculate_aaeandk(data_dict, int(k)))
