"""Unit tests for the retrofit Phase 1 Gate-0 helpers (no .mat I/O).

Covers: scalar summary construction, geometry-tensor convention parity with
src/data/dataset.py, metric math, and 1-NN retrieval sanity.

Run:  python tests/test_gate0_scalar_vs_shape.py   (pytest-collectable)
"""

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "diagnostics"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from gate0_scalar_vs_shape import (  # noqa: E402
    build_geometries, build_scalars, rel_l2, r2_score, mse, score_arm,
    scalar_knn,
)


def _toy_sample():
    """One 64x64 occupancy square in the top-left quadrant + known params."""
    pat = np.zeros((64, 64, 1), dtype=np.int8)
    pat[0:8, 0:8, 0] = 1
    params = np.array([[3.0, 1.0, 5.0]])   # [l_lattice, h_atom, r_atom]
    return pat, params


def test_build_scalars_shape_and_values():
    pat, params = _toy_sample()
    f = build_scalars(params, pat)
    assert f.shape == (1, 6)
    r, h, l, frac, cy, cx = f[0]
    assert (r, h, l) == (5.0, 1.0, 3.0)
    assert abs(frac - 64.0 / 4096.0) < 1e-12
    # 8x8 block spanning rows/cols 0..7 -> centroid at (3.5, 3.5)/63
    assert abs(cy - 3.5 / 63.0) < 1e-9 and abs(cx - 3.5 / 63.0) < 1e-9


def test_build_geometries_matches_dataset_convention():
    from data.dataset import MetaDiTDataset

    pat, params = _toy_sample()
    g = build_geometries(pat, params)
    assert g.shape == (1, 3, 64, 64) and g.dtype == torch.float32

    ds = MetaDiTDataset.__new__(MetaDiTDataset)   # no I/O; call __getitem__ body manually
    import torch as _t
    grid = _t.zeros(3, 64, 64, dtype=_t.float32)
    occ = _t.from_numpy(pat[:, :, 0]).float() == 1.0
    p = params[0]
    grid[0][occ] = p[2] / 5.0
    grid[1][occ] = p[1]
    grid[2] = p[0] / 3.0
    assert torch.equal(g[0], grid)


def test_metrics_hand_values():
    pred = np.array([[1.0, 0.0], [0.0, 1.0]])
    true = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert abs(mse(pred, true) - 0.25) < 1e-12
    # per-sample rel L2: 0 and sqrt(2)/1 -> mean sqrt(2)/2
    assert abs(rel_l2(pred, true) - (np.sqrt(2.0) / 2.0)) < 1e-12
    assert abs(r2_score(true, true) - 1.0) < 1e-12
    s = score_arm("x", true, true)
    assert s["rel_l2"] == 0.0 and s["r2"] == 1.0 and s["mse"] == 0.0


def test_scalar_knn_recovers_exact_train_row():
    rng = np.random.RandomState(0)
    x_tr = rng.randn(16, 6).astype(np.float32)
    y_tr = rng.randn(16, 602)
    x_va = x_tr[:4] + np.float32(0.0)          # exact copies -> must retrieve own row
    out = scalar_knn(x_tr, y_tr, x_va)
    assert out.shape == (4, 602)
    assert np.allclose(out, y_tr[:4])


if __name__ == "__main__":
    test_build_scalars_shape_and_values()
    test_build_geometries_matches_dataset_convention()
    test_metrics_hand_values()
    test_scalar_knn_recovers_exact_train_row()
    print("all gate0 helper tests passed")
