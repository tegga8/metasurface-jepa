"""Stub-based tests for the NN-scoping probe helpers (no checkpoint needed).

Run: python -m pytest tests/test_nn_scoping.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "diagnostics"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import torch

import nn_scoping as ns


def test_novelty_quartiles_orders_by_distance():
    """Quartile 0 must be the closest neighbours; means must follow the order."""
    dists = [4.0, 1.0, 3.0, 2.0]          # ascending order of indices: 1, 3, 2, 0
    vals = [10.0, 20.0, 30.0, 40.0]       # value per item
    qs = ns.novelty_quartiles(dists, vals)
    # 4 items -> 4 quartiles of 1; q0=idx1(20) q1=idx3(40) q2=idx2(30) q3=idx0(10)
    assert [q[0] for q in qs] == [20.0, 40.0, 30.0, 10.0]


def test_novelty_quartiles_multiple_value_arrays():
    dists = [0.0, 1.0, 2.0, 3.0]
    a = [1.0, 2.0, 3.0, 4.0]
    b = [10.0, 20.0, 30.0, 40.0]
    qs = ns.novelty_quartiles(dists, a, b)
    assert [q[0] for q in qs] == [1.0, 2.0, 3.0, 4.0]
    assert [q[1] for q in qs] == [10.0, 20.0, 30.0, 40.0]


def test_retain_visible_keeps_visible_and_fills_masked():
    pred = torch.zeros(1, 1, 64, 64)
    occ_input = torch.ones(1, 1, 64, 64)
    mask = torch.ones(1, 16, 16)
    mask[:, :8, :] = 0.0                     # top half of the grid is masked
    out = ns.retain_visible(pred, occ_input, mask)
    assert out.shape == (1, 1, 64, 64)
    assert float(out[0, 0, :32, :].mean()) == 0.0     # masked -> pred
    assert float(out[0, 0, 32:, :].mean()) == 1.0     # visible -> input


def test_retain_visible_uses_input_on_visible_only():
    pred = torch.full((1, 1, 64, 64), 0.3)
    occ_input = torch.full((1, 1, 64, 64), 0.9)
    mask = torch.ones(1, 16, 16)             # everything visible
    out = ns.retain_visible(pred, occ_input, mask)
    assert torch.allclose(out, occ_input)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
