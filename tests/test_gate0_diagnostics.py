"""Deterministic tests for the Gate 0 / Gate 0.5 diagnostics.

Run:  python tests/test_gate0_diagnostics.py        (also collectable by pytest)

Covers the pure helpers shared by:
    scripts/diagnostics/gate0_occupancy_audit.py
    scripts/diagnostics/gate0_5_trivial_baseline.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "diagnostics"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np

import gate0_occupancy_audit as g0
import gate0_5_trivial_baseline as g05


def _sample_pattern(seed=0):
    """Small deterministic synthetic occupancy: filled disk on a 16x16 grid."""
    rng = np.random.RandomState(seed)
    pat = np.zeros((64, 64, 4), dtype=np.int8)
    for k in range(4):
        cy, cx = rng.randint(10, 54, size=2)
        rr = 3.5 + rng.randint(0, 4) * 0.5
        yy, xx = np.mgrid[:64, :64]
        pat[:, :, k] = ((yy - cy) ** 2 + (xx - cx) ** 2) <= rr ** 2
    return pat


def _sample_params(seed=0):
    rng = np.random.RandomState(seed)
    return np.stack([
        np.round(rng.uniform(2.5, 3.0, 4), 2),
        np.round(rng.uniform(0.5, 1.0, 4), 2),
        np.round(rng.uniform(3.5, 5.0, 4), 2),
    ], axis=1)


def test_quantize_params_exact_bucketing():
    p = np.array([[2.5, 0.5, 3.5], [2.99, 0.82, 4.95], [2.5, 0.5, 3.5]])
    q = g0.quantize_params(p)
    assert np.array_equal(q[0], q[2]), "identical params must bucket together"
    assert np.array_equal(q[1], [2.99, 0.82, 4.95])
    # float noise at the 3rd decimal must not move the bucket
    p2 = np.array([2.5 + 1e-9, 0.5 - 1e-9, 3.5 + 1e-9])
    assert np.array_equal(g0.quantize_params(p2[None])[0], [2.5, 0.5, 3.5])


def test_pack_popcount_iou():
    pat = _sample_pattern()
    flat = pat.reshape(64 * 64, 4).T  # (4, 4096)
    packed = g0.pack_patterns(pat)     # (4, 512)
    for k in range(4):
        assert np.packbits(flat[k]) is not None
    a, b = packed[0], packed[1]
    h = g0.popcount_hamming(a[None], b[None])[0]
    direct = int(np.count_nonzero(pat[:, :, 0] != pat[:, :, 1]))
    assert h == direct, f"hamming {h} != direct {direct}"
    iou = g0.mask_iou(a[None], b[None])[0]
    inter = int(np.count_nonzero(pat[:, :, 0] & pat[:, :, 1]))
    union = int(np.count_nonzero(pat[:, :, 0] | pat[:, :, 1]))
    assert abs(iou - inter / union) < 1e-12
    # identical masks give IoU 1 and hamming 0
    assert g0.mask_iou(a[None], a[None])[0] == 1.0
    assert g0.popcount_hamming(a[None], a[None])[0] == 0


def test_visible_param_features_recovers_values():
    """Under the exact dataset encoding, visible pixels recover grid-encoded
    l/3, h, r/5 exactly (the encoding src/data/dataset.py paints and the
    geometry encoder consumes)."""
    rng = np.random.RandomState(3)
    for k in range(20):
        l, h, r = np.round(rng.uniform(2.5, 3.0), 2), \
                  np.round(rng.uniform(0.5, 1.0), 2), \
                  np.round(rng.uniform(3.5, 5.0), 2)
        pat = _sample_pattern(seed=k)
        occ = pat[:, :, 0] == 1
        G = torch.zeros(3, 64, 64)
        G[0][torch.from_numpy(occ)] = r / 5.0
        G[1][torch.from_numpy(occ)] = h
        G[2] = l / 3.0
        M = torch.ones(16, 16)                       # everything visible
        f = g05.visible_param_features(G, M)
        assert abs(f[0] - l / 3.0) < 1e-4, (f[0], l / 3.0)
        assert abs(f[1] - h) < 1e-4, (f[1], h)
        assert abs(f[2] - r / 5.0) < 1e-4, (f[2], r / 5.0)
        assert f[3] == 1.0 and f[4] == 1.0 and f[5] == 1.0


def test_visible_param_features_masked_zeros():
    """Fully masked geometry: all flags 0, values 0 (l too — no visible pixels)."""
    rng = np.random.RandomState(4)
    l, h, r = np.round(rng.uniform(2.5, 3.0), 2), \
              np.round(rng.uniform(0.5, 1.0), 2), \
              np.round(rng.uniform(3.5, 5.0), 2)
    pat = _sample_pattern(seed=1)
    occ = pat[:, :, 0] == 1
    G = torch.zeros(3, 64, 64)
    G[0][torch.from_numpy(occ)] = r / 5.0
    G[1][torch.from_numpy(occ)] = h
    G[2] = l / 3.0
    M = torch.zeros(16, 16)                          # nothing visible
    f = g05.visible_param_features(G, M)
    assert list(f) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_lookup_render_exact_and_nn():
    rng = np.random.RandomState(5)
    keys = np.stack([np.full(8, 2.5), np.full(8, 0.5),
                     np.linspace(3.5, 5.0, 8)], axis=1)
    masks = rng.randint(0, 2, size=(8, 64, 64)).astype(np.uint8)
    lookup = g05.build_lookup(keys, masks)
    # exact-match prediction renders the exact training mask
    pred = np.array([[2.5, 0.5, 3.5]])
    rendered, exact, dist = g05.render_occupancy_lookup(pred, lookup, keys)
    assert exact[0] and np.array_equal(rendered[0], masks[0])
    assert dist[0] == 0.0
    # off-grid prediction falls back to nearest neighbor, flagged inexact
    pred2 = np.array([[2.51, 0.51, 3.51]])
    rendered2, exact2, dist2 = g05.render_occupancy_lookup(pred2, lookup, keys)
    assert not exact2[0] and dist2[0] == 0.03
    assert np.array_equal(rendered2[0], masks[0])


import torch  # noqa: E402


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)