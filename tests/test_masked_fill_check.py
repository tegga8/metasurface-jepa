"""Tests for the masked-fill verification helpers (scripts/diagnostics/masked_fill_check.py).

Stub-based and fast (no checkpoint / no released weights needed):
- visible-region identity (the hard retention guarantee)
- filled-region IoU/F1
- seam mismatch at the mask boundary
- texture stats (roughness / neighbour agreement)
- locality probe: a local operator must score lower than a global one

Run:  python -m pytest tests/test_masked_fill_check.py -v
      python tests/test_masked_fill_check.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "diagnostics"))

import pytest
import torch
import torch.nn.functional as F

import masked_fill_check as mfc


def _mask(b=1):
    """All visible except a centred 4x4-token block (=> 16x16 pixel hole)."""
    m = torch.ones(b, 16, 16)
    m[:, 6:10, 6:10] = 0.0
    return m


def _occ(b=1, seed=0):
    torch.manual_seed(seed)
    o = (torch.rand(b, 1, 64, 64) > 0.5).float()
    return o


def test_visible_identity_is_zero_when_retained_and_positive_when_not():
    occ, mask = _occ(), _mask()
    vis = mfc.upsample_vis(mask)
    deployed = occ * vis.float() + 1.0 * (~vis).float()  # the decode_geometry rule
    assert mfc.visible_identity_maxdiff(deployed, occ, mask) == 0.0
    tampered = deployed.clone()
    tampered[vis] = 1.0 - tampered[vis]
    assert mfc.visible_identity_maxdiff(tampered, occ, mask) == 1.0


def test_region_iou_f1_perfect_and_none():
    occ, mask = _occ(), _mask()
    b = occ > 0.5
    filled = ~mfc.upsample_vis(mask)
    perfect = mfc.region_iou_f1(b, b, filled)
    assert perfect["iou"] == pytest.approx(1.0)
    assert perfect["f1"] == pytest.approx(1.0)
    assert perfect["n_pixels"] > 0
    none = mfc.region_iou_f1(torch.zeros_like(b), b, filled)
    assert none["iou"] == 0.0 and none["f1"] == 0.0


def test_seam_mismatch_zero_when_correct_and_one_when_inverted():
    occ, mask = _occ(), _mask()
    b = occ > 0.5
    assert mfc.masked_fill_report(b.float(), occ, occ, mask)["seam_mismatch"] == 0.0
    inv = (~b).float()
    rep = mfc.masked_fill_report(inv, occ, occ, mask)
    assert rep["seam_mismatch"] == pytest.approx(1.0)


def test_texture_ratio_one_when_identical_and_below_one_when_oversmoothed():
    occ, mask = _occ(), _mask()
    b = occ > 0.5
    rep = mfc.masked_fill_report(b.float(), occ, occ, mask)
    assert rep["filled_roughness_ratio"] == pytest.approx(1.0)
    blurred = F.avg_pool2d(occ, 3, stride=1, padding=1)
    rep_blur = mfc.masked_fill_report(blurred, occ, occ, mask)
    assert rep_blur["filled_roughness_ratio"] < 1.0


def test_masked_fill_report_has_expected_keys():
    occ, mask = _occ(), _mask()
    rep = mfc.masked_fill_report((occ > 0.5).float(), occ, occ, mask)
    for k in ("visible_identity_maxdiff", "filled_region", "seam_band",
              "seam_mismatch", "filled_roughness_ratio", "filled_agreement_pred"):
        assert k in rep


class _LocalStub:
    """Prediction depends only on a 3x3 neighbourhood -> far pixels cannot matter."""

    def __call__(self, occ, mask):
        return F.avg_pool2d(occ, 3, stride=1, padding=1)


class _GlobalStub:
    """Prediction depends on the whole-image mean -> near and far matter equally."""

    def __call__(self, occ, mask):
        return occ.mean(dim=(1, 2, 3), keepdim=True).expand_as(occ)


def test_locality_probe_separates_local_from_global():
    occ, mask = _occ(b=2), _mask(b=2)
    local = mfc.locality_probe(_LocalStub(), occ, mask, k=32)
    glob = mfc.locality_probe(_GlobalStub(), occ, mask, k=32)
    assert local["locality_ratio"] < 0.5
    assert local["localized"] is True
    assert glob["locality_ratio"] > 0.8
    assert glob["localized"] is False
    assert local["locality_ratio"] < glob["locality_ratio"]


def test_locality_probe_not_applicable_when_no_visible_context():
    occ = _occ(1)
    mask = torch.zeros(1, 16, 16)  # fully masked -> no visible pixels
    r = mfc.locality_probe(_LocalStub(), occ, mask, k=8)
    assert r["applicable"] is False
    assert r["locality_ratio"] is None
    assert r["localized"] is None


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
