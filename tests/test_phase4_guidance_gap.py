"""Tests for guidance-gap diagnostic (§20.3).

Tests:
- compute_guidance_gap returns valid dict with expected keys
- normalized gap is gap / std
- guidance_gap_sweep returns results for all ratios
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

import torch
import torch.nn as nn
import pytest


class _StubReleasedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 256))
    def forward(self, S):
        return self.net(S.transpose(1, 2))


def _build_model():
    from assembly import UnifiedJEPA
    torch.manual_seed(0)
    model = UnifiedJEPA(
        hidden=192, num_heads=6, geo_depth=2,
        predictor_depth=4, goal_tokens=16,
        num_predictor_heads=6, scalar_hidden=128,
        n_film_blocks=2, spec_dim=256)
    stub = _StubReleasedEncoder()
    for p in stub.parameters():
        p.requires_grad_(False)
    stub.eval()
    model.spectrum_path.released = stub
    model.ema.target.load_state_dict(model.occupancy_encoder.state_dict())
    model.scalar_mlp_ema.target.load_state_dict(model.scalar_encoder.state_dict())
    model.eval()
    return model


def _test_data():
    from data.mask import BlockMasker
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    occ[:, :, :32, :32] = 1.0
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]])
    spec = torch.randn(2, 2, 301)
    sk = torch.ones(2, 3, dtype=torch.bool)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=42)
    M = masker.sample(occ, ratio=0.5)
    return occ, sv, spec, sk, M, masker


def test_compute_guidance_gap_returns_dict():
    from diagnostics.guidance_gap import compute_guidance_gap
    model = _build_model()
    occ, sv, spec, sk, M, _ = _test_data()
    result = compute_guidance_gap(model, occ, sv, sk, spec, M)
    assert isinstance(result, dict)
    assert "guidance_gap" in result
    assert "normalized_guidance_gap" in result
    assert "z_real_std" in result
    assert "z_null_std" in result
    assert result["guidance_gap"] >= 0
    assert result["normalized_guidance_gap"] >= 0


def test_guidance_gap_nonnegative():
    from diagnostics.guidance_gap import compute_guidance_gap
    model = _build_model()
    occ, sv, spec, sk, M, _ = _test_data()
    result = compute_guidance_gap(model, occ, sv, sk, spec, M)
    assert result["guidance_gap"] >= 0
    assert result["normalized_guidance_gap"] >= 0


def test_guidance_gap_sweep_returns_all_ratios():
    from diagnostics.guidance_gap import guidance_gap_sweep
    model = _build_model()
    occ, sv, spec, sk, _, masker = _test_data()
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    results = guidance_gap_sweep(model, occ, sv, spec, masker, ratios)
    assert len(results) == len(ratios)
    for r in ratios:
        assert r in results
        assert isinstance(results[r], float)
        assert results[r] >= 0


def test_guidance_gap_does_not_mutate_model_mode():
    from diagnostics.guidance_gap import compute_guidance_gap
    model = _build_model()
    was_training = model.training
    occ, sv, spec, sk, M, _ = _test_data()
    _ = compute_guidance_gap(model, occ, sv, sk, spec, M)
    assert model.training == was_training


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
