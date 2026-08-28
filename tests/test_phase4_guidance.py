"""Tests for classifier-free guidance (§3.5.1) and goal dropout.

Tests:
- cfg_combine: correct combination formula
- goal_dropout: correct null replacement probability
- cfg_forward: produces guided prediction with valid gap
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


def test_cfg_combine_formula():
    """Z_guided = z_null + w * (z_real - z_null)."""
    from predictor.guidance import cfg_combine
    z_real = torch.tensor([1.0, 2.0, 3.0])
    z_null = torch.tensor([0.0, 0.0, 0.0])
    w = 2.0
    z_guided = cfg_combine(z_real, z_null, w)
    expected = z_null + w * (z_real - z_null)
    assert torch.allclose(z_guided, expected), \
        f"cfg_combine: expected {expected}, got {z_guided}"


def test_cfg_combine_w1_is_real():
    """w=1 → guided == real."""
    from predictor.guidance import cfg_combine
    z_real = torch.randn(4, 8)
    z_null = torch.randn(4, 8)
    z_guided = cfg_combine(z_real, z_null, w=1.0)
    assert torch.allclose(z_guided, z_real), "w=1 should return z_real"


def test_cfg_combine_w0_is_null():
    """w=0 → guided == null."""
    from predictor.guidance import cfg_combine
    z_real = torch.randn(4, 8)
    z_null = torch.randn(4, 8)
    z_guided = cfg_combine(z_real, z_null, w=0.0)
    assert torch.allclose(z_guided, z_null), "w=0 should return z_null"


def test_goal_dropout_prob_zero():
    """p=0 → always keeps real."""
    from predictor.guidance import goal_dropout
    for _ in range(100):
        assert goal_dropout("real", 0.0) == "real"


def test_goal_dropout_prob_one():
    """p=1 → always null."""
    from predictor.guidance import goal_dropout
    for _ in range(100):
        assert goal_dropout("real", 1.0) == "null"


def test_goal_dropout_null_is_idempotent():
    """null stays null regardless of p."""
    from predictor.guidance import goal_dropout
    for _ in range(50):
        assert goal_dropout("null", 0.5) == "null"


def test_goal_dropout_rate_approximately_correct():
    """With p=0.1, roughly 10% become null over many trials."""
    from predictor.guidance import goal_dropout
    rng = torch.Generator().manual_seed(42)
    n_null = sum(1 for _ in range(1000) if goal_dropout("real", 0.1, rng) == "null")
    assert 70 <= n_null <= 130, f"Expected ~100 nulls, got {n_null}"


def test_cfg_forward_shapes():
    """cfg_forward returns guided z_hat with correct shape."""
    from predictor.guidance import cfg_forward
    model = _build_model()
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    occ[:, :, :32, :32] = 1.0
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]])
    spec = torch.randn(2, 2, 301)
    sk = torch.ones(2, 3, dtype=torch.bool)
    M = torch.ones(2, 16, 16)
    M[:, :8, :8] = 0  # mask out some

    z_guided, info = cfg_forward(model, occ, sv, sk, spec, M, w=1.0)
    assert z_guided.shape == (2, 256, 192), f"Expected (2,256,192), got {z_guided.shape}"
    assert "guidance_gap" in info
    assert "normalized_guidance_gap" in info
    assert info["guidance_gap"] > 0  # real ≠ null at init


@torch.no_grad()
def test_cfg_forward_w0_equals_null():
    """w=0 → guided == null prediction."""
    from predictor.guidance import cfg_forward
    model = _build_model()
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]])
    spec = torch.randn(2, 2, 301)
    sk = torch.ones(2, 3, dtype=torch.bool)
    M = torch.ones(2, 16, 16)
    M[:, :8, :8] = 0

    z_guided, info = cfg_forward(model, occ, sv, sk, spec, M, w=0.0)
    assert torch.allclose(z_guided, info["z_hat_null"], atol=1e-5)


@torch.no_grad()
def test_cfg_forward_w1_equals_real():
    """w=1 → guided == real prediction."""
    from predictor.guidance import cfg_forward
    model = _build_model()
    occ = (torch.rand(2, 1, 64, 64) > 0.5).float()
    sv = torch.tensor([[1.5, 0.8, 10.0], [2.0, 1.2, 12.0]])
    spec = torch.randn(2, 2, 301)
    sk = torch.ones(2, 3, dtype=torch.bool)
    M = torch.ones(2, 16, 16)
    M[:, :8, :8] = 0

    z_guided, info = cfg_forward(model, occ, sv, sk, spec, M, w=1.0)
    assert torch.allclose(z_guided, info["z_hat_real"], atol=1e-5)


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
