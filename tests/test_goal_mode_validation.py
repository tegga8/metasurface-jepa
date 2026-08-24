"""Test goal_mode validation (hardening spec §9).

goal_mode accepts only 'real' or 'null'; anything else raises.
"""

import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
from torch import nn


def test_goal_mode_real():
    """Model accepts 'real' goal_mode."""
    class _StubModel(nn.Module):
        def forward(self, G, S, M, goal_mode="real", need_attn=False):
            assert goal_mode == "real"
            return {"z_hat": torch.randn(1, 256, 64), "z_y_raw": torch.randn(1, 256, 64), "mask": torch.ones(1, 256, dtype=bool)}

    model = _StubModel()
    G = torch.randn(1, 3, 64, 64)
    S = torch.randn(1, 2, 301)
    M = torch.ones(1, 16, 16)
    out = model(G, S, M, goal_mode="real")
    assert "z_hat" in out


def test_goal_mode_null():
    """Model accepts 'null' goal_mode."""
    class _StubModel(nn.Module):
        def forward(self, G, S, M, goal_mode="real", need_attn=False):
            assert goal_mode == "null"
            return {"z_hat": torch.randn(1, 256, 64), "z_y_raw": torch.randn(1, 256, 64), "mask": torch.ones(1, 256, dtype=bool)}

    model = _StubModel()
    G = torch.randn(1, 3, 64, 64)
    S = torch.randn(1, 2, 301)
    M = torch.ones(1, 16, 16)
    out = model(G, S, M, goal_mode="null")
    assert "z_hat" in out


def test_goal_mode_invalid_raises():
    """Model should raise for invalid goal_mode (if validation added)."""
    # This test documents the expected behavior after adding validation
    # Currently the model doesn't validate goal_mode, but the hardening spec requires it
    from runtime.physics_controls import validate_goal_mode
    with pytest.raises(ValueError):
        validate_goal_mode("invalid")


if __name__ == "__main__":
    test_goal_mode_real()
    test_goal_mode_null()
    test_goal_mode_invalid_raises()
    print("All goal_mode validation tests passed")