"""Test derangement controls for shuffled spectrum (hardening spec §8)."""

import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
from runtime.physics_controls import derangement_permutation, make_shuffled_spectrum, validate_goal_mode


def test_derangement_permutation():
    """Derangement has no fixed points."""
    for b in [2, 3, 4, 5, 10, 100]:
        perm = derangement_permutation(b, "cpu", seed=42)
        assert perm.shape == (b,)
        assert not torch.any(perm == torch.arange(b)), f"Fixed point found for b={b}"
        # All elements present exactly once
        assert torch.equal(torch.sort(perm).values, torch.arange(b))


def test_derangement_permutation_deterministic():
    """Same seed gives same derangement."""
    p1 = derangement_permutation(10, "cpu", seed=123)
    p2 = derangement_permutation(10, "cpu", seed=123)
    assert torch.equal(p1, p2)


def test_derangement_batch_size_1_raises():
    """Batch size 1 cannot be deranged."""
    with pytest.raises(ValueError, match="batch_size >= 2"):
        derangement_permutation(1, "cpu")


def test_make_shuffled_spectrum():
    """Shuffled spectrum uses derangement."""
    S = torch.randn(10, 2, 301)
    S_shuf = make_shuffled_spectrum(S, seed=42)
    assert S_shuf.shape == S.shape
    # No sample should match its original
    for i in range(10):
        assert not torch.equal(S_shuf[i], S[i]), f"Sample {i} matched its own spectrum"


def test_make_shuffled_spectrum_batch_1():
    """Batch size 1 returns same (no derangement possible)."""
    S = torch.randn(1, 2, 301)
    S_shuf = make_shuffled_spectrum(S)
    assert torch.equal(S_shuf, S)


def test_validate_goal_mode_real():
    validate_goal_mode("real")  # Should not raise


def test_validate_goal_mode_null():
    validate_goal_mode("null")  # Should not raise


def test_validate_goal_mode_invalid():
    with pytest.raises(ValueError, match="goal_mode must be 'real' or 'null'"):
        validate_goal_mode("invalid")
    with pytest.raises(ValueError, match="goal_mode must be 'real' or 'null'"):
        validate_goal_mode("shuffled")
    with pytest.raises(ValueError, match="goal_mode must be 'real' or 'null'"):
        validate_goal_mode("")


if __name__ == "__main__":
    test_derangement_permutation()
    test_derangement_permutation_deterministic()
    test_derangement_batch_size_1_raises()
    test_make_shuffled_spectrum()
    test_make_shuffled_spectrum_batch_1()
    test_validate_goal_mode_real()
    test_validate_goal_mode_null()
    test_validate_goal_mode_invalid()
    print("All derangement controls tests passed")