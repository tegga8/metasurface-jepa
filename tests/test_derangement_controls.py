"""Test derangement controls for shuffled spectrum (hardening spec §8)."""

import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
from runtime.physics_controls import (
    derangement_permutation, derange_batch_tensor,
    make_shuffled_spectrum, validate_goal_mode,
)


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


def test_derange_batch_tensor_scalars():
    """Generic derangement applies to scalar tensors (cleanup item 4)."""
    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    X_shuf = derange_batch_tensor(X, seed=0)
    assert X_shuf.shape == X.shape
    # Derangement: no row keeps its own position.
    for i in range(4):
        assert not torch.equal(X_shuf[i], X[i]), f"row {i} kept its position"
    # Rows are a permutation of the input rows.
    for i in range(4):
        assert any(torch.equal(X_shuf[i], X[j]) for j in range(4))


def test_derange_batch_tensor_deterministic():
    """Same seed → same derangement for scalar tensors."""
    X = torch.randn(6, 3)
    d1 = derange_batch_tensor(X, seed=7)
    d2 = derange_batch_tensor(X, seed=7)
    assert torch.equal(d1, d2)


def test_derange_batch_tensor_batch_1_raises():
    """Generic derangement raises on B < 2 (explicit infeasible)."""
    X = torch.randn(1, 3)
    with pytest.raises(ValueError, match="batch size"):
        derange_batch_tensor(X)


def test_derange_batch_tensor_matches_permutation():
    """derange_batch_tensor output equals X[derangement_permutation(...)]."""
    X = torch.randn(8, 3)
    perm = derangement_permutation(8, "cpu", seed=3)
    X_shuf = derange_batch_tensor(X, seed=3)
    assert torch.equal(X_shuf, X[perm])


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