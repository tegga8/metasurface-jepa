"""Unit tests for the EMA-target diversity diagnostic's same_token_cos().

Regression guard for the indexing bug (2026-08-17): the old implementation's einsum
"btd,bsd->tbs" aliased the second operand's TOKEN dim onto the batch label, producing
G (T, B, T) instead of (T, B, B). With B > T it raised
    IndexError: index 256 is out of bounds for dimension 1 with size 256
and with B <= T it silently computed within-sample token-pair cosines instead of
cross-sample cosines at the same spatial token.

These tests pin the correct semantics: different samples i, j compared at the SAME
spatial token t, returning a finite scalar.

Run:  python tests/test_same_token_cos.py        (also collectable by pytest)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "diagnostics"))

import numpy as np
import torch
import torch.nn.functional as F

from check_ema_target_diversity import same_token_cos  # noqa: E402

B, T, D = 4, 256, 384


def test_returns_finite_scalar_bfour():
    X = torch.randn(B, T, D)
    v = same_token_cos(X)
    assert isinstance(v, float)
    assert np.isfinite(v), "same_token_cos must return a finite scalar"
    assert -1.0 <= v <= 1.0


def test_no_index_error_large_batch():
    """B > T was the exact failure regime (index 256 out of bounds for dim of size 256)."""
    X = torch.randn(512, 256, 384)
    v = same_token_cos(X)
    assert np.isfinite(v)
    assert -1.0 <= v <= 1.0


def test_identical_samples_give_one():
    """All samples identical at every token -> same-token cross-sample cosine == 1."""
    X = torch.randn(B, T, D)
    X[1] = X[0].clone()
    X[2] = X[0].clone()
    X[3] = X[0].clone()
    assert abs(same_token_cos(X) - 1.0) < 1e-6


def test_semantics_match_manual_loop():
    """Cross-sample cosine at the SAME spatial token, averaged over token & sample pairs.

    Independent reference implementation (explicit per-token loop over sample pairs)
    must match the vectorized function — this is the check that caught the old
    einsum computing within-sample token-pair cosines instead.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(B, T, D).astype(np.float32)
    Xt = torch.from_numpy(X)
    Xn = F.normalize(Xt, dim=-1)

    expected = 0.0
    n_pairs = 0
    for i in range(B):
        for j in range(i + 1, B):
            cos_ij = (Xn[i] * Xn[j]).sum(dim=-1)  # (T,) same-token cosine per token
            expected += cos_ij.sum().item()
            n_pairs += 1
    expected /= (n_pairs * T)

    got = same_token_cos(Xt)
    assert abs(got - expected) < 1e-5, f"semantics mismatch: got {got:.8f}, expected {expected:.8f}"


def test_single_token_reconstruction():
    """Reconstruct a hand-computed value: two identical samples + two orthogonal ones."""
    rng = np.random.RandomState(1)
    a = rng.randn(D).astype(np.float32)
    b = rng.randn(D).astype(np.float32)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    cos_ab = float(a @ b)
    X = np.zeros((B, T, D), dtype=np.float32)
    for t in range(T):
        X[0, t], X[1, t], X[2, t], X[3, t] = a, b, a, b
    got = same_token_cos(torch.from_numpy(X))
    expected = (4.0 * cos_ab + 2.0 * 1.0) / 6.0  # pairs: (0,1),(0,3),(1,2),(2,3)=cos_ab; (0,2),(1,3)=1
    assert abs(got - expected) < 1e-6, f"hand-computed mismatch: got {got:.8f}, expected {expected:.8f}"


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
