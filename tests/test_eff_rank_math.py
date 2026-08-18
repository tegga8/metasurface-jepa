"""Unit tests for the effective-rank math correction (audit items 2 and 3).

eff_rank_unnorm / eff_rank_frac previously returned the Shannon entropy H and the
normalized entropy H/log(D) while being NAMED effective rank. The effective rank
(Roy–Vetterli) is exp(H) — in #-of-dims units — and the fraction is exp(H)/D.

Run:  python tests/test_eff_rank_math.py        (also collectable by pytest)
"""

import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch

from diagnostics.representation_health import (  # noqa: E402
    COLLAPSED_ANCHOR, eff_ranks, goal_token_stats,
)


def _spectrum_matrix(singular_values):
    """X (B, D) with exact singular values via SVD construction."""
    B, D = 64, 16
    x0 = torch.randn(B, D)
    U, _, Vt = torch.linalg.svd(x0)
    return U[:, :D] @ torch.diag(torch.tensor(singular_values, dtype=torch.float32)) @ Vt


def test_uniform_spectrum_is_full_effective_rank():
    """Uniform eigenvalues -> H = log(D) -> exp(H) = D, fraction 1.0 (float32
    SVD round-trip noise accounted for)."""
    X = _spectrum_matrix([1.0] * 16)
    s = eff_ranks(X)
    assert abs(s["eff_rank_unnorm"] - 16.0) < 0.1, f"got {s['eff_rank_unnorm']:.6f}"
    assert abs(s["eff_rank_frac"] - 1.0) < 0.01, f"got {s['eff_rank_frac']:.6f}"


def test_rank_one_spectrum_has_effective_rank_one():
    """One dominant singular value -> H ~ 0 -> exp(H) ~ 1."""
    X = _spectrum_matrix([1.0] + [0.0] * 15)
    s = eff_ranks(X)
    assert abs(s["eff_rank_unnorm"] - 1.0) < 0.01, f"got {s['eff_rank_unnorm']:.6f}"
    assert abs(s["eff_rank_frac"] - 1.0 / 16.0) < 0.001, f"got {s['eff_rank_frac']:.6f}"


def test_median_spectrum_between_1_and_d():
    """Effective rank must lie strictly between 1 and D for a spread spectrum."""
    X = _spectrum_matrix([8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625] + [0.0] * 8)
    s = eff_ranks(X)
    assert 1.0 < s["eff_rank_unnorm"] < 16.0, f"got {s['eff_rank_unnorm']:.6f}"


def test_entropy_scale_would_fail_these_bounds():
    """Guard against regression to the entropy scale: exp(H) for a spread spectrum
    is strictly larger than H itself, and equals D for uniform (H only equals log D)."""
    X = _spectrum_matrix([1.0] * 16)
    s = eff_ranks(X)
    assert s["eff_rank_unnorm"] > math.log(16.0) + 10.0  # exp(log 16) = 16 >> log 16


def test_goal_token_effective_rank_uses_exp_scale():
    """goal_token_stats: well-spread goal tokens per sample -> exp(H) near the number
    of (token) dimensions; identical tokens -> 0 (degenerate catch, unchanged)."""
    B, G, D = 4, 16, 16
    u = torch.linalg.svd(torch.randn(B, G, D)).U          # orthonormal rows per sample
    r = goal_token_stats(u)["goal_token_effective_rank"]
    assert r > 12.0, f"got {r:.4f} (entropy scale could never exceed {math.log(G):.3f})"
    assert r <= 16.5, f"got {r:.4f}"

    same = u[:, :1, :].expand(B, G, D).contiguous()
    r_same = goal_token_stats(same)["goal_token_effective_rank"]
    assert r_same == 0.0


def test_collapsed_anchor_on_exp_scale():
    """The collapsed anchor must have moved to the exp(H) scale (item 2 fix)."""
    assert abs(COLLAPSED_ANCHOR["eff_rank_unnorm"] - math.exp(2.5986)) < 1e-3
    assert abs(COLLAPSED_ANCHOR["eff_rank_frac"] - math.exp(2.5986) / 384.0) < 1e-5


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