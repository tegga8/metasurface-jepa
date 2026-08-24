"""Test Barlow N<2 contract (hardening spec §9).

Barlow must not silently return zero for invalid N < 2; use an explicit contract.
"""

import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch
from losses.barlow import barlow_twins_loss


def test_barlow_n2_contract_n1_raises():
    """N=1 must raise ValueError."""
    z_p = torch.randn(1, 64)
    z_t = torch.randn(1, 64)
    with pytest.raises(ValueError, match="N >= 2"):
        barlow_twins_loss(z_p, z_t)


def test_barlow_n2_contract_n0_raises():
    """N=0 must raise ValueError."""
    z_p = torch.randn(0, 64)
    z_t = torch.randn(0, 64)
    with pytest.raises(ValueError, match="N >= 2"):
        barlow_twins_loss(z_p, z_t)


def test_barlow_n2_works_n2():
    """N=2 should work."""
    z_p = torch.randn(2, 64)
    z_t = torch.randn(2, 64)
    loss, info = barlow_twins_loss(z_p, z_t)
    assert torch.isfinite(loss)
    assert "diag_term" in info
    assert "off_diag_term" in info


def test_barlow_n2_works_large_n():
    """Large N should work."""
    z_p = torch.randn(32, 64)
    z_t = torch.randn(32, 64)
    loss, info = barlow_twins_loss(z_p, z_t)
    assert torch.isfinite(loss)
    assert info["diag_term"] >= 0
    assert info["off_diag_term"] >= 0


if __name__ == "__main__":
    test_barlow_n2_contract_n1_raises()
    test_barlow_n2_contract_n0_raises()
    test_barlow_n2_works_n2()
    test_barlow_n2_works_large_n()
    print("All Barlow N<2 contract tests passed")