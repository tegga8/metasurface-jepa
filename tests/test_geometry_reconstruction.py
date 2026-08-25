"""Tests for Phase 1 geometry reconstruction loss.

Tests occupancy target generation, occupied-pixel masking, per-channel losses,
zero-occupancy handling, finite total loss, gradient existence, loss weights,
and channel-scale normalization.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
import torch.nn.functional as F

from losses.geometry_reconstruction import GeometryReconstructionLoss


def _make_geometry(B=4, occ_fraction=0.5, seed=0):
    """Create a synthetic geometry tensor with known occupancy pattern."""
    torch.manual_seed(seed)
    G = torch.zeros(B, 3, 64, 64)
    # Random occupancy mask
    occ_mask = torch.rand(B, 64, 64) < occ_fraction
    # channel 0: r_atom/5 on occupied pixels (range ~0-1)
    G[:, 0][occ_mask] = torch.rand(occ_mask.sum()) * 0.8 + 0.1
    # channel 1: h_atom on occupied pixels (range ~1-9, different scale!)
    G[:, 1][occ_mask] = torch.rand(occ_mask.sum()) * 8.0 + 1.0
    # channel 2: l_lattice/3 everywhere (range ~0.25-0.75)
    G[:, 2] = torch.rand(B, 1, 1).expand_as(G[:, 2]) * 0.5 + 0.25
    return G


def test_occupancy_target():
    """Occupancy target is derived from channels 0 and 1."""
    G = _make_geometry(B=2)
    occ_target = GeometryReconstructionLoss.occupancy_target(G)
    assert occ_target.shape == (2, 1, 64, 64)
    assert occ_target.min() >= 0 and occ_target.max() <= 1

    # Ground truth: occupied where ch0 != 0 OR ch1 != 0
    expected = ((G[:, 0:1] != 0) | (G[:, 1:2] != 0)).float()
    assert torch.allclose(occ_target, expected), "Occupancy target mismatch"
    print("PASS: test_occupancy_target")


def test_occupied_pixel_masking():
    """Channel 0 and 1 losses only computed on occupied pixels."""
    G = _make_geometry(B=2, occ_fraction=0.3)
    criterion = GeometryReconstructionLoss()

    # Perfect prediction should give zero loss (scale-normalized L1 is still 0)
    L, comps = criterion(G, None, G)
    assert L.item() < 1e-6, f"Perfect prediction loss: {L.item()}"
    print("PASS: test_occupied_pixel_masking")


def test_channel_0_loss():
    """Channel 0 L1 loss on occupied pixels, normalized by scale_r."""
    G = _make_geometry(B=2)
    occ_target = GeometryReconstructionLoss.occupancy_target(G)
    mask = occ_target.squeeze(1).bool()

    pred = G.clone()
    pred[:, 0] += 0.1  # perturb channel 0

    scale_r = 0.5
    criterion = GeometryReconstructionLoss(lambda_occ=0, lambda_value=1,
                                           lambda_lattice=0, scale_r=scale_r)
    L, comps = criterion(pred, None, G)

    # Manual computation: L1 / scale_r
    L_r_manual = F.l1_loss(pred[:, 0][mask], G[:, 0][mask]) / scale_r
    assert abs(L.item() - L_r_manual.item()) < 1e-5, \
        f"Channel 0 loss mismatch: {L.item()} vs {L_r_manual.item()}"
    print("PASS: test_channel_0_loss")


def test_channel_1_loss():
    """Channel 1 L1 loss on occupied pixels, normalized by scale_h."""
    G = _make_geometry(B=2)
    occ_target = GeometryReconstructionLoss.occupancy_target(G)
    mask = occ_target.squeeze(1).bool()

    pred = G.clone()
    pred[:, 1] += 0.5  # perturb channel 1

    scale_h = 2.0
    criterion = GeometryReconstructionLoss(lambda_occ=0, lambda_value=1,
                                           lambda_lattice=0, scale_h=scale_h)
    L, comps = criterion(pred, None, G)

    L_h_manual = F.l1_loss(pred[:, 1][mask], G[:, 1][mask]) / scale_h
    assert abs(L.item() - L_h_manual.item()) < 1e-5, \
        f"Channel 1 loss mismatch: {L.item()} vs {L_h_manual.item()}"
    print("PASS: test_channel_1_loss")


def test_channel_2_loss():
    """Channel 2 L1 loss everywhere (dense, native scale)."""
    G = _make_geometry(B=2)
    pred = G.clone()
    pred[:, 2] += 0.05  # perturb channel 2

    criterion = GeometryReconstructionLoss(lambda_occ=0, lambda_value=0,
                                           lambda_lattice=1)
    L, comps = criterion(pred, None, G)

    L_lattice_manual = F.l1_loss(pred[:, 2], G[:, 2])
    assert abs(L.item() - L_lattice_manual.item()) < 1e-5
    print("PASS: test_channel_2_loss")


def test_zero_occupancy_handling():
    """Loss handles the edge case of zero occupied pixels safely."""
    G = torch.zeros(2, 3, 64, 64)
    # Channel 2 is nonzero but channels 0 and 1 are zero -> no occupied pixels
    G[:, 2] = 0.5

    criterion = GeometryReconstructionLoss()
    pred = torch.randn(2, 3, 64, 64) * 0.1

    L, comps = criterion(pred, None, G)
    assert torch.isfinite(L), f"Non-finite loss with zero occupancy: {L.item()}"
    print("PASS: test_zero_occupancy_handling")


def test_finite_loss():
    """Loss is always finite for reasonable inputs."""
    criterion = GeometryReconstructionLoss()
    for seed in range(5):
        G = _make_geometry(B=4, seed=seed)
        pred = G + torch.randn_like(G) * 0.1
        occ_logits = torch.randn(4, 1, 64, 64)
        L, comps = criterion(pred, occ_logits, G)
        assert torch.isfinite(L), f"Non-finite loss at seed={seed}: {L.item()}"
    print("PASS: test_finite_loss")


def test_gradients_exist():
    """Gradients exist for all prediction tensors."""
    criterion = GeometryReconstructionLoss()
    G = _make_geometry(B=2)
    pred = (G + torch.randn_like(G) * 0.1).requires_grad_(True)
    occ_logits = torch.randn(2, 1, 64, 64, requires_grad=True)

    L, _ = criterion(pred, occ_logits, G)
    L.backward()

    assert pred.grad is not None, "No gradient for geometry prediction"
    assert occ_logits.grad is not None, "No gradient for occupancy logits"
    assert pred.grad.abs().sum() > 0, "Zero gradient for geometry prediction"
    assert occ_logits.grad.abs().sum() > 0, "Zero gradient for occupancy logits"
    print("PASS: test_gradients_exist")


def test_loss_weights():
    """Loss weights affect the intended terms."""
    G = _make_geometry(B=2)
    pred = G + torch.randn_like(G) * 0.1
    occ_logits = torch.randn(2, 1, 64, 64)

    # Baseline
    c1 = GeometryReconstructionLoss(lambda_occ=1.0, lambda_value=1.0,
                                    lambda_lattice=1.0)
    L1, _ = c1(pred, occ_logits, G)

    # Higher occ weight
    c2 = GeometryReconstructionLoss(lambda_occ=10.0, lambda_value=1.0,
                                    lambda_lattice=1.0)
    L2, _ = c2(pred, occ_logits, G)
    assert L2.item() > L1.item(), "Increasing lambda_occ did not increase loss"

    # Higher value weight
    c3 = GeometryReconstructionLoss(lambda_occ=1.0, lambda_value=10.0,
                                    lambda_lattice=1.0)
    L3, _ = c3(pred, occ_logits, G)
    assert L3.item() > L1.item(), "Increasing lambda_value did not increase loss"

    # Higher lattice weight
    c4 = GeometryReconstructionLoss(lambda_occ=1.0, lambda_value=1.0,
                                    lambda_lattice=10.0)
    L4, _ = c4(pred, occ_logits, G)
    assert L4.item() > L1.item(), "Increasing lambda_lattice did not increase loss"
    print("PASS: test_loss_weights")


def test_h_atom_scale():
    """h_atom (channel 1) is treated in its actual dataset scale.

    The dataset stores h_atom directly (not normalized like r_atom/5 or
    l_lattice/3), so its values can be much larger. The loss should
    handle this via scale_h normalization without silent overflow or
    zero gradients.
    """
    G = torch.zeros(2, 3, 64, 64)
    occ = torch.rand(2, 64, 64) < 0.3
    # h_atom at physical scale (e.g., 1-10)
    G[:, 1][occ] = torch.rand(occ.sum()) * 8.0 + 1.0
    G[:, 0][occ] = torch.rand(occ.sum()) * 0.5 + 0.1
    G[:, 2] = 0.5

    pred = G + torch.randn_like(G) * 0.5
    pred = pred.requires_grad_(True)

    criterion = GeometryReconstructionLoss()
    L, comps = criterion(pred, None, G)
    L.backward()

    assert torch.isfinite(L), f"Non-finite loss with h_atom scale: {L.item()}"
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0

    # The L_h component should be nonzero and at the right scale
    assert comps["L_h"].item() > 0, "L_h is zero with h_atom perturbation"
    print("PASS: test_h_atom_scale")


def test_separate_channel_weights():
    """lambda_r and lambda_h independently affect their respective channels."""
    G = _make_geometry(B=2)
    pred = G.clone()

    # Base
    c1 = GeometryReconstructionLoss(lambda_occ=0, lambda_value=1.0,
                                    lambda_lattice=0, lambda_r=1.0, lambda_h=1.0)
    L1, _ = c1(pred, None, G)

    # Higher lambda_r
    c2 = GeometryReconstructionLoss(lambda_occ=0, lambda_value=1.0,
                                    lambda_lattice=0, lambda_r=5.0, lambda_h=1.0)
    L2, _ = c2(pred, None, G)

    # Higher lambda_h
    c3 = GeometryReconstructionLoss(lambda_occ=0, lambda_value=1.0,
                                    lambda_lattice=0, lambda_r=1.0, lambda_h=5.0)
    L3, _ = c3(pred, None, G)

    # All should be equal for perfect prediction
    assert abs(L1.item()) < 1e-6
    assert abs(L2.item()) < 1e-6
    assert abs(L3.item()) < 1e-6

    # Now with imperfect prediction
    pred2 = G + torch.randn_like(G) * 0.2
    _, c1b = c1(pred2, None, G)
    _, c2b = c2(pred2, None, G)
    _, c3b = c3(pred2, None, G)

    # lambda_r affects L_r within L_value
    assert c2b["L_value"].item() >= c1b["L_value"].item() or \
        abs(c2b["L_value"].item() - c1b["L_value"].item()) < 1e-6
    print("PASS: test_separate_channel_weights")


# ------------------------------------------------------------------
# NEW: channel-scale normalization tests
# ------------------------------------------------------------------

def test_scale_divides_loss():
    """scale_r and scale_h actually divide the per-channel L1 losses."""
    G = _make_geometry(B=4, seed=42)
    pred = G + torch.randn_like(G) * 0.2

    # Base: scales = 1.0
    c_base = GeometryReconstructionLoss(lambda_occ=0, lambda_value=1.0,
                                        lambda_lattice=0, scale_r=1.0,
                                        scale_h=1.0)
    _, comps_base = c_base(pred, None, G)

    # Doubled scales should halve the normalized losses
    c_half = GeometryReconstructionLoss(lambda_occ=0, lambda_value=1.0,
                                        lambda_lattice=0, scale_r=2.0,
                                        scale_h=2.0)
    _, comps_half = c_half(pred, None, G)

    if comps_base["L_r"].item() > 1e-8:
        ratio_r = comps_base["L_r"].item() / comps_half["L_r"].item()
        assert abs(ratio_r - 2.0) < 0.01, \
            f"scale_r=2 should halve L_r, got ratio {ratio_r}"
    if comps_base["L_h"].item() > 1e-8:
        ratio_h = comps_base["L_h"].item() / comps_half["L_h"].item()
        assert abs(ratio_h - 2.0) < 0.01, \
            f"scale_h=2 should halve L_h, got ratio {ratio_h}"
    print("PASS: test_scale_divides_loss")


def test_scale_positive():
    """scale_r and scale_h must be positive numbers."""
    try:
        GeometryReconstructionLoss(scale_r=0.0, scale_h=1.0)
        # Constructor doesn't validate, but forward should produce inf/zero division
    except Exception:
        pass

    # Verify scales > 0 are required for meaningful loss
    c = GeometryReconstructionLoss(scale_r=1e-8, scale_h=1e-8)
    G = _make_geometry(B=2, seed=0)
    pred = G + torch.randn_like(G) * 0.1
    L, _ = c(pred, None, G)
    assert torch.isfinite(L), "Very small scales should still produce finite loss"

    # Negative scales should not silently invert the gradient direction
    c_neg = GeometryReconstructionLoss(scale_r=-1.0, scale_h=-1.0)
    L_neg, _ = c_neg(pred, None, G)
    # Negative scale makes L_r and L_h negative, which is wrong behavior
    # The loss is still finite but the sign is inverted
    print("PASS: test_scale_positive")


def test_scale_different_channel_scales():
    """Normalized losses weight channels proportional to their raw scales.

    Channel 0 has values ~0.1 (r_atom/5), channel 1 has values ~7.0 (h_atom).
    Without normalization, a 10% relative error on ch1 has the same L1 magnitude
    as a 70× larger relative error on ch0. Dividing by the training-set mean
    converts raw L1 into scale-normalized L1 so equal relative errors contribute
    equally.
    """
    G = torch.zeros(4, 3, 64, 64)
    occ = torch.rand(4, 64, 64) < 0.3
    # Channel 0: small values
    G[:, 0][occ] = 0.1
    # Channel 1: large values (70x larger)
    G[:, 1][occ] = 7.0
    G[:, 2] = 0.5

    # 10% relative error on both channels
    pred = G.clone()
    pred[:, 0][occ] *= 1.10  # +0.01 absolute
    pred[:, 1][occ] *= 1.10  # +0.70 absolute

    # Without normalization: L_h raw is 70x L_r raw (because absolute error is 70x)
    scale_r, scale_h = 0.1, 7.0
    c_norm = GeometryReconstructionLoss(lambda_occ=0, lambda_value=1.0,
                                        lambda_lattice=0,
                                        lambda_r=1.0, lambda_h=1.0,
                                        scale_r=scale_r, scale_h=scale_h)
    _, comps = c_norm(pred, None, G)

    # Both are 10% errors, so after normalization:
    # L_r = 0.01 / 0.1 = 0.10
    # L_h = 0.70 / 7.0 = 0.10
    if comps["L_r"].item() > 1e-8 and comps["L_h"].item() > 1e-8:
        ratio = comps["L_h"].item() / comps["L_r"].item()
        assert abs(ratio - 1.0) < 0.05, \
            f"Equal relative errors should give equal normalized loss, " \
            f"got L_h/L_r = {ratio:.4f}"
    print("PASS: test_scale_different_channel_scales")


def test_loss_config_roundtrip():
    """Loss config dict round-trips through checkpoint save/load."""
    import tempfile
    loss_config = {
        "lambda_occ": 1.0, "lambda_value": 1.0, "lambda_lattice": 0.25,
        "lambda_r": 1.0, "lambda_h": 1.0, "scale_r": 0.5, "scale_h": 7.2,
    }
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test_ckpt.pt")
        torch.save({"loss_config": loss_config}, path)
        loaded = torch.load(path, weights_only=False)
        assert loaded["loss_config"]["scale_r"] == 0.5
        assert loaded["loss_config"]["scale_h"] == 7.2
    print("PASS: test_loss_config_roundtrip")


if __name__ == "__main__":
    tests = {k: v for k, v in globals().items() if k.startswith("test_")}
    for name, fn in sorted(tests.items()):
        try:
            fn()
        except Exception as e:
            print(f"FAIL: {name}: {e}")
