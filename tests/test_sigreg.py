"""Regression tests for src/losses/sigreg.py — CUDA device-mismatch fix."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
import torch

from losses.sigreg import sigreg_loss


def test_sigreg_cpu_finite():
    z = torch.randn(64, 384)
    loss, info = sigreg_loss(z, num_slices=8, num_points=32, seed=0)
    assert torch.isfinite(loss)
    assert info["num_slices"] == 8
    assert info["seed"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sigreg_cuda_finite():
    z = torch.randn(64, 384, device="cuda")
    loss, info = sigreg_loss(z, num_slices=8, num_points=32, seed=0)
    assert torch.isfinite(loss)
    assert loss.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sigreg_cuda_gradients_finite():
    # FIX C gate: the CUDA path must backpropagate finite gradients (the ECF
    # complex exp is differentiable through its real intermediate).
    z = torch.randn(64, 384, device="cuda", requires_grad=True)
    loss, _ = sigreg_loss(z, num_slices=8, num_points=32, seed=0)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_sigreg_gradients_finite():
    z = torch.randn(64, 384, requires_grad=True)
    loss, _ = sigreg_loss(z, num_slices=8, num_points=32, seed=0)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_sigreg_deterministic_same_seed():
    z = torch.randn(64, 384)
    loss1, _ = sigreg_loss(z, num_slices=8, num_points=32, seed=0)
    loss2, _ = sigreg_loss(z, num_slices=8, num_points=32, seed=0)
    assert torch.equal(loss1, loss2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sigreg_slice_directions_on_z_device():
    # Regression test for the original bug: U and the subsample index must be
    # created on z.device, not default CPU, or z @ U.T / proj[idx] crash.
    z = torch.randn(300, 384, device="cuda")  # n > num_points forces the idx path
    loss, info = sigreg_loss(z, num_slices=8, num_points=256, seed=0)
    assert torch.isfinite(loss)
    assert info["num_points"] == 256


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sigreg_cuda_subsampling_shape():
    z = torch.randn(500, 384, device="cuda")
    loss, info = sigreg_loss(z, num_slices=4, num_points=100, seed=1)
    assert torch.isfinite(loss)
    assert info["num_points"] == 100


def _check_phi_contract(monkeypatch, z, num_slices, num_points, seed):
    """Pins the ECF broadcast contract (line-66 fix): the tensor fed to torch.exp
    must be (N, T, S), so phi = mean over dim 0 is (T, S) == (len(t_grid),
    num_slices). phi itself is internal to sigreg_loss, so the shape is captured
    by spying on the module's torch.exp call."""
    import losses.sigreg as sigreg_mod

    t_grid_len = len(sigreg_mod.DEFAULT_T_GRID)
    phi_shape = {}
    real_exp = sigreg_mod.torch.exp

    def _spy(exp_input, *a, **k):
        if exp_input.ndim == 3:  # only the phi ECF call is (N, T, S); the
            phi_shape["phi_shape"] = tuple(exp_input.shape)[1:]  # target ECF (5,) is skipped
        return real_exp(exp_input, *a, **k)

    monkeypatch.setattr(sigreg_mod.torch, "exp", _spy)
    loss, info = sigreg_mod.sigreg_loss(
        z, num_slices=num_slices, num_points=num_points, seed=seed)
    assert "phi_shape" in phi_shape, "no (N, T, S) ECF exp call observed"
    assert phi_shape["phi_shape"] == (t_grid_len, num_slices), \
        f"phi.shape must be (len(t_grid)={t_grid_len}, num_slices={num_slices}), got {phi_shape['phi_shape']}"
    assert info["num_points"] == min(z.shape[0], num_points)
    assert torch.isfinite(loss)
    assert loss.shape == ()
    return loss


def test_sigreg_phi_shape_cpu(monkeypatch):
    # N=64, S=8, T=5, n > num_points=32 (subsampling path) — finite loss + grads.
    z = torch.randn(64, 384, requires_grad=True)
    loss = _check_phi_contract(monkeypatch, z, num_slices=8, num_points=32, seed=0)
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_sigreg_phi_shape_cpu_no_subsample(monkeypatch):
    # N=64 <= num_points: no index path.
    z = torch.randn(64, 384)
    _check_phi_contract(monkeypatch, z, num_slices=8, num_points=256, seed=0)


def test_sigreg_phi_shape_cpu_subsampling_path(monkeypatch):
    # N>num_points forces the randperm index path; phi contract must hold post-idx.
    z = torch.randn(300, 384)
    _check_phi_contract(monkeypatch, z, num_slices=8, num_points=256, seed=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sigreg_phi_shape_cuda(monkeypatch):
    z = torch.randn(64, 384, device="cuda")
    _check_phi_contract(monkeypatch, z, num_slices=8, num_points=32, seed=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sigreg_phi_shape_cuda_subsampling(monkeypatch):
    z = torch.randn(300, 384, device="cuda")
    _check_phi_contract(monkeypatch, z, num_slices=8, num_points=256, seed=1)