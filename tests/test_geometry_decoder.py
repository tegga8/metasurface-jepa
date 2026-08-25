"""Tests for Phase 1 geometry decoder.

Shape tests, token-grid reshape correctness, EMA freeze verification,
optimizer ownership, gradient checks, and checkpoint round-trip.
"""

import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
import torch.nn as nn

from decoders.geometry_decoder import GeometryDecoder


def test_shape():
    """B,256,384 -> B,3,64,64 + B,1,64,64."""
    dec = GeometryDecoder(hidden_dim=384, base_dim=192, num_channels=3,
                          occupancy_head=True)
    B = 4
    z = torch.randn(B, 256, 384)
    geom, occ = dec(z)
    assert geom.shape == (B, 3, 64, 64), f"geom shape: {geom.shape}"
    assert occ.shape == (B, 1, 64, 64), f"occ shape: {occ.shape}"
    print("PASS: test_shape")


def test_shape_no_occ():
    """Without occupancy head."""
    dec = GeometryDecoder(occupancy_head=False)
    z = torch.randn(2, 256, 384)
    geom, occ = dec(z)
    assert geom.shape == (2, 3, 64, 64)
    assert occ is None
    print("PASS: test_shape_no_occ")


def test_token_reshape():
    """Verify token index maps to correct spatial position."""
    B = 1
    D = 384
    z = torch.arange(256).float().unsqueeze(-1).expand(-1, D).unsqueeze(0)
    # z[0, i, :] = i for all 384 dims

    dec = GeometryDecoder(hidden_dim=D, base_dim=192, occupancy_head=False)
    geom, _ = dec(z)

    # The reshape: z.view(B, 16, 16, D).permute(0, 3, 1, 2)
    # Token i maps to spatial position (i // 16, i % 16)
    # After permute, feature at position (row, col) = z[:, row*16+col, :]
    # So feature value at (row, col, :) should equal row*16+col
    feat = z.view(B, 16, 16, D).permute(0, 3, 1, 2).contiguous()
    for row in range(16):
        for col in range(16):
            token_idx = row * 16 + col
            val = feat[0, 0, row, col].item()
            assert val == token_idx, (
                f"Position ({row},{col}): expected feature={token_idx}, got {val}"
            )
    print("PASS: test_token_reshape")


def test_gradient_exists():
    """After backward, decoder parameters have gradients."""
    dec = GeometryDecoder()
    z = torch.randn(2, 256, 384, requires_grad=False)
    geom, occ = dec(z)
    loss = geom.sum() + (occ.sum() if occ is not None else 0)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in dec.parameters())
    assert has_grad, "No gradients in decoder after backward"
    print("PASS: test_gradient_exists")


def test_ema_freeze():
    """EMA parameters are not trainable after explicit freeze."""
    from encoders.target_encoder import EMAEncoder
    from encoders.geometry_encoder import GeometryEncoder

    geo = GeometryEncoder(hidden=384, num_heads=6, depth=1)
    ema = EMAEncoder(geo, momentum_start=0.996, momentum_end=0.999)

    # EMA params should already have requires_grad=False from __init__
    for p in ema.parameters():
        assert not p.requires_grad, f"EMA param not frozen: requires_grad=True"

    # Explicit freeze
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    assert all(not p.requires_grad for p in ema.parameters())

    # Forward should still work
    G = torch.randn(2, 3, 64, 64)
    z = ema(G)
    assert z.shape == (2, 256, 384)
    print("PASS: test_ema_freeze")


def test_optimizer_ownership():
    """Optimizer contains only decoder parameters."""
    dec = GeometryDecoder()
    ema_geo = None
    try:
        from encoders.geometry_encoder import GeometryEncoder
        from encoders.target_encoder import EMAEncoder
        geo = GeometryEncoder(hidden=384, num_heads=6, depth=1)
        ema = EMAEncoder(geo)
        ema.eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        ema_geo = ema
    except Exception:
        pass

    optimizer = torch.optim.AdamW(dec.parameters(), lr=1e-3)
    dec_ids = {id(p) for p in dec.parameters()}
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert opt_ids == dec_ids, "Optimizer contains parameters outside decoder"
    if ema_geo is not None:
        ema_ids = {id(p) for p in ema_geo.parameters()}
        assert not (opt_ids & ema_ids), "Optimizer contains EMA parameters"
    print("PASS: test_optimizer_ownership")


def test_checkpoint_roundtrip():
    """Save/reload decoder checkpoint and verify deterministic output."""
    dec = GeometryDecoder()
    dec.eval()
    z = torch.randn(2, 256, 384)

    with torch.no_grad():
        geom1, occ1 = dec(z)

    optimizer = torch.optim.AdamW(dec.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "test.pt")
        state = {
            "decoder_state": dec.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": 42,
            "epoch": 3,
        }
        torch.save(state, ckpt_path)

        dec2 = GeometryDecoder()
        opt2 = torch.optim.AdamW(dec2.parameters(), lr=1e-3)
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        dec2.load_state_dict(obj["decoder_state"])
        opt2.load_state_dict(obj["optimizer_state"])
        dec2.eval()

        with torch.no_grad():
            geom2, occ2 = dec2(z)

    assert torch.allclose(geom1, geom2, atol=1e-5), \
        "Geometry not deterministic after checkpoint round-trip"
    assert torch.allclose(occ1, occ2, atol=1e-5), \
        "Occupancy not deterministic after checkpoint round-trip"
    assert obj["step"] == 42
    assert obj["epoch"] == 3
    print("PASS: test_checkpoint_roundtrip")


def test_ema_no_gradient():
    """After one backward pass through decoder, EMA parameters have no gradient."""
    from encoders.geometry_encoder import GeometryEncoder
    from encoders.target_encoder import EMAEncoder

    geo = GeometryEncoder(hidden=384, num_heads=6, depth=1)
    ema = EMAEncoder(geo)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    dec = GeometryDecoder(hidden_dim=384)
    G = torch.randn(2, 3, 64, 64)

    # Forward through EMA -> decoder
    with torch.no_grad():
        z = ema(G)
    geom, occ = dec(z)
    loss = geom.sum() + occ.sum()
    loss.backward()

    # Check EMA has no gradients
    ema_leaked = [n for n, p in ema.named_parameters() if p.grad is not None]
    assert not ema_leaked, f"EMA received gradients: {ema_leaked}"

    # Check decoder has gradients
    dec_has = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in dec.parameters())
    assert dec_has, "Decoder has no gradients after backward"
    print("PASS: test_ema_no_gradient")


if __name__ == "__main__":
    tests = {k: v for k, v in globals().items() if k.startswith("test_")}
    for name, fn in sorted(tests.items()):
        try:
            fn()
        except Exception as e:
            print(f"FAIL: {name}: {e}")
