"""imp1 regression tests: mirror-parameterised (quadrant) occupancy decoder.

Covers: the base variant is untouched (module surface, no extra blocks, default
flag); the quadrant variant is exactly mirror-symmetric by construction at
[B,1,64,64]; mirror_quadrant places the head's 32x32 exactly in the top-left and
is parameter-free; the extra refinement blocks exist and carry gradient; and the
config/model plumbing reaches the decoder.

Run: python -m pytest tests/test_imp_decoder.py -v
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch  # noqa: E402

from decoders.occupancy_decoder import (  # noqa: E402
    OccupancyDecoder,
    mirror_quadrant,
)


def _z(b=2, hidden=32):
    torch.manual_seed(0)
    return torch.randn(b, 256, hidden)


def _s(b=2):
    torch.manual_seed(1)
    return torch.randn(b, 6)


def test_base_variant_surface_unchanged():
    d = OccupancyDecoder(hidden=32, base_dim=16, scalar_hidden=16)
    assert d.variant == "base"
    keys = set(d.state_dict())
    assert not any(k.startswith("extra") for k in keys), \
        "base variant must not gain new parameters (base-v1 checkpoints load)"
    out = d(_z(), _s())
    assert out.shape == (2, 1, 64, 64)


def test_quadrant_output_is_exactly_mirror_symmetric():
    d = OccupancyDecoder(hidden=32, base_dim=16, scalar_hidden=16,
                         variant="quadrant", depth=3)
    out = d(_z(), _s())
    assert out.shape == (2, 1, 64, 64)
    assert torch.equal(out, torch.flip(out, dims=[3])), "not left-right mirror"
    assert torch.equal(out, torch.flip(out, dims=[2])), "not up-down mirror"
    assert torch.equal(out, torch.flip(out, dims=[2, 3])), "not rot180"


def test_mirror_quadrant_is_an_exact_placement():
    q = torch.randn(1, 1, 32, 32)
    f = mirror_quadrant(q)
    assert f.shape == (1, 1, 64, 64)
    assert torch.equal(f[:, :, :32, :32], q)
    assert torch.equal(f[:, :, :32, 32:], torch.flip(q, dims=[3]))
    assert torch.equal(f[:, :, 32:, :32], torch.flip(q, dims=[2]))
    assert torch.equal(f[:, :, 32:, 32:], torch.flip(q, dims=[2, 3]))


def test_extra_blocks_exist_and_get_gradient():
    d = OccupancyDecoder(hidden=32, base_dim=16, scalar_hidden=16,
                         variant="quadrant", depth=3)
    assert len(d.extra) == 2, "depth=3 must give 1 upsample + 2 refinements"
    d(_z(), _s()).sum().backward()
    assert all(p.grad is not None for p in d.parameters())
    assert any(p.grad.abs().sum() > 0 for p in d.extra.parameters()), \
        "refinement blocks received no gradient"


def test_model_level_plumbing_reaches_the_decoder():
    from assembly import UnifiedJEPA

    imp = UnifiedJEPA(hidden=32, num_heads=2, geo_depth=2, predictor_depth=2,
                      num_predictor_heads=2, n_film_blocks=2,
                      decoder_variant="quadrant", decoder_depth=3)
    assert imp.occupancy_decoder.variant == "quadrant"
    base = UnifiedJEPA(hidden=32, num_heads=2, geo_depth=2, predictor_depth=2,
                       num_predictor_heads=2, n_film_blocks=2)
    assert base.occupancy_decoder.variant == "base"


def test_imp1_config_selects_the_variant():
    import yaml

    with open(os.path.join(REPO_ROOT, "configs", "unified_imp1.yaml"),
              encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["decoder"]["variant"] == "quadrant"
    assert int(cfg["decoder"]["depth"]) == 3
    # base config must NOT carry the key (base v1 stays byte-identical)
    with open(os.path.join(REPO_ROOT, "configs", "unified.yaml"),
              encoding="utf-8") as f:
        base = yaml.safe_load(f)
    assert base.get("decoder") is None


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if fails else 0)
