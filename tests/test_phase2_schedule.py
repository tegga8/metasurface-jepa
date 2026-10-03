"""Phase 2b/2c — mask-ratio curriculum and physics-ramp schedule.

2b: per-sample mask ratios + a ramped distribution with P(full mask) raised.
2c: physics held at 0 until `lambda_phys_start_step`, then ramped.

Run:  python -m pytest tests/test_phase2_schedule.py -v
      python tests/test_phase2_schedule.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "train"))

import pytest
import torch
import yaml

from data.mask import BlockMasker
from train_unified import _mask_probs_at, _physics_lambda_at, sample_mask_ratios


def _cfg(ramp=1000):
    return {"curriculum": {
        "train_mask_ratios": [0.25, 0.5, 0.75, 1.0],
        "train_mask_ratio_probs": [0.10, 0.25, 0.30, 0.35],
        "mask_schedule": {"ramp_steps": ramp,
                          "start_mask_ratio_probs": [0.55, 0.30, 0.10, 0.05]},
    }}


def test_mask_probs_ramp_start_to_target():
    cfg = _cfg(ramp=1000)
    rs0, p0 = _mask_probs_at(cfg, 0)
    _, pN = _mask_probs_at(cfg, 1000)
    assert rs0 == [0.25, 0.5, 0.75, 1.0]
    assert abs(float(p0[-1]) - 0.05) < 1e-6      # P(full) at step 0
    assert abs(float(pN[-1]) - 0.35) < 1e-6      # P(full) at/after ramp
    mid = float(_mask_probs_at(cfg, 500)[1][-1])
    assert float(p0[-1]) < mid < float(pN[-1]), "full-mask mass must rise"
    for _, probs in ((0, p0), (1000, pN)):
        assert abs(float(probs.sum()) - 1.0) < 1e-6


def test_sample_mask_ratios_per_sample_and_nonzero():
    cfg = _cfg()
    rng = torch.Generator().manual_seed(0)
    ratios = sample_mask_ratios(cfg, rng, b=8, step=0)
    assert len(ratios) == 8
    assert all(r > 0.0 for r in ratios), "0.0 is excluded"
    assert all(r in (0.25, 0.5, 0.75, 1.0) for r in ratios)


def test_per_sample_masks_track_each_ratio():
    occ = (torch.rand(3, 1, 64, 64) > 0.5).float()
    mk = BlockMasker(placement="random", seed=3)
    M = mk.sample_per_sample(occ, [0.0, 1.0, 0.5])
    assert (M[0] > 0.5).all(), "ratio 0.0 -> all visible"
    assert (M[1] < 0.5).all(), "ratio 1.0 -> all masked"
    achieved = float((M[2] < 0.5).float().mean().item())
    assert abs(achieved - 0.5) < 0.15, f"ratio 0.5 achieved {achieved}"


def test_physics_lambda_schedule():
    assert _physics_lambda_at(0, 3.32, 100, 200) == 0.0
    assert _physics_lambda_at(99, 3.32, 100, 200) == 0.0
    assert _physics_lambda_at(300, 3.32, 100, 200) == pytest.approx(3.32)
    mid = _physics_lambda_at(200, 3.32, 100, 200)
    assert 0.0 < mid < 3.32
    # Constant when no start and no ramp (previous behaviour).
    assert _physics_lambda_at(5, 1.5, 0, 0) == 1.5
    # Ramp with no explicit start: previous 0->target behaviour.
    assert _physics_lambda_at(0, 2.0, 0, 100) == pytest.approx(0.02)


def test_shipped_config_sets_phase2_schedule():
    with open(os.path.join(REPO_ROOT, "configs", "unified.yaml")) as f:
        cfg = yaml.safe_load(f)
    cur = cfg["curriculum"]
    assert cur["mask_schedule"]["ramp_steps"] > 0
    i = cur["train_mask_ratios"].index(1.0)
    assert cur["train_mask_ratio_probs"][i] > 0.15, "P(full mask) must exceed 0.15"
    assert cfg["staging"].get("lambda_phys_start_step", 0) > 0
    assert cfg["train"].get("log_grad_share_every_steps", 0) > 0


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
