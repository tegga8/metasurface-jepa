"""Tests for length-proportional staging (resolve_staging_steps, 2026-10-05).

Guards the operator retune: the 10k-era schedule is preserved bit-identically
when the fractions reproduce it, the 70k run gets the designed proportions
instead of a ~7 %-of-run compression, and absent fractions keep the old
absolute behavior untouched.

Run:  python -m pytest tests/test_staging_fractions.py -v
      python tests/test_staging_fractions.py
"""

import copy
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "train"))

from train_unified import _physics_lambda_at, resolve_staging_steps


def _cfg():
    return {
        "train": {"total_steps": 10000},
        "staging": {
            "lambda_phys_start_step": 2000,
            "lambda_phys_ramp_steps": 3000,
            "multi_target_start_step": 1000,
            "multi_target_ramp_steps": 2000,
            "lambda_phys_start_frac": 0.20,
            "lambda_phys_ramp_frac": 0.30,
            "multi_target_start_frac": 0.10,
            "multi_target_ramp_frac": 0.20,
        },
        "curriculum": {"mask_schedule": {"ramp_steps": 3000, "ramp_frac": 0.30}},
    }


def test_fractions_reproduce_the_10k_schedule():
    cfg = resolve_staging_steps(_cfg(), 10000)
    s = cfg["staging"]
    assert s["lambda_phys_start_step"] == 2000
    assert s["lambda_phys_ramp_steps"] == 3000
    assert s["multi_target_start_step"] == 1000
    assert s["multi_target_ramp_steps"] == 2000
    assert cfg["curriculum"]["mask_schedule"]["ramp_steps"] == 3000


def test_70k_run_keeps_designed_proportions():
    cfg = resolve_staging_steps(_cfg(), 70000)
    s = cfg["staging"]
    assert s["lambda_phys_start_step"] == 14000          # 20 %
    assert s["lambda_phys_ramp_steps"] == 21000          # 30 %
    assert s["multi_target_start_step"] == 7000
    assert s["multi_target_ramp_steps"] == 14000
    assert cfg["curriculum"]["mask_schedule"]["ramp_steps"] == 21000


def test_absent_fractions_preserve_absolute_behavior():
    cfg = _cfg()
    del cfg["staging"]["lambda_phys_start_frac"]
    del cfg["staging"]["lambda_phys_ramp_frac"]
    del cfg["curriculum"]["mask_schedule"]["ramp_frac"]
    before = copy.deepcopy(cfg)
    resolve_staging_steps(cfg, 70000)
    assert cfg["staging"]["lambda_phys_start_step"] == 2000   # untouched
    assert cfg["staging"]["lambda_phys_ramp_steps"] == 3000
    assert cfg["curriculum"]["mask_schedule"]["ramp_steps"] == 3000
    assert cfg["staging"]["multi_target_start_step"] == 7000  # frac still applied


def test_physics_ramp_shape_at_70k_after_resolution():
    cfg = resolve_staging_steps(_cfg(), 70000)
    s = cfg["staging"]
    lam = 3.32
    assert _physics_lambda_at(0, lam, s["lambda_phys_start_step"],
                              s["lambda_phys_ramp_steps"]) == 0.0
    assert _physics_lambda_at(13999, lam, s["lambda_phys_start_step"],
                              s["lambda_phys_ramp_steps"]) == 0.0
    mid = _physics_lambda_at(24500, lam, s["lambda_phys_start_step"],
                             s["lambda_phys_ramp_steps"])
    assert abs(mid - lam * 0.5) < 0.02 * lam
    full = _physics_lambda_at(35000, lam, s["lambda_phys_start_step"],
                              s["lambda_phys_ramp_steps"])
    assert full == pytest.approx(lam, rel=1e-6)


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
