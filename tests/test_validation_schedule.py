"""Test validation schedule config validation (hardening spec §7).

Requires 0 < val_every_steps < total_steps when best-checkpoint selection is enabled.
"""

import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pytest
from scripts.train.train_milestone_b import validate_config


def test_validate_config_val_every_steps_zero_raises():
    cfg = {
        "train": {"val_every_steps": 0, "warmup_steps": 0, "epochs": 10, "batch_size": 8, "grad_accum": 1},
        "data": {"max_train_samples": 100},
    }
    # steps_per_epoch = 100/8 = 12.5 -> 13, total_steps = 130
    with pytest.raises(ValueError, match="val_every_steps must be > 0"):
        validate_config(cfg, total_steps=130)


def test_validate_config_val_every_steps_ge_total_raises():
    cfg = {
        "train": {"val_every_steps": 200, "warmup_steps": 0, "epochs": 10, "batch_size": 8, "grad_accum": 1},
        "data": {"max_train_samples": 100},
    }
    # total_steps = 130, val_every_steps=200 >= 130
    with pytest.raises(ValueError, match="val_every_steps.*must be < total_steps"):
        validate_config(cfg, total_steps=130)


def test_validate_config_val_every_steps_ok():
    cfg = {
        "train": {"val_every_steps": 10, "warmup_steps": 0, "epochs": 10, "batch_size": 8, "grad_accum": 1},
        "data": {"max_train_samples": 100},
    }
    # total_steps = 130, val_every_steps=10 < 130
    validate_config(cfg, total_steps=130)  # Should not raise


def test_validate_config_val_every_steps_equal_total_raises():
    cfg = {
        "train": {"val_every_steps": 130, "warmup_steps": 0, "epochs": 10, "batch_size": 8, "grad_accum": 1},
        "data": {"max_train_samples": 100},
    }
    with pytest.raises(ValueError, match="val_every_steps.*must be < total_steps"):
        validate_config(cfg, total_steps=130)


if __name__ == "__main__":
    test_validate_config_val_every_steps_zero_raises()
    test_validate_config_val_every_steps_ge_total_raises()
    test_validate_config_val_every_steps_ok()
    test_validate_config_val_every_steps_equal_total_raises()
    print("All validation schedule tests passed")