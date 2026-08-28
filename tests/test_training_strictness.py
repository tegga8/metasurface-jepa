"""Tests for strict real-data / released-weights enforcement (Fixes 5, 6, 16).

Verifies:
- Real mode with missing dataset split → RuntimeError (no silent synthetic).
- Real mode with missing released spectrum weights → RuntimeError.
- Smoke mode (explicit flag) allows synthetic data + dummy spectrum weights.
- Preflight requires real data + released weights.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "train"))

import pytest
import yaml
import torch


def _load_cfg():
    with open(os.path.join(REPO_ROOT, "configs", "unified.yaml")) as f:
        return yaml.safe_load(f)


def test_real_mode_missing_data_raises():
    """Fix 5: real training with a missing dataset split must raise, never
    silently fall back to synthetic data."""
    from train_unified import train
    cfg = _load_cfg()
    cfg["data"]["train_split"] = "data/metadit/split_data/NONEXISTENT.mat"
    with pytest.raises(RuntimeError, match="real dataset split"):
        train(cfg, no_train=True, device="cpu", use_synthetic_smoke=False)


def test_real_mode_missing_spectrum_weights_raises():
    """Fix 6: real mode with missing released spectrum encoder must raise."""
    from train_unified import _ensure_spectrum_weights
    missing = os.path.join(REPO_ROOT, "data/metadit/weights/NONEXISTENT.pth")
    with pytest.raises(RuntimeError, match="released spectrum encoder"):
        _ensure_spectrum_weights(missing, "cpu", allow_dummy=False)


def test_smoke_mode_allows_dummy_spectrum_weights():
    """Fix 6: explicit smoke mode permits creating a dummy spectrum encoder."""
    from train_unified import _ensure_spectrum_weights
    import tempfile
    tmpdir = tempfile.mkdtemp()
    dummy = os.path.join(tmpdir, "dummy_spec.pth")
    path = _ensure_spectrum_weights(dummy, "cpu", allow_dummy=True)
    assert os.path.exists(path)
    os.remove(path)


def test_smoke_mode_synthetic_runs():
    """Fix 16: --use-synthetic-smoke runs with synthetic data (no real paths).

    Uses a temp directory for the dummy spectrum checkpoint so the real
    weights directory is never polluted by the smoke dummy.
    """
    import tempfile
    from train_unified import train
    tmpdir = tempfile.mkdtemp()
    cfg = _load_cfg()
    cfg["data"]["train_split"] = "data/metadit/split_data/NONEXISTENT.mat"
    cfg["data"]["val_split"] = "data/metadit/split_data/NONEXISTENT.mat"
    cfg["weights"]["spectrum"] = os.path.join(tmpdir, "dummy_spec.pth")
    cfg["data"]["use_synthetic"] = True
    report = train(cfg, no_train=True, device="cpu",
                   use_synthetic_smoke=True)
    assert "final_loss" in report


@pytest.mark.skipif(
    not os.path.exists(os.path.join(REPO_ROOT, "data/metadit/split_data/train_set.mat")),
    reason="real training split not present")
def test_preflight_requires_real_data():
    """Fix 17: preflight must require the real training split."""
    from train_unified import preflight
    cfg = _load_cfg()
    cfg["data"]["train_split"] = "data/metadit/split_data/NONEXISTENT.mat"
    with pytest.raises(RuntimeError, match="requires the real training split"):
        preflight(cfg, device="cpu")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(REPO_ROOT, "data/metadit/split_data/train_set.mat")),
    reason="real training split not present")
def test_preflight_passes_on_real_data():
    """Fix 17: preflight passes end-to-end on real data (shapes, finite loss,
    gradient ownership)."""
    from train_unified import preflight
    cfg = _load_cfg()
    result = preflight(cfg, device="cpu")
    checks = result["checks"]
    assert checks["occupancy_shape"] == [2, 1, 64, 64]
    assert checks["z_x_shape"] == [2, 256, 192]
    assert checks["z_hat_shape"] == [2, 256, 192]
    assert checks["scalar_pred_shape"] == [2, 3]
    assert checks["assembled_geometry_shape"] == [2, 3, 64, 64]
    assert checks["surrogate_prediction_shape"] == [2, 2, 301]
    assert checks["loss_finite"] is True
    own = result["gradient_ownership"]
    assert own["student_params_with_grad"] > 0
    assert own["decoder_params_with_grad"] > 0
    assert own["predictor_params_with_grad"] > 0
    assert own["surrogate_params_with_grad"] == 0
    assert own["ema_params_with_grad"] == 0
    assert own["scalar_mlp_ema_params_with_grad"] == 0
    assert own["released_params_with_grad"] == 0


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
