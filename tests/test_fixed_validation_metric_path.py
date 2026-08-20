"""Unit tests for the FixedValidation metric path (audit item 4).

The prediction metric (cos_err) must be computed with need_attn=False — the
predictor's attention-extraction path (need_weights branch) was a divergence
source between in-loop validation and the winner eval. Goal-token utilization
remains monitored via the spectrum-path attention weights, which never touch
the model's forward outputs.

Run:  python tests/test_fixed_validation_metric_path.py   (collectable by pytest)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch

from diagnostics.representation_health import token_space_stats  # noqa: E402
from train.engine import FixedValidation  # noqa: E402


class FakeSpectrumPath:
    def __call__(self, S, goal_mode="real", need_weights=False):
        B = S.shape[0]
        return (torch.randn(B, 4), torch.randn(B, 2, 3), torch.rand(B, 1, 2, 4))


class FakeModel:
    """Minimal model-shaped object recording every forward's kwargs."""

    def __init__(self):
        self.spectrum_path = FakeSpectrumPath()
        self.calls = []
        self.training = False

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def __call__(self, G, S, M, goal_mode="real", need_attn=False, with_target=True):
        self.calls.append({"goal_mode": goal_mode, "need_attn": need_attn})
        B = G.shape[0]
        mask = (M.view(B, -1) == 0)                      # (B, 256) bool, 1 = masked
        return {"z_hat": torch.randn(B, 256, 3), "z_y": torch.randn(B, 256, 3),
                "mask": mask}


def _fixed_val_and_model():
    torch.manual_seed(0)
    G = torch.randn(2, 3, 64, 64)
    S = torch.randn(2, 301)
    fv = FixedValidation([(G, S), (G, S)], ratio=0.5, mask_seed=12345, device="cpu")
    model = FakeModel()
    raw = token_space_stats(torch.randn(4, 2, 3))
    proj_stats = token_space_stats(torch.randn(4, 2, 3))
    return fv, model, raw, proj_stats


def test_evaluate_forwards_without_attention():
    fv, model, raw, proj_stats = _fixed_val_and_model()
    metrics, health = fv.evaluate(model, None, raw, proj_stats)
    assert model.calls, "evaluate must forward the model"
    assert all(c["need_attn"] is False for c in model.calls), (
        "prediction metric must not request predictor attention "
        f"(saw need_attn in {model.calls})")
    assert all(c["goal_mode"] == "real" for c in model.calls)


def test_evaluate_metrics_are_prediction_only():
    fv, model, raw, proj_stats = _fixed_val_and_model()
    metrics, _ = fv.evaluate(model, None, raw, proj_stats)
    assert "cos_err_r0.5" in metrics
    assert "goal_token_entropy" not in metrics
    assert "goal_token_log_entropy" not in metrics
    assert model.calls[-1]["need_attn"] is False


def test_mode_restored_after_evaluate():
    fv, model, raw, proj_stats = _fixed_val_and_model()
    model.train()
    fv.evaluate(model, None, raw, proj_stats)
    assert model.training is True


def test_health_goal_stats_still_available():
    """Goal-token utilization moves via the spectrum-path weights (never touch
    the prediction forward), so health['goal']/health['attention'] must exist."""
    fv, model, raw, proj_stats = _fixed_val_and_model()
    _, health = fv.evaluate(model, None, raw, proj_stats)
    assert "goal" in health and "attention" in health
    assert "goal_token_effective_rank" in health["goal"]


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