"""Regression tests for BUGLOG Tier 2 rows #7-#9, #11 (2026-08-18).

#7  healthy_references() must project on the model's device — no .cpu() before
    proj and no .cuda() hack (the projection input must be the exact tensor the
    EMA encoder produced).
#8  fixed_validation_from_loader must honor n_samples exactly (trim the final
    batch), including n_samples < batch_size.
#9  mask coverage std must use population std (unbiased std of one sample is NaN).
#11 load_into_model must refuse silent non-strict loads.

#12 was the AdaptiveController state round-trip — the adaptive ladder was
removed in the architecture repair (controller and LOSS_LADDER deleted); the
resume-integrity contract it protected now lives in the §30 checkpoint tests
(tests/test_checkpoint_resume.py) and the engine's load_checkpoint strictness.

(#10 — global_step counts optimizer steps under grad_accum>1 — is verified by
integration runs with grad_accum=1 vs 2; see the Milestone B BUGLOG.)

Run:  python tests/test_tier2_fixes.py   (pytest-collectable)
"""

import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn

from assembly import load_into_model, saveable_state_dict  # noqa: E402
from train.engine import (  # noqa: E402
    FixedValidation, fixed_validation_from_loader, healthy_references,
)
from torch.utils.data import Dataset  # noqa: E402


# --------------------------------------------------------------------------
# #7 — healthy_references device contract
# --------------------------------------------------------------------------

def test_healthy_references_projects_ema_output_object():
    """The projection must receive the exact tensor object the EMA encoder
    returned (no .cpu()/transform before proj, no .cuda() round-trip hack)."""
    torch.manual_seed(0)
    G = torch.randn(2, 3, 64, 64)
    S = torch.randn(2, 301)
    fv = FixedValidation([(G, S), (G, S)], ratio=0.5, mask_seed=12345, device="cpu")

    sentinel = torch.randn(2, 256, 3)
    proj_seen = {"ema_object": False, "cuda_called": False, "calls": 0}

    class _Ema:
        def __call__(self, G):
            return sentinel

    class _Proj:
        def __call__(self, x):
            proj_seen["calls"] += 1
            if x is sentinel:
                proj_seen["ema_object"] = True     # Bug #14: z_y must be projected
            return x * 2.0                          # in-place, never transformed

    class _Objective:
        """Objective-shaped carrier for the projector (spec §17: the projection
        lives on the objective, never on the model)."""
        name = "tier2"
        projector = _Proj()

    class _Model:
        def __init__(self):
            self.ema = _Ema()
            self.training = False

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def __call__(self, G_, S_, M):
            B = G_.shape[0]
            mask = (M.view(B, -1) == 0)
            return {"z_hat": torch.randn(B, 256, 3), "mask": mask}

    original_cuda = torch.Tensor.cuda
    def _raise_cuda(self, *a, **k):
        proj_seen["cuda_called"] = True
        return original_cuda(self, *a, **k)
    torch.Tensor.cuda = _raise_cuda
    try:
        refs = healthy_references(_Model(), fv, objective=_Objective())
    finally:
        torch.Tensor.cuda = original_cuda

    assert proj_seen["ema_object"], (
        "proj input must be the exact EMA output object — a .cpu()/transform "
        "before projection breaks the device contract on GPU")
    # Bug #13: z_hat is also projected (for pooled pred stats in the same space).
    assert proj_seen["calls"] == 4, (
        "expected 2 batches x (ema z_y projection + z_hat projection)")
    assert not proj_seen["cuda_called"], "no .cuda() round-trip allowed"
    assert set(refs) == {"raw", "proj", "pred"}


# --------------------------------------------------------------------------
# #8 — fixed_validation_from_loader exact n_samples
# --------------------------------------------------------------------------

class _TinyDS(Dataset):
    """Deterministic per index (no global-RNG dependence)."""

    def __len__(self):
        return 12

    def __getitem__(self, i):
        torch.manual_seed(1000 + i)
        return torch.randn(3, 64, 64), torch.randn(301)


def test_loader_honors_exact_n_samples_not_multiple_of_batch():
    fv = fixed_validation_from_loader(_TinyDS(), n_samples=10, batch_size=4,
                                      device="cpu")
    assert fv.mask_statistics["n_samples"] == 10
    assert sum(g.shape[0] for g, _ in fv.batches) == 10


def test_loader_n_samples_smaller_than_batch():
    """n=2 with batch_size=4 must yield 2 samples, not 4 (old behavior)."""
    fv = fixed_validation_from_loader(_TinyDS(), n_samples=2, batch_size=4,
                                      device="cpu")
    assert fv.mask_statistics["n_samples"] == 2
    assert sum(g.shape[0] for g, _ in fv.batches) == 2


def test_loader_deterministic_across_calls():
    a = fixed_validation_from_loader(_TinyDS(), n_samples=10, batch_size=4,
                                     device="cpu")
    b = fixed_validation_from_loader(_TinyDS(), n_samples=10, batch_size=4,
                                     device="cpu")
    assert torch.equal(a.batches[0][0], b.batches[0][0])
    assert a.mask_statistics == b.mask_statistics


# --------------------------------------------------------------------------
# #9 — mask coverage std finite at n_batches=1
# --------------------------------------------------------------------------

def test_single_batch_mask_std_is_finite():
    torch.manual_seed(0)
    G = torch.randn(1, 3, 64, 64)
    S = torch.randn(1, 301)
    fv = FixedValidation([(G, S)], ratio=0.5, mask_seed=12345, device="cpu")
    std = fv.mask_statistics["actual_mask_ratio_std"]
    assert math.isfinite(std), f"single-batch mask std must be finite, got {std}"
    assert std == 0.0, "population std of a single observation is 0"


# --------------------------------------------------------------------------
# #11 — load_into_model strict contract
# --------------------------------------------------------------------------

class _Tiny(nn.Module):
    def __init__(self, with_extra=False):
        super().__init__()
        self.l = nn.Linear(3, 3)
        if with_extra:
            self.extra = nn.Parameter(torch.zeros(1))


class _WithReleased(nn.Module):
    """Frozen released-style submodule must be filtered on save and load."""

    def __init__(self):
        super().__init__()
        self.sub = nn.Module()
        self.sub.released_enc = nn.Linear(2, 2)
        self.l = nn.Linear(3, 3)


def test_load_roundtrip_strict_ok():
    m = _Tiny()
    sd = saveable_state_dict(m)
    m2 = _Tiny()
    load_into_model(m2, sd, "cpu")          # strict by default → must succeed
    assert torch.equal(m2.l.weight, m.l.weight)


def test_load_raises_on_key_mismatch():
    m = _Tiny()
    sd = saveable_state_dict(m)
    m3 = _Tiny(with_extra=True)             # model has a param the ckpt lacks
    try:
        load_into_model(m3, sd, "cpu")
    except RuntimeError as e:
        assert "missing" in str(e) or "unexpected" in str(e)
        return
    raise AssertionError("load_into_model must raise on key mismatch, not load "
                         "silently with strict=False")


def test_load_filters_released_keys_before_strict():
    m = _WithReleased()
    sd = saveable_state_dict(m)
    assert not any(".released." in k for k in sd)
    m2 = _WithReleased()
    load_into_model(m2, sd, "cpu")          # released filter → no mismatch


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