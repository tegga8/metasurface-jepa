"""Regression tests for BUGLOG Tier 2 rows #7-#12 (2026-08-18).

#7  healthy_references() must project on the model's device — no .cpu() before
    proj and no .cuda() hack (the projection input must be the exact tensor the
    EMA encoder produced).
#8  fixed_validation_from_loader must honor n_samples exactly (trim the final
    batch), including n_samples < batch_size.
#9  mask coverage std must use population std (unbiased std of one sample is NaN).
#11 load_into_model must refuse silent non-strict loads.
#12 AdaptiveController state_dict/load_state_dict must round-trip patience
    counters, histories, transitions, and phase bookkeeping exactly, and the
    restored counters must drive subsequent decisions.

(#10 — global_step counts optimizer steps under grad_accum>1 — is verified by
integration runs of the adaptive smoke with grad_accum=1 vs 2; see BUGLOG.)

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
from train.adaptive import AdaptiveController  # noqa: E402
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
    proj_seen = {"same_object": False, "cuda_called": False}

    class _Ema:
        def __call__(self, G):
            return sentinel

    class _Proj:
        def __call__(self, x):
            proj_seen["same_object"] = (x is sentinel)
            return x * 2.0

    class _Model:
        def __init__(self):
            self.proj = _Proj()
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
        refs = healthy_references(_Model(), fv)
    finally:
        torch.Tensor.cuda = original_cuda

    assert proj_seen["same_object"], (
        "proj input must be the exact EMA output object — a .cpu()/transform "
        "before projection breaks the device contract on GPU")
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


# --------------------------------------------------------------------------
# #12 — AdaptiveController state round-trip
# --------------------------------------------------------------------------

def test_controller_state_roundtrip_exact():
    c = AdaptiveController({"max_total_steps": 100, "warmup_steps": 5,
                            "plateau_patience": 2, "min_delta": 1e-5,
                            "collapse_patience": 2}, ["jepa", "lejepa"])
    c.start_phase("jepa", 0)
    c.on_validation(0.9, "HEALTHY", 10)
    c.on_validation(0.7, "HEALTHY", 20)     # improvement
    c.on_validation(0.72, "HEALTHY", 30)    # plateau_bad = 1

    sd = c.state_dict()
    c2 = AdaptiveController({}, ["unrelated"])   # wrong config on purpose
    c2.max_total_steps = 7
    c2.load_state_dict(sd)

    assert c2.max_total_steps == 100 and c2.objectives == ["jepa", "lejepa"]
    assert c2.transitions == c.transitions
    p = c2.phase
    assert p.objective == "jepa" and p.idx == 0
    assert p.start_global_step == 0
    assert p.plateau_bad == 1 and p.collapse_bad == 0
    assert p.best_metric == 0.7 and p.best_step == 20
    assert p.metric_history == [0.9, 0.7, 0.72]
    assert p.health_history == ["HEALTHY", "HEALTHY", "HEALTHY"]


def test_restored_counters_drive_future_decisions():
    """The restored plateau counter must actually trigger switching on the next
    non-improvement — the point of bug #12 (resume was restarting patience)."""
    c = AdaptiveController({"max_total_steps": 100, "warmup_steps": 0,
                            "plateau_patience": 2, "min_delta": 1e-5,
                            "collapse_patience": 2}, ["jepa", "lejepa"])
    c.start_phase("jepa", 0)
    c.on_validation(0.9, "HEALTHY", 10)     # improvement (best=0.9, bad=0)
    c.on_validation(0.91, "HEALTHY", 20)    # non-improvement -> plateau_bad = 1

    c2 = AdaptiveController({}, ["x"])
    c2.load_state_dict(c.state_dict())
    assert c2.phase.plateau_bad == 1
    d = c2.on_validation(0.92, "HEALTHY", 30)   # plateau_bad -> 2 -> switch
    assert d["action"] == "switch", (
        "restored plateau counter must fire on the next non-improvement, "
        f"got action={d['action']}")
    assert d["transition"]["next"] == "lejepa"


def test_controller_state_without_phase():
    c = AdaptiveController({"max_total_steps": 50}, ["jepa"])
    sd = c.state_dict()
    assert sd["phase"] is None
    c2 = AdaptiveController({}, ["x"])
    c2.load_state_dict(sd)
    assert c2.phase is None and c2.max_total_steps == 50


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