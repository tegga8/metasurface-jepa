"""Regression tests for BUGLOG Tier 3 rows #13-#21 (2026-08-18), minus the
controller rows removed in the architecture repair.

#13 FixedValidation prediction-health stats pool the PROJECTED prediction
    (ph_), not raw z_hat — they change when the objective's projector weights
    change (spec §17: the projector lives on the objective, never model.proj).
#14 healthy_references projects reference RAW embeddings through the CANDIDATE
    OBJECTIVE's projector, never a reference/random head (separate random init =
    different coordinate system). objective=None keeps proj stats == raw stats.
#16 no itertools.cycle(loader) anywhere in the training path.
#17 CUDA RNG state is saved/restored with checkpoints; CPU-only environments
    skip the CUDA part safely. (The full checkpoint save/restore round-trip now
    lives in tests/test_checkpoint_resume.py — §30, replacing the removed
    phase-checkpoint path.)
#18 IntervalLossAccumulator reports exact per-interval training-loss means,
    reset after each report, correct under any grad_accum and phase boundaries.
#19 FixedValidation metric aggregation is global (loss_sum / mask_count), so
    the metric is invariant to batch partitioning.
#20 jepa_loss raises on a zero-token mask instead of silently falling back to
    the full-token mean (masked-only objective is undefined).
#21 classify_health returns explicit UNAVAILABLE for n_geoms < 2 instead of
    letting NaN flow into a HEALTHY/COLLAPSED verdict.

Run:  python tests/test_tier3_fixes.py   (pytest-collectable)
"""

import math
import os
import random
import sys

import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from data.mask import BlockMasker  # noqa: E402
from diagnostics.representation_health import (  # noqa: E402
    classify_health, token_space_stats,
)
from losses.jepa_loss import jepa_loss  # noqa: E402
from train.engine import (  # noqa: E402
    FixedValidation, IntervalLossAccumulator, collect_rng_state, healthy_references,
    restore_rng_state,
)


# --------------------------------------------------------------------------
# shared fake model (same shape-contract as test_fixed_validation_metric_path)
# --------------------------------------------------------------------------

class FakeSpectrumPath:
    def __call__(self, S, goal_mode="real", need_weights=False):
        B = S.shape[0]
        return (torch.randn(B, 4), torch.randn(B, 2, 3), torch.rand(B, 1, 2, 4))


class FakeModel:
    """Deterministic model-shaped object (outputs are a pure function of the
    inputs, no RNG — required for metric-invariance tests); carries NO
    projector of its own (spec §17: the projection lives on the objective)."""

    def __init__(self):
        self.spectrum_path = FakeSpectrumPath()
        self.training = False

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def __call__(self, G, S, M, goal_mode="real", need_attn=False, with_target=True):
        B = G.shape[0]
        mask = (M.view(B, -1) == 0)
        z = G.view(B, 256, 3, 16).mean(dim=-1)       # (B, 256, 3), goal-dependent
        z_hat = z * (0.5 if goal_mode == "null" else 1.0)
        z_y = z * 0.9 + 0.1
        return {"z_hat": z_hat, "z_y_raw": z_y, "mask": mask}


class _ScaleProj(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = float(k)

    def forward(self, x):
        if self.k == 1.0:
            return x
        return x * self.k


def _objective(k):
    """Objective-shaped carrier for the projector (spec §17: the projection
    lives on the objective, never on the model)."""

    class _Obj:
        name = "tier3"
        projector = None if k is None else _ScaleProj(k)

    return _Obj()


def _fixed_val(batch_sizes, seed=0, n_per=2):
    torch.manual_seed(seed)
    batches = []
    for b in batch_sizes:
        for _ in range(b):
            batches.append((torch.randn(n_per, 3, 64, 64), torch.randn(n_per, 301)))
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=12345)
    M = masker.sample(torch.randn(n_per, 3, 64, 64), 0.5)
    fv = FixedValidation(batches, ratio=0.5, mask_seed=12345, device="cpu")
    fv.masks = [M.to("cpu").clone() for _ in batches]
    return fv


def _refs_dict(seed=0, n=4):
    return token_space_stats(torch.randn(n, 4, 3))


# --------------------------------------------------------------------------
# #13 — prediction-health stats built from the PROJECTED prediction
# --------------------------------------------------------------------------

def test_prediction_health_tracks_projection_weights():
    """Raw z_hat pooling is invariant to the projector; projected pooling must
    change when the OBJECTIVE's projector changes (linear scale k -> mean_std
    scales |k|) — the projection belongs to the objective, not the model (§17)."""
    fv = _fixed_val([2, 2])
    _, h1 = fv.evaluate(FakeModel(), _objective(1.0), _refs_dict(), _refs_dict())
    _, h5 = fv.evaluate(FakeModel(), _objective(5.0), _refs_dict(), _refs_dict())
    s1 = h1["pred"]["mean_std"]
    s5 = h5["pred"]["mean_std"]
    assert math.isfinite(s1) and math.isfinite(s5)
    assert abs(s5 - 5.0 * s1) < 1e-4, (
        "projected pooled stats must scale with the projection, got "
        f"k1={s1:.6e} k5={s5:.6e}")


# --------------------------------------------------------------------------
# #14 — healthy reference projected through the CANDIDATE's proj head
# --------------------------------------------------------------------------

class _RefModel:
    """released-init style reference: EMA returns fixed Z; no own projector
    (spec §17); forward returns z_hat = Z_aug (T dims must match)."""

    def __init__(self, ema_out):
        self.ema = _EmaStub(ema_out)
        self.training = False

    def __call__(self, G, S, M):
        B = G.shape[0]
        mask = (M.view(B, -1) == 0)
        return {"z_hat": self.ema(G), "mask": mask}


class _EmaStub:
    def __init__(self, z):
        self.z = z

    def __call__(self, G):
        return self.z


def test_healthy_references_use_objective_projector():
    Z = torch.randn(2, 256, 8)               # T must match the 256-token mask grid
    fv = _fixed_val([2, 2], n_per=2)
    refs_model = _RefModel(ema_out=Z)
    refs = healthy_references(refs_model, fv, objective=_objective(10.0))
    expected = token_space_stats(torch.cat([Z * 10.0] * 4, dim=0))  # one per batch
    for k in ("eff_rank_unnorm", "eff_rank_frac", "token_std", "same_token_cos",
              "n_geoms"):
        assert abs(refs["proj"][k] - expected[k]) < 1e-6, f"proj stat {k} mismatch"
    for k in ("mean", "p05", "median", "p95", "min"):
        assert abs(refs["proj"]["pairwise_cos"][k]
                   - expected["pairwise_cos"][k]) < 1e-6, f"pairwise {k} mismatch"
    wrong = token_space_stats(torch.cat([Z * 2.0] * 4, dim=0))
    assert abs(refs["proj"]["token_std"] - wrong["token_std"]) > 1e-3, (
        "refs proj stats must be projected through the candidate objective's "
        "projector, not a reference/other head")


def test_healthy_references_default_identity_projection():
    """objective=None keeps proj stats == raw stats (identity projection) — the
    only default left after the repair removed model-level projection heads."""
    Z = torch.randn(2, 256, 8)
    fv = _fixed_val([2, 2], n_per=2)
    refs_model = _RefModel(ema_out=Z)
    refs = healthy_references(refs_model, fv)
    assert refs["proj"] == refs["raw"], (
        "objective=None must keep proj stats == raw stats (identity projection)")


# --------------------------------------------------------------------------
# #16 — no itertools.cycle in the training path
# --------------------------------------------------------------------------

def test_no_cycle_in_training_path():
    path = os.path.join(REPO_ROOT, "scripts", "train", "train_milestone_b.py")
    with open(path, "r") as f:
        src = f.read()
    assert "from itertools import" not in src, "itertools import still present"
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "cycle(" not in code, "cycle( still present in the training path"
    assert "enumerate(epoch_iterator)" in src, "explicit epoch iteration missing"


# --------------------------------------------------------------------------
# #17 — CUDA RNG state in checkpoint save/resume
# --------------------------------------------------------------------------

def test_collect_restore_rng_cpu_roundtrip():
    """State captured after draw N must make the next draw equal draw N+1 of the
    original stream (checkpoint semantics: save after a step, restore before the
    next step). torch + numpy + python RNG all roundtrip via the engine functions."""
    torch.manual_seed(7)
    draw1 = torch.randn(3).clone()
    state = collect_rng_state()                # captured after draw1
    draw2 = torch.randn(3).clone()             # the draw restore must replay
    _ = torch.randn(3)                         # advance the stream
    restore_rng_state(state)
    assert torch.equal(torch.randn(3), draw2), \
        "torch RNG must replay the exact next draw after restore"

    np.random.seed(11)
    n1 = np.random.rand(3).copy()
    state = collect_rng_state()                # captured after n1
    n2 = np.random.rand(3).copy()
    _ = np.random.rand(3)
    restore_rng_state(state)
    assert np.array_equal(np.random.rand(3), n2), "numpy RNG must roundtrip"

    random.seed(5)
    p1 = [random.random() for _ in range(3)]
    state = collect_rng_state()                # captured after p1
    p2 = [random.random() for _ in range(3)]
    _ = [random.random() for _ in range(3)]
    restore_rng_state(state)
    replay = [random.random() for _ in range(3)]
    assert [round(v, 12) for v in replay] == [round(v, 12) for v in p2], \
        "python RNG must roundtrip"


def test_cuda_rng_state_roundtrip_when_available():
    if not torch.cuda.is_available():
        print("SKIP (no CUDA) test_cuda_rng_state_roundtrip_when_available")
        return
    torch.cuda.manual_seed(11)
    c1 = [s.clone() for s in torch.cuda.get_rng_state_all()]
    _ = torch.randn(3, device="cuda")
    restore_rng_state({"torch_cuda_rng": c1})
    assert all(torch.equal(a, b) for a, b in
               zip(torch.cuda.get_rng_state_all(), c1))


def test_cuda_state_restore_skipped_safely_on_cpu():
    """A checkpoint saved on GPU (torch_cuda_rng present) restored on a CPU-only
    machine must skip the CUDA part, not error (and vice versa: None is fine)."""
    state = collect_rng_state()
    restore_rng_state(state)                 # None (CPU) or valid list (CUDA)
    fake_gpu_state = {"torch_rng": torch.get_rng_state(),
                      "numpy_rng": __import__("numpy").random.get_state(),
                      "python_rng": __import__("random").getstate(),
                      "torch_cuda_rng": [torch.zeros(1, dtype=torch.uint8)]}
    restore_rng_state(fake_gpu_state)        # must not raise on CPU


# --------------------------------------------------------------------------
# #18 — IntervalLossAccumulator exact per-interval means
# --------------------------------------------------------------------------

def test_interval_loss_accumulator_exact_means():
    acc = IntervalLossAccumulator()
    seq = [1.0, 2.0, 3.5]
    for v in seq:
        acc.add(v)
    assert abs(acc.report() - (sum(seq) / len(seq))) < 1e-12
    assert acc.count == 0 and acc.sum == 0.0, "report() must reset"
    acc.add(0.5)
    acc.add(1.5)
    assert abs(acc.report() - 1.0) < 1e-12
    assert acc.report() == 0.0, "empty interval reports 0.0, not NaN"


# --------------------------------------------------------------------------
# #19 — batch-partition-invariant validation aggregation
# --------------------------------------------------------------------------

def _same_masks_fv(groups_a, groups_b, seed=0):
    torch.manual_seed(seed)
    G = torch.randn(96, 3, 64, 64)
    S = torch.randn(96, 301)
    masker = BlockMasker(placement="random", grid=16, min_side=3,
                         k_range=(1, 4), seed=12345)
    M = masker.sample(G, 0.5)                # per-sample masks, partition-free
    fv_a = FixedValidation([(G[a:b], S[a:b]) for a, b in groups_a],
                           ratio=0.5, mask_seed=12345, device="cpu")
    fv_b = FixedValidation([(G[a:b], S[a:b]) for a, b in groups_b],
                           ratio=0.5, mask_seed=12345, device="cpu")
    fv_a.masks = [M[a:b].clone() for a, b in groups_a]
    fv_b.masks = [M[a:b].clone() for a, b in groups_b]
    return fv_a, fv_b


def test_validation_metric_invariant_to_batch_partition():
    """Same 96 samples and per-sample masks, split [32,32,32] vs [16,40,40]:
    with per-batch-mean averaging the two splits give different means; global
    aggregation must agree. Same deterministic model instance for both."""
    fv_a, fv_b = _same_masks_fv([(0, 32), (32, 64), (64, 96)],
                                [(0, 16), (16, 56), (56, 96)])
    raw, proj = _refs_dict(n=96), _refs_dict(n=96)
    model = FakeModel()
    obj = _objective(1.0)
    m_a, _ = fv_a.evaluate(model, obj, raw, proj)
    m_b, _ = fv_b.evaluate(model, obj, raw, proj)
    assert abs(m_a["cos_err_r0.5"] - m_b["cos_err_r0.5"]) < 1e-9, (
        f"metric depends on batch partitioning: {m_a['cos_err_r0.5']} vs "
        f"{m_b['cos_err_r0.5']}")


def test_null_gap_invariant_to_batch_partition():
    fv_a, fv_b = _same_masks_fv([(0, 32), (32, 64), (64, 96)],
                                [(0, 16), (16, 56), (56, 96)])
    obj = _objective(1.0)
    ratio_key = f"cos_err_r{fv_a.ratio:g}"
    a = fv_a.null_gap(FakeModel(), obj)
    b = fv_b.null_gap(FakeModel(), obj)
    for name in ("real", "null", "gap"):
        key = f"{name}_{ratio_key}"
        assert abs(a[key] - b[key]) < 1e-9, f"{name} metric depends on partitioning"


# --------------------------------------------------------------------------
# #20 — explicit zero-mask policy
# --------------------------------------------------------------------------

def test_zero_mask_raises_value_error():
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)
    mask = torch.zeros(2, 4, dtype=torch.bool)      # no masked tokens at all
    raised = False
    try:
        jepa_loss(pred, target, mask)
    except ValueError as e:
        raised = True
        assert "no masked tokens" in str(e)
    assert raised, "zero-mask input must raise, not silently fall back to d.mean()"
    # and with proj too
    try:
        jepa_loss(pred, target, mask, proj=_ScaleProj(2.0))
    except ValueError:
        pass
    else:
        raise AssertionError("zero-mask with proj must raise too")


def test_nonzero_mask_still_works():
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[0, 0] = True
    loss, per = jepa_loss(pred, target, mask)
    assert math.isfinite(loss.item()) and per.shape == (2,)


# --------------------------------------------------------------------------
# #21 — n_samples < 2 -> explicit UNAVAILABLE, never a NaN verdict
# --------------------------------------------------------------------------

def test_classify_health_unavailable_for_one_sample():
    raw1 = token_space_stats(torch.randn(1, 4, 3))     # n_geoms = 1
    healthy = token_space_stats(torch.randn(4, 4, 3))  # n_geoms = 4
    status, signals = classify_health(raw1, healthy, healthy, healthy)
    assert status == "UNAVAILABLE"
    assert not status in ("HEALTHY", "COLLAPSED"), (
        "a one-sample health diagnostic must never produce HEALTHY/COLLAPSED")
    assert "n_geoms=1" in signals.get("reason", "")
    assert signals["votes"] == 0


def test_classify_health_unavailable_for_healthy_side_too():
    raw_ok = token_space_stats(torch.randn(4, 4, 3))
    small = token_space_stats(torch.randn(1, 4, 3))
    status, _ = classify_health(raw_ok, raw_ok, small, raw_ok)
    assert status == "UNAVAILABLE"


def test_one_sample_fixed_validation_yields_unavailable():
    torch.manual_seed(0)
    G = torch.randn(1, 3, 64, 64)
    S = torch.randn(1, 301)
    fv = FixedValidation([(G, S)], ratio=0.5, mask_seed=12345, device="cpu")
    model = FakeModel()
    _, health = fv.evaluate(model, _objective(1.0), _refs_dict(), _refs_dict())
    assert health["status"] == "UNAVAILABLE", (
        f"one-sample validation must be UNAVAILABLE, got {health['status']}")


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