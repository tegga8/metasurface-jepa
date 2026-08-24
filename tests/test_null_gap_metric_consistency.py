"""Regression tests: FixedValidation metric consistency (audit item 6 — the
0.888873-vs-0.087273 divergence, root cause 2026-08-18).

Root cause: `torch.nn.functional.normalize(x, -1)` binds the second positional
argument to `p` (norm order), NOT `dim` — with `dim` defaulting to 1 (the
256-token dim). The probe/wrapper metric paths passed `-1` positionally,
producing p=-1 normalization over the token dim and garbage cos_err values
(0.888873) while the canonical `_acc_stats` used `dim=-1` (0.087273).

These tests pin the contract that any metric path touching prediction quality
(evaluate / null_gap) is (a) the identical formula with dim=-1, (b) invariant
to call order, and (c) pure — no input or weight mutation. Projection is the
OBJECTIVE's (spec §17), so the frozen-projection assertions check the objective
projector, never a model attribute.

Run:  python tests/test_null_gap_metric_consistency.py   (pytest-collectable)
"""

import copy
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch

from diagnostics.representation_health import token_space_stats  # noqa: E402
from predictor.gclct import GCLCT  # noqa: E402
from train.engine import FixedValidation  # noqa: E402


def _wrapper_cos_err(z_hat, z_y, mask, pos_style):
    """The metric formula as used by engine paths, parameterized by how the
    normalize call is written — this is exactly the divergence that occurred."""
    if pos_style == "positional":
        d = (1.0 - torch.nn.functional.cosine_similarity(
            torch.nn.functional.normalize(z_hat, -1),
            torch.nn.functional.normalize(z_y, -1), -1)).clamp(min=0)
    else:
        d = (1.0 - torch.nn.functional.cosine_similarity(
            torch.nn.functional.normalize(z_hat, dim=-1),
            torch.nn.functional.normalize(z_y, dim=-1), dim=-1)).clamp(min=0)
    return d[mask].mean().item()


def test_normalize_positional_minus_one_is_not_dim():
    """The pitfall itself: `normalize(x, -1)` binds p=-1 (and dim stays 1), so it
    is NOT a shortcut for `normalize(x, dim=-1)`. Guard against 'simplifications'."""
    torch.manual_seed(0)
    x = torch.randn(1, 256, 384)
    pos = torch.nn.functional.normalize(x, -1)
    kw = torch.nn.functional.normalize(x, dim=-1)
    assert not torch.allclose(pos, kw, atol=1e-3), (
        "positional -1 must not equal dim=-1 normalization (it binds p, not dim)")
    assert pos.shape == kw.shape
    # the observed divergence scale: garbage cos_err (~0.89) vs real (~0.09)
    mask = torch.randint(0, 2, (1, 256)).bool()
    bad = _wrapper_cos_err(x, x + 0.1, mask, "positional")
    good = _wrapper_cos_err(x, x + 0.1, mask, "keyword")
    assert abs(bad - good) > 0.1, f"positional-vs-keyword diff too small: {bad} vs {good}"


class _FakeSpectrumPath:
    def __call__(self, S, goal_mode="real", need_weights=False):
        B = S.shape[0]
        return (torch.randn(B, 4), torch.randn(B, 2, 3), torch.rand(B, 1, 2, 4))


class _Objective:
    """Objective-shaped carrier for the projection head (spec §17). Frozen
    during evaluate/null_gap (no autograd), like the real objective projector."""

    def __init__(self, seed=0):
        self.name = "nullgap"
        torch.manual_seed(seed)
        self.projector = torch.nn.Linear(3, 3)


class _FakeModel:
    """Deterministic model-shaped object: outputs are a pure function of the
    inputs (no RNG), so every metric path sees bit-identical forwards. Carries
    NO projector of its own (spec §17). Records every call."""

    def __init__(self, seed=0):
        self.seed = seed
        self.spectrum_path = _FakeSpectrumPath()
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
        # z_hat is goal-conditioned (null goal weakens it), z_y is the EMA-style
        # target from the FULL geometry: goal-independent, exactly like self.ema(G).
        z_hat = G.view(B, 256, 3, 16).mean(dim=-1)       # (B, 256, 3)
        z_hat = z_hat * (0.5 if goal_mode == "null" else 1.0)
        z_y = G.view(B, 256, 3, 16).mean(dim=-1) * 0.9 + 0.1
        return {"z_hat": z_hat, "z_y_raw": z_y, "mask": mask}


def _fixed_val_and_model():
    torch.manual_seed(1)
    G = torch.randn(2, 3, 64, 64)
    S = torch.randn(2, 301)
    fv = FixedValidation([(G, S), (G, S)], ratio=0.5, mask_seed=12345, device="cpu")
    model = _FakeModel()
    obj = _Objective()
    raw = token_space_stats(torch.randn(4, 2, 3))
    proj_stats = token_space_stats(torch.randn(4, 2, 3))
    return fv, model, obj, raw, proj_stats


def test_null_gap_real_equals_evaluate_cos_err():
    """null_gap's real-mode cos_err and evaluate's cos_err must be bit-identical
    (same batches, masks, formula, dim=-1 normalization, same objective projector)."""
    fv, model, obj, raw, proj_stats = _fixed_val_and_model()
    ratio_key = f"cos_err_r{fv.ratio:g}"
    gap_metrics = fv.null_gap(model, obj)
    real = gap_metrics[f"real_{ratio_key}"]
    metrics, _ = fv.evaluate(model, obj, raw, proj_stats)
    assert real == metrics[ratio_key], (
        f"null_gap real {real} != evaluate cos_err {metrics[ratio_key]}")


def test_null_gap_null_equals_evaluate_null_mode():
    """null_gap's null-mode cos_err must equal evaluate(goal_mode='null'). The
    target z_y is EMA-style (goal-independent, from the full geometry), so the
    null prediction is scored against the same target in both paths."""
    fv, model, obj, raw, proj_stats = _fixed_val_and_model()
    ratio_key = f"cos_err_r{fv.ratio:g}"
    gap_metrics = fv.null_gap(model, obj)
    null = gap_metrics[f"null_{ratio_key}"]
    metrics, _ = fv.evaluate(model, obj, raw, proj_stats, goal_mode="null")
    assert null == metrics[ratio_key], (
        f"null_gap null {null} != evaluate(null) {metrics[ratio_key]}")


def test_null_gap_gap_positive_with_distinct_goals():
    """A non-null goal must change z_hat on masked tokens: gap > 0."""
    fv, model, obj, raw, proj_stats = _fixed_val_and_model()
    ratio_key = f"cos_err_r{fv.ratio:g}"
    gap_metrics = fv.null_gap(model, obj)
    gap = gap_metrics[f"gap_{ratio_key}"]
    assert gap > 0.0, f"goal gap collapsed: {gap}"


def test_order_invariance_evaluate_nullgap_evaluate():
    """Metric values must not depend on which path ran first (canonical →
    diagnostic → canonical), i.e. no hidden state pollution between paths."""
    fv, model, obj, raw, proj_stats = _fixed_val_and_model()
    ratio_key = f"cos_err_r{fv.ratio:g}"
    m1, _ = fv.evaluate(model, obj, raw, proj_stats)
    gap_metrics = fv.null_gap(model, obj)
    m2, _ = fv.evaluate(model, obj, raw, proj_stats)
    assert m1[ratio_key] == m2[ratio_key], (
        f"cos_err changed across call order: {m1[ratio_key]} -> "
        f"{m2[ratio_key]}")
    # reversed order as well
    fv2, model2, obj2, _, _ = _fixed_val_and_model()
    gap_metrics2 = fv2.null_gap(model2, obj2)
    m3, _ = fv2.evaluate(model2, obj2, raw, proj_stats)
    gap_metrics3 = fv2.null_gap(model2, obj2)
    assert gap_metrics2 == gap_metrics3 and m3[ratio_key] == m1[ratio_key]


def test_evaluate_and_null_gap_do_not_mutate_inputs():
    """G/S/M tensors must be bit-identical after evaluate and null_gap."""
    fv, model, obj, raw, proj_stats = _fixed_val_and_model()
    G0, S0 = fv.batches[0]
    M0 = fv.masks[0].clone()
    g0, s0, m0 = G0.clone(), S0.clone(), M0.clone()
    fv.evaluate(model, obj, raw, proj_stats)
    fv.null_gap(model, obj)
    assert torch.equal(G0, g0) and torch.equal(S0, s0) and torch.equal(M0, m0), (
        "evaluate/null_gap mutated fixed inputs")


def test_projector_weights_invariant_under_evaluate_and_null_gap():
    """The objective's projection head must be frozen (no grad accumulation)
    across both paths."""
    fv, model, obj, raw, proj_stats = _fixed_val_and_model()
    w0 = obj.projector.weight.detach().clone()
    b0 = obj.projector.bias.detach().clone()
    fv.evaluate(model, obj, raw, proj_stats)
    fv.null_gap(model, obj)
    assert torch.equal(obj.projector.weight, w0) and torch.equal(obj.projector.bias, b0)
    assert obj.projector.weight.grad is None and obj.projector.bias.grad is None


def test_model_mode_restored_after_null_gap():
    fv, model, obj, raw, proj_stats = _fixed_val_and_model()
    model.train()
    fv.null_gap(model, obj)
    assert model.training is True
    assert all(c["need_attn"] is False for c in model.calls)


def test_need_weights_branch_does_not_change_predictions():
    """Cross-attention must produce the same z_hat with or without weight
    extraction — a numerical drift here would re-open the in-loop vs winner-eval
    divergence the audit removed (need_attn=True path)."""
    torch.manual_seed(3)
    g = GCLCT(depth=2, hidden=16, num_heads=2)
    g.eval()
    queries = torch.randn(1, 256, 16)
    kv = torch.randn(1, 80, 16)
    c = torch.randn(1, 16)
    with torch.no_grad():
        out_plain, w_plain = g(queries, kv, c, need_weights=False)
        out_w, w_list = g(queries, kv, c, need_weights=True)
    assert w_plain == [] and len(w_list) == g.depth
    assert torch.allclose(out_plain, out_w, atol=1e-5, rtol=1e-5), (
        f"need_weights changed predictions, max diff "
        f"{(out_plain - out_w).abs().max().item():.2e}")


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