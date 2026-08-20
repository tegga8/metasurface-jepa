"""FIX B regression: the healthy released-init reference must be ONE
deterministic state, independent of the ambient RNG stream.

Failure being fixed: two equivalent local smoke runs (same code, same config,
no explicit seed) produced different health states — Run A HEALTHY vs Run B
WARNING (votes=0) — because refs_model was freshly/re-randomly built per run,
consuming whatever ambient RNG state the process happened to hold. A health
gate that flips run-to-run on the same validation set is not scientific.

Required invariant (final pre-training directive, FIX B):

    fixed released MetaDiT reference
            |
    raw reference embeddings
            |
    CANDIDATE OBJECTIVE's projector      <-- the projected comparison space (§17)
            |
    projected healthy reference stats

This suite verifies that evaluating the SAME fixed validation set twice under
different ambient RNG states — with the SAME released reference state built via
build_deterministic_reference() — produces:
  - identical raw reference statistics
  - identical projected reference statistics
  - identical health classification
and that the candidate objective's projector (not a separately random head)
defines the projected comparison space.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch
import torch.nn as nn

from train.engine import FixedValidation, build_deterministic_reference, healthy_references

REFS_SEED = 777


class _ScaleProj(nn.Module):
    """Deterministic stand-in for a projection head: multiply by k."""

    def __init__(self, k):
        super().__init__()
        self.k = float(k)

    def forward(self, x):
        return x * self.k


class _DetSpectrumPath:
    """Deterministic, goal-mode-invariant spectrum-path stand-in (the smoke-scale
    real path is deterministic too; this keeps the unit test free of model code)."""

    def __call__(self, S, goal_mode="real", need_weights=False):
        B = S.shape[0]
        return (torch.zeros(B, 4), torch.zeros(B, 2, 3),
                torch.full((B, 1, 2, 4), 1.0 / 8.0))


class _Objective:
    """Objective-shaped carrier for the projection head (spec §17: the
    projection lives on the objective, never on the model)."""

    def __init__(self, k):
        self.name = "determinism"
        self.projector = _ScaleProj(k)


class _RefModel(nn.Module):
    """Released-init-style reference stand-in: a random-initialized linear
    encoder (the part build_deterministic_reference must fix — its init consumes
    the ambient torch RNG, like build_model's random components) plus a
    deterministic forward that returns the encoder output as z_hat/z_y. Carries
    NO projector of its own."""

    def __init__(self, hidden=4, T=256, grid=64):
        super().__init__()
        self.hidden = hidden
        self.T = T
        self.linear = nn.Linear(3 * grid * grid, T * hidden)
        self.spectrum_path = _DetSpectrumPath()

    def ema(self, G):
        return self.linear(G.flatten(1)).view(G.shape[0], self.T, self.hidden)

    def forward(self, G, S, M, goal_mode="real", need_attn=False, with_target=True):
        B = G.shape[0]
        mask = (M.view(B, -1) == 0)
        z = self.ema(G)
        return {"z_hat": z, "z_y": z, "mask": mask}


def _make_ref_model():
    return _RefModel()   # consumes ambient RNG (nn.Linear init), like build_model


def _fixed_val(n_batches=4, n_per=2, seed=0):
    """Deterministic fixed validation set (fixed batches, fixed masks)."""
    torch.manual_seed(seed)
    batches = [(torch.randn(n_per, 3, 64, 64), torch.randn(n_per, 301))
               for _ in range(n_batches)]
    return FixedValidation(batches, ratio=0.5, mask_seed=12345, device="cpu")


def _build_reference_pair():
    """Two builds of the SAME reference state, each under a DIFFERENT ambient
    RNG state (mimics two independent processes with time-based ambient RNG)."""
    torch.manual_seed(1111)
    m1 = build_deterministic_reference(_make_ref_model, seed=REFS_SEED)
    torch.manual_seed(2222)
    m2 = build_deterministic_reference(_make_ref_model, seed=REFS_SEED)
    return m1, m2


def _flat_params(model):
    return [p.detach().clone() for p in model.parameters()]


def test_deterministic_reference_weights_identical_across_ambient_rng():
    m1, m2 = _build_reference_pair()
    for p1, p2 in zip(_flat_params(m1), _flat_params(m2)):
        assert torch.equal(p1, p2), (
            "the released-init reference model must be a pure function of the "
            "fixed seed, not of the ambient RNG state")


def test_deterministic_reference_restores_ambient_rng():
    """The reference build must not perturb the ambient RNG stream (it is
    wrapped in fork_rng): continuing to draw after the build must give the same
    draws as an untouched stream."""
    torch.manual_seed(42)
    torch.randn(1)                     # advance one draw
    expected_b = torch.randn(1)        # draw 2 of the untouched stream
    torch.manual_seed(42)
    a = torch.randn(1)
    build_deterministic_reference(_make_ref_model, seed=REFS_SEED)
    b = torch.randn(1)
    assert torch.equal(b, expected_b), (
        "reference build must leave the ambient stream untouched")
    assert not torch.equal(a, b), "sanity: the stream is actually consuming RNG"


def test_healthy_reference_stats_identical_across_ambient_rng():
    """Same fixed validation set + same released reference state -> identical
    RAW and PROJECTED reference statistics, whatever the ambient RNG state."""
    fv = _fixed_val()
    m1, m2 = _build_reference_pair()
    obj = _Objective(5.0)
    refs1 = healthy_references(m1, fv, objective=obj)
    refs2 = healthy_references(m2, fv, objective=obj)

    assert refs1["raw"] == refs2["raw"], "raw reference stats must be identical"
    assert refs1["proj"] == refs2["proj"], "projected reference stats must be identical"
    assert refs1["pred"] == refs2["pred"], "reference prediction stats must be identical"
    for stats in (refs1["raw"], refs2["raw"], refs1["proj"], refs2["proj"]):
        assert stats["n_geoms"] >= 2


def test_health_classification_identical_across_ambient_rng():
    """The same candidate evaluated against the same reference state must get
    the same health status and the same collapse votes under both ambient RNG
    states — the Run A HEALTHY vs Run B WARNING (votes=0) flip is fixed."""
    fv = _fixed_val()
    m1, m2 = _build_reference_pair()
    obj = _Objective(5.0)
    refs1 = healthy_references(m1, fv, objective=obj)
    refs2 = healthy_references(m2, fv, objective=obj)

    _, h1 = fv.evaluate(m1, obj, refs1["raw"], refs1["proj"])
    _, h2 = fv.evaluate(m2, obj, refs2["raw"], refs2["proj"])
    assert h1["status"] == h2["status"]
    assert h1["signals"]["votes"] == h2["signals"]["votes"]
    assert h1["signals"] == h2["signals"]


def test_objective_projector_defines_projected_comparison_space():
    """The projected reference stats must be measured through the CANDIDATE
    OBJECTIVE's projector — a k=10 objective projector gives 5x the projected
    token_std of a k=2 projector on the same raw embeddings (raw side unchanged)."""
    fv = _fixed_val()
    m1, _ = _build_reference_pair()

    refs_2 = healthy_references(m1, fv, objective=_Objective(2.0))
    refs_10 = healthy_references(m1, fv, objective=_Objective(10.0))
    assert refs_2["raw"] == refs_10["raw"], "raw stats must not move with the objective"
    s2, s10 = refs_2["proj"]["token_std"], refs_10["proj"]["token_std"]
    assert abs(s10 - 5.0 * s2) < 1e-4, f"projected stats must scale with proj k: {s2} vs {s10}"