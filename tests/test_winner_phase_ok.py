"""Unit tests for the adaptive-ladder winner eligibility (audit items 5 + 15).

Tier 1 (item 5) introduced HEALTHY-only gating; Tier 3 (item 15) made the gate
explicit and health-blind-proof: eligibility now requires a HEALTHY-gated best
(best_healthy_cos_err present = a real HEALTHY checkpoint exists), never the
best-prediction metric (which can be a WARNING/COLLAPSED step). Winner selection
lives in train.adaptive.select_winner so it is testable without the training
script's heavy imports.

Run:  python tests/test_winner_phase_ok.py        (also collectable by pytest)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from train.adaptive import goal_score, phase_ok, select_winner  # noqa: E402


def _report(**over):
    base = {"objective": "jepa", "phase": 0, "representation_status": "HEALTHY",
            "unstable_steps": 0, "best_cos_err": 0.5,
            "best_healthy_cos_err": 0.5, "step_of_best_healthy": 3,
            "best_healthy_checkpoint": "phase_00_jepa_best_healthy.pt"}
    base.update(over)
    return base


def test_healthy_phase_is_eligible():
    assert phase_ok(_report())


def test_warning_phase_without_healthy_checkpoint_is_excluded():
    """The Tier-1 bug: only COLLAPSED was excluded. The Tier-3 rule: no HEALTHY
    checkpoint exists (WARNING status alone) -> not eligible, even with a low
    prediction-best cosine error."""
    assert not phase_ok(_report(representation_status="WARNING",
                                best_healthy_cos_err=None))
    assert not phase_ok(_report(representation_status="WARNING"))


def test_collapsed_phase_is_excluded():
    assert not phase_ok(_report(representation_status="COLLAPSED",
                                best_healthy_cos_err=None))


def test_unknown_or_missing_status_is_excluded():
    assert not phase_ok(_report(representation_status=None,
                                best_healthy_cos_err=None))
    assert not phase_ok(_report(representation_status="UNSTABLE",
                                best_healthy_cos_err=None))


def test_unstable_phase_is_excluded():
    assert not phase_ok(_report(unstable_steps=3))


def test_missing_best_healthy_metric_is_excluded():
    """No HEALTHY checkpoint ever appeared -> not eligible (no silent fallback to
    the best-prediction metric, which may be a WARNING/COLLAPSED step)."""
    assert not phase_ok(_report(best_cos_err=0.05, best_healthy_cos_err=None))


def test_goal_score_contract():
    """Tiebreak contract only (criteria polarity is a DEFERRED audit item):
    prefer best-HEALTHY-step goal health; fall back to the prediction-step goal
    health for older reports; missing returns 0.0."""
    hi = _report(best_healthy_health={"goal": {"goal_token_pairwise_cosine_mean": 0.9}})
    lo = _report(best_healthy_health={"goal": {"goal_token_pairwise_cosine_mean": 0.1}})
    none = _report()
    fallback = _report(goal_token_health={"goal_token_pairwise_cosine_mean": 0.7})
    assert goal_score(hi) == -0.9
    assert goal_score(lo) == -0.1
    assert goal_score(none) == 0.0
    assert goal_score(fallback) == -0.7


def test_winner_requires_healthy_checkpoint_and_ignores_better_warning():
    """A phase with a great prediction-best but NO HEALTHY checkpoint must lose
    to a phase with an (objectively worse) HEALTHY checkpoint. And if NO phase
    has a HEALTHY checkpoint, select_winner returns None (no-clean-winner), NOT
    the best WARNING/COLLAPSED result."""
    no_healthy = _report(objective="jepa", phase=0, best_cos_err=0.05,
                         best_healthy_cos_err=None)
    healthy = _report(objective="lejepa", phase=1, best_cos_err=0.5,
                      best_healthy_cos_err=0.5,
                      best_healthy_checkpoint="phase_01_lejepa_best_healthy.pt")
    w = select_winner([no_healthy, healthy])
    assert w is not None
    assert w["objective"] == "lejepa" and w["checkpoint"] == "phase_01_lejepa_best_healthy.pt"
    assert w["best_cos_err"] == 0.5

    assert select_winner([no_healthy]) is None, (
        "no HEALTHY checkpoint anywhere -> explicit no-clean-winner, never the "
        "best WARNING/COLLAPSED result")


def test_winner_prefers_lower_best_healthy():
    a = _report(objective="jepa", phase=0, best_healthy_cos_err=0.4)
    b = _report(objective="lejepa", phase=1, best_healthy_cos_err=0.3,
                best_healthy_checkpoint="phase_01_lejepa_best_healthy.pt")
    w = select_winner([a, b])
    assert w["objective"] == "lejepa" and w["best_cos_err"] == 0.3


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