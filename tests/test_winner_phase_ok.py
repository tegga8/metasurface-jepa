"""Unit tests for the adaptive-ladder winner eligibility (audit item 5).

Winner selection previously excluded only COLLAPSED phases — a WARNING (degraded
but not collapsed) phase could be crowned. Winner eligibility now requires the
representation status to be HEALTHY, with no unstable steps and a recorded best
metric.

Run:  python tests/test_winner_phase_ok.py        (also collectable by pytest)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "train"))

from train_milestone_b import _goal_score, _phase_ok  # noqa: E402


def _report(**over):
    base = {"objective": "jepa", "phase": 0, "representation_status": "HEALTHY",
            "unstable_steps": 0, "best_cos_err": 0.5}
    base.update(over)
    return base


def test_healthy_phase_is_eligible():
    assert _phase_ok(_report())


def test_warning_phase_is_excluded():
    """The bug: only COLLAPSED was excluded; WARNING phases passed the gate."""
    assert not _phase_ok(_report(representation_status="WARNING"))


def test_collapsed_phase_is_excluded():
    assert not _phase_ok(_report(representation_status="COLLAPSED"))


def test_unknown_or_missing_status_is_excluded():
    assert not _phase_ok(_report(representation_status=None))
    assert not _phase_ok(_report(representation_status="UNSTABLE"))


def test_unstable_phase_is_excluded():
    assert not _phase_ok(_report(unstable_steps=3))


def test_missing_best_metric_is_excluded():
    assert not _phase_ok(_report(best_cos_err=None))


def test_goal_score_contract():
    """Tiebreak contract only (criteria polarity is a DEFERRED audit item): present
    goal-token pairwise cosine returns its negated value, missing returns 0.0."""
    hi = _report(goal_token_health={"goal_token_pairwise_cosine_mean": 0.9})
    lo = _report(goal_token_health={"goal_token_pairwise_cosine_mean": 0.1})
    none = _report()
    assert _goal_score(hi) == -0.9
    assert _goal_score(lo) == -0.1
    assert _goal_score(none) == 0.0


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