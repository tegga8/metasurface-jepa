"""Regression tests for the ladder summary writer (Batch 4 fix).

A phase that ends before its first validation has best_cos_err=None; the
summary table must format it as "n/a" instead of crashing (observed in the
six-objective smoke: jepa_var ended at the global budget with 0 validations).
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from train.engine import write_ladder_summary


def _phase_report(objective, start, end, best_cos_err, status="HEALTHY"):
    return {
        "objective": objective,
        "start_global_step": start,
        "end_global_step": end,
        "best_cos_err": best_cos_err,
        "representation_status": status,
        "unstable_steps": 0,
    }


def test_summary_with_none_best_cos_err(tmp_path):
    reports = [
        _phase_report("jepa", 0, 3, 0.123456),
        _phase_report("jepa_var", 3, 4, None),   # ended before first val
    ]
    json_path, md_path = write_ladder_summary(
        tmp_path, reports,
        {"max_total_steps": 28, "objectives": ["jepa", "jepa_var"]},
        {"n_samples": 4}, {"no_clean_winner": True})
    md = open(md_path, encoding="utf-8").read()
    assert "n/a" in md, "None best_cos_err must be formatted as n/a"
    assert "0.123456" in md
    data = json.load(open(json_path, encoding="utf-8"))
    assert data["reports"][1]["best_cos_err"] is None


def test_summary_all_none_best(tmp_path):
    # A fully-empty ladder (no phase ever validated) must still produce a summary.
    reports = [_phase_report("lejepa", 0, 1, None, status="WARNING")]
    json_path, md_path = write_ladder_summary(
        tmp_path, reports, {"max_total_steps": 28,
                            "objectives": ["lejepa"]},
        {"n_samples": 4}, {"no_clean_winner": True})
    assert os.path.exists(json_path) and os.path.exists(md_path)
    assert "n/a" in open(md_path, encoding="utf-8").read()
