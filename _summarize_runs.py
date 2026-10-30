"""Compact metric table from run artifacts (scenarios.json + benchmark_A.json)."""

import json
import os
import sys


def g(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def metrics(sc_path, bm_path=None):
    sc = json.load(open(sc_path, encoding="utf-8"))
    m = {
        "gateA": g(sc, "scenario_A_rns", "gap", "real_beats_shuffled_fraction"),
        "gateB": g(sc, "scenario_B_rns", "gap", "real_beats_shuffled_fraction"),
        "gateC": g(sc, "scenario_C_rns", "gap", "real_beats_shuffled_fraction"),
        "scal1": g(sc, "scalar_dependence_one_known",
                   "real_beats_shuffled_fraction"),
        "scal2": g(sc, "scalar_dependence_two_known",
                   "real_beats_shuffled_fraction"),
        "specA": g(sc, "scenario_A_pure_inverse", "spectrum_error"),
        "specB": g(sc, "scenario_B_partial", "spectrum_error"),
        "specC": g(sc, "scenario_C_retrofit", "spectrum_error"),
        "occA": g(sc, "scenario_A_pure_inverse", "pred_occupancy_fraction"),
        "trueA": g(sc, "scenario_A_pure_inverse", "true_occupancy_fraction"),
        "iouA": g(sc, "scenario_A_pure_inverse", "masked_region", "iou"),
    }
    if bm_path and os.path.exists(bm_path):
        bm = json.load(open(bm_path, encoding="utf-8"))
        m["MAE"] = g(bm, "primary", "MAE")
        m["AAE"] = g(bm, "primary", "AAE")
        m["occIou"] = g(bm, "secondary", "occupancy_iou_masked")
    return m


runs = []
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    sc = os.path.join(path, "scenarios.json")
    bm = os.path.join(path, "benchmark_A.json")
    if os.path.exists(sc):
        runs.append((name, metrics(sc, bm)))
    else:
        print(f"# {name}: no scenarios.json at {path}")

keys = ["MAE", "AAE", "gateA", "gateB", "gateC", "scal1", "scal2",
        "specA", "specB", "specC", "occA", "trueA", "iouA", "occIou"]
print(f"{'metric':8s}" + "".join(f"{n:>12s}" for n, _ in runs))
for k in keys:
    row = f"{k:8s}"
    for _, m in runs:
        v = m.get(k)
        row += f"{v:>12.4f}" if isinstance(v, float) else f"{'-':>12s}"
    print(row)
