"""Scenario-C crossed seed study — analysis (Task §2/§3/§4).

Reads the 9 crossed_<ckpt>_e<E>.json artifacts (3 training checkpoints x 3 eval seeds)
and reports: per-checkpoint gate (mean ± std over eval seeds); the training-seed result
(grand mean, sample std, 95 % Student-t CI, df=2, replication unit = checkpoint); the
A5 per-sample tail per checkpoint and pooled; and sample-aligned failure consistency
across checkpoints (the 512 val items are fixed, so align by index).

Post-processing only — no model, dataset, or compute.

Run:
    python scripts/diagnostics/scenario_c_crossed.py --glob "<dir>/crossed_*.json"
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def _stats(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def _ci95(values):
    v = np.asarray(values, dtype=float)
    n = v.size
    if n < 2:
        return {"mean": float(v.mean()) if n else None, "std": None,
                "ci_lower": None, "ci_upper": None, "n": int(n)}
    mean = float(v.mean())
    std = float(v.std(ddof=1))
    hw = T95.get(n - 1, 12.706) * std / np.sqrt(n)
    return {"mean": mean, "std": std, "ci_lower": mean - hw, "ci_upper": mean + hw,
            "n": int(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    args = ap.parse_args()

    cells = defaultdict(dict)          # ckpt -> eval_seed -> parsed
    per_ckpt_item_diff = defaultdict(list)   # ckpt -> list of per-item mean diff arrays
    for f in sorted(glob.glob(args.glob)):
        base = os.path.basename(f)
        if not base.startswith("crossed_"):
            continue
        ckpt = base.split("_")[1]
        eseed = base.split("_e")[1].split(".")[0]
        d = json.load(open(f))
        rns = d.get("scenario_C_rns", {})
        gap = rns.get("gap", {})
        r = np.asarray(gap.get("per_sample_real", []), dtype=float)
        s = np.asarray(gap.get("per_sample_shuffled", []), dtype=float)
        retro = d.get("scenario_C_retrofit", {})
        cells[ckpt][eseed] = {
            "gate": gap.get("real_beats_shuffled_fraction"),
            "real": r, "shuffled": s, "diff": s - r,
            "spectrum_error": retro.get("spectrum_error"),
            "masked_iou": (retro.get("masked_region") or {}).get("iou"),
            "occ_frac": retro.get("pred_occupancy_fraction"),
        }

    ckpts = sorted(cells)
    per_ckpt = {}
    for c in ckpts:
        gates = [cells[c][e]["gate"] for e in sorted(cells[c])]
        diffs = [cells[c][e]["diff"] for e in sorted(cells[c])]
        L = min(len(x) for x in diffs)
        per_ckpt_item_diff[c] = np.mean([x[:L] for x in diffs], axis=0)
        allr = np.concatenate([cells[c][e]["real"] for e in sorted(cells[c])])
        alls = np.concatenate([cells[c][e]["shuffled"] for e in sorted(cells[c])])
        alld = np.concatenate([cells[c][e]["diff"] for e in sorted(cells[c])])
        per_ckpt[c] = {
            "n_eval_seeds": len(gates),
            "gate_mean": float(np.mean(gates)), "gate_std": float(np.std(gates, ddof=1))
            if len(gates) > 1 else None, "gate_raw": gates,
            "real": _stats(allr), "shuffled": _stats(alls),
            "paired_diff": _stats(alld),
            "frac_real_lt_shuffled": float((alld > 0).mean()),
            "fail_diff_mean": float(alld[alld <= 0].mean()) if (alld <= 0).any() else 0.0,
            "win_diff_mean": float(alld[alld > 0].mean()) if (alld > 0).any() else 0.0,
            "masked_iou_mean": float(np.mean([cells[c][e]["masked_iou"]
                                              for e in cells[c]])),
            "occ_frac_mean": float(np.mean([cells[c][e]["occ_frac"] for e in cells[c]])),
        }

    ckpt_means = [per_ckpt[c]["gate_mean"] for c in ckpts]
    overall = _ci95(ckpt_means)

    # sample-aligned failure consistency (items fixed by validation index)
    L = min(len(v) for v in per_ckpt_item_diff.values())
    fails = np.stack([(per_ckpt_item_diff[c][:L] <= 0) for c in ckpts], axis=0)
    fail_freq = fails.sum(axis=0)          # per item, how many checkpoints fail it
    consistency = {
        "n_items": int(L),
        "items_failing_in_k_checkpoints": {
            str(k): int((fail_freq == k).sum()) for k in range(len(ckpts) + 1)},
        f"frac_failing_in_all_{len(ckpts)}": float((fail_freq == len(ckpts)).mean()),
        "frac_failing_in_ge2": float((fail_freq >= 2).mean()),
        "frac_failing_in_ge1": float((fail_freq >= 1).mean()),
    }

    report = {
        "checkpoints": ckpts,
        "per_checkpoint": per_ckpt,
        "overall_training_seed": overall,
        "all_nine_raw_gates": {f"{c}_e{e}": cells[c][e]["gate"]
                               for c in ckpts for e in sorted(cells[c])},
        "failure_consistency": consistency,
        "decision": ("above 0.75" if overall["ci_lower"] and overall["ci_lower"] > 0.75
                     else "below 0.75" if overall["ci_upper"] and overall["ci_upper"] < 0.75
                     else "UNRESOLVED (CI straddles 0.75)"),
    }
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
