"""A5 — Scenario C per-sample tail analysis (reads already-produced eval JSONs).

Post-processing only: no model, no dataset, no compute. Describes where the C
paired gate succeeds/fails across evaluation seeds — is the shortfall broad or a
narrow hard tail?

Run:
    python scripts/diagnostics/scenario_c_tail.py --glob "<dir>/ev/*.json"
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def _stats(x):
    x = np.asarray(x, dtype=float)
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p75": float(np.percentile(x, 75)), "p90": float(np.percentile(x, 90)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
            "max": float(x.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    rows, pr, ps = [], [], []
    for f in files:
        d = json.load(open(f))
        rns = d.get("scenario_C_rns", {})
        gap = rns.get("gap", {})
        r = np.asarray(gap.get("per_sample_real", []), dtype=float)
        s = np.asarray(gap.get("per_sample_shuffled", []), dtype=float)
        if r.size == 0 or s.size != r.size:
            continue
        diff = s - r                       # > 0 => real closer than shuffled
        retro = d.get("scenario_C_retrofit", {})
        rows.append({
            "file": os.path.basename(f),
            "gate": gap.get("real_beats_shuffled_fraction"),
            "spectrum_error": retro.get("spectrum_error"),
            "masked_iou": (retro.get("masked_region") or {}).get("iou"),
            "real_mean": float(r.mean()), "real_median": float(np.median(r)),
            "shuffled_mean": float(s.mean()),
            "diff_mean": float(diff.mean()), "diff_median": float(np.median(diff)),
            "frac_real_lt_shuffled": float((diff > 0).mean()),
            "frac_ties": float((diff == 0).mean()),
            "worst_real": float(r.max()),
        })
        pr.append(r); ps.append(s)

    pr = np.concatenate(pr) if pr else np.array([])
    ps = np.concatenate(ps) if ps else np.array([])
    diff = ps - pr
    pooled = {
        "real": _stats(pr),
        "shuffled": _stats(ps),
        "paired_diff": _stats(diff),
        "frac_real_lt_shuffled": float((diff > 0).mean()),
        "frac_fail_or_tie": float((diff <= 0).mean()),
        # failure magnitude: how big are the failing differences?
        "fail_diff_mean": float(diff[diff <= 0].mean()) if (diff <= 0).any() else 0.0,
        "win_diff_mean": float(diff[diff > 0].mean()) if (diff > 0).any() else 0.0,
    }
    report = {"n_files": len(rows), "per_file": rows, "pooled_per_sample": pooled}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
