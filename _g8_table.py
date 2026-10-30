"""Assemble the gate-8 table across every arm measured so far."""

import glob
import json
import os

BASE = r"D:\metamaterials_scratch\fs"
ORDER = ["base_v1", "imp1_s0", "imp2_s0",
         "scale_s1", "scale_s2", "scale_l1"]
LABEL = {"base_v1": "base v1 (11.4 M)", "imp1_s0": "imp1 s0 (mirror decoder)",
         "imp2_s0": "imp2 s0 (mirror-pair masks)", "scale_s1": "S1 small",
         "scale_s2": "S2 slim", "scale_l1": "L1 wide"}

rows = {}
for f in glob.glob(os.path.join(BASE, "*.json")):
    tag = os.path.basename(f).replace(".json", "")
    if tag not in ORDER:
        continue
    j = json.load(open(f, encoding="utf-8"))
    p = j.get("pooled", {})
    easy = p.get("easy (ratio 0.25, scalars known)", {})
    hard = p.get("hard (ratio 1.00, scalars unknown)", {})
    rows[tag] = (easy, hard)

print(f"{'arm':30s} {'easy win':>9s} {'hard win':>9s} {'hard margin':>12s} "
      f"{'CI':>20s} {'gate':>5s}")
for tag in ORDER:
    if tag not in rows:
        continue
    e, h = rows[tag]
    ci = h.get("ci95", [float("nan")] * 2)
    print(f"{LABEL.get(tag, tag):30s} {e.get('pooled_win_rate', float('nan')):9.4f} "
          f"{h.get('pooled_win_rate', float('nan')):9.4f} "
          f"{h.get('paired_diff_mean', float('nan')):+12.4f} "
          f"[{ci[0]:+.4f},{ci[1]:+.4f}] "
          f"{'PASS' if h.get('passes') else 'FAIL':>5s}")
print(f"\nn pooled per stratum per arm: "
      f"{rows.get('base_v1', ({}, {}))[1].get('n_pooled')}")
