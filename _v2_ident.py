"""Compare the candidate checkpoint's full-split signature with v2's recorded one.

Ledger (Phase 6b, length-proportional staging arm = v2 = sched_fix), seed 0:
    MAE 0.0507,  gates A/B/C = 0.996 / 0.970 / 0.905
base v1 for contrast (3 seeds): 0.0500 / 0.0494 / 0.0476,  0.997 / 0.977 / 0.930
Gates B and C are the discriminators: v2's 0.970/0.905 sit clearly below v1's
0.977/0.930, which a same-config checkpoint cannot reach by seed noise alone.
"""

import json
import os
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "scenarioA_full.json"
j = json.load(open(p, encoding="utf-8"))
prim = j.get("primary", {})
sec = j.get("secondary", j.get("gates", {}))
print(f"file: {os.path.basename(p)}")
print(f"  primary : {({k: round(v, 4) if isinstance(v, float) else v for k, v in prim.items()})}")
print(f"  secondary: {({k: round(v, 4) if isinstance(v, float) else v for k, v in sec.items()})}")
print("\nrecorded signatures:")
print("  v2 (sched_fix) seed0 : MAE 0.0507, A/B/C 0.996 / 0.970 / 0.905")
print("  base v1 (3 seeds)    : 0.0500 / 0.0494 / 0.0476, 0.997 / 0.977 / 0.930")
