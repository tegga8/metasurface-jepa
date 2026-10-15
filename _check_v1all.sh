#!/bin/bash
# Precondition checks on the 638208 output (per VALIDATION_PLAN.md section 3).
# Read-only: lists, parses JSONs, counts flags. No simulation.
set -u
cd "$HOME/runs/v1all" || { echo "NO v1all DIR"; exit 1; }

echo "==== v1all tree ===="
ls -la
echo
for d in */ ; do
  n=$(ls "$d" | wc -l)
  echo "-- $d : $n entries"
  ls "$d" | head -3
done
echo
echo "==== per-design checks ===="
~/envs/meep/bin/python - <<'PYEOF'
import json, glob, os
base = os.path.expanduser("~/runs/v1all")
for d in sorted(os.listdir(base)):
    p = os.path.join(base, d)
    if not os.path.isdir(p):
        continue
    files = sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
    print("### set:", d, "| json files:", len(files))
    if not files:
        continue
    print("    samples:", [os.path.relpath(f, base) for f in files[:2]])
    names, fails, capped, capped_ring = [], [], [], []
    for f in files:
        try:
            j = json.load(open(f))
        except Exception as e:
            print("    PARSE FAIL", f, e)
            continue
        for e in (j.get("designs") or []):
            nm = e.get("name")
            names.append(nm)
            if not e.get("passivity_ok"):
                fails.append((nm, round(float(e.get("peak_ratio") or -1), 4)))
            if e.get("hit_cap"):
                capped.append(nm)
                capped_ring.append(e.get("ringdown"))
    prefixes = sorted({(nm or "").split("/")[0] for nm in names})
    dupes = len(names) - len(set(names))
    print("    designs:", len(names), "| unique:", len(set(names)), "| dupes:", dupes)
    print("    name prefixes:", prefixes)
    print("    passivity failures:", len(fails), fails[:10])
    print("    ringdown-capped:", len(capped), capped[:10], capped_ring[:10])
PYEOF
