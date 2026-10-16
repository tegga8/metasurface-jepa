#!/bin/bash
#SBATCH -p small -N 1 -n 4 -t 04:00:00
#SBATCH --output=/home/mm24r002/logs/t3res_%j.out
# Settle v1_spec_p20: is its surviving flux excess (>1.02 after every correction)
# a property of those decodes, or a solver-setup systematic?
#
# The empty-cell control showed the measurement itself is exact (identical system
# -> exactly 1.0000, zero spread), so the excess is a real difference between the
# sample and the air reference. The remaining question is whether that difference
# is physical or an artefact of the discretisation/PML. One variable: resolution.
# If the excess shrinks toward 1 as res rises it is a solver artefact and the row
# stays unquotable; if it holds, it is a real finding about v1's +20 % row.
#
# Three designs spanning the observed range (worst, middle, best) at res 32 (the
# shipped setting, for a same-code comparison) and res 48.
set -u
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MEEP_PY="$HOME/envs/meep/bin/python"
cd "$HOME/metasurface-jepa" || exit 1
NPZ="$HOME/runs/phase3/out/jepa_v1_spec_p20.npz"
[ -f "$NPZ" ] || { echo "MISSING $NPZ"; exit 1; }
W="$HOME/runs/t3res"
mkdir -p "$W/logs"

for RES in 32 48; do
  for K in 3 5 0; do      # sym03 worst (2.11), sym05 mildest (1.16), sym00 mid
    OUT="$W/res${RES}_k${K}.json"
    [ -f "$OUT" ] && continue
    "$MEEP_PY" _meep_novel.py "$NPZ" "$OUT" --start "$K" --end $((K + 1)) \
        --max-ring 2500 --res "$RES" > "$W/logs/res${RES}_k${K}.log" 2>&1
    echo "res$RES idx$K exit=$?"
  done
done

echo "=== v1_spec_p20 vs resolution (corrected flux statistic) ==="
"$MEEP_PY" - <<'PY'
import glob, json, os
import numpy as np
W = os.path.expanduser("~/runs/t3res")
rows = {}
for f in sorted(glob.glob(os.path.join(W, "res*_k*.json"))):
    j = json.load(open(f))
    tag = os.path.basename(f).replace(".json", "")
    for e in j.get("designs", []):
        for ref in j.get("refs", {}).values():
            fm = np.asarray(e["flux_mag"], float)
            fe = np.abs(np.asarray(ref["flux"], float))
            keep = fe >= 1e-2 * np.median(fe)
            sel = fm[keep]
            sel = sel[np.isfinite(sel)]
            rows.setdefault(e.get("name"), {})[tag] = float(sel.max())
for name, d in rows.items():
    parts = "  ".join(f"{k}={v:.4f}" for k, v in sorted(d.items()))
    print(f"  {name:20s} {parts}")
PY
echo "T3RES DONE"
