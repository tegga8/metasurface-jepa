#!/bin/bash
#SBATCH -p small -N 1 -n 16 -t 06:00:00
#SBATCH --output=/home/mm24r002/logs/t3res64b_%j.out
# Second half of the res-64 point, chained to the first with afterany so it runs
# whether or not that job reached its 6 h wall.
#
# The first job computes the res-64 empty-cell reference (and starts a design run
# it will not finish: res-64 costs ~13x res-32, ~4 h per FDTD run). The harness
# writes the JSON as soon as the reference exists, so this job adopts it via
# --ref-from (the protocol guard accepts it: same loss tangent, same window) and
# only needs the design run. Falls back to computing its own reference if the
# first job died before writing one.
set -u
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MEEP_PY="$HOME/envs/meep/bin/python"
cd "$HOME/metasurface-jepa" || exit 1
NPZ="$HOME/runs/phase3/out/jepa_v1_spec_p20.npz"
[ -f "$NPZ" ] || { echo "MISSING $NPZ"; exit 1; }
W="$HOME/runs/t3res"
REF="$W/res64_k3.json"
if [ ! -f "$REF" ]; then
  echo "no res-64 reference from the first job; computing one here"
  REF=""
else
  echo "adopting res-64 reference from $REF"
fi

for K in 3 5; do
  OUT="$W/res64_k${K}.json"
  [ -f "$OUT" ] && continue
  EXTRA=""
  [ -n "$REF" ] && [ "$K" != "3" ] && EXTRA="--ref-from $REF"
  "$MEEP_PY" _meep_novel.py "$NPZ" "$OUT" --start "$K" --end $((K + 1)) \
      --max-ring 2500 --res 64 $EXTRA > "$W/logs/res64_k${K}.log" 2>&1
  echo "res64 idx$K exit=$?"
done

echo "=== res sweep (corrected flux statistic, floor 1e-2 and 5e-2) ==="
"$MEEP_PY" - <<'PY'
import glob, json, os
import numpy as np
W = os.path.expanduser("~/runs/t3res")
for f in sorted(glob.glob(os.path.join(W, "res*_k*.json"))):
    j = json.load(open(f))
    for e in j.get("designs", []):
        for ref in j.get("refs", {}).values():
            fm = np.asarray(e["flux_mag"], float)
            fe = np.abs(np.asarray(ref["flux"], float))
            out = []
            for floor in (1e-2, 5e-2):
                keep = fe >= floor * np.median(fe)
                sel = fm[keep]
                sel = sel[np.isfinite(sel)]
                out.append(f"floor{floor:g}={sel.max():.4f}")
            print(f"  {os.path.basename(f):14s} {e.get('name','?'):18s} "
                  + "  ".join(out))
PY
echo "T3RES64B DONE"
