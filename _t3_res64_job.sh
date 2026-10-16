#!/bin/bash
#SBATCH -p small -N 1 -n 8 -t 06:00:00
#SBATCH --output=/home/mm24r002/logs/t3res64_%j.out
# Third resolution point for the surviving v1_spec_p20 excess.
#
# res 32 -> 48 moved the corrected statistic down on both designs tested
# (1.1063 -> 1.0835 and 1.1639 -> 1.0701) but not below the 1.02 tolerance, so
# the trend is not yet decisive: converging toward 1 (a discretisation artefact,
# row unquotable) and plateauing at ~1.07 (a real residual) look the same from
# two points. One design (sym03, the worst at res 32) at res 64 decides it.
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
OUT="$W/res64_k3.json"
if [ ! -f "$OUT" ]; then
  "$MEEP_PY" _meep_novel.py "$NPZ" "$OUT" --start 3 --end 4 \
      --max-ring 2500 --res 64 > "$W/logs/res64_k3.log" 2>&1
  echo "res64 idx3 exit=$?"
fi
echo "=== res sweep, sym03 (corrected flux statistic) ==="
"$MEEP_PY" - <<'PY'
import glob, json, os
import numpy as np
W = os.path.expanduser("~/runs/t3res")
for f in sorted(glob.glob(os.path.join(W, "res*_k3.json"))):
    j = json.load(open(f))
    for e in j.get("designs", []):
        for ref in j.get("refs", {}).values():
            fm = np.asarray(e["flux_mag"], float)
            fe = np.abs(np.asarray(ref["flux"], float))
            for floor in (1e-2, 5e-2):
                keep = fe >= floor * np.median(fe)
                sel = fm[keep]
                sel = sel[np.isfinite(sel)]
                print(f"  {os.path.basename(f):14s} floor={floor:g} "
                      f"corrected={sel.max():.4f} "
                      f"excluded={100 * (1 - keep.mean()):.1f}%")
PY
echo "T3RES64 DONE"
