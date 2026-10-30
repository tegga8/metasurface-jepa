#!/bin/bash
#SBATCH -p small -N 1 -n 4 -t 03:00:00
#SBATCH --output=/home/mm24r002/logs/t3ctrl_%j.out
# Instrument control for the task-3 passivity gate: apply it to an EMPTY cell.
#
# Sample and reference are then physically identical, so whatever ratio comes out
# is the statistic's own error -- noise floor plus systematic offset. The gate
# calls anything above 1.02 a solver failure, so this measures whether the
# tolerance is even resolvable by the instrument. Run twice per scalar combo, so
# run-to-run variation is visible as well as the single-run ratio.
set -u
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MEEP_PY="$HOME/envs/meep/bin/python"
cd "$HOME/metasurface-jepa" || exit 1
W="$HOME/runs/t3ctrl"
mkdir -p "$W/logs"
for TAG in m20 p20; do
  NPZ="$HOME/_empty_${TAG}.npz"
  [ -f "$NPZ" ] || { echo "MISSING $NPZ"; continue; }
  for RUN in a b; do
    "$MEEP_PY" _meep_novel.py "$NPZ" "$W/${TAG}_${RUN}.json" \
        --start 0 --end 1 --max-ring 2500 > "$W/logs/${TAG}_${RUN}.log" 2>&1
    echo "$TAG/$RUN exit=$?"
  done
done
echo "=== empty-cell passivity (should be 1.000) ==="
"$MEEP_PY" - <<'PY'
import glob, json, os
import numpy as np
W = os.path.expanduser("~/runs/t3ctrl")
for f in sorted(glob.glob(os.path.join(W, "*.json"))):
    j = json.load(open(f))
    for e in j.get("designs", []):
        fm = np.asarray(e["flux_mag"], float)
        mm = np.asarray(e["modes"]["def"]["mag"], float)
        fe = np.abs(np.asarray(list(j["refs"].values())[0]["flux"], float))
        med = np.median(fe)
        band = fe >= 1e-2 * med
        print(f"{os.path.basename(f):10s} peak_raw={e['peak_ratio']:.4f} "
              f"flux_all={np.nanmax(fm):.4f} flux_cond={np.nanmax(fm[band]):.4f} "
              f"mode_all={np.nanmax(mm):.4f} edge_max={np.nanmax(fm[~band]):.4f}")
PY
echo "T3CTRL DONE"
