#!/bin/bash
#SBATCH -p small -N 1 -n 8 -t 02:00:00
#SBATCH --output=/home/mm24r002/logs/p23decode_%j.out
# imp2 decode for phases 2 and 3 (inference only; the Meep passes are a separate job).
#
# The driver reads b40_merged.json (phase-3 scalar rows) and random_spectra.npz
# (phase-2 targets) from the PARENT of its out_dir, so both are staged in
# ~/runs/phase3f. Those targets are the CORRECTED feasible A'/B' set, regenerated
# locally and verified against the ledger's recorded screen (A' l1n 0.3480 /
# corr 0.9592, B' 4.4850 -- identical to the values v1/v2/MetaDiT were scored on),
# so imp2's phase-2 numbers are directly comparable to theirs.
#
# NOTE: the checkout's driver is the same version that produced the jepa_v1/v2 sets
# (it predates --task2-only, which is not needed here -- we want both phases).
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
W="$HOME/imp2"
cd "$W" || exit 1
B="$HOME/runs/phase3f"
mkdir -p "$B/out"
cp -f "$HOME/runs/phase3/b40_merged.json" "$B/b40_merged.json" || exit 1
ls -l "$B/b40_merged.json" "$B/random_spectra.npz" "$W/runs_out/final.pt" || exit 1
echo "=== $(date) on $(hostname) ==="

python _phase3_models.py "$B/out" \
    --jepa "imp2:$W/runs_out/final.pt" --skip-metadit
echo "driver exit=$?"

echo "=== produced sets ==="
python - <<'PY'
import glob, os
import numpy as np
for f in sorted(glob.glob(os.path.expanduser("~/runs/phase3f/out/*.npz"))):
    z = np.load(f, allow_pickle=True)
    names = z["names"].tolist() if "names" in z else []
    print(f"  {os.path.basename(f):34s} n={len(z['patterns']):3d} "
          f"scalars={np.asarray(z['scalars']).shape} e.g. {names[:2]}")
PY
echo "P23DECODE DONE"
