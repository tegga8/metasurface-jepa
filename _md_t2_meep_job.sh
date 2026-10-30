#!/bin/bash
#SBATCH -p small -N 1 -n 24 -t 12:00:00
#SBATCH --output=/home/mm24r002/logs/md_t2_meep_%j.out
# MetaDiT-S task-2 physics: full-wave Meep on its decodes of the two random
# spectrum sets (A plausible / B implausible control), conditioned on the same
# targets our JEPA received. Separate per-set output dirs.
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MEEP_PY="$HOME/envs/meep/bin/python"
cd "$HOME/metasurface-jepa" || exit 1
REF="$HOME/runs/t1/ref.json"
OUTBASE="$HOME/runs/md_t2"
echo "=== $(date) on $(hostname) nproc=$(nproc) ==="
ls -l "$REF" || exit 1
for tag in metadit_A_interpolated metadit_B_synthetic; do
  NPZ="$HOME/runs/phase3/out_md/$tag.npz"
  OUT="$OUTBASE/$tag"
  if [ ! -f "$NPZ" ]; then echo "MISSING $NPZ - skipping"; continue; fi
  N=$("$MEEP_PY" -c "import numpy as np; print(len(np.load('$NPZ', allow_pickle=True)['patterns']))")
  rm -rf "$OUT"; mkdir -p "$OUT/logs"
  echo "=== SET $tag n=$N start $(date +%T) ==="
  srun --ntasks="$N" --ntasks-per-node="$N" --cpus-per-task=1 \
      "$HOME/worker.sh" "$NPZ" "$OUT" 0 "$OUT/logs" 2500 --ref-from "$REF"
  echo "  srun exit=$? end $(date +%T) files=$(ls "$OUT"/e_0_*.json 2>/dev/null | wc -l)/$N"
done
echo "MD_T2_MEEP DONE"
