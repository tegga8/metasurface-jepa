#!/bin/bash
#SBATCH -p small -N 1 -n 38 -t 12:00:00
#SBATCH --output=/home/mm24r002/logs/imp_phys_%j.out
# Full-wave Meep on the imp1 / imp2 decodes of the novel geometries (task-1
# setting: full mask and 50% mask, per-set output dirs, shared dataset-mean
# empty-cell reference). Same instrument as every other task-1 number.
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MEEP_PY="$HOME/envs/meep/bin/python"
cd "$HOME/metasurface-jepa" || exit 1
REF="$HOME/runs/t1/ref.json"
BASE="$HOME/runs/imp_phys"
echo "=== $(date) on $(hostname) nproc=$(nproc) ==="
ls -l "$REF" || exit 1
run_set () {   # $1 = npz   $2 = tag
  local npz="$1" tag="$2"
  local out="$BASE/$tag"
  [ -f "$npz" ] || { echo "MISSING $npz - skipping"; return 1; }
  local n
  n=$("$MEEP_PY" -c "import numpy as np; print(len(np.load('$npz', allow_pickle=True)['patterns']))")
  rm -rf "$out"; mkdir -p "$out/logs"
  echo "=== SET $tag n=$n start $(date +%T) ==="
  srun --ntasks="$n" --ntasks-per-node="$n" --cpus-per-task=1 \
      "$HOME/worker.sh" "$npz" "$out" 0 "$out/logs" 2500 --ref-from "$REF"
  echo "  srun exit=$? end $(date +%T) files=$(ls "$out"/e_0_*.json 2>/dev/null | wc -l)/$n"
}
run_set "$BASE/dec_imp1/decoded_full.npz" imp1_full
run_set "$BASE/dec_imp1/decoded_half.npz" imp1_half
run_set "$BASE/dec_imp2/decoded_full.npz" imp2_full
run_set "$BASE/dec_imp2/decoded_half.npz" imp2_half

echo "=== summary (design files per set) ==="
for t in imp1_full imp1_half imp2_full imp2_half; do
  echo "  $t: $(ls "$BASE/$t"/e_0_*.json 2>/dev/null | wc -l) design files"
done
echo "IMP_PHYS DONE"
