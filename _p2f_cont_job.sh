#!/bin/bash
#SBATCH -p small -N 1 -n 24 -t 24:00:00
#SBATCH --output=/home/mm24r002/logs/p2fc_%j.out
# Continuation of 642242, which died at the MetaDiT step because _phase3_models
# expected b40_merged.json next to the output dir. The screen and both JEPA
# decodes already exist, so this job only needs: MetaDiT's task-2 decode
# (--task2-only) and then the four Meep sets.
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MEEP_PY="$HOME/envs/meep/bin/python"
cd "$HOME/metasurface-jepa" || exit 1
WORK="$HOME/runs/p2f"
echo "=== $(date) on $(hostname) nproc=$(nproc) ==="
ls -l "$WORK/random_spectra.npz" \
      "$WORK/dec_v1/decoded_A_interpolated.npz" \
      "$WORK/dec_v2/decoded_A_interpolated.npz" || exit 1

echo "--- MetaDiT-S task-2 decode (A' feasible / B' infeasible) ---"
python _phase3_models.py "$WORK/md_out" --n 24 --task2-only || exit 1
ls -l "$WORK/md_out"

echo "--- Meep on the decodes ---"
run_set () {   # $1 = design npz   $2 = set tag
  local npz="$1" tag="$2"
  local out="$WORK/meep/$tag"
  [ -f "$npz" ] || { echo "MISSING $npz - skipping"; return 1; }
  rm -rf "$out"; mkdir -p "$out/logs"
  local n
  n=$("$MEEP_PY" -c "import numpy as np; print(len(np.load('$npz', allow_pickle=True)['patterns']))")
  echo "=== SET $tag n=$n start $(date +%T) ==="
  srun --ntasks="$n" --ntasks-per-node="$n" --cpus-per-task=1 \
      "$HOME/worker.sh" "$npz" "$out" 0 "$out/logs" 2500 --ref-from "$HOME/runs/t1/ref.json"
  echo "  srun exit=$? end $(date +%T) files=$(ls "$out"/e_0_*.json 2>/dev/null | wc -l)/$n"
}
run_set "$WORK/dec_v1/decoded_A_interpolated.npz" jepa_v1_A
run_set "$WORK/dec_v1/decoded_B_synthetic.npz"    jepa_v1_B
run_set "$WORK/dec_v2/decoded_A_interpolated.npz" jepa_v2_A
run_set "$WORK/md_out/metadit_A_interpolated.npz" metadit_A

echo "=== summary ==="
for t in jepa_v1_A jepa_v1_B jepa_v2_A metadit_A; do
  echo "  $t: $(ls "$WORK/meep/$t"/e_0_*.json 2>/dev/null | wc -l) design files"
done
echo "P2FC DONE"
