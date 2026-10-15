#!/bin/bash
#SBATCH -p small -N 1 -n 24 -t 24:00:00
#SBATCH --output=/home/mm24r002/logs/p2f_%j.out
# Phase-2 RE-SPECIFIED, one job end to end:
#   1. generate realizability-screened targets (A' feasible, B' infeasible control)
#   2. decode with JEPA v1, JEPA v2 and MetaDiT-S   (CPU)
#   3. full-wave Meep on the decodes, per-set output dirs
# The screen is what the original phase-2 lacked: set A sat 1.87 from the nearest
# real spectrum (corpus density ~0.33 in the same units), i.e. it was scored against
# a target no corpus geometry can realize.
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MEEP_PY="$HOME/envs/meep/bin/python"
cd "$HOME/metasurface-jepa" || exit 1
WORK="$HOME/runs/p2f"
mkdir -p "$WORK"
echo "=== $(date) on $(hostname) nproc=$(nproc) ==="
ls -l checkpoints/unified/final.pt ckpt_v2_seed0.pt || exit 1
ls -l "$HOME/runs/t1/ref.json" || exit 1

echo "--- 1) screened targets ---"
python _make_feasible_spectra.py "$WORK/random_spectra.npz" 24 || exit 1

echo "--- 2a) JEPA v1 decode ---"
python _decode_random_spectra.py "$WORK/random_spectra.npz" \
    "$HOME/metasurface-jepa/checkpoints/unified/final.pt" "$WORK/dec_v1" || exit 1

echo "--- 2b) JEPA v2 decode ---"
python _decode_random_spectra.py "$WORK/random_spectra.npz" \
    "$HOME/metasurface-jepa/ckpt_v2_seed0.pt" "$WORK/dec_v2" || exit 1

echo "--- 2c) MetaDiT-S decode (CPU, ~150 s/item) ---"
python _phase3_models.py "$WORK/md_out" --n 24 || exit 1

echo "--- 3) Meep on the decodes ---"
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
echo "P2F DONE"
