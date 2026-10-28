#!/bin/bash
#SBATCH -p small -N 1 -n 18 -t 24:00:00
#SBATCH --output=/home/mm24r002/logs/seedtest_%j.out
# Seed test, step 2: full-wave Meep on the 90 candidate geometries
# (18 targets x 5 guidance strengths), decoded from frozen base v1.
#
# The shuffled control needs NO extra solves: it re-scores the same measurements
# against a deranged target. That is the non-degenerate form of the control (see
# _seedtest_score.py) and it halves the cost.
#
# Same harness, shared empty-cell reference and ringdown cap as every earlier
# phase-2 run, so the errors are comparable. Resumable: existing designs are skipped.
set -u
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MEEP_PY="$HOME/envs/meep/bin/python"
cd "$HOME/imp2" || exit 1
DEC="$HOME/runs/seedtest/dec"
OUT="$HOME/runs/seedtest/meep"
mkdir -p "$OUT"
echo "=== $(date) on $(hostname) nproc=$(nproc) ==="
ls -l "$HOME/worker.sh" "$HOME/runs/t1/ref.json" || exit 1
ls "$DEC"/decoded_real_*.npz || exit 1

run_set () {   # $1 = design npz   $2 = set tag
  local npz="$1" tag="$2"
  local out="$OUT/$tag"
  mkdir -p "$out/logs"
  local n have
  n=$("$MEEP_PY" -c "import numpy as np; print(len(np.load('$npz', allow_pickle=True)['patterns']))")
  have=$(ls "$out"/e_0_*.json 2>/dev/null | wc -l)
  if [ "$have" -ge "$n" ]; then echo "=== SET $tag: already complete ($have/$n)"; return 0; fi
  echo "=== SET $tag n=$n have=$have start $(date +%T) ==="
  srun --ntasks="$n" --ntasks-per-node="$n" --cpus-per-task=1 \
      "$HOME/worker.sh" "$npz" "$out" 0 "$out/logs" 2500 --ref-from "$HOME/runs/t1/ref.json"
  echo "  srun exit=$? end $(date +%T) files=$(ls "$out"/e_0_*.json 2>/dev/null | wc -l)/$n"
}

for W in w050 w075 w100 w125 w150; do
  run_set "$DEC/decoded_real_${W}.npz" "real_${W}"
done

echo "=== summary ==="
for d in "$OUT"/*/; do
  echo "  $(basename "$d"): $(ls "$d"/e_0_*.json 2>/dev/null | wc -l) design files"
done
echo "SEEDTEST DONE"
