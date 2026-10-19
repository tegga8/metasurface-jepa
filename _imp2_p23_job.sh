#!/bin/bash
#SBATCH -p small -N 1 -n 24 -t 12:00:00
#SBATCH --output=/home/mm24r002/logs/imp2p23_%j.out
# imp2 through phases 2 and 3, end to end, mirroring the job that produced the
# v1/v2/MetaDiT numbers so the comparison is like-for-like:
#   - same targets (the realizability-screened A'/B', regenerated locally and
#     verified against the ledger's screen: A' l1n 0.3480 / corr 0.9592, B' 4.4850)
#   - same decoder harness (_decode_random_spectra.py) and same Meep worker
#   - same shared empty-cell reference (runs/t1/ref.json), same --max-ring 2500
#
# Phase 3's decode sets already exist from the driver run (jepa_imp2_B_true /
# spec_m20 / spec_p20, 10 novel geometries each); phase 2 is decoded here.
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MEEP_PY="$HOME/envs/meep/bin/python"
cd "$HOME/imp2" || exit 1
WORK="$HOME/runs/p23imp2"
mkdir -p "$WORK/meep"
echo "=== $(date) on $(hostname) nproc=$(nproc) ==="
ls -l "$HOME/worker.sh" "$HOME/runs/t1/ref.json" \
      "$HOME/runs/phase3f/out/jepa_imp2_B_true.npz" runs_out/final.pt || exit 1

echo "--- phase 2 decode (imp2) ---"
python _decode_random_spectra.py "$HOME/runs/phase3f/random_spectra.npz" \
    "$HOME/imp2/runs_out/final.pt" "$WORK/dec_imp2" || exit 1
ls -l "$WORK/dec_imp2"/*.npz || exit 1

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

# phase 2
run_set "$WORK/dec_imp2/decoded_A_interpolated.npz" jepa_imp2_A
run_set "$WORK/dec_imp2/decoded_B_synthetic.npz"    jepa_imp2_B
# phase 3
run_set "$HOME/runs/phase3f/out/jepa_imp2_B_true.npz"   jepa_imp2_B_true
run_set "$HOME/runs/phase3f/out/jepa_imp2_spec_m20.npz" jepa_imp2_spec_m20
run_set "$HOME/runs/phase3f/out/jepa_imp2_spec_p20.npz" jepa_imp2_spec_p20

echo "=== summary ==="
for t in jepa_imp2_A jepa_imp2_B jepa_imp2_B_true jepa_imp2_spec_m20 jepa_imp2_spec_p20; do
  echo "  $t: $(ls "$WORK/meep/$t"/e_0_*.json 2>/dev/null | wc -l) design files"
done
echo "IMP2P23 DONE"
