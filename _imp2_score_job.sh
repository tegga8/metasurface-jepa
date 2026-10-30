#!/bin/bash
#SBATCH -p small -N 1 -n 4 -t 01:00:00
#SBATCH --output=/home/mm24r002/logs/imp2score_%j.out
# Score imp2's phase-2 and phase-3 full-wave results with the same comparators
# used for v1/v2/MetaDiT, on the same targets and the same shared reference.
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/pylibs${PYTHONPATH:+:$PYTHONPATH}"
cd "$HOME/imp2" || exit 1
W="$HOME/runs/p23imp2"
mkdir -p "$W/meep_p2" "$W/meep_p3"
for t in jepa_imp2_A jepa_imp2_B; do ln -sfn "$W/meep/$t" "$W/meep_p2/$t"; done
for t in jepa_imp2_B_true jepa_imp2_spec_m20 jepa_imp2_spec_p20; do
  ln -sfn "$W/meep/$t" "$W/meep_p3/$t"
done
ls -l "$W/meep_p2" "$W/meep_p3" | head -12
echo "=== PHASE 3 (scalar sensitivity) ==="
python _compare_phase3_physics.py "$W/meep_p3" \
    --cond-dir "$HOME/runs/phase3f/out" \
    --orig "$HOME/runs/phase3/b40_merged.json" --orig-rows B_true
echo "p3 exit=$?"
echo
echo "=== PHASE 2 (spectrum-first, corrected A'/B') ==="
python _compare_phase2.py "$W/meep_p2" \
    "$HOME/runs/phase3f/random_spectra.npz" "$HOME/imp2/_meep_convention.json"
echo "p2 exit=$?"
echo IMP2SCORE DONE
