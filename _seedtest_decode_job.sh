#!/bin/bash
#SBATCH -p small -N 1 -n 8 -t 01:00:00
#SBATCH --output=/home/mm24r002/logs/seeddec_%j.out
# Seed test, step 1: decode 180 candidates (2 arms x 5 guidance strengths x 18
# targets) with frozen base v1. Inference only.
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
cd "$HOME/imp2" || exit 1
echo "=== $(date) on $(hostname) ==="
ls -l batch40_designs.npz "$HOME/../mm24r002/runs/phase3f/random_spectra.npz" 2>/dev/null
ls -l "$HOME/metasurface-jepa/checkpoints/unified/final.pt" || exit 1
python _seedtest_decode.py
echo "exit=$?"
ls -l "$HOME/runs/seedtest/dec" | head -13
echo SEEDDEC DONE
