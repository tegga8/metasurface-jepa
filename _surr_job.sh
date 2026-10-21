#!/bin/bash
#SBATCH -p small -N 1 -n 8 -t 01:00:00
#SBATCH --output=/home/mm24r002/logs/surrtransfer_%j.out
# Surrogate->Meep transfer check on imp2's existing phase-2 decodes. No Meep runs:
# it forward-passes the already-decoded geometries through the frozen surrogate and
# compares that prediction against the Meep measurement of the SAME geometry.
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
cd "$HOME/imp2" || exit 1
echo "=== $(date) on $(hostname) ==="
ls -l "$HOME/runs/p23imp2/dec_imp2/"*.npz | head -3
python _surrogate_transfer.py
echo "exit=$?"
echo SURRT DONE
