#!/bin/bash
#SBATCH -p small -N 1 -n 8 -t 02:00:00
#SBATCH --output=/home/mm24r002/logs/gate8s0_%j.out
# Gate 8 for S0 tiny -- the last rung of the size ladder.
#
# S0's checkpoint was not staged locally; it was downloaded from the Kaggle kernel
# output (akashkesav/metasurface-jepa-scale-s0-tiny) and key-diffed against
# configs/scaling/unified_s0_tiny.yaml first: 0 missing trainable keys, 0 unexpected
# keys, 0 shape mismatches (only the 106 frozen released-encoder tensors absent, as
# by design). CPU variant because a Kaggle-trained checkpoint's CUDA RNG state can
# fail to restore here (see the GPU attempts of 2026-10-16).
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
W="$HOME/imp2"
cd "$W" || exit 1
mkdir -p "$HOME/runs/gate8"
echo "=== $(date) on $(hostname) ==="
ls -l "$HOME/scale_ckpts/s0.pt" || exit 1
python scripts/diagnostics/gate8_stratified.py \
    configs/scaling/unified_s0_tiny.yaml "$HOME/scale_ckpts/s0.pt" \
    --n 512 --eval-seeds 3 4 5 --out "$HOME/runs/gate8/scale_s0.json"
echo "exit=$? end $(date +%T)"
echo "GATE8_S0 DONE"
