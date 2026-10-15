#!/bin/bash
#SBATCH -p sgpu -N 1 --gres=gpu:1 -n 8 -t 04:00:00
#SBATCH --output=/home/mm24r002/logs/phase3_%j.out
# Task 3 (scalar sensitivity) geometry rows for JEPA v1 + v2, and MetaDiT-S on
# identical conditions; plus MetaDiT-S on the task-2 random-spectrum sets.
# Writes design npz files that worker.sh / _meep_novel.py can run.
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2
cd "$HOME/metasurface-jepa" || exit 1
mkdir -p "$HOME/runs/phase3/out"
echo "=== $(date) on $(hostname) ==="
ls -l "$HOME/runs/phase3/b40_merged.json" "$HOME/runs/phase3/random_spectra.npz" \
      "$HOME/metasurface-jepa/ckpt_v2_seed0.pt" || exit 1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1
python _phase3_models.py "$HOME/runs/phase3/out" --n 10 \
    --jepa "v1:$HOME/metasurface-jepa/checkpoints/unified/final.pt" \
           "v2:$HOME/metasurface-jepa/ckpt_v2_seed0.pt"
echo "exit=$?"
ls -l "$HOME/runs/phase3/out" | tail -20
echo "PHASE3 DONE"
