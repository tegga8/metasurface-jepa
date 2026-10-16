#!/bin/bash
#SBATCH -p small -N 1 -n 16 -t 06:00:00
#SBATCH --output=/home/mm24r002/logs/fs_s0_%j.out
# imp2 seed 0's full-split benchmark. Its checkout is ~/imp2 (the seed jobs used
# imp2_s1/imp2_s2), so the generic loop skipped it; run it from there.
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
W="$HOME/imp2"
cd "$W" || exit 1
echo "=== $(date) on $(hostname) cwd=$(pwd) ==="
ls -l configs/unified_imp2.yaml runs_out/final.pt || exit 1
python scripts/benchmark/benchmark_metadit.py --config configs/unified_imp2.yaml \
    --checkpoint runs_out/final.pt --split test --scenario A --device cpu \
    --samples 0 --batch-size 8 --out runs_out/benchmark_full.json
echo "exit=$? end $(date +%T)"
echo "IMP2_S0_FULLSPLIT_DONE"
