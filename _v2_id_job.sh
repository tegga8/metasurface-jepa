#!/bin/bash
#SBATCH -p small -N 1 -n 16 -t 06:00:00
#SBATCH --output=/home/mm24r002/logs/v2id_%j.out
# Identify the v2 (sched_fix) checkpoint AND answer gate 8 for it.
#
# The ledger names v2 as the failed length-proportional-staging arm with a recorded
# signature (seed 0: MAE 0.0507, gates A/B/C 0.996/0.970/0.905) -- gate B and C are
# the discriminators, since base v1 sits at 0.977/0.930. So the same job both
# confirms WHICH checkpoint this is and produces the last missing gate-8 row, with
# the provenance stated rather than assumed from the filename.
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HOME/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
W="$HOME/imp2"
cd "$W" || exit 1
mkdir -p "$HOME/runs/gate8" "$HOME/runs/v2id"
CK="$HOME/scale_ckpts/v2_schedfix_seed0.pt"
ls -l "$CK" || exit 1
echo "=== $(date) on $(hostname) ==="

echo "=== identity: full-split scenario A (expect MAE ~0.0507, A/B/C ~0.996/0.970/0.905) ==="
python scripts/benchmark/benchmark_metadit.py --config configs/unified.yaml \
    --checkpoint "$CK" --split test --scenario A --device cpu \
    --samples 0 --batch-size 8 --out "$HOME/runs/v2id/scenarioA_full.json"
echo "  benchmark exit=$?"

echo "=== gate 8 for v2 ==="
python scripts/diagnostics/gate8_stratified.py configs/unified.yaml "$CK" \
    --n 512 --eval-seeds 3 4 5 --out "$HOME/runs/gate8/v2_schedfix.json"
echo "  gate8 exit=$? end $(date +%T)"
echo "V2ID DONE"
