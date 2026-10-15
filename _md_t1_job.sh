#!/bin/bash
#SBATCH -p small -N 1 -n 38 -t 04:00:00
#SBATCH --output=/home/mm24r002/logs/md_t1_%j.out
# MetaDiT-S task-1 physics: full-wave Meep on MetaDiT's 38 decodes of the novel
# geometries, conditioned on the identical [2,301] targets our JEPA received.
# Same harness, same empty-cell reference (dataset-mean scalars) as every other
# result: Meep-vs-Meep, so the fitted convention cancels on both sides.
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MEEP_PY="$HOME/envs/meep/bin/python"
cd "$HOME/metasurface-jepa" || exit 1
REF="$HOME/runs/t1/ref.json"
OUT="$HOME/runs/md_t1"
echo "=== $(date) on $(hostname) nproc=$(nproc) ==="
"$MEEP_PY" -c "import meep; print('meep', meep.__version__)" || exit 1
bash -n "$HOME/worker.sh" || { echo "worker.sh SYNTAX ERROR"; exit 1; }
ls -l "$REF" metadit_decoded_full.npz || exit 1
rm -rf "$OUT"; mkdir -p "$OUT/logs"
srun --ntasks=38 --ntasks-per-node=38 --cpus-per-task=1 \
    "$HOME/worker.sh" metadit_decoded_full.npz "$OUT" 0 "$OUT/logs" 2500 \
    --ref-from "$REF"
echo "srun exit=$? end $(date +%T)"
n=$(ls "$OUT"/e_0_*.json 2>/dev/null | wc -l)
echo "design files: $n / 38"
echo "MD_T1 DONE"
